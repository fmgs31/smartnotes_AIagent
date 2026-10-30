#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Build and deploy both Cloud Run services
#
# Prerequisites:
#   gcloud CLI installed and authenticated  (gcloud auth login)
#   Docker installed (used by Cloud Build)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── LOAD FROM ENV ─────────────────────────────────────────────────────────────
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${GCP_REGION:?GCP_REGION is required}"
: "${GOOGLE_OAUTH_CLIENT_ID:?GOOGLE_OAUTH_CLIENT_ID is required}"
: "${GOOGLE_OAUTH_CLIENT_SECRET:?GOOGLE_OAUTH_CLIENT_SECRET is required}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${APP_API_KEY:?APP_API_KEY is required}"

PROJECT_ID="$GCP_PROJECT_ID"
REGION="$GCP_REGION"

MCP_SERVICE="smartnotes-mcp"
GATEWAY_SERVICE="smartnotes-gateway"

# One least-privilege runtime identity per service (instead of the default compute
# SA, which has project Editor).
MCP_SA="$MCP_SERVICE@$PROJECT_ID.iam.gserviceaccount.com"
GATEWAY_SA="$GATEWAY_SERVICE@$PROJECT_ID.iam.gserviceaccount.com"

# Secret Manager names
SECRET_OPENAI="openai-api-key"
SECRET_APP_API_KEY="app-api-key"
SECRET_OAUTH_CLIENT="google-oauth-client-secret"

# Comma-separated list of allowed Google account emails.
# Update this and re-run only the gateway deploy section to add/remove users.
ALLOWED_EMAILS="${ALLOWED_EMAILS:-email_0@gmail.com,email_1@gmail.com,email_2@gmail.com}"

# Google Workspace tools to enable (space-separated)
TOOLS="${TOOLS:-drive docs sheets}"
TOOL_TIER="complete"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    firestore.googleapis.com \
    secretmanager.googleapis.com \
    iam.googleapis.com \
    --project "$PROJECT_ID"

echo "==> Ensuring Artifact Registry repository exists in $REGION..."
gcloud artifacts repositories create cloud-run-source-deploy \
    --repository-format docker \
    --location "$REGION" \
    --project "$PROJECT_ID" 2>/dev/null || true

echo "==> Setting Artifact Registry cleanup policy (delete untagged images after 1 day)..."
POLICYFILE=$(mktemp)
echo '[{"name":"delete-untagged","action":{"type":"Delete"},"condition":{"tagState":"UNTAGGED","olderThan":"86400s"}}]' > "$POLICYFILE"
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
    --location "$REGION" \
    --project "$PROJECT_ID" \
    --policy "$POLICYFILE"
rm "$POLICYFILE"

echo "==> Ensuring runtime service accounts exist..."
for SA_NAME in "$MCP_SERVICE" "$GATEWAY_SERVICE"; do
    gcloud iam service-accounts describe "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --project "$PROJECT_ID" >/dev/null 2>&1 \
    || { gcloud iam service-accounts create "$SA_NAME" \
            --display-name "$SA_NAME runtime" \
            --project "$PROJECT_ID" \
         && sleep 15; }  # new SAs take a few seconds before they can be granted roles
done

echo "==> Granting the gateway Firestore access..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$GATEWAY_SA" \
    --role "roles/datastore.user" \
    --condition None >/dev/null

# Creates the secret on first run; afterwards adds a version only if the value
# changed, so redeploys don't pile up billable versions. Then grants $3 access.
upsert_secret() {
    local name="$1" value="$2" accessor="$3"
    if ! gcloud secrets describe "$name" --project "$PROJECT_ID" >/dev/null 2>&1; then
        echo "    creating $name"
        printf '%s' "$value" | gcloud secrets create "$name" \
            --replication-policy automatic --data-file - --project "$PROJECT_ID"
    elif [ "$(gcloud secrets versions access latest --secret "$name" --project "$PROJECT_ID")" != "$value" ]; then
        echo "    $name changed, adding new version"
        printf '%s' "$value" | gcloud secrets versions add "$name" \
            --data-file - --project "$PROJECT_ID"
    fi
    gcloud secrets add-iam-policy-binding "$name" \
        --member "serviceAccount:$accessor" \
        --role "roles/secretmanager.secretAccessor" \
        --project "$PROJECT_ID" >/dev/null
}

# Destroys every non-destroyed version except the newest. Superseded values are
# useless (services read :latest) and still count toward the free tier.
prune_secret_versions() {
    local name="$1"
    gcloud secrets versions list "$name" \
        --filter "state!=DESTROYED" --sort-by "~createTime" \
        --format "value(name.basename())" --project "$PROJECT_ID" \
    | tail -n +2 \
    | while read -r version; do
        echo "    destroying $name version $version"
        gcloud secrets versions destroy "$version" --secret "$name" \
            --project "$PROJECT_ID" --quiet >/dev/null
    done
}

echo "==> Syncing secrets to Secret Manager..."
upsert_secret "$SECRET_OAUTH_CLIENT" "$GOOGLE_OAUTH_CLIENT_SECRET" "$MCP_SA"
upsert_secret "$SECRET_OPENAI"       "$OPENAI_API_KEY"             "$GATEWAY_SA"
upsert_secret "$SECRET_APP_API_KEY"  "$APP_API_KEY"                "$GATEWAY_SA"

# ── 1. Build and deploy the MCP server ───────────────────────────────────────
echo ""
echo "==> Building MCP server image..."
gcloud builds submit "servers/google_workspace_mcp" \
    --tag "$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$MCP_SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID"

echo "==> Deploying MCP server (phase 1 — without base URI, URL not yet known)..."
# --ingress all: publicly reachable. Cloud Run-to-Cloud Run traffic exits via the
# public internet even within the same project, so --ingress internal blocks it.
# --no-allow-unauthenticated: only principals with roles/run.invoker (the gateway
# SA, granted below) get through. The gateway sends its IAM ID token in
# X-Serverless-Authorization, so Authorization still carries the user's Google
# token to the server's own OAuth 2.1 layer (MCP_ENABLE_OAUTH21).
gcloud run deploy "$MCP_SERVICE" \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$MCP_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$MCP_SA" \
    --ingress all \
    --no-allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 2 \
    --set-env-vars "\
WORKSPACE_MCP_STATELESS_MODE=true,\
MCP_ENABLE_OAUTH21=true,\
EXTERNAL_OAUTH21_PROVIDER=true,\
GOOGLE_OAUTH_CLIENT_ID=$GOOGLE_OAUTH_CLIENT_ID,\
TOOLS=$TOOLS,\
TOOL_TIER=$TOOL_TIER" \
    --set-secrets "GOOGLE_OAUTH_CLIENT_SECRET=$SECRET_OAUTH_CLIENT:latest"

# Belt and braces: make sure no leftover public binding survives from the old
# --allow-unauthenticated deploys.
gcloud run services remove-iam-policy-binding "$MCP_SERVICE" \
    --member allUsers --role roles/run.invoker \
    --region "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1 || true

echo "==> Allowing the gateway to invoke the MCP server..."
gcloud run services add-iam-policy-binding "$MCP_SERVICE" \
    --member "serviceAccount:$GATEWAY_SA" \
    --role roles/run.invoker \
    --region "$REGION" --project "$PROJECT_ID" >/dev/null

MCP_URL=$(gcloud run services describe "$MCP_SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format "value(status.url)")

echo "MCP service URL: $MCP_URL"

echo "==> Updating MCP server (phase 2 — setting base URI to its own Cloud Run URL)..."
# Required for correct behaviour behind Cloud Run's HTTPS reverse proxy.
# Without this, fastmcp's OAuth 2.1 handshake fails during MCP initialize.
gcloud run services update "$MCP_SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --update-env-vars "WORKSPACE_EXTERNAL_URL=$MCP_URL,WORKSPACE_MCP_BASE_URI=$MCP_URL"

# ── 2. Build and deploy the gateway ──────────────────────────────────────────
echo ""
echo "==> Building gateway image..."
gcloud builds submit . \
    --tag "$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$GATEWAY_SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID"

echo "==> Deploying gateway..."
# Write env vars to a temp YAML file — --set-env-vars can't handle @ or , inline.
ENVFILE=$(mktemp)
# Secrets are not in this file: they come from Secret Manager via --set-secrets.
cat > "$ENVFILE" << YAML
ALLOWED_EMAILS: "${ALLOWED_EMAILS}"
GOOGLE_WORKSPACE_URL: "${MCP_URL}/mcp/"
FIRESTORE_ENABLED: "true"
MCP_IAM_AUTH: "true"
YAML

# --allow-unauthenticated: the gateway is called from the mobile client, which
# can't mint Cloud Run ID tokens. Auth is the API key + Google token allowlist.
gcloud run deploy "$GATEWAY_SERVICE" \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$GATEWAY_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$GATEWAY_SA" \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 2 \
    --env-vars-file "$ENVFILE" \
    --set-secrets "OPENAI_API_KEY=$SECRET_OPENAI:latest,APP_API_KEY=$SECRET_APP_API_KEY:latest"

rm "$ENVFILE"

# Only after both services run on the new :latest — older revisions may still
# have been starting instances from the previous versions until now.
echo "==> Pruning superseded secret versions..."
for name in "$SECRET_OAUTH_CLIENT" "$SECRET_OPENAI" "$SECRET_APP_API_KEY"; do
    prune_secret_versions "$name"
done

GATEWAY_URL=$(gcloud run services describe "$GATEWAY_SERVICE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format "value(status.url)")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Deployment complete."
echo ""
echo "  MCP server:  $MCP_URL"
echo "  Gateway:     $GATEWAY_URL"
echo ""
echo "Mobile app endpoints:"
echo "  Text:  POST $GATEWAY_URL/run"
echo "  Audio: POST $GATEWAY_URL/run-audio"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
