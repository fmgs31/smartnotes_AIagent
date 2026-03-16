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

MCP_SERVICE="smarnotes-mcp"
GATEWAY_SERVICE="smarnotes-gateway"

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

# The gateway's Cloud Run service account needs Firestore access.
# Grant it once (safe to re-run):
#   gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#     --member "serviceAccount:$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
#     --role "roles/datastore.user"

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
# Security is provided by the server's own OAuth 2.1 layer (MCP_ENABLE_OAUTH21).
# --allow-unauthenticated: Cloud Run IAM auth is skipped because it conflicts with
# the GCP bearer token the gateway sends — OAuth 2.1 is the actual auth layer.
gcloud run deploy "$MCP_SERVICE" \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$MCP_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --ingress all \
    --allow-unauthenticated \
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
GOOGLE_OAUTH_CLIENT_SECRET=$GOOGLE_OAUTH_CLIENT_SECRET,\
TOOLS=$TOOLS,\
TOOL_TIER=$TOOL_TIER"

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
cat > "$ENVFILE" << YAML
OPENAI_API_KEY: "${OPENAI_API_KEY}"
APP_API_KEY: "${APP_API_KEY}"
ALLOWED_EMAILS: "${ALLOWED_EMAILS}"
GOOGLE_WORKSPACE_URL: "${MCP_URL}/mcp/"
FIRESTORE_ENABLED: "true"
YAML

gcloud run deploy "$GATEWAY_SERVICE" \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$GATEWAY_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 2 \
    --env-vars-file "$ENVFILE"

rm "$ENVFILE"

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
