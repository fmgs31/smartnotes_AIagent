# SmarNotes

A FastAPI gateway that turns a spoken or typed instruction into actions on your
Google Drive, Docs and Sheets.

You send free-form text (or an audio clip, transcribed with Whisper). An LLM
planner reads your request plus your personal *guidelines*, then decides one
tool call at a time — search a folder, inspect a document, append a paragraph,
add a spreadsheet row — dispatching each call to a Google Workspace MCP server
and feeding the result back into the next planning step. Progress is streamed
back over Server-Sent Events, and the final summary plus the Drive items that
were touched are persisted per user in Firestore so later requests can refer to
them by name.

---

## Architecture

Two services. The **gateway** (this repo) holds the planner loop and auth; the
**MCP server** (a vendored fork, see [Local setup](#local-setup)) holds the
Google Workspace tools. In production both run on Cloud Run.

```
Client (mobile app)
  ─POST /run or /run-audio─►  app/main.py
                                │
                                ▼
                      Auth:  X-API-Key header  +  Google OAuth token
                             (tokeninfo lookup → email allowlist)
                                │
                                ▼
                          Agent.stream()   ── app/agent.py
                              │
                              ├─ open MCP session(s), discover tools (cached)
                              │
                              └─ loop, up to MAX_PLAN_STEPS:
                                   Planner.plan_next_step()      ← LLM call
                                     → NextStepPlan {next_tool, arguments, done}
                                   Agent._execute_step()
                                     → MCPClient.call_tool()     ← Google Workspace
                                     → Agent._refine_result()    ← one-shot repair
                                     → result appended to planning context
                              │
                              └─ when done=true:
                                   Planner.summarize()           ← 2nd LLM call
                                     → FullTaskSummary {text, items}
                                     → items persisted to Firestore
```

The planner never talks to Google directly and never sees raw credentials — it
only ever emits a tool id and an arguments object.

---

## Repository layout

### `app/` — the gateway

| Module | Role |
| --- | --- |
| [main.py](app/main.py) | FastAPI app, endpoints, singleton wiring |
| [agent.py](app/agent.py) | Orchestrator: planning loop, tool dispatch, loop detection, SSE events |
| [planner.py](app/planner.py) | Builds the planner prompt, calls the model with a strict JSON schema |
| [model_registry.py](app/model_registry.py) | Model key → full name + prompt + vendor client (OpenAI Responses API) |
| [mcp_client.py](app/mcp_client.py) | `fastmcp` client over Streamable HTTP, with retries |
| [server_registry.py](app/server_registry.py) | Loads [servers/servers.json](servers/servers.json); applies allow/block lists |
| [tool_refiners.py](app/tool_refiners.py) | Shims for MCP-server quirks — one automatic retry of a tool call, transparent to the planner |
| [agent_memory.py](app/agent_memory.py) / [agent_memory_firestore.py](app/agent_memory_firestore.py) | Per-user cache of frequently used Drive items |
| [auth.py](app/auth.py) | API-key dependency + Google token validation and email allowlist |
| [schemas.py](app/schemas.py) | Pydantic models: request, planner output, events, final summary |
| [config.py](app/config.py) | Settings from environment / `.env` |
| [error_mapping.py](app/error_mapping.py), [logging_config.py](app/logging_config.py), [utils.py](app/utils.py) | Exception → HTTP mapping, logging setup, small helpers |

### Everything else

| Path | Contents |
| --- | --- |
| `servers/google_workspace_mcp/` | The MCP server. **Not tracked by git** — see [Local setup](#local-setup) |
| [servers/servers.json](servers/servers.json) | MCP server registry + blocked tools |
| [app/prompts/](app/prompts/) | Planner and summarizer prompts — see [Prompts](#prompts) |
| [tests/](tests/) | Unit tests (pytest, no network) |
| [functional_tests/](functional_tests/) | End-to-end tests against a real LLM and a fake Drive |
| [deploy_all.sh](deploy_all.sh) | Builds and deploys both Cloud Run services |
| [access_token_getter.py](access_token_getter.py), [token_utils.py](token_utils.py) | Local OAuth helper — mints and refreshes `gcp_token.json` |
| [send_text_request.py](send_text_request.py), [send_audio_request.py](send_audio_request.py) | Manual smoke-test clients |

---

## Prerequisites

- Python 3.12 (the container image is `python:3.12-slim`). Locally the project
  uses the conda env `llm`: `conda activate llm`.
- A Google Cloud project with the Drive, Docs and Sheets APIs enabled, and an
  OAuth 2.0 client (Desktop type for local use).
- An OpenAI API key.
- For deployment: the `gcloud` CLI, authenticated (`gcloud auth login`).

---

## Local setup

**1. Clone the MCP server into `servers/google_workspace_mcp`.**

It is deliberately excluded by [.gitignore](.gitignore) so the upstream project
stays independently patchable — do not vendor it into this repo's history.

```bash
git clone git@github.com:fmgs31/google_workspace_mcp.git servers/google_workspace_mcp
```

(This is a fork of the `workspace-mcp` project; follow its own README to install
its dependencies.)

**2. Install the gateway's dependencies.**

```bash
conda activate llm
pip install -r requirements-test.txt   # includes requirements.txt + pytest
```

**3. Create a `.env` file** in the repo root — see [Configuration](#configuration).

**4. Start both services.** The MCP server listens on `8000` by default; point
the gateway at it and run the gateway on another port:

```bash
# terminal 1 — MCP server (from servers/google_workspace_mcp)
python main.py --transport streamable-http --tools drive docs sheets

# terminal 2 — gateway
uvicorn app.main:app --port 8001 --reload
```

Check it is up: `curl http://localhost:8001/health` → `{"ok":true}`

**5. Get a Google access token.** Put your OAuth client file at
`client_secret.json`, then:

```bash
python access_token_getter.py
```

This opens a browser, and writes `gcp_token.json`. `token_utils.get_valid_token`
refreshes it automatically from then on.

**6. Fire a request.**

```bash
python send_text_request.py      # types a request
python send_audio_request.py     # records one from the microphone
```

`send_text_request.py` hardcodes the instruction in `TEXT` near the top of the
file; both scripts hardcode `user_guidelines`. Edit them there.
`send_audio_request.py` records until you press Enter and needs `pyaudio`, which
is not in `requirements.txt` — `pip install pyaudio` if you want it.

---

## Configuration

All settings are read from the environment, falling back to a `.env` file in the
repo root. Names are case-insensitive; the fields are defined in
[app/config.py](app/config.py).

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | **yes** | — | Used for both the planner and Whisper transcription |
| `APP_API_KEY` | **yes** | — | Shared secret the client sends as the `X-API-Key` header |
| `ALLOWED_EMAILS` | no | `""` | Comma-separated allowlist of Google accounts. Takes priority over the JSON file |
| `ALLOWED_USERS_PATH` | no | `allowed_users.json` | Fallback allowlist file, shape `{"allowed_emails": [...]}`. Only consulted when `ALLOWED_EMAILS` is empty |
| `GOOGLE_WORKSPACE_URL` | no | `http://localhost:8000/mcp/` | Overrides the `url` of the `gw` server in `servers.json` (via its `url_env_var_name`) |
| `MAX_PLAN_STEPS` | no | `5` | Hard cap on planner iterations per request |
| `MCP_TIMEOUT_S` | no | `20.0` | Per-call MCP timeout |
| `MCP_MAX_RETRIES` | no | `3` | Retries on MCP transport errors |
| `MCP_RETRY_INITIAL_WAIT_S` / `MCP_RETRY_MAX_WAIT_S` | no | `1.5` / `10.0` | Backoff bounds for those retries |
| `FIRESTORE_ENABLED` | no | `false` | Enables per-user Drive-item memory. Off ⇒ no persistence |
| `FIRESTORE_PROJECT` | no | auto (ADC) | Override the Firestore project id |
| `SERVERS_CONFIG_PATH` | no | `servers/servers.json` | MCP server registry location |
| `MODEL_PROMPT_MAP_PATH` | no | `app/prompts/model_prompt_map.json` | Prompt registry location |

---

## API

Both endpoints require the `X-API-Key` header and return
`text/event-stream`. `GET /health` is unauthenticated.

### `POST /run`

JSON body ([`InboundRequest`](app/schemas.py)):

| Field | Required | Meaning |
| --- | --- | --- |
| `text` | yes | The user's instruction |
| `mcp_token` | yes | Google OAuth access token — used both to call Google and to identify/authorize the user |
| `user_guidelines` | no | Free-text personal rules injected into the planner prompt (e.g. "'nueva idea suelta' → create a doc in `Ideas Sueltas`") |
| `planner_model` | no | Model key from the prompt registry; falls back to `default_model` |

```bash
curl -N http://localhost:8001/run \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"List the first four folders","mcp_token":"ya29...."}'
```

### `POST /run-audio`

`multipart/form-data`: an `audio` file part plus the form fields `mcp_token`,
`user_guidelines` and `planner_model`. The clip is transcribed with `whisper-1`
and then handled exactly like `/run`.

### SSE events

Each line is `data: <json>`. The `type` field is one of
[`EventType`](app/schemas.py):

| `type` | Payload | When |
| --- | --- | --- |
| `transcript` | `{text}` | First event of `/run-audio` only |
| `step_planned` | `{next_tool, arguments, next_tool_explanation}` | Once per planning step |
| `result` | `{text, items}` | Final summary. `items` are the Drive files/folders touched |
| `error` | `{detail}` | Terminates the stream |

---

## Tools and safety

Tools are exposed by the MCP server and addressed by the planner as
`{server_id}.{tool_name}` — with the single registered server `gw`, that means
`gw.search_drive_files`, `gw.create_doc`, `gw.modify_sheet_values`, and so on.
Sources: `gdrive/drive_tools.py`, `gdocs/docs_tools.py`, `gsheets/sheets_tools.py`
inside the MCP server.

**Blocked tools.** [servers/servers.json](servers/servers.json) removes these
from the planner's menu entirely:

`import_to_google_doc`, `create_drive_file`, `find_and_replace_doc`,
`update_drive_file`, `manage_drive_access`, `remove_drive_permission`,
`set_drive_file_permissions`, `update_drive_permission`, `batch_update_doc`

The rationale is twofold: nothing may change *sharing or permissions*, and
document creation has exactly one sanctioned path — `create_doc`, which produces
a native Google Doc in place. `create_drive_file` is blocked because it creates
plain-text files that then need conversion; use `create_doc` for documents and
`create_spreadsheet` for sheets.

**Runaway guards.**

- `MAX_PLAN_STEPS` caps the loop.
- Loop detection signs each step as `(tool_id, arguments)`. A repeated
  *mutating* call is rejected anywhere in the session; a repeated *read-only*
  call is only rejected if it immediately follows an identical one.

---

## Tool refiners

[app/tool_refiners.py](app/tool_refiners.py) is a post-call interception layer
for MCP-server quirks. After every tool call, `Agent._refine_result` offers the
result to each refiner in turn; the first one that applies gets **exactly one**
follow-up call to the same tool, and that follow-up's result replaces the
original. The planner never sees the broken intermediate — from its point of
view the tool simply returned the right answer. No extra planning step, no extra
LLM round-trip, no prompt rule.

Failed calls reach refiners too: `_call_tool` catches the exception, feeds
`[Tool error] …` through the same pass, and re-raises untouched if nothing
applied.

The alternative was to teach the planner to recover from these itself — index
arithmetic plus multi-step retries, in a nano model. It demonstrably could not:
it looped until `MAX_PLAN_STEPS` and the request died.

### `TabAwareInspectRefiner` — `total_length: 0` on every document

All Google Docs have been tab-based since October 2024. The MCP server fetches
with `includeTabsContent=True`, which per Google's documented contract leaves
the legacy top-level `body` empty — content lives in
`tabs[].documentTab.body`. But `inspect_doc_structure` still analyses that empty
body when no `tab_id` is passed, so it reports `total_length: 0` next to a
populated `tabs` list, for *every* document, single-tab ones included.

The planner reads `0`, writes at index 0, the Docs API rejects it (index 0 sits
before the first section break), the planner re-inspects, gets `0` again — an
unbreakable loop.

Repair: re-inspect with the first `tab_id` parsed out of the result itself,
recursing into `child_tabs`. Never hardcodes `t.0` — tab ids are not guaranteed
to follow that form.

### `InsertIndexBoundsRefiner` — `total_length` is not a valid insertion index

`total_length` is the segment *end* index, and the Docs API requires an
insertion index strictly below it. So the natural append
(`start_index = total_length`) fails 100% of the time with
`Index 65004 must be less than the end index of the referenced segment, 65004`.

Repair: match that error and retry at the bound the API itself reported, minus
one — no local arithmetic, no assumptions about document length. Insertions
only: the refiner bails when `end_index` is present, so it can never turn a
replacement into an overwrite.

### Two properties worth keeping

**Single pass, structurally.** The ABC contract is that refined arguments must
not re-trigger `applies`. `TabAwareInspectRefiner` adds a `tab_id` and its
`applies` bails when one is set; `InsertIndexBoundsRefiner` checks
`start_index != last_valid`. This cannot become the retry loop it was built to
eliminate. [tests/test_tool_refiners.py](tests/test_tool_refiners.py) pins that
contract.

**Self-retiring.** Each refiner keys off a *symptom*, not a workaround flag.
When the server-side fixes land, the symptoms stop appearing, `applies` returns
`False` forever, and the classes can be deleted with nothing else to unwind.

---

## Prompts

[app/prompts/model_prompt_map.json](app/prompts/model_prompt_map.json) is the
single source of truth for which model runs with which prompt:

```json
{
  "default_model": "gpt-5-nano",
  "models": {
    "gpt-5-nano": {
      "full_name": "gpt-5-nano-2025-08-07",
      "vendor": "open_ai",
      "prompt_file": "planner/v14.txt",
      "summarizer_prompt_file": "summarizer/v13.txt"
    }
  }
}
```

- **Add a model**: add an entry. The key is what a client sends as
  `planner_model`.
- **Add a prompt version**: drop a new file in [app/prompts/planner/](app/prompts/planner/)
  or [app/prompts/summarizer/](app/prompts/summarizer/) and repoint the entry.
- If a model has no `summarizer_prompt_file`, the planner emits the final
  summary itself in the same call instead of a second one.

The `app/prompts/v1/ … v14/` directories are an **archive** of superseded
prompts. They are not loaded; only the files named in the JSON are.

---

## Testing

### Unit tests

Fast, hermetic, no network. [pytest.ini](pytest.ini) limits collection to
`tests/`.

```bash
conda activate llm
pytest
```

### Functional tests

Real OpenAI calls against an **in-memory fake Drive** — no Google credentials
needed and nothing is written to disk, but each run costs tokens. The fake Drive
is built from the file tree in `functional_tests/drive/` and deep-copied per
test, so cases are isolated.

```bash
python -m functional_tests.run --list             # show the 12 cases
python -m functional_tests.run                    # run all
python -m functional_tests.run --test append_terapia -v
python -m functional_tests.run --model gpt-5.4-nano
```

Coverage includes the guideline triggers (append to a therapy doc, file a loose
idea into the right subfolder, add a board-game idea to a spreadsheet), safety
cases (a deletion request must be refused; a full-content replacement must not
destroy the original), the memory chain, and the tab-aware document refiner.

These are never collected by `pytest` — `testpaths` excludes them on purpose, so
a normal test run stays free and offline.

---

## Deployment

[deploy_all.sh](deploy_all.sh) builds and deploys both Cloud Run services with
Cloud Build. Load your `.env` into the shell first:

```bash
set -a && source .env && set +a
./deploy_all.sh
```

The script hard-requires `GCP_PROJECT_ID`, `GCP_REGION`,
`GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `OPENAI_API_KEY` and
`APP_API_KEY`, and exits immediately if any is unset. `ALLOWED_EMAILS` and
`TOOLS` have defaults you will want to override.

What it does:

1. Enables the required GCP APIs and ensures the Artifact Registry repo exists.
2. Builds and deploys `smarnotes-mcp` **in two phases** — the server needs
   `WORKSPACE_MCP_BASE_URI` set to its own Cloud Run URL for the OAuth 2.1
   handshake to work behind the HTTPS proxy, and that URL does not exist until
   after the first deploy.
3. Builds and deploys `smarnotes-gateway`, pointing `GOOGLE_WORKSPACE_URL` at
   the MCP service and enabling Firestore.

Both services are deployed `--allow-unauthenticated`: Cloud Run IAM auth would
collide with the Google bearer token the gateway forwards, so the real auth
layers are the MCP server's own OAuth 2.1 and the gateway's API key + email
allowlist.

**One-time IAM grant.** The gateway's service account needs Firestore access.
The exact command is in a comment at the top of
[deploy_all.sh](deploy_all.sh) — run it once per project
(`roles/datastore.user` on the compute service account).

Adding or removing users needs no rebuild: update the `ALLOWED_EMAILS` env var
on the gateway service.

---

## Secrets

None of the following are tracked, and none should ever be committed:

| File | What it is |
| --- | --- |
| `.env` | API keys and deployment variables |
| `client_secret.json` | Google OAuth client credentials |
| `gcp_token.json` | Your cached access/refresh token |
| `allowed_users.json` | Local email allowlist |
| `servers/google_workspace_mcp/` | The vendored MCP server (separate repo) |
