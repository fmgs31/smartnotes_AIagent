from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    

    # Servers config
    servers_config_path: str = "servers/servers.json"

    # Planner prompt map: JSON file of the form
    #   { "default_model": "<prefix>",
    #     "models": { "<prefix>": {"full_name": "...", "vendor": "...", "prompt_file": "..."} } }
    # The resolved model is matched against "models" keys via str.startswith; if no entry
    # matches, the prefix under "default_model" is used the same way.
    model_prompt_map_path: str = "app/prompts/model_prompt_map.json"

    # App-level API key (gates access from mobile app)
    app_api_key: str

    # Comma-separated list of allowed emails (takes priority over the JSON file).
    # Set this env var on Cloud Run to add/remove users without rebuilding the image.
    # Example: ALLOWED_EMAILS=alice@gmail.com,bob@gmail.com
    allowed_emails: str = ""

    # Fallback: path to a JSON file listing allowed emails (used locally when
    # ALLOWED_EMAILS env var is not set).
    allowed_users_path: str = "allowed_users.json"

    # MCP client — retry/timeout settings apply to all servers (Cloud Run platform-level)
    mcp_timeout_s: float = 20.0
    mcp_max_retries: int = 3
    mcp_retry_initial_wait_s: float = 1.5
    mcp_retry_max_wait_s: float = 10.0

    # Orchestrator
    max_plan_steps: int = 5

    # Firestore memory — set FIRESTORE_ENABLED=true to enable Drive item persistence.
    # Project auto-detected via ADC on Cloud Run; override with FIRESTORE_PROJECT if needed.
    firestore_enabled: bool = False
    firestore_project: str | None = None


settings = Settings()
