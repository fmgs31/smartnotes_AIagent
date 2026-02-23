from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    # openai_model: str = "gpt-5.2"
    openai_model: str = "gpt-5-mini-2025-08-07"
    # openai_model: str = "gpt-5-nano-2025-08-07"

    # Servers config
    servers_config_path: str = "servers.json"

    # App-level API key (gates access from mobile app)
    app_api_key: str

    # Path to the JSON file listing allowed user emails
    allowed_users_path: str = "allowed_users.json"

    # Runtime
    mcp_timeout_s: float = 20.0
    max_retries: int = 0
    max_plan_steps: int = 5


settings = Settings()
