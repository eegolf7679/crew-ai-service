from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth — shared bearer token used by callers (CIO KB)
    SHARED_TOKEN: str = ""

    # LLM (configured later by the consumer; both styles supported)
    LOVABLE_API_KEY: str = ""
    LOVABLE_AI_BASE_URL: str = "https://ai.gateway.lovable.dev/v1"
    DEFAULT_MODEL: str = "google/gemini-2.5-flash"

    # Server
    PORT: int = 8000
    LOG_LEVEL: str = "info"


settings = Settings()