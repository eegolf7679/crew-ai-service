from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth — shared bearer token used by callers (CIO KB)
    SHARED_TOKEN: str = ""

    # LLM (configured later by the consumer; both styles supported)
    LOVABLE_API_KEY: str = ""
    LOVABLE_AI_BASE_URL: str = "https://ai.gateway.lovable.dev/v1"
    DEFAULT_MODEL: str = "google/gemini-2.5-flash"

    # Optional fallback OpenAI-compatible key (used by litellm if set)
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""

    # Vectara (kb_search tool)
    VECTARA_API_KEY: str = ""
    VECTARA_CUSTOMER_ID: str = ""
    VECTARA_CORPUS_KEY: str = ""
    VECTARA_BASE_URL: str = "https://api.vectara.io"

    # Web search (web_search tool) — pick whichever provider is set
    TAVILY_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    BRAVE_API_KEY: str = ""

    # Generic HTTP tool tuning
    HTTP_CALL_MAX_BYTES: int = 8192
    HTTP_CALL_TIMEOUT_S: float = 20.0

    # Server
    PORT: int = 8000
    LOG_LEVEL: str = "info"


settings = Settings()