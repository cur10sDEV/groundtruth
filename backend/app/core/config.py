from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-prod"
    environment: str = "development"

    # Postgres
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"

    # MinIO (S3)
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "documents"
    s3_region: str = "us-east-1"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "chunks"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400

    # RabbitMQ
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    # LLM routing (LiteLLM). Primary + fallback.
    llm_primary_model: str = "openai/gpt-4o"
    llm_fallback_model: str = "anthropic/claude-3-5-sonnet-20241022"
    llm_api_key_primary: str = ""
    llm_api_key_fallback: str = ""
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-english-v3.0"
    reranker_api_key: str = ""

    # Flagsmith
    flagsmith_api_url: str = "http://localhost:8001/api/v1/"
    flagsmith_server_key: str = ""

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3002"

    # CORS (comma-separated allowed browser origins)
    cors_origins: str = "http://localhost:3000"

    # Auth
    jwt_secret: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Rate limit
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    # Presigned ingestion
    upload_max_bytes: int = 50 * 1024 * 1024
    presign_expiry_seconds: int = 900
    reaper_pending_after_seconds: int = 3600
    cleanup_interval_seconds: int = 300

    # Tokens
    max_input_tokens: int = 8000
    max_output_tokens: int = 2048
    max_context_tokens: int = 4000

    # Optional dedicated guard model (feature-flagged; empty disables)
    guard_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
