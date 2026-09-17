import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine default env file location: check local .env first, then /web/.env
_local_env = Path(".env")
_web_env = Path("/web/.env")
_env_file = str(_local_env) if _local_env.is_file() else (str(_web_env) if _web_env.is_file() else ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # MySQL / MariaDB Settings
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "xenforo"
    MYSQL_POOL_SIZE: int = 10
    MYSQL_CHARSET: str = "utf8mb4"
    MYSQL_COLLATION: str = "utf8mb4_unicode_ci"

    # Elasticsearch Settings
    ELASTICSEARCH_HOST: str = "http://127.0.0.1:9200"
    ELASTICSEARCH_USER: str = ""
    ELASTICSEARCH_PASSWORD: str = ""
    ELASTICSEARCH_INDEX: str = "wf_wf"

    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_DIMENSIONS: int = 1536
    OPENAI_MAX_RETRIES: int = 5
    OPENAI_REQUEST_TIMEOUT: float = 30.0

    # Local Embedding Microservice (port 8765)
    USE_LOCAL_EMBEDDING_SERVICE: bool = True
    LOCAL_EMBEDDING_URL: str = "http://127.0.0.1:8765"

    # Redis & Multi-tier Caching Settings (shared with XFAI)
    CACHE_ENABLED: bool = True
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    QUERY_VECTOR_CACHE_TTL: int = 77760000  # 900 days TTL, identical to XFAI
    SEARCH_RESULT_CACHE_TTL: int = 1800  # 30 minutes TTL, identical to XFAI

    # Vector Search Configuration
    VECTOR_BACKEND: Literal["faiss", "mariadb"] = "faiss"
    FAISS_INDEX_PATH: str = "faiss_index.bin"
    FAISS_INDEX_TYPE: Literal["flat", "hnsw"] = "flat"
    FAISS_METRIC: Literal["cosine", "l2"] = "cosine"
    FAISS_HNSW_M: int = 32
    FAISS_HNSW_EF_SEARCH: int = 64
    FAISS_HNSW_EF_CONSTRUCTION: int = 64

    # Hybrid Search Settings
    HYBRID_ALGORITHM: Literal["rrf", "weighted"] = "rrf"
    RRF_K: int = 60
    HYBRID_ALPHA: float = 0.6  # Weight for vector search when using weighted mode

    # API Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"


settings = Settings()
