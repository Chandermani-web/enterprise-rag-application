"""Central, typed configuration. Every setting can be overridden via environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _alias(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    # ── Environment ──────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    log_json: bool = False
    data_dir: Path = Path("data")
    state_dir: Path = Path(".state")  # sqlite db for chat memory + ingestion manifest

    # ── LLM providers (ordered fallback chain) ───────────────────────────────
    openrouter_api_key: SecretStr | None = Field(None, validation_alias=_alias("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"))
    openrouter_fallback_api_key: SecretStr | None = Field(
        None, validation_alias=_alias("OPEN_ROUTER_FALLBACK_API_KEY", "OPENROUTER_FALLBACK_API_KEY")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    openrouter_fallback_model: str = ""  # empty -> same as openrouter_model
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    mistral_api_key: SecretStr | None = None
    mistral_model: str = "mistral-small-latest"

    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 1  # per provider (in addition to cross-provider fallback)
    breaker_failure_threshold: int = 3  # consecutive failures before a provider is skipped
    breaker_cooldown_s: float = 60.0

    # ── Embeddings ───────────────────────────────────────────────────────────
    # huggingface: HF Inference API (falls back to the SAME model run locally via fastembed)
    # local:       fastembed/ONNX only      mistral: mistral-embed (1024d, no cross-model fallback)
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    huggingface_api_key: SecretStr | None = None
    mistral_embedding_model: str = "mistral-embed"

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = Field("", validation_alias=_alias("QDRANT_CLUSTER_ENDPOINT", "QDRANT_URL"))
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "enterprise_rag"
    qdrant_cache_collection: str = "enterprise_rag_cache"
    qdrant_in_memory: bool = False  # tests / quick demos without a cluster

    # ── Retrieval ────────────────────────────────────────────────────────────
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 15  # stage 1 (vector search)
    rerank_top_n: int = 5  # stage 2 (cross-encoder) -> context handed to the LLM
    rerank_enabled: bool = True
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    min_relevance_score: float = 0.05  # reranker score below this == "not grounded"
    min_vector_score: float = 0.35  # used when the reranker is unavailable
    max_context_chars: int = 12000

    # ── Web search fallback (Tavily) ─────────────────────────────────────────
    tavily_api_key: SecretStr | None = None
    web_fallback_enabled: bool = True
    web_max_results: int = 4

    # ── Semantic cache ───────────────────────────────────────────────────────
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.93  # cosine similarity
    semantic_cache_ttl_s: int = 3600

    # ── Memory ───────────────────────────────────────────────────────────────
    history_turns: int = 6

    # ── Guardrails ───────────────────────────────────────────────────────────
    max_input_chars: int = 4000
    guardrails_enabled: bool = True
    redact_pii: bool = False  # also mask e-mails / phone numbers before text leaves your network
    domain_description: str = "enterprise technical documentation"

    # ── API security ─────────────────────────────────────────────────────────
    api_auth_key: SecretStr | None = None  # if set, clients must send X-API-Key
    cors_origins: str = "*"
    rate_limit_per_minute: int = 60
    max_upload_mb: int = 50

    # ── Streamlit ────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def _checks(self) -> "Settings":
        if self.embedding_provider not in {"huggingface", "local", "mistral"}:
            raise ValueError("EMBEDDING_PROVIDER must be one of huggingface|local|mistral")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.embedding_provider == "mistral":
            self.embedding_dim = 1024  # mistral-embed is fixed at 1024
        return self

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}

    def secret(self, name: str) -> str | None:
        v = getattr(self, name)
        return v.get_secret_value() if v else None


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.state_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
