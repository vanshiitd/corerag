"""Application configuration.

Every tunable in CoreRAG lives here as a single, validated, environment-driven
settings object. Nothing else in the codebase should hardcode a model name,
threshold, URL, or secret -- read it from :func:`get_settings` instead.

Values are loaded from environment variables and an optional ``.env`` file (see
``.env.example``). The two API keys default to empty and are enforced as present
by :meth:`Settings._validate_consistency`, so a misconfigured deployment fails
fast at startup rather than at first request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration, validated at instantiation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application --------------------------------------------------------
    app_name: str = "CoreRAG"
    environment: Literal["local", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # ---- API hardening (P6.3) ------------------------------------------------
    # "*" is fine for local dev; the hosted demo sets this to the real deployed
    # Vercel origin via env, not a wildcard, once P6.2 assigns one.
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Applied to /query only (the expensive, LLM-calling endpoint) -- a public
    # demo needs abuse protection, not a metered-product-grade limiter.
    rate_limit_per_minute: int = Field(default=20, ge=1)

    # ---- Corpus / arXiv ingestion (locked: cs.DC + cs.AR + cs.LG) -----------
    arxiv_categories: list[str] = Field(default_factory=lambda: ["cs.DC", "cs.AR", "cs.LG"])
    # cs.LG is filtered on *systems-specific* abstract terms -- deliberately NOT
    # bare "inference" or bare "LLM"/"large language model", which are hopelessly
    # overloaded (e.g. "LLM-Generated Clinical Notes" is a medical NLP-application
    # paper, not an AI-systems one, but matches any bare-"LLM" filter).
    arxiv_cslg_keywords: list[str] = Field(
        default_factory=lambda: [
            "inference serving",
            "model serving",
            "LLM serving",
            "LLM inference",
            "inference latency",
            "inference throughput",
            "KV cache",
            "key-value cache",
            "continuous batching",
            "dynamic batching",
            "speculative decoding",
            "tensor parallelism",
            "model parallelism",
            "quantization",
            "GPU memory",
        ]
    )
    arxiv_max_papers: int = Field(default=150, ge=1)
    arxiv_date_floor: str = "2022-01-01"
    arxiv_pdf_dir: str = "data/raw/pdfs"
    arxiv_metadata_path: str = "data/raw/metadata.jsonl"
    pdf_parser: Literal["grobid", "pymupdf"] = "grobid"  # falls back to pymupdf on error

    # ---- Chunking (locked) --------------------------------------------------
    chunk_size_tokens: int = Field(default=512, ge=1)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    # ---- Contextualization (optional LLM enhancement; A/B arm) ---------------
    contextualize_strategy: Literal["none", "doc_summary", "per_chunk"] = "none"
    contextualize_model: str = "gpt-4o-mini"
    # per_chunk resends the document on every call; OpenAI's TPM rate limit counts
    # full tokens regardless of prompt-cache billing discount, so the document is
    # truncated (front-biased: abstract/intro carry most of the "what is this
    # paper about" signal) to keep one document's calls well under the TPM budget.
    contextualize_doc_token_budget: int = Field(default=3000, ge=1)
    contextualize_concurrency: int = Field(default=4, ge=1)
    # Proactive rate limit: a single large paper's per-chunk calls can approach the
    # account's whole TPM budget (observed live: 200k/min), and consecutive papers
    # compound it -- retry-after-429 alone isn't enough once demand is *structurally*
    # above capacity. Paced below the observed limit to leave headroom.
    contextualize_tokens_per_minute: int = Field(default=150_000, ge=1)

    # ---- Embeddings: local dense + sparse (hybrid, via FastEmbed) -----------
    dense_embed_model: str = "BAAI/bge-base-en-v1.5"  # FastEmbed-native ONNX (no torch)
    dense_embed_dim: int = Field(default=768, ge=1)
    sparse_embed_model: str = "Qdrant/bm25"

    # ---- Reranker: local cross-encoder --------------------------------------
    reranker_model: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    # Default is pytorch-cpu, not cpu-onnx: benchmarked on Apple M5, ONNX Runtime's
    # generic CPU EP was ~1.7-2x SLOWER than PyTorch's native CPU backend for this
    # ModernBERT architecture (across fp32/int8/uint8/quantized variants alike --
    # not a quantization issue), and CoreML EP crashed/hung under the async +
    # asyncio.to_thread pattern this service actually uses. Re-benchmark on the
    # eventual x86 cloud host (P6) -- this may not hold there.
    reranker_mode: Literal["pytorch-cpu", "cpu-onnx", "gpu", "service"] = "pytorch-cpu"
    reranker_onnx_file: str = "onnx/model_int8.onnx"
    reranker_service_url: str | None = None
    reranker_max_length: int = Field(default=512, ge=1)

    # ---- Retrieval: two-stage -----------------------------------------------
    # 30, not 50: P2.4 benchmark showed reranking K=50 costs ~5.9s p50 (true CPU,
    # not accelerated) vs ~3.9s at K=30 -- a meaningful latency win for a modest
    # recall tradeoff, reranking a 6x funnel down to rerank_top_n=5.
    retrieval_k: int = Field(default=30, ge=1)  # hybrid candidates from Qdrant (stage 1)
    rerank_top_n: int = Field(default=5, ge=1)  # survivors after rerank (stage 2)

    # ---- LLM: generation (Groq) ---------------------------------------------
    groq_api_key: SecretStr = SecretStr("")
    generation_model: str = "llama-3.3-70b-versatile"  # confirmed live on Groq 2026-08-03
    generation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=1024, ge=1)

    # ---- LLM: agents -- router + grader (OpenAI) ----------------------------
    # openrouter_api_key is kept as an optional, unenforced fallback (used briefly
    # for google/gemma-4-31b-it:free while OpenAI credits weren't yet purchased).
    openai_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    agent_model: str = "gpt-4o-mini"
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ---- Reflection loop ----------------------------------------------------
    max_reflection_retries: int = Field(default=2, ge=0)
    grader_relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # ---- Semantic cache -----------------------------------------------------
    cache_enabled: bool = True
    cache_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=86_400, ge=0)
    cache_version: str = "v1"  # bump to invalidate cached answers after re-ingest

    # ---- Qdrant -------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "corerag"
    qdrant_collection_version: str = "v1"

    # ---- Redis --------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # ---- GROBID (ingestion only) --------------------------------------------
    grobid_url: str = "http://localhost:8070"

    # ---- Langfuse (observability; optional; Cloud, not self-hosted) ---------
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    # LANGFUSE_BASE_URL, not LANGFUSE_HOST -- confirmed against langfuse/skills
    # docs 2026-08-03; the old field name silently ignored the env var.
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # ---- Derived ------------------------------------------------------------
    @property
    def langfuse_enabled(self) -> bool:
        """True only when both Langfuse keys are configured."""
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    @property
    def qdrant_collection_name(self) -> str:
        """Versioned collection name, so re-ingestion can target a fresh namespace."""
        return f"{self.qdrant_collection}_{self.qdrant_collection_version}"

    # ---- Consistency validation ---------------------------------------------
    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if not self.groq_api_key.get_secret_value():
            raise ValueError("GROQ_API_KEY must be set")
        if not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY must be set")
        if self.rerank_top_n > self.retrieval_k:
            raise ValueError("rerank_top_n cannot exceed retrieval_k")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` singleton."""
    return Settings()
