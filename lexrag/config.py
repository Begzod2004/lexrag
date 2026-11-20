"""Barcha sozlamalar .env yoki muhit o'zgaruvchilaridan o'qiladi."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # .env bo'lsa yuklaymiz (majburiy emas)
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("LEXRAG_ENV_FILE", ".env"), override=False)
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on", "ha"}


def _int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


def _float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


@dataclass
class LLMEndpoint:
    provider: str  # anthropic | openai | qwen (openai-compatible)
    model: str
    api_key: str | None
    base_url: str | None
    timeout: float = 60.0
    max_tokens: int = 1200

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class Settings:
    # --- yo'llar ---
    data_dir: Path = field(default_factory=lambda: Path(_env("LEXRAG_DATA_DIR", "data")))

    # --- API ---
    api_keys: list[str] = field(default_factory=lambda: [k.strip() for k in (_env("LEXRAG_API_KEYS", "") or "").split(",") if k.strip()])
    rate_limit_per_sec: float = field(default_factory=lambda: _float("LEXRAG_RATE_LIMIT_PER_SEC", 5.0))
    rate_limit_burst: int = field(default_factory=lambda: _int("LEXRAG_RATE_LIMIT_BURST", 10))
    sync_timeout_s: float = field(default_factory=lambda: _float("LEXRAG_SYNC_TIMEOUT_S", 110.0))
    question_timeout_s: float = field(default_factory=lambda: _float("LEXRAG_QUESTION_TIMEOUT_S", 100.0))
    max_parallel_questions: int = field(default_factory=lambda: _int("LEXRAG_MAX_PARALLEL", 10))
    result_ttl_days: int = field(default_factory=lambda: _int("LEXRAG_RESULT_TTL_DAYS", 30))
    answer_cache: bool = field(default_factory=lambda: _bool("LEXRAG_ANSWER_CACHE", True))
    timezone: str = field(default_factory=lambda: _env("LEXRAG_TZ", "Asia/Tashkent"))

    # --- index / retrieval ---
    index_name: str = field(default_factory=lambda: _env("LEXRAG_INDEX", "current"))
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "openai"))  # openai|st|hash|none
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-large"))
    embedding_dims: int = field(default_factory=lambda: _int("EMBEDDING_DIMS", 1024))
    embedding_base_url: str = field(default_factory=lambda: _env("EMBEDDING_BASE_URL", _env("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    embedding_api_key: str | None = field(default_factory=lambda: _env("EMBEDDING_API_KEY", _env("OPENAI_API_KEY")))
    dense_dtype: str = field(default_factory=lambda: _env("DENSE_DTYPE", "float32"))  # float32 (tez) | float16 (kam xotira)
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))  # RRF silliqlash konstantasi
    pool_size: int = field(default_factory=lambda: _int("LEXRAG_POOL_SIZE", 40))
    gen_k: int = field(default_factory=lambda: _int("LEXRAG_GEN_K", 8))
    reranker: str = field(default_factory=lambda: _env("RERANKER", "llm"))  # llm|cross_encoder|none
    cross_encoder_model: str = field(default_factory=lambda: _env("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"))
    query_rewrite: bool = field(default_factory=lambda: _bool("QUERY_REWRITE", True))
    adaptive_contexts: bool = field(default_factory=lambda: _bool("ADAPTIVE_CONTEXTS", True))
    min_contexts: int = field(default_factory=lambda: _int("MIN_CONTEXTS", 3))
    min_context_score: float = field(default_factory=lambda: _float("MIN_CONTEXT_SCORE", 0.35))
    doc_id_mode: str = field(default_factory=lambda: _env("DOC_ID_MODE", "canonical"))  # canonical|by_language

    # --- LLM ---
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = field(default_factory=lambda: _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    claude_answer_model: str = field(default_factory=lambda: _env("CLAUDE_ANSWER_MODEL", "claude-sonnet-5-5"))
    claude_fast_model: str = field(default_factory=lambda: _env("CLAUDE_FAST_MODEL", "claude-haiku-5-5"))
    openai_api_key: str | None = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_answer_model: str = field(default_factory=lambda: _env("OPENAI_ANSWER_MODEL", "gpt-5.6-sol"))
    openai_fast_model: str = field(default_factory=lambda: _env("OPENAI_FAST_MODEL", "gpt-5.6-luna"))
    qwen_api_key: str | None = field(default_factory=lambda: _env("QWEN_API_KEY"))
    qwen_base_url: str | None = field(default_factory=lambda: _env("QWEN_BASE_URL"))
    qwen_model: str = field(default_factory=lambda: _env("QWEN_MODEL", "Qwen/Qwen3-8B-AWQ"))
    llm_timeout_s: float = field(default_factory=lambda: _float("LLM_TIMEOUT_S", 60.0))
    llm_fallback: bool = field(default_factory=lambda: _bool("LLM_FALLBACK", True))

    # --- crawler ---
    crawl_delay_s: float = field(default_factory=lambda: _float("CRAWL_DELAY_S", 0.4))
    crawl_workers: int = field(default_factory=lambda: _int("CRAWL_WORKERS", 3))
    crawl_languages: list[str] = field(default_factory=lambda: (_env("CRAWL_LANGUAGES", "uz-Cyrl,uz-Latn,ru") or "").split(","))

    # ---------- derived ----------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index" / self.index_name

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lexrag.sqlite3"

    def _endpoints(self, role: str) -> list[LLMEndpoint]:
        """role: 'answer' yoki 'fast'. Birinchisi asosiy, qolganlari fallback."""
        eps: list[LLMEndpoint] = []
        claude = None
        if self.anthropic_api_key:
            claude = LLMEndpoint(
                "anthropic",
                self.claude_answer_model if role == "answer" else self.claude_fast_model,
                self.anthropic_api_key,
                self.anthropic_base_url,
                self.llm_timeout_s,
            )
        openai = None
        if self.openai_api_key:
            openai = LLMEndpoint(
                "openai",
                self.openai_answer_model if role == "answer" else self.openai_fast_model,
                self.openai_api_key,
                self.openai_base_url,
                self.llm_timeout_s,
            )
        qwen = None
        if self.qwen_base_url:
            qwen = LLMEndpoint("qwen", self.qwen_model, self.qwen_api_key, self.qwen_base_url, self.llm_timeout_s)
        order = {
            "anthropic": [claude, openai, qwen],
            "openai": [openai, claude, qwen],
            "qwen": [qwen, claude, openai],
            "auto": [claude, openai, qwen],
        }.get(self.llm_provider, [claude, openai, qwen])
        eps = [e for e in order if e is not None]
        if not self.llm_fallback:
            eps = eps[:1]
        return eps

    @property
    def answer_endpoints(self) -> list[LLMEndpoint]:
        return self._endpoints("answer")

    @property
    def fast_endpoints(self) -> list[LLMEndpoint]:
        return self._endpoints("fast")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(s: Settings) -> None:
    global _settings
    _settings = s
