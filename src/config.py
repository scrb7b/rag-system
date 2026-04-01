from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent / ".env"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "ollama"] = "ollama"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    sparse_embed_model: str = "Qdrant/bm25"
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    qdrant_location: str = ":memory:"
    qdrant_collection: str = "knowledge_base"

    top_k: int = Field(default=10, ge=1)
    rerank_candidates: int = Field(default=20, ge=1)
    enable_reranking: bool = False

    score_threshold: float = 0.0

    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=1024, ge=128)

    log_level: str = "INFO"
    log_json: bool = False


settings = Config()
