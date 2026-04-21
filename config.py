"""
config.py – Centralised configuration via Pydantic BaseSettings.
All secrets are read from environment variables / .env file.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────────────
    app_name: str = "Movie Recommender AI"
    app_version: str = "2.0.0"
    debug: bool = False

    # ── OpenAI ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 512

    # ── Data ───────────────────────────────────────────────────────────
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "saved_models"

    # MovieLens 100K paths (auto-downloaded if missing)
    ratings_file: str = "data/raw/ratings.csv"
    movies_file: str = "data/raw/movies.csv"
    tags_file: str = "data/raw/tags.csv"

    # ── Model Hyperparameters ──────────────────────────────────────────
    # Collaborative Filtering (SVD)
    svd_n_factors: int = 150
    svd_n_epochs: int = 30
    svd_lr_all: float = 0.005
    svd_reg_all: float = 0.02

    # Neural Collaborative Filtering
    ncf_embedding_dim: int = 64
    ncf_hidden_layers: list = [256, 128, 64]
    ncf_dropout: float = 0.3
    ncf_learning_rate: float = 1e-3
    ncf_batch_size: int = 1024
    ncf_epochs: int = 20

    # Content-Based
    tfidf_max_features: int = 15_000
    sentence_model: str = "all-MiniLM-L6-v2"

    # Hybrid weights (must sum to 1.0)
    weight_collaborative: float = 0.45
    weight_content: float = 0.30
    weight_neural: float = 0.25

    # RAG / FAISS
    faiss_index_path: str = "saved_models/faiss.index"
    top_k_retrieval: int = 20

    # ── API ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    rate_limit_per_minute: int = 200

    # ── Redis Cache ────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "movie-recommender"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
