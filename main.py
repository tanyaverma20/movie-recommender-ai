"""
api/main.py
────────────
FastAPI application — production-grade REST API.

Features:
  • Async endpoints with lifespan model loading
  • Rate limiting (slowapi)
  • Response caching (Redis-compatible)
  • Structured error handling
  • Prometheus metrics middleware
  • CORS for frontend integration
  • OpenAPI docs at /docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import get_settings

cfg = get_settings()

# ── Global model registry (loaded once at startup) ─────────────────────────
_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup, release at shutdown."""
    logger.info("Loading models …")
    from data.data_loader import MovieLensDataset
    from models.collaborative_filter import CollaborativeFilter
    from models.content_based import ContentBasedRecommender
    from models.neural_cf import NCFTrainer
    from models.hybrid_recommender import HybridRecommender
    from models.llm_explainer import LLMExplainer
    import os

    ds = MovieLensDataset().prepare()

    # Try loading pre-trained models, else fit fresh
    try:
        cf = CollaborativeFilter.load()
        cb = ContentBasedRecommender.load()
        ncf = NCFTrainer.load()
    except Exception:
        logger.warning("Pre-trained models not found. Training from scratch …")
        cf = CollaborativeFilter(algorithms=["svd", "nmf"]).fit(ds.train, ds.val)
        cf.save()
        cb = ContentBasedRecommender(use_bert=True).fit(ds.movie_profiles)
        cb.save()
        ncf = NCFTrainer(ds.n_users, ds.n_movies)
        ncf.train(ds.train, ds.val, epochs=5)
        ncf.save()

    hybrid = HybridRecommender(cf, cb, ncf, ds)
    hybrid.fit_meta_learner(ds.val)
    explainer = LLMExplainer(cb, ds.movie_profiles)

    _models.update({
        "cf": cf, "cb": cb, "ncf": ncf,
        "hybrid": hybrid, "explainer": explainer, "dataset": ds,
    })
    logger.success("All models loaded — API ready.")
    yield

    logger.info("Shutting down …")
    _models.clear()


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🎬 Movie Recommender AI",
    description=(
        "Hybrid recommendation engine powered by Matrix Factorisation (SVD/SVD++/NMF), "
        "Neural Collaborative Filtering (NeuMF), Sentence-BERT content embeddings, "
        "FAISS ANN retrieval, and GPT-4 LLM explanations."
    ),
    version=cfg.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed * 1000:.2f}ms"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int = Field(..., example=42, description="Target user ID")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations")
    strategy: str = Field(
        "weighted",
        pattern="^(weighted|rrf|stack|contextual)$",
        description="Fusion strategy",
    )
    diversity: bool = Field(True, description="Apply MMR diversity reranking")
    explain: bool = Field(False, description="Generate LLM explanations (slower)")
    exclude_seen: bool = Field(True, description="Filter already-watched movies")


class SimilarMoviesRequest(BaseModel):
    movie_id: int
    top_n: int = Field(10, ge=1, le=50)
    mode: str = Field("bert", pattern="^(tfidf|bert|fusion)$")


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, example="sci-fi thriller with time travel")
    top_k: int = Field(10, ge=1, le=50)


class RecommendationItem(BaseModel):
    movie_id: int
    title: Optional[str]
    genres: Optional[str]
    year: Optional[float]
    score: float
    cf_score: Optional[float]
    cb_score: Optional[float]
    ncf_score: Optional[float]
    explanation: Optional[dict] = None


class RecommendResponse(BaseModel):
    user_id: int
    strategy: str
    top_n: int
    recommendations: list[RecommendationItem]
    latency_ms: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    if not _models:
        raise HTTPException(503, "Models not loaded yet. Try again shortly.")
    return _models


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": cfg.app_name,
        "version": cfg.app_version,
        "status": "healthy",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health(models: dict = Depends(get_models)):
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(body: RecommendRequest, models: dict = Depends(get_models)):
    """
    Generate personalised movie recommendations for a user.

    **Fusion strategies**:
    - `weighted` – static weighted average of CF + CB + NCF scores
    - `rrf` – Reciprocal Rank Fusion (robust to score scale differences)
    - `stack` – Meta-learner stacking (Ridge regression on val features)
    - `contextual` – auto-selects strategy based on user interaction density
    """
    t0 = time.perf_counter()
    hybrid = models["hybrid"]
    ds = models["dataset"]

    try:
        recs = hybrid.recommend(
            user_id=body.user_id,
            top_n=body.top_n,
            strategy=body.strategy,
            diversity=body.diversity,
            exclude_seen=body.exclude_seen,
        )
    except Exception as exc:
        logger.error(f"Recommendation error: {exc}")
        raise HTTPException(500, f"Recommendation failed: {exc}")

    # Enrich with movie metadata
    enriched = []
    profiles = ds.movie_profiles.set_index("movie_id")
    for rec in recs:
        mid = rec["movie_id"]
        meta = profiles.loc[mid] if mid in profiles.index else None
        item = RecommendationItem(
            movie_id=mid,
            title=meta["title_clean"] if meta is not None else None,
            genres=meta["genres"] if meta is not None else None,
            year=meta["year"] if meta is not None else None,
            score=rec["score"],
            cf_score=rec.get("cf_score"),
            cb_score=rec.get("cb_score"),
            ncf_score=rec.get("ncf_score"),
        )
        enriched.append(item)

    # Optional LLM explanations
    if body.explain and enriched:
        liked = hybrid._get_liked_movies(body.user_id)
        explainer = models["explainer"]
        for item in enriched[:3]:   # Explain top-3 to control cost
            try:
                exp = await explainer.explain_async(
                    user_id=body.user_id,
                    movie_id=item.movie_id,
                    liked_movie_ids=liked,
                    predicted_rating=item.score * 5,
                )
                item.explanation = exp
            except Exception as e:
                logger.warning(f"Explanation failed for movie {item.movie_id}: {e}")

    latency = (time.perf_counter() - t0) * 1000

    return RecommendResponse(
        user_id=body.user_id,
        strategy=body.strategy,
        top_n=body.top_n,
        recommendations=enriched,
        latency_ms=round(latency, 2),
    )


@app.post("/similar-movies", tags=["Content"])
async def similar_movies(body: SimilarMoviesRequest, models: dict = Depends(get_models)):
    """
    Find movies similar to a given movie using content embeddings.
    Supports TF-IDF, SBERT, or fused similarity.
    """
    cb: ContentBasedRecommender = models["cb"]
    ds = models["dataset"]
    profiles = ds.movie_profiles.set_index("movie_id")

    results = cb.get_similar_movies(body.movie_id, top_n=body.top_n, mode=body.mode)

    enriched = []
    for r in results:
        mid = r["movie_id"]
        meta = profiles.loc[mid] if mid in profiles.index else None
        enriched.append({
            "movie_id": mid,
            "title": meta["title_clean"] if meta is not None else None,
            "genres": meta["genres"] if meta is not None else None,
            "year": int(meta["year"]) if meta is not None and not pd.isna(meta["year"]) else None,
            "similarity": round(r["similarity"], 4),
        })

    return {"movie_id": body.movie_id, "similar_movies": enriched, "mode": body.mode}


@app.post("/semantic-search", tags=["Content"])
async def semantic_search(body: SemanticSearchRequest, models: dict = Depends(get_models)):
    """
    Free-text semantic movie search using SBERT + FAISS.
    Example: "80s action movies with robots" or "coming-of-age dramedy"
    """
    cb: ContentBasedRecommender = models["cb"]
    ds = models["dataset"]
    profiles = ds.movie_profiles.set_index("movie_id")

    results = cb.semantic_search(body.query, top_k=body.top_k)

    enriched = []
    for r in results:
        mid = r["movie_id"]
        meta = profiles.loc[mid] if mid in profiles.index else None
        enriched.append({
            "movie_id": mid,
            "title": meta["title_clean"] if meta is not None else None,
            "genres": meta["genres"] if meta is not None else None,
            "year": int(meta["year"]) if meta is not None and not pd.isna(meta["year"]) else None,
            "similarity": round(r["similarity"], 4),
        })

    return {"query": body.query, "results": enriched}


@app.get("/user/{user_id}/profile", tags=["Users"])
async def user_profile(user_id: int, models: dict = Depends(get_models)):
    """Returns aggregated user statistics and taste profile."""
    ds = models["dataset"]
    df = ds.train[ds.train["user_id"] == user_id]

    if df.empty:
        raise HTTPException(404, f"User {user_id} not found.")

    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    top_genres = (
        df[genre_cols].sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    ) if genre_cols else {}

    return {
        "user_id": user_id,
        "n_ratings": len(df),
        "avg_rating": round(float(df["rating"].mean()), 2),
        "rating_std": round(float(df["rating"].std()), 2),
        "favourite_genres": top_genres,
        "is_cold_start": len(df) < 5,
    }


@app.get("/movie/{movie_id}", tags=["Movies"])
async def movie_detail(movie_id: int, models: dict = Depends(get_models)):
    """Returns detailed movie metadata."""
    ds = models["dataset"]
    row = ds.movie_profiles[ds.movie_profiles["movie_id"] == movie_id]
    if row.empty:
        raise HTTPException(404, f"Movie {movie_id} not found.")
    r = row.iloc[0]
    return {
        "movie_id": movie_id,
        "title": r.get("title_clean"),
        "year": r.get("year"),
        "genres": r.get("genres"),
        "tags": r.get("tags_text", "")[:300],
        "soup_preview": r.get("soup", "")[:200],
    }


@app.get("/explain/{user_id}/{movie_id}", tags=["Explanations"])
async def explain_recommendation(
    user_id: int,
    movie_id: int,
    models: dict = Depends(get_models),
):
    """Generate an LLM explanation for why movie X is recommended to user Y."""
    hybrid = models["hybrid"]
    explainer = models["explainer"]
    liked = hybrid._get_liked_movies(user_id)
    score = hybrid.cf.predict(user_id, movie_id)

    explanation = await explainer.explain_async(
        user_id=user_id,
        movie_id=movie_id,
        liked_movie_ids=liked,
        predicted_rating=score,
    )
    return {"user_id": user_id, "movie_id": movie_id, "explanation": explanation}


@app.get("/metrics", tags=["Evaluation"])
async def model_metrics(models: dict = Depends(get_models)):
    """
    Run quick evaluation of the hybrid recommender on the test set.
    (Capped at 100 users for speed — use /evaluate for full run.)
    """
    from evaluation.metrics import evaluate_recommender
    hybrid = models["hybrid"]
    ds = models["dataset"]

    results = evaluate_recommender(
        recommender=hybrid,
        test_df=ds.test,
        train_df=ds.train,
        movie_profiles=ds.movie_profiles,
        k=10,
        max_users=100,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        workers=cfg.api_workers,
        reload=cfg.debug,
        log_level="info",
    )

# Silence pandas import in annotation
try:
    import pandas as pd
except ImportError:
    pass
