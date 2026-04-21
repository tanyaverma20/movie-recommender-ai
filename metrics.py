"""
evaluation/metrics.py
──────────────────────
Comprehensive recommendation system evaluation suite.

Metrics Implemented:
  Rating Prediction:
    • RMSE – Root Mean Squared Error
    • MAE  – Mean Absolute Error

  Ranking Quality (top-N):
    • Precision@K  – fraction of top-K that are relevant
    • Recall@K     – fraction of relevant items in top-K
    • F1@K
    • NDCG@K       – Normalised Discounted Cumulative Gain (position-aware)
    • MRR          – Mean Reciprocal Rank
    • Hit Rate@K   – at least one relevant item in top-K

  Coverage & Diversity:
    • Catalogue Coverage     – fraction of items ever recommended
    • Intra-List Diversity   – avg pairwise distance within recommendation list
    • Novelty                – avg self-information of recommended items (surprisal)
    • Serendipity            – unexpected & relevant (beyond obvious popular items)

Usage:
    from evaluation.metrics import evaluate_recommender
    results = evaluate_recommender(recommender, test_df, k=10)
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# Rating Prediction Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Ranking Metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommended items that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items that appear in top-K."""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.
    Rewards placing relevant items higher in the ranked list.

    NDCG = DCG / IDCG
    DCG  = Σ rel_i / log2(i+2)  for i in 0..K-1
    IDCG = DCG of a perfect ranking
    """
    top_k = recommended[:k]

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )

    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(recommended: list, relevant: set) -> float:
    """
    MRR: reciprocal rank of the first hit.
    Useful when the user cares only about the first relevant item.
    """
    for rank, item in enumerate(recommended, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """1 if at least one relevant item is in top-K, else 0."""
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Coverage & Diversity Metrics
# ─────────────────────────────────────────────────────────────────────────────

def catalogue_coverage(all_recommendations: list[list], total_items: int) -> float:
    """
    Fraction of all items in the catalogue that appear in any recommendation list.
    Low coverage → popularity bias (model only recommends blockbusters).
    """
    recommended_items = {item for recs in all_recommendations for item in recs}
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def intra_list_diversity(
    recommended: list[int],
    item_embeddings: dict[int, np.ndarray],
) -> float:
    """
    Average pairwise cosine distance between recommended items' embeddings.
    High ILD → diverse recommendation list (less echo chamber).
    """
    vecs = [item_embeddings[m] for m in recommended if m in item_embeddings]
    if len(vecs) < 2:
        return 0.0

    distances = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos_sim = np.dot(vecs[i], vecs[j]) / (
                np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-10
            )
            distances.append(1.0 - cos_sim)  # cosine distance

    return float(np.mean(distances))


def novelty(
    recommended: list[int],
    item_popularity: dict[int, float],
    n_users: int,
) -> float:
    """
    Average self-information (surprisal) of recommended items.
    novelty(i) = -log2(p(i))  where p(i) = #users_who_rated_i / n_users

    High novelty → recommending long-tail / niche items.
    """
    scores = []
    for item in recommended:
        p = item_popularity.get(item, 0.0)
        if p > 0:
            scores.append(-math.log2(p / n_users))
    return float(np.mean(scores)) if scores else 0.0


def serendipity(
    recommended: list[int],
    relevant: set[int],
    popular_items: set[int],
    k: int,
) -> float:
    """
    Serendipity: items that are both relevant AND non-obvious (not just popular).
    serendipity@K = |{i ∈ top-K : relevant & not popular}| / K
    """
    top_k = recommended[:k]
    serendipitous = sum(
        1 for item in top_k if item in relevant and item not in popular_items
    )
    return serendipitous / k if k > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Suite
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_recommender(
    recommender,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    movie_profiles: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 500,
) -> dict:
    """
    Runs the full evaluation suite on a test set.

    Parameters:
        recommender     : HybridRecommender instance
        test_df         : test ratings DataFrame
        train_df        : training ratings (to compute popularity)
        movie_profiles  : from build_movie_profiles()
        k               : cut-off for ranking metrics
        relevance_threshold: rating ≥ this → relevant
        max_users       : cap for speed

    Returns:
        dict of all metrics
    """
    logger.info(f"Running evaluation (k={k}, threshold={relevance_threshold}) …")

    # Pre-compute popularity
    popularity = train_df.groupby("movie_id")["rating"].count().to_dict()
    n_users = train_df["user_id"].nunique()
    popular_items = {
        m for m, c in popularity.items()
        if c >= np.percentile(list(popularity.values()), 80)
    }

    # Embeddings for ILD
    bert_matrix = getattr(recommender.cb, "bert_matrix", None)
    mid_to_idx = getattr(recommender.cb, "movie_id_to_idx", {})
    if bert_matrix is not None:
        item_embs = {
            mid: bert_matrix[idx]
            for mid, idx in mid_to_idx.items()
        }
    else:
        item_embs = {}

    # Aggregate per-user ground truth
    user_relevant = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    metrics_per_user: dict[str, list] = defaultdict(list)
    all_recs: list[list] = []
    users = list(user_relevant.keys())[:max_users]

    for user_id in users:
        relevant = user_relevant[user_id]
        try:
            recs = recommender.recommend(user_id, top_n=k, strategy="weighted")
        except Exception as e:
            logger.warning(f"Skipping user {user_id}: {e}")
            continue

        rec_ids = [r["movie_id"] for r in recs]
        all_recs.append(rec_ids)

        metrics_per_user["precision"].append(precision_at_k(rec_ids, relevant, k))
        metrics_per_user["recall"].append(recall_at_k(rec_ids, relevant, k))
        metrics_per_user["f1"].append(f1_at_k(rec_ids, relevant, k))
        metrics_per_user["ndcg"].append(ndcg_at_k(rec_ids, relevant, k))
        metrics_per_user["mrr"].append(mean_reciprocal_rank(rec_ids, relevant))
        metrics_per_user["hit_rate"].append(hit_rate_at_k(rec_ids, relevant, k))
        metrics_per_user["ild"].append(intra_list_diversity(rec_ids, item_embs))
        metrics_per_user["novelty"].append(
            novelty(rec_ids, popularity, n_users)  # type: ignore[arg-type]
        )
        metrics_per_user["serendipity"].append(
            serendipity(rec_ids, relevant, popular_items, k)
        )

    total_movies = movie_profiles["movie_id"].nunique()
    coverage = catalogue_coverage(all_recs, total_movies)

    results = {
        f"precision@{k}": round(float(np.mean(metrics_per_user["precision"])), 4),
        f"recall@{k}": round(float(np.mean(metrics_per_user["recall"])), 4),
        f"f1@{k}": round(float(np.mean(metrics_per_user["f1"])), 4),
        f"ndcg@{k}": round(float(np.mean(metrics_per_user["ndcg"])), 4),
        "mrr": round(float(np.mean(metrics_per_user["mrr"])), 4),
        f"hit_rate@{k}": round(float(np.mean(metrics_per_user["hit_rate"])), 4),
        "catalogue_coverage": round(coverage, 4),
        "intra_list_diversity": round(float(np.mean(metrics_per_user["ild"])), 4),
        "novelty": round(float(np.mean(metrics_per_user["novelty"])), 4),
        "serendipity": round(float(np.mean(metrics_per_user["serendipity"])), 4),
        "n_users_evaluated": len(users),
    }

    # Pretty log
    logger.success("─" * 50)
    for metric, value in results.items():
        logger.success(f"  {metric:30s}: {value}")
    logger.success("─" * 50)

    return results


def rating_prediction_metrics(
    model,
    test_df: pd.DataFrame,
    predict_fn,
) -> dict:
    """
    Evaluates RMSE and MAE for rating prediction.
    predict_fn(user_id, movie_id) → float
    """
    y_true, y_pred = [], []
    for row in test_df.itertuples():
        try:
            pred = predict_fn(row.user_id, row.movie_id)
            y_true.append(row.rating)
            y_pred.append(pred)
        except Exception:
            continue

    if not y_true:
        return {"rmse": None, "mae": None}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    return {
        "rmse": round(rmse(y_true_arr, y_pred_arr), 4),
        "mae": round(mae(y_true_arr, y_pred_arr), 4),
        "n_predictions": len(y_true),
    }
