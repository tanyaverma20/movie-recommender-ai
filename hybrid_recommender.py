"""
models/hybrid_recommender.py
──────────────────────────────
Adaptive Hybrid Recommender — combines all three models.

Fusion Strategies:
  1. Weighted Score Fusion   – static weights from config
  2. Rank Aggregation        – Borda count over ranked lists
  3. Learned Stacking        – Ridge regression meta-learner trained on val set
  4. Contextual Routing      – selects strategy per user based on data density

Strong ML Concepts Applied:
  • Cascade filtering (cheap CF first → expensive SBERT only for top-K)
  • Popularity debiasing (MMR – Maximal Marginal Relevance for diversity)
  • Context-aware switching (cold vs warm users)
  • Meta-learner stacking (Ridge blending)
  • Reciprocal Rank Fusion (RRF) for robust rank aggregation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from typing import Optional

from config import get_settings
from models.collaborative_filter import CollaborativeFilter
from models.content_based import ContentBasedRecommender
from models.neural_cf import NCFTrainer

cfg = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score: 1 / (k + rank)."""
    return 1.0 / (k + rank)


def mmr_rerank(
    candidates: list[dict],
    bert_matrix: np.ndarray,
    movie_id_to_idx: dict,
    lambda_: float = 0.7,
    top_n: int = 10,
) -> list[dict]:
    """
    Maximal Marginal Relevance (MMR) for diversity.

    Balances relevance (λ × score) against redundancy
    ((1 - λ) × max_sim_to_already_selected).

    λ = 1 → pure relevance;  λ = 0 → maximum diversity.
    """
    if len(candidates) <= top_n:
        return candidates

    selected: list[dict] = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x["score"])
        else:
            sel_idxs = [movie_id_to_idx[s["movie_id"]] for s in selected if s["movie_id"] in movie_id_to_idx]
            sel_vecs = bert_matrix[sel_idxs]

            mmr_scores = []
            for c in remaining:
                c_idx = movie_id_to_idx.get(c["movie_id"])
                if c_idx is None:
                    mmr_scores.append(lambda_ * c["score"])
                    continue
                c_vec = bert_matrix[c_idx]
                sim_to_sel = float(np.max(sel_vecs @ c_vec))
                mmr = lambda_ * c["score"] - (1 - lambda_) * sim_to_sel
                mmr_scores.append(mmr)
            best = remaining[int(np.argmax(mmr_scores))]

        selected.append(best)
        remaining.remove(best)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Recommender
# ─────────────────────────────────────────────────────────────────────────────

class HybridRecommender:
    """
    Orchestrates CF + Content-Based + NCF into a unified recommendation engine.

    Usage:
        rec = HybridRecommender(cf, cb, ncf, dataset)
        results = rec.recommend(user_id=42, top_n=10, strategy="stack")
    """

    STRATEGIES = ("weighted", "rrf", "stack", "contextual")

    def __init__(
        self,
        cf: CollaborativeFilter,
        cb: ContentBasedRecommender,
        ncf: NCFTrainer,
        dataset,                             # MovieLensDataset instance
        meta_learner: Ridge | None = None,
    ):
        self.cf = cf
        self.cb = cb
        self.ncf = ncf
        self.ds = dataset
        self.meta_learner = meta_learner     # fitted on validation set

        # All movie IDs in training set
        self._all_movie_ids = list(dataset.movie_encoder.classes_)

    # ── Core Entry Point ──────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        strategy: str = "weighted",
        diversity: bool = True,
        exclude_seen: bool = True,
    ) -> list[dict]:
        """
        Generate top-N recommendations for a user.

        Parameters:
            strategy: "weighted" | "rrf" | "stack" | "contextual"
            diversity: apply MMR reranking for diversity
            exclude_seen: filter out already-rated movies
        """
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}"

        seen = self._get_seen_movies(user_id) if exclude_seen else []
        user_density = len(seen)              # proxy for warm/cold

        # Cold-start: fewer than 5 ratings → pure content-based
        if user_density < 5:
            logger.info(f"User {user_id} is cold-start (< 5 ratings). Using content-based.")
            return self._cold_start(user_id, seen, top_n)

        # Contextual routing
        if strategy == "contextual":
            strategy = "stack" if user_density >= 20 else "rrf"

        candidates = self._generate_candidates(user_id, seen, n_candidates=cfg.top_k_retrieval * 3)

        if strategy == "weighted":
            scored = self._weighted_fusion(user_id, candidates)
        elif strategy == "rrf":
            scored = self._reciprocal_rank_fusion(user_id, candidates)
        elif strategy == "stack":
            scored = self._stacking(user_id, candidates)
        else:
            scored = self._weighted_fusion(user_id, candidates)

        scored.sort(key=lambda x: x["score"], reverse=True)

        if diversity and self.cb.bert_matrix is not None:
            scored = mmr_rerank(
                scored,
                self.cb.bert_matrix,
                self.cb.movie_id_to_idx,
                lambda_=0.7,
                top_n=top_n,
            )

        return scored[:top_n]

    # ── Candidate Generation (Cascade) ────────────────────────────────────

    def _generate_candidates(
        self, user_id: int, seen: list[int], n_candidates: int
    ) -> list[int]:
        """
        Two-stage retrieval:
          1. CF recommends top-K candidates cheaply.
          2. CB (SBERT) broadens with semantically diverse candidates.
        """
        cf_cands = [
            r["movie_id"]
            for r in self.cf.recommend(user_id, self._all_movie_ids, top_n=n_candidates, exclude_seen=seen)
        ]

        liked = self._get_liked_movies(user_id)
        cb_cands = [
            r["movie_id"]
            for r in self.cb.recommend_for_user(liked or cf_cands[:5], top_n=n_candidates // 2, exclude_ids=seen)
        ]

        # Union, dedup, preserve order
        seen_set = set(seen)
        all_cands, dedup = [], set()
        for m in cf_cands + cb_cands:
            if m not in dedup and m not in seen_set:
                all_cands.append(m)
                dedup.add(m)

        return all_cands[:n_candidates]

    # ── Fusion Strategies ─────────────────────────────────────────────────

    def _weighted_fusion(self, user_id: int, candidates: list[int]) -> list[dict]:
        """
        Simple weighted average of three model scores.
        Weights are normalised so all three signals contribute proportionally.
        """
        user_idx = self._uid_to_idx(user_id)
        results = []
        for movie_id in candidates:
            cf_score = self.cf.predict(user_id, movie_id) / 5.0
            cb_score = self._cb_score(user_id, movie_id)
            ncf_score = self._ncf_score(user_idx, movie_id)

            score = (
                cfg.weight_collaborative * cf_score
                + cfg.weight_content * cb_score
                + cfg.weight_neural * ncf_score
            )
            results.append({
                "movie_id": movie_id,
                "score": float(score),
                "cf_score": round(cf_score, 4),
                "cb_score": round(cb_score, 4),
                "ncf_score": round(ncf_score, 4),
            })
        return results

    def _reciprocal_rank_fusion(self, user_id: int, candidates: list[int]) -> list[dict]:
        """
        RRF merges ranked lists without requiring score normalisation.
        Robust to outlier scores and scale differences between models.
        """
        user_idx = self._uid_to_idx(user_id)

        cf_ranked = sorted(candidates, key=lambda m: self.cf.predict(user_id, m), reverse=True)
        ncf_ranked = sorted(candidates, key=lambda m: self._ncf_score(user_idx, m), reverse=True)
        cb_ranked = sorted(candidates, key=lambda m: self._cb_score(user_id, m), reverse=True)

        rrf: dict[int, float] = {}
        for rank, m in enumerate(cf_ranked, 1):
            rrf[m] = rrf.get(m, 0) + _rrf_score(rank)
        for rank, m in enumerate(ncf_ranked, 1):
            rrf[m] = rrf.get(m, 0) + _rrf_score(rank)
        for rank, m in enumerate(cb_ranked, 1):
            rrf[m] = rrf.get(m, 0) + _rrf_score(rank)

        return [{"movie_id": m, "score": s} for m, s in rrf.items()]

    def _stacking(self, user_id: int, candidates: list[int]) -> list[dict]:
        """
        Meta-learner stacking: train a Ridge regressor on validation set
        features [cf_score, cb_score, ncf_score] → target rating.
        Falls back to weighted fusion if meta-learner not available.
        """
        if self.meta_learner is None:
            logger.warning("Meta-learner not fitted; falling back to weighted fusion.")
            return self._weighted_fusion(user_id, candidates)

        user_idx = self._uid_to_idx(user_id)
        rows = []
        for movie_id in candidates:
            rows.append([
                self.cf.predict(user_id, movie_id) / 5.0,
                self._cb_score(user_id, movie_id),
                self._ncf_score(user_idx, movie_id),
            ])
        X = np.array(rows)
        scores = self.meta_learner.predict(X)
        return [
            {"movie_id": m, "score": float(s)}
            for m, s in zip(candidates, scores)
        ]

    def fit_meta_learner(self, val_df: pd.DataFrame) -> None:
        """
        Fits the Ridge meta-learner on validation set predictions.
        Should be called after all three base models are trained.
        """
        X_rows, y = [], []
        for _, row in val_df.iterrows():
            uid, mid, rating = row["user_id"], row["movie_id"], row["rating"]
            uid_idx = self._uid_to_idx(uid)
            if uid_idx is None:
                continue
            cf_s = self.cf.predict(uid, mid) / 5.0
            cb_s = self._cb_score(uid, mid)
            ncf_s = self._ncf_score(uid_idx, mid)
            X_rows.append([cf_s, cb_s, ncf_s])
            y.append(rating / 5.0)

        if not X_rows:
            logger.warning("No valid rows for meta-learner fitting.")
            return

        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(np.array(X_rows), y)
        train_score = self.meta_learner.score(np.array(X_rows), y)
        logger.success(f"Meta-learner R² on val: {train_score:.4f}")

    # ── Cold Start ────────────────────────────────────────────────────────

    def _cold_start(self, user_id: int, seen: list[int], top_n: int) -> list[dict]:
        """
        For cold-start users: recommend by popularity + diversity.
        Uses Bayesian average ratings from item features.
        """
        popular = (
            self.ds.ratings.groupby("movie_id")["bayesian_avg"]
            .max()
            .reset_index()
            .sort_values("bayesian_avg", ascending=False)
        )
        popular = popular[~popular["movie_id"].isin(set(seen))]
        top_ids = popular["movie_id"].head(top_n * 2).tolist()

        if self.cb.bert_matrix is not None:
            results = [{"movie_id": m, "score": float(popular[popular.movie_id == m]["bayesian_avg"].values[0])} for m in top_ids if m in self.cb.movie_id_to_idx]
            results = mmr_rerank(results, self.cb.bert_matrix, self.cb.movie_id_to_idx, top_n=top_n)
        else:
            results = [{"movie_id": m, "score": 0.0} for m in top_ids[:top_n]]

        return results

    # ── Score Helpers ─────────────────────────────────────────────────────

    def _cb_score(self, user_id: int, movie_id: int) -> float:
        liked = self._get_liked_movies(user_id)
        if not liked or movie_id not in self.cb.movie_id_to_idx:
            return 0.0
        idx = self.cb.movie_id_to_idx[movie_id]
        liked_idxs = [self.cb.movie_id_to_idx[m] for m in liked if m in self.cb.movie_id_to_idx]
        if not liked_idxs or self.cb.bert_matrix is None:
            return 0.0
        profile = self.cb.bert_matrix[liked_idxs].mean(axis=0)
        profile /= np.linalg.norm(profile) + 1e-10
        return float(np.dot(profile, self.cb.bert_matrix[idx]))

    def _ncf_score(self, user_idx: int | None, movie_id: int) -> float:
        if user_idx is None:
            return 0.0
        try:
            movie_idx = int(self.ds.movie_encoder.transform([movie_id])[0])
        except Exception:
            return 0.0
        return self.ncf.predict(user_idx, movie_idx)

    # ── Lookup Helpers ─────────────────────────────────────────────────────

    def _get_seen_movies(self, user_id: int) -> list[int]:
        mask = self.ds.train["user_id"] == user_id
        return self.ds.train.loc[mask, "movie_id"].tolist()

    def _get_liked_movies(self, user_id: int, threshold: float = 3.5) -> list[int]:
        df = self.ds.train
        mask = (df["user_id"] == user_id) & (df["rating"] >= threshold)
        return df.loc[mask, "movie_id"].tolist()

    def _uid_to_idx(self, user_id: int) -> int | None:
        try:
            return int(self.ds.user_encoder.transform([user_id])[0])
        except Exception:
            return None
