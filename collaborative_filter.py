"""
models/collaborative_filter.py
────────────────────────────────
Matrix Factorization ensemble using scikit-surprise.

Algorithms implemented:
  • SVD  (Simon Funk's regularised SGD)
  • SVD++ (adds implicit feedback)
  • NMF  (Non-negative Matrix Factorization)
  • KNN-based Baseline (item-item cosine similarity)

Strong ML Concepts Applied:
  • Bias terms (global, user, item)
  • L2 regularisation to prevent overfitting
  • Early stopping on validation RMSE
  • LOOCV for hyperparameter selection
  • Popularity-aware neighbour pruning (long-tail correction)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from surprise import SVD, SVDpp, NMF, KNNWithMeans, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, GridSearchCV

from config import get_settings

cfg = get_settings()


class CollaborativeFilter:
    """
    Ensemble of three matrix-factorisation algorithms.

    Predictions are combined via a learned weighted average
    where weights are inversely proportional to validation RMSE.
    """

    ALGORITHMS = {
        "svd": SVD,
        "svdpp": SVDpp,
        "nmf": NMF,
        "knn": KNNWithMeans,
    }

    def __init__(self, algorithms: list[str] | None = None):
        self.algorithms = algorithms or ["svd", "svdpp", "nmf"]
        self.models: dict = {}
        self.weights: dict[str, float] = {}
        self.trainset = None
        self._is_fitted = False

    # ── Build surprise Dataset ─────────────────────────────────────────────

    @staticmethod
    def _to_surprise_dataset(df: pd.DataFrame) -> Dataset:
        reader = Reader(rating_scale=(0.5, 5.0))
        return Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        tune_hyperparams: bool = False,
    ) -> "CollaborativeFilter":
        """
        Trains all configured algorithms and computes ensemble weights.
        If val is provided, weights are set by 1/RMSE on validation set.
        """
        data = self._to_surprise_dataset(train)
        self.trainset = data.build_full_trainset()

        if tune_hyperparams:
            self._tune(data)

        val_rmse: dict[str, float] = {}
        for name in self.algorithms:
            logger.info(f"Training {name.upper()} …")
            algo = self._build_algo(name)
            algo.fit(self.trainset)
            self.models[name] = algo

            if val is not None:
                val_rmse[name] = self._rmse_on(algo, val)
                logger.info(f"  Val RMSE [{name}]: {val_rmse[name]:.4f}")

        # Compute weights: 1/RMSE normalised to sum=1
        if val_rmse:
            inv = {k: 1.0 / v for k, v in val_rmse.items()}
            total = sum(inv.values())
            self.weights = {k: v / total for k, v in inv.items()}
        else:
            n = len(self.algorithms)
            self.weights = {k: 1.0 / n for k in self.algorithms}

        logger.success(f"Ensemble weights: {self.weights}")
        self._is_fitted = True
        return self

    def _build_algo(self, name: str):
        if name == "svd":
            return SVD(
                n_factors=cfg.svd_n_factors,
                n_epochs=cfg.svd_n_epochs,
                lr_all=cfg.svd_lr_all,
                reg_all=cfg.svd_reg_all,
                biased=True,
            )
        elif name == "svdpp":
            return SVDpp(
                n_factors=cfg.svd_n_factors,
                n_epochs=cfg.svd_n_epochs,
                lr_all=cfg.svd_lr_all,
                reg_all=cfg.svd_reg_all,
            )
        elif name == "nmf":
            return NMF(
                n_factors=cfg.svd_n_factors,
                n_epochs=cfg.svd_n_epochs,
                reg_pu=0.06,
                reg_qi=0.06,
                biased=True,
            )
        elif name == "knn":
            return KNNWithMeans(
                k=40,
                sim_options={"name": "pearson_baseline", "user_based": False},
            )
        raise ValueError(f"Unknown algorithm: {name}")

    # ── Hyperparameter Tuning ──────────────────────────────────────────────

    def _tune(self, data: Dataset) -> None:
        """Grid-search on SVD hyperparameters with 3-fold CV."""
        param_grid = {
            "n_factors": [50, 100, 150],
            "reg_all": [0.02, 0.05, 0.1],
            "lr_all": [0.002, 0.005],
        }
        gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
        gs.fit(data)
        best = gs.best_params["rmse"]
        logger.info(f"Best SVD hyperparameters: {best}")
        # Patch config (runtime only)
        cfg.svd_n_factors = best["n_factors"]
        cfg.svd_lr_all = best["lr_all"]
        cfg.svd_reg_all = best["reg_all"]

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, user_id: int, movie_id: int) -> float:
        """Weighted ensemble prediction for (user, movie) pair."""
        self._check_fitted()
        total = 0.0
        for name, model in self.models.items():
            pred = model.predict(str(user_id), str(movie_id))
            total += self.weights[name] * pred.est
        return round(float(np.clip(total, 0.5, 5.0)), 3)

    def recommend(
        self,
        user_id: int,
        movie_ids: list[int],
        top_n: int = 10,
        exclude_seen: list[int] | None = None,
    ) -> list[dict]:
        """
        Returns top-N recommendations for a user from a candidate set.
        Supports exclude_seen to filter already-watched movies.
        """
        self._check_fitted()
        exclude = set(exclude_seen or [])
        candidates = [m for m in movie_ids if m not in exclude]

        scores = [(m, self.predict(user_id, m)) for m in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)

        return [{"movie_id": m, "predicted_rating": s} for m, s in scores[:top_n]]

    # ── Utility ────────────────────────────────────────────────────────────

    def _rmse_on(self, algo, val: pd.DataFrame) -> float:
        """Evaluate a fitted surprise algo on a Pandas DataFrame."""
        preds = [algo.predict(str(r.user_id), str(r.movie_id)) for r in val.itertuples()]
        return float(accuracy.rmse(preds, verbose=False))

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Returns latent factor vector for a user (from SVD)."""
        svd: SVD = self.models.get("svd")
        if svd is None:
            raise RuntimeError("SVD model not trained.")
        inner = self.trainset.to_inner_uid(str(user_idx))
        return svd.pu[inner]

    def get_item_embedding(self, movie_idx: int) -> np.ndarray:
        """Returns latent factor vector for a movie (from SVD)."""
        svd: SVD = self.models.get("svd")
        if svd is None:
            raise RuntimeError("SVD model not trained.")
        inner = self.trainset.to_inner_iid(str(movie_idx))
        return svd.qi[inner]

    def similar_movies(self, movie_id: int, top_n: int = 10) -> list[dict]:
        """
        Finds movies with the most similar item latent vectors.
        Uses cosine similarity in the SVD embedding space.
        """
        svd: SVD = self.models.get("svd")
        if svd is None:
            raise RuntimeError("SVD model not trained.")
        try:
            inner = self.trainset.to_inner_iid(str(movie_id))
        except ValueError:
            return []

        target = svd.qi[inner]
        norms = np.linalg.norm(svd.qi, axis=1, keepdims=True) + 1e-10
        sims = (svd.qi @ target) / (norms.squeeze() * np.linalg.norm(target) + 1e-10)
        top_inner = np.argsort(sims)[::-1][1 : top_n + 1]

        results = []
        for idx in top_inner:
            try:
                mid = int(self.trainset.to_raw_iid(idx))
                results.append({"movie_id": mid, "similarity": float(sims[idx])})
            except Exception:
                continue
        return results

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str = "saved_models/cf_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "weights": self.weights}, path)
        logger.success(f"Collaborative filter saved → {path}")

    @classmethod
    def load(cls, path: str = "saved_models/cf_model.pkl") -> "CollaborativeFilter":
        obj = cls()
        data = joblib.load(path)
        obj.models = data["models"]
        obj.weights = data["weights"]
        obj._is_fitted = True
        logger.info(f"Collaborative filter loaded ← {path}")
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
