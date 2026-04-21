"""
train.py
─────────
Master training script — trains all models end-to-end with MLflow tracking.

Run:
    python train.py --tune --epochs 20 --experiment my-run

Strong ML Concepts:
  • Experiment tracking with MLflow
  • Reproducibility via random seed fixing
  • Modular, sequential training pipeline
  • Comprehensive evaluation after each stage
"""

from __future__ import annotations

import argparse
import random

import mlflow
import numpy as np
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

from config import get_settings
from data.data_loader import MovieLensDataset
from evaluation.metrics import evaluate_recommender, rating_prediction_metrics
from models.collaborative_filter import CollaborativeFilter
from models.content_based import ContentBasedRecommender
from models.hybrid_recommender import HybridRecommender
from models.neural_cf import NCFTrainer

cfg = get_settings()
console = Console()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train Movie Recommender AI")
    p.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (slower)")
    p.add_argument("--epochs", type=int, default=cfg.ncf_epochs, help="NCF training epochs")
    p.add_argument("--experiment", type=str, default="default-run", help="MLflow run name")
    p.add_argument("--no-bert", action="store_true", help="Skip SBERT (faster, CPU-only)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def log_rich_table(metrics: dict, title: str) -> None:
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(str(k), str(v))
    console.print(table)


def main():
    args = parse_args()
    set_seed(args.seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=args.experiment):
        mlflow.log_params({
            "seed": args.seed,
            "svd_n_factors": cfg.svd_n_factors,
            "ncf_embedding_dim": cfg.ncf_embedding_dim,
            "ncf_epochs": args.epochs,
            "use_bert": not args.no_bert,
        })

        # ── 1. Data ──────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing dataset")
        logger.info("=" * 60)
        ds = MovieLensDataset().prepare()

        mlflow.log_params({
            "n_users": ds.n_users,
            "n_movies": ds.n_movies,
            "n_train": len(ds.train),
            "n_val": len(ds.val),
            "n_test": len(ds.test),
        })

        # ── 2. Collaborative Filtering ───────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2: Training Collaborative Filter (SVD ensemble)")
        logger.info("=" * 60)
        cf = CollaborativeFilter(algorithms=["svd", "svdpp", "nmf"])
        cf.fit(ds.train, ds.val, tune_hyperparams=args.tune)
        cf.save()

        cf_pred_metrics = rating_prediction_metrics(
            cf,
            ds.test,
            predict_fn=cf.predict,
        )
        log_rich_table(cf_pred_metrics, "CF Rating Prediction Metrics")
        mlflow.log_metrics({f"cf_{k}": v for k, v in cf_pred_metrics.items() if v is not None})

        # ── 3. Content-Based ─────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3: Training Content-Based Recommender (TF-IDF + SBERT)")
        logger.info("=" * 60)
        cb = ContentBasedRecommender(use_bert=not args.no_bert)
        cb.fit(ds.movie_profiles)
        cb.save()
        logger.success("Content-based model fitted and saved.")

        # ── 4. Neural CF ──────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4: Training Neural Collaborative Filtering (NeuMF)")
        logger.info("=" * 60)
        ncf = NCFTrainer(ds.n_users, ds.n_movies)
        history = ncf.train(ds.train, ds.val, epochs=args.epochs)
        ncf.save()

        if history["hr@10"]:
            best_hr = max(history["hr@10"])
            best_ndcg = max(history["ndcg@10"])
            mlflow.log_metrics({"ncf_best_hr@10": best_hr, "ncf_best_ndcg@10": best_ndcg})
            log_rich_table({"best_hr@10": best_hr, "best_ndcg@10": best_ndcg}, "NCF Metrics")

        # ── 5. Hybrid ─────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 5: Building Hybrid Recommender")
        logger.info("=" * 60)
        hybrid = HybridRecommender(cf, cb, ncf, ds)
        hybrid.fit_meta_learner(ds.val)

        # ── 6. Full Evaluation ────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 6: Evaluating Hybrid Recommender on Test Set")
        logger.info("=" * 60)
        ranking_metrics = evaluate_recommender(
            recommender=hybrid,
            test_df=ds.test,
            train_df=ds.train,
            movie_profiles=ds.movie_profiles,
            k=10,
            max_users=500,
        )
        log_rich_table(ranking_metrics, "Hybrid Recommender – Full Evaluation (k=10)")
        mlflow.log_metrics(ranking_metrics)

        logger.success("Training complete! Run `uvicorn api.main:app --reload` to start the API.")


if __name__ == "__main__":
    main()
