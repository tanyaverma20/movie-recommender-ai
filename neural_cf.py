"""
models/neural_cf.py
────────────────────
Neural Collaborative Filtering (NCF) — He et al., 2017

Architecture: NeuMF = GMF ⊕ MLP
  • GMF  (Generalised Matrix Factorisation) – element-wise product of embeddings
  • MLP  (Multi-Layer Perceptron)           – deep interaction layers
  • NeuMF fuses both towers → final sigmoid rating prediction

Strong ML Concepts Applied:
  • Dual embedding tables (GMF embeddings vs MLP embeddings)
  • Batch Normalisation + Dropout regularisation
  • Binary Cross-Entropy loss with negative sampling
  • Cosine Annealing LR scheduler (warm restarts)
  • Gradient clipping for stability
  • He initialisation for ReLU networks
  • PyTorch Lightning-style training loop with early stopping
  • Evaluation via Hit Rate @K and NDCG @K
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import get_settings

cfg = get_settings()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RatingsDataset(Dataset):
    """
    Converts a ratings DataFrame into PyTorch tensors.
    Supports negative sampling: for each positive (u,i) pair, sample
    `neg_ratio` unobserved (u,j) pairs as negatives.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_movies: int,
        neg_ratio: int = 4,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.n_movies = n_movies
        self.neg_ratio = neg_ratio
        self.is_train = is_train

        # Pre-build positive item sets per user for fast negative sampling
        self.user_pos: dict[int, set] = (
            df.groupby("user_id_idx")["movie_id_idx"].apply(set).to_dict()
        )

        if is_train:
            self.data = self._build_with_negatives()
        else:
            # Val/test: include all interactions as positives only
            self.data = list(
                zip(df["user_id_idx"], df["movie_id_idx"], df["rating"])
            )

    def _build_with_negatives(self):
        records = []
        for _, row in self.df.iterrows():
            u, i, r = int(row["user_id_idx"]), int(row["movie_id_idx"]), float(row["rating"])
            # Normalise rating to [0,1] for BCE loss
            records.append((u, i, r / 5.0))
            pos_set = self.user_pos.get(u, set())
            sampled = 0
            while sampled < self.neg_ratio:
                j = np.random.randint(self.n_movies)
                if j not in pos_set:
                    records.append((u, j, 0.0))
                    sampled += 1
        return records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, r = self.data[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Neural Matrix Factorisation (NeuMF).

    Two pathways:
      GMF  : u_gmf ⊙ i_gmf                              → linear projection
      MLP  : [u_mlp; i_mlp] → FC → BN → ReLU → Dropout  → repeated N times
    Fusion: concat(GMF_out, MLP_out) → FC(1) → sigmoid

    Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [256, 128, 64]

        # GMF embeddings
        self.emb_user_gmf = nn.Embedding(n_users, emb_dim)
        self.emb_item_gmf = nn.Embedding(n_items, emb_dim)

        # MLP embeddings (separate – different representation space)
        self.emb_user_mlp = nn.Embedding(n_users, emb_dim)
        self.emb_item_mlp = nn.Embedding(n_items, emb_dim)

        # MLP tower
        mlp_input_dim = emb_dim * 2
        mlp_layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for out_dim in hidden_layers:
            mlp_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Fusion layer
        self.fusion = nn.Linear(emb_dim + hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """He (Kaiming) initialisation – optimal for ReLU activations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        # GMF path
        u_gmf = self.emb_user_gmf(user_ids)
        i_gmf = self.emb_item_gmf(item_ids)
        gmf_out = u_gmf * i_gmf                              # element-wise product

        # MLP path
        u_mlp = self.emb_user_mlp(user_ids)
        i_mlp = self.emb_item_mlp(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        logit = self.fusion(fused).squeeze(-1)
        return self.sigmoid(logit)

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Returns the concatenated GMF+MLP user embedding (for downstream tasks)."""
        uid = torch.tensor([user_id], device=DEVICE)
        with torch.no_grad():
            u_gmf = self.emb_user_gmf(uid).cpu().numpy()
            u_mlp = self.emb_user_mlp(uid).cpu().numpy()
        return np.concatenate([u_gmf, u_mlp], axis=-1).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NCFTrainer:
    """
    Training wrapper for NeuMF with:
      • BCE loss + L2 regularisation (weight_decay)
      • CosineAnnealingWarmRestarts scheduler
      • Gradient clipping
      • Early stopping on validation Hit Rate@10
      • Hit Rate@K and NDCG@K evaluation
    """

    def __init__(self, n_users: int, n_movies: int):
        self.model = NeuMF(
            n_users=n_users,
            n_items=n_movies,
            emb_dim=cfg.ncf_embedding_dim,
            hidden_layers=cfg.ncf_hidden_layers,
            dropout=cfg.ncf_dropout,
        ).to(DEVICE)

        self.n_movies = n_movies
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.ncf_learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        self.criterion = nn.BCELoss()
        self._best_hit_rate = 0.0
        self._patience = 3
        self._no_improve = 0
        self._is_fitted = False

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int | None = None,
    ) -> dict[str, list[float]]:
        epochs = epochs or cfg.ncf_epochs
        train_ds = RatingsDataset(train_df, self.n_movies, neg_ratio=4, is_train=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.ncf_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        history: dict[str, list[float]] = {"loss": [], "hr@10": [], "ndcg@10": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for users, items, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                preds = self.model(users, items)
                loss = self.criterion(preds, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            self.scheduler.step(epoch)

            if val_df is not None:
                hr, ndcg = self.evaluate(val_df, k=10)
                history["hr@10"].append(hr)
                history["ndcg@10"].append(ndcg)
                logger.info(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")

                # Early stopping
                if hr > self._best_hit_rate:
                    self._best_hit_rate = hr
                    self._no_improve = 0
                    self.save()
                else:
                    self._no_improve += 1
                    if self._no_improve >= self._patience:
                        logger.info(f"Early stopping at epoch {epoch}.")
                        break
            else:
                logger.info(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

        self._is_fitted = True
        return history

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, val_df: pd.DataFrame, k: int = 10) -> tuple[float, float]:
        """
        Computes Hit Rate@K and NDCG@K using the leave-one-out protocol:
          For each user, rank the positive test item among 99 random negatives.
          HR@K  = 1 if positive item appears in top-K.
          NDCG@K = 1/log2(rank+1) if positive item in top-K.
        """
        self.model.eval()
        hr_list, ndcg_list = [], []

        with torch.no_grad():
            for user_id_idx, group in val_df.groupby("user_id_idx"):
                pos_items = group["movie_id_idx"].tolist()
                if not pos_items:
                    continue
                test_item = pos_items[0]

                # 99 random negatives + 1 positive
                neg_items = np.random.randint(0, self.n_movies, size=99).tolist()
                candidates = [test_item] + neg_items

                users_t = torch.full((100,), user_id_idx, dtype=torch.long, device=DEVICE)
                items_t = torch.tensor(candidates, dtype=torch.long, device=DEVICE)
                scores = self.model(users_t, items_t).cpu().numpy()

                ranked = np.argsort(scores)[::-1]
                rank = int(np.where(ranked == 0)[0][0]) + 1  # 1-based rank of positive

                hr_list.append(1.0 if rank <= k else 0.0)
                ndcg_list.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)

        return float(np.mean(hr_list)), float(np.mean(ndcg_list))

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, user_idx: int, movie_idx: int) -> float:
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([user_idx], device=DEVICE)
            i = torch.tensor([movie_idx], device=DEVICE)
            return float(self.model(u, i).item())

    def recommend(
        self,
        user_idx: int,
        candidate_movie_idxs: list[int],
        top_n: int = 10,
    ) -> list[dict]:
        self.model.eval()
        with torch.no_grad():
            users = torch.full((len(candidate_movie_idxs),), user_idx, dtype=torch.long, device=DEVICE)
            items = torch.tensor(candidate_movie_idxs, dtype=torch.long, device=DEVICE)
            scores = self.model(users, items).cpu().numpy()

        ranked = np.argsort(scores)[::-1][:top_n]
        return [
            {"movie_idx": candidate_movie_idxs[i], "score": float(scores[i])}
            for i in ranked
        ]

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str = "saved_models/ncf_model.pt") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "n_movies": self.n_movies,
                "config": {
                    "n_users": self.model.emb_user_gmf.num_embeddings,
                    "n_items": self.model.emb_item_gmf.num_embeddings,
                    "emb_dim": cfg.ncf_embedding_dim,
                    "hidden_layers": cfg.ncf_hidden_layers,
                    "dropout": cfg.ncf_dropout,
                },
            },
            path,
        )
        logger.success(f"NCF model saved → {path}")

    @classmethod
    def load(cls, path: str = "saved_models/ncf_model.pt") -> "NCFTrainer":
        ckpt = torch.load(path, map_location=DEVICE)
        c = ckpt["config"]
        trainer = cls(n_users=c["n_users"], n_movies=ckpt["n_movies"])
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.eval()
        trainer._is_fitted = True
        logger.info(f"NCF model loaded ← {path}")
        return trainer
