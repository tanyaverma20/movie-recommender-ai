"""
models/content_based.py
────────────────────────
Dual-mode content-based recommender.

Mode 1 – TF-IDF + Cosine Similarity (fast, interpretable)
Mode 2 – Sentence-BERT Embeddings + FAISS ANN (semantic, scalable)

Strong ML Concepts Applied:
  • TF-IDF with sublinear_tf & L2 normalisation
  • Sentence-transformers (Bi-encoder) for dense retrieval
  • FAISS IndexFlatIP (inner-product ANN for cosine sim on unit vecs)
  • Weighted feature fusion (genre + text + temporal)
  • Cold-start support via content (no interaction data needed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config import get_settings

cfg = get_settings()


class ContentBasedRecommender:
    """
    Two-tower content-based recommender.

    Attributes:
        movie_profiles : pd.DataFrame  – enriched movie metadata
        tfidf_matrix   : np.ndarray    – (n_movies, vocab) TF-IDF matrix
        bert_matrix    : np.ndarray    – (n_movies, 384) SBERT embeddings
        faiss_index    : faiss.Index   – ANN index over SBERT embeddings
        movie_id_to_idx: dict          – movie_id → row index mapping
    """

    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert
        self.tfidf: TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.bert_model: SentenceTransformer | None = None
        self.bert_matrix: np.ndarray | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.movie_profiles: pd.DataFrame | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self._genre_cols: list[str] = []
        self._is_fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(self, movie_profiles: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Builds TF-IDF and/or SBERT matrices from movie soup text.

        Parameters:
            movie_profiles : output of data_loader.build_movie_profiles()
        """
        self.movie_profiles = movie_profiles.reset_index(drop=True)
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_profiles["movie_id"])}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        self._genre_cols = [c for c in self.movie_profiles.columns if c.startswith("genre_")]

        # ── TF-IDF ────────────────────────────────────────────────────────
        logger.info("Building TF-IDF matrix …")
        self.tfidf = TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,          # log normalisation
            min_df=2,
            max_df=0.95,
            analyzer="word",
            stop_words="english",
        )
        tfidf_raw = self.tfidf.fit_transform(self.movie_profiles["soup"].fillna(""))
        self.tfidf_matrix = normalize(tfidf_raw, norm="l2").toarray().astype(np.float32)
        logger.info(f"TF-IDF shape: {self.tfidf_matrix.shape}")

        # ── Sentence-BERT ─────────────────────────────────────────────────
        if self.use_bert:
            logger.info(f"Encoding with SBERT ({cfg.sentence_model}) …")
            self.bert_model = SentenceTransformer(cfg.sentence_model)
            soups = self.movie_profiles["soup"].fillna("").tolist()
            self.bert_matrix = self.bert_model.encode(
                soups,
                batch_size=256,
                show_progress_bar=True,
                normalize_embeddings=True,   # unit vectors → inner product = cosine
                convert_to_numpy=True,
            ).astype(np.float32)
            logger.info(f"SBERT shape: {self.bert_matrix.shape}")
            self._build_faiss_index()

        self._is_fitted = True
        logger.success("Content-based recommender fitted.")
        return self

    def _build_faiss_index(self) -> None:
        """Builds a FAISS IndexFlatIP (exact inner-product) index."""
        dim = self.bert_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.bert_matrix)
        logger.info(f"FAISS index built — {self.faiss_index.ntotal:,} vectors (dim={dim})")

    # ── Similarity ────────────────────────────────────────────────────────

    def _get_combined_vector(self, movie_idx: int) -> np.ndarray:
        """
        Fuses TF-IDF, SBERT, and genre one-hot into a single vector
        with learnable (configurable) mixture weights.
        """
        tfidf_vec = self.tfidf_matrix[movie_idx]                        # (vocab,)
        genre_vec = self.movie_profiles.iloc[movie_idx][self._genre_cols].values.astype(np.float32)
        genre_vec = genre_vec / (np.linalg.norm(genre_vec) + 1e-10)

        if self.use_bert:
            bert_vec = self.bert_matrix[movie_idx]                      # (384,)
            # Weighted concatenation (genre signal is small but useful)
            combined = np.concatenate([
                0.5 * tfidf_vec[:512],          # truncate for speed
                0.4 * bert_vec,
                0.1 * genre_vec,
            ])
        else:
            combined = np.concatenate([0.9 * tfidf_vec[:1024], 0.1 * genre_vec])

        norm = np.linalg.norm(combined) + 1e-10
        return (combined / norm).astype(np.float32)

    def get_similar_movies(
        self,
        movie_id: int,
        top_n: int = 10,
        mode: str = "bert",              # "tfidf" | "bert" | "fusion"
    ) -> list[dict]:
        """
        Returns the top-N most similar movies by content.

        mode:
            "tfidf"  → TF-IDF cosine similarity (interpretable)
            "bert"   → SBERT cosine via FAISS (semantic)
            "fusion" → combined vector
        """
        self._check_fitted()
        if movie_id not in self.movie_id_to_idx:
            logger.warning(f"movie_id {movie_id} not in index.")
            return []

        idx = self.movie_id_to_idx[movie_id]

        if mode == "bert" and self.use_bert:
            # ANN search in FAISS
            query = self.bert_matrix[idx : idx + 1]
            distances, indices = self.faiss_index.search(query, top_n + 1)
            results = []
            for dist, i in zip(distances[0], indices[0]):
                if i == idx:
                    continue
                results.append({
                    "movie_id": self.idx_to_movie_id[i],
                    "similarity": float(dist),
                })
            return results[:top_n]

        elif mode == "tfidf":
            sim_row = cosine_similarity(
                self.tfidf_matrix[idx : idx + 1], self.tfidf_matrix
            )[0]
            top_idxs = np.argsort(sim_row)[::-1][1 : top_n + 1]
            return [
                {"movie_id": self.idx_to_movie_id[i], "similarity": float(sim_row[i])}
                for i in top_idxs
            ]

        else:  # fusion
            vec = self._get_combined_vector(idx).reshape(1, -1)
            all_vecs = np.vstack([self._get_combined_vector(i) for i in range(len(self.movie_profiles))])
            sims = (all_vecs @ vec.T).squeeze()
            top_idxs = np.argsort(sims)[::-1][1 : top_n + 1]
            return [
                {"movie_id": self.idx_to_movie_id[i], "similarity": float(sims[i])}
                for i in top_idxs
            ]

    # ── Candidate generation (user profile) ───────────────────────────────

    def recommend_for_user(
        self,
        liked_movie_ids: list[int],
        top_n: int = 10,
        exclude_ids: list[int] | None = None,
        mode: str = "bert",
    ) -> list[dict]:
        """
        Builds a user profile vector by averaging liked-movie embeddings,
        then retrieves nearest neighbours via FAISS.

        This is the classic 'user profile = mean of item embeddings' approach.
        """
        self._check_fitted()
        exclude = set(exclude_ids or []) | set(liked_movie_ids)

        valid_idxs = [self.movie_id_to_idx[m] for m in liked_movie_ids if m in self.movie_id_to_idx]
        if not valid_idxs:
            return []

        # User profile = mean embedding (weighted by recency would be better in prod)
        if mode == "bert" and self.use_bert:
            profile_vec = self.bert_matrix[valid_idxs].mean(axis=0, keepdims=True)
            profile_vec /= np.linalg.norm(profile_vec) + 1e-10
            distances, indices = self.faiss_index.search(profile_vec.astype(np.float32), top_n + len(exclude))
            results = []
            for dist, i in zip(distances[0], indices[0]):
                mid = self.idx_to_movie_id.get(i)
                if mid and mid not in exclude:
                    results.append({"movie_id": mid, "similarity": float(dist)})
                    if len(results) == top_n:
                        break
            return results
        else:
            # TF-IDF fallback
            profile_vec = self.tfidf_matrix[valid_idxs].mean(axis=0, keepdims=True)
            sims = cosine_similarity(profile_vec, self.tfidf_matrix)[0]
            top_idxs = np.argsort(sims)[::-1]
            results = []
            for i in top_idxs:
                mid = self.idx_to_movie_id.get(i)
                if mid and mid not in exclude:
                    results.append({"movie_id": mid, "similarity": float(sims[i])})
                    if len(results) == top_n:
                        break
            return results

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a freeform text query (for RAG retrieval)."""
        if self.bert_model is None:
            raise RuntimeError("BERT model not loaded.")
        vec = self.bert_model.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def semantic_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Full semantic search: encode query → FAISS → return top-K movies.
        Used as the retrieval step in the RAG pipeline.
        """
        self._check_fitted()
        if not self.use_bert:
            raise RuntimeError("BERT mode required for semantic search.")
        q_vec = self.encode_query(query).reshape(1, -1)
        distances, indices = self.faiss_index.search(q_vec, top_k)
        return [
            {
                "movie_id": self.idx_to_movie_id[i],
                "similarity": float(d),
            }
            for i, d in zip(indices[0], distances[0])
        ]

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, dir_path: str = "saved_models/cb") -> None:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "tfidf": self.tfidf,
                "tfidf_matrix": self.tfidf_matrix,
                "bert_matrix": self.bert_matrix,
                "movie_profiles": self.movie_profiles,
                "movie_id_to_idx": self.movie_id_to_idx,
                "idx_to_movie_id": self.idx_to_movie_id,
                "_genre_cols": self._genre_cols,
                "use_bert": self.use_bert,
            },
            p / "content_model.pkl",
        )
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(p / "faiss.index"))
        logger.success(f"Content-based model saved → {dir_path}")

    @classmethod
    def load(cls, dir_path: str = "saved_models/cb") -> "ContentBasedRecommender":
        p = Path(dir_path)
        data = joblib.load(p / "content_model.pkl")
        obj = cls(use_bert=data["use_bert"])
        for k, v in data.items():
            setattr(obj, k, v)
        faiss_path = p / "faiss.index"
        if faiss_path.exists():
            obj.faiss_index = faiss.read_index(str(faiss_path))
            obj.bert_model = SentenceTransformer(cfg.sentence_model)
        obj._is_fitted = True
        logger.info(f"Content-based model loaded ← {dir_path}")
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
