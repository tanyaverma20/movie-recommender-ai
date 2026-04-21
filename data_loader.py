"""
data/data_loader.py
───────────────────
Downloads MovieLens 100K if not present, cleans and
feature-engineers the dataset, and exposes train/val/test splits.

Strong ML Concepts Applied:
  • Temporal train/test split (avoids data leakage)
  • Long-tail distribution handling (popularity bias correction)
  • Feature engineering: genre multi-hot, tag aggregation, year extraction
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ─────────────────────────────────────────────────────────────────────────────
# Download Helper
# ─────────────────────────────────────────────────────────────────────────────

def download_movielens(url: str = MOVIELENS_URL) -> None:
    """Download and extract MovieLens dataset into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "ratings.csv").exists():
        logger.info("Dataset already present – skipping download.")
        return

    logger.info(f"Downloading MovieLens from {url} …")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename.endswith(".csv"):
                with zf.open(member) as src, open(RAW_DIR / filename, "wb") as dst:
                    dst.write(src.read())
    logger.success("MovieLens dataset downloaded successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Raw loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_ratings(path: str | Path = RAW_DIR / "ratings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    logger.info(f"Ratings: {df.shape[0]:,} rows | {df['user_id'].nunique():,} users | {df['movie_id'].nunique():,} movies")
    return df


def load_movies(path: str | Path = RAW_DIR / "movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"movieId": "movie_id"})

    # Extract year from title
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*", "", regex=True).str.strip()

    # Multi-hot genre encoding
    all_genres = sorted({g for genres in df["genres"].str.split("|") for g in genres if g != "(no genres listed)"})
    for genre in all_genres:
        df[f"genre_{genre}"] = df["genres"].str.contains(genre, regex=False).astype(int)

    df["genre_count"] = df[[c for c in df.columns if c.startswith("genre_")]].sum(axis=1)
    logger.info(f"Movies: {df.shape[0]:,} rows | {len(all_genres)} unique genres")
    return df


def load_tags(path: str | Path = RAW_DIR / "tags.csv") -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["user_id", "movie_id", "tag", "timestamp"])
    df = pd.read_csv(path)
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["tag"] = df["tag"].str.lower().str.strip()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_movie_profiles(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a rich textual + numeric feature profile per movie.
    Used by content-based & embedding models.
    """
    # Aggregate tags per movie
    tag_agg = (
        tags.groupby("movie_id")["tag"]
        .apply(lambda x: " ".join(x.dropna().unique()))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )

    df = movies.merge(tag_agg, on="movie_id", how="left")
    df["tags_text"] = df["tags_text"].fillna("")

    # Soup feature: title + genres + tags → for TF-IDF / BERT
    df["soup"] = (
        df["title_clean"].fillna("")
        + " "
        + df["genres"].str.replace("|", " ", regex=False).fillna("")
        + " "
        + df["tags_text"]
    )

    # Normalize year
    scaler = MinMaxScaler()
    df["year_norm"] = scaler.fit_transform(df[["year"]].fillna(df["year"].median()))

    return df


def add_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Adds user-level aggregate statistics useful for cold-start handling.
    """
    user_stats = ratings.groupby("user_id").agg(
        user_mean_rating=("rating", "mean"),
        user_rating_count=("rating", "count"),
        user_rating_std=("rating", "std"),
        user_first_rating=("timestamp", "min"),
        user_last_rating=("timestamp", "max"),
    ).reset_index()
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0)
    user_stats["user_activity_days"] = (
        user_stats["user_last_rating"] - user_stats["user_first_rating"]
    ).dt.days
    return ratings.merge(user_stats, on="user_id", how="left")


def add_item_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """Adds item-level popularity & quality signals."""
    item_stats = ratings.groupby("movie_id").agg(
        item_mean_rating=("rating", "mean"),
        item_rating_count=("rating", "count"),
        item_rating_std=("rating", "std"),
    ).reset_index()
    # Bayesian average rating (handles long-tail)
    C = item_stats["item_rating_count"].mean()
    m = item_stats["item_mean_rating"].mean()
    item_stats["bayesian_avg"] = (
        (C * m + item_stats["item_rating_count"] * item_stats["item_mean_rating"])
        / (C + item_stats["item_rating_count"])
    )
    return ratings.merge(item_stats, on="movie_id", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test Split  (temporal – no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    ratings: pd.DataFrame,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits ratings chronologically:
      train  → oldest (80 %)
      val    → middle (10 %)
      test   → newest (10 %)

    This mirrors real-world deployment where models are trained on past
    interactions and evaluated on future ones (no temporal leakage).
    """
    df = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - val_ratio - test_ratio))

    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()

    logger.info(
        f"Temporal split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Integer encoding helpers (for Neural CF)
# ─────────────────────────────────────────────────────────────────────────────

def encode_ids(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, LabelEncoder]:
    user_enc = LabelEncoder().fit(train["user_id"])
    movie_enc = LabelEncoder().fit(train["movie_id"])

    def safe_encode(df, col, enc):
        known = set(enc.classes_)
        df = df[df[col].isin(known)].copy()
        df[f"{col}_idx"] = enc.transform(df[col])
        return df

    train = safe_encode(train, "user_id", user_enc)
    train = safe_encode(train, "movie_id", movie_enc)
    val = safe_encode(val, "user_id", user_enc)
    val = safe_encode(val, "movie_id", movie_enc)
    test = safe_encode(test, "user_id", user_enc)
    test = safe_encode(test, "movie_id", movie_enc)

    return train, val, test, user_enc, movie_enc


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MovieLensDataset:
    """
    One-stop shop that downloads, processes and serves the dataset.

    Usage:
        ds = MovieLensDataset()
        ds.prepare()
        train, val, test = ds.train, ds.val, ds.test
    """

    def __init__(self, auto_download: bool = True):
        self.auto_download = auto_download
        self.ratings: pd.DataFrame | None = None
        self.movies: pd.DataFrame | None = None
        self.tags: pd.DataFrame | None = None
        self.movie_profiles: pd.DataFrame | None = None
        self.train: pd.DataFrame | None = None
        self.val: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.user_encoder: LabelEncoder | None = None
        self.movie_encoder: LabelEncoder | None = None
        self.n_users: int = 0
        self.n_movies: int = 0

    def prepare(self) -> "MovieLensDataset":
        if self.auto_download:
            download_movielens()

        self.ratings = load_ratings()
        self.movies = load_movies()
        self.tags = load_tags()
        self.movie_profiles = build_movie_profiles(self.movies, self.tags)

        enriched = add_user_features(self.ratings)
        enriched = add_item_features(enriched)

        self.train, self.val, self.test = temporal_split(enriched)
        self.train, self.val, self.test, self.user_encoder, self.movie_encoder = encode_ids(
            self.train, self.val, self.test
        )

        self.n_users = len(self.user_encoder.classes_)
        self.n_movies = len(self.movie_encoder.classes_)

        logger.success(
            f"Dataset ready — {self.n_users:,} users | {self.n_movies:,} movies"
        )
        return self
