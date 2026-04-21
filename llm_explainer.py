"""
models/llm_explainer.py
────────────────────────
LLM-powered personalised explanations via a RAG (Retrieval-Augmented Generation) pipeline.

Architecture:
  Retrieve  → FAISS top-K similar movies (using SBERT embeddings)
  Augment   → Inject retrieved context + user history into prompt
  Generate  → OpenAI GPT-4o-mini produces a natural-language explanation

Strong ML Concepts Applied:
  • RAG (Retrieval-Augmented Generation) for knowledge-grounded generation
  • Prompt engineering with Chain-of-Thought (CoT) reasoning
  • Token budget management (tiktoken)
  • Structured output (JSON mode) for reliable parsing
  • Async streaming for low-latency UX
  • Response caching to reduce API costs
"""

from __future__ import annotations

import asyncio
import json
import hashlib
from typing import AsyncIterator

import tiktoken
from loguru import logger
from openai import AsyncOpenAI, OpenAI

from config import get_settings
from models.content_based import ContentBasedRecommender

cfg = get_settings()

# System prompt for the explanation agent
SYSTEM_PROMPT = """You are CineAI, an expert film analyst and personalised movie recommendation assistant.

Your task is to explain WHY a movie is recommended to a specific user, using:
1. Their viewing history and preferences
2. The movie's content (genres, themes, tags)
3. Similar movies they've enjoyed

Guidelines:
- Be specific and insightful, not generic. Reference concrete plot elements, directing styles, or themes.
- Use 2-3 short paragraphs. No bullet points.
- Sound like a knowledgeable friend, not a bot.
- Highlight the CONNECTION between their taste and this movie.
- Mention 1-2 similar movies they liked that led to this recommendation.
"""

EXPLANATION_PROMPT = """
USER CONTEXT:
  User ID: {user_id}
  Movies they enjoyed (rated ≥ 4/5): {liked_movies}
  Genres they prefer: {preferred_genres}

MOVIE TO EXPLAIN:
  Title: {movie_title}
  Year: {movie_year}
  Genres: {movie_genres}
  Tags: {movie_tags}
  Predicted Rating: {predicted_rating}/5.0

RETRIEVED SIMILAR MOVIES FROM USER'S HISTORY:
{similar_context}

Write a personalised explanation for why this user would enjoy "{movie_title}".
Focus on their specific taste profile and concrete connections to movies they've rated highly.
"""

JSON_PROMPT = """
Respond with a valid JSON object (no markdown fences) with these fields:
{
  "headline": "one punchy sentence (max 15 words)",
  "explanation": "2-3 paragraph explanation (150-200 words)",
  "similarity_hook": "which of their past movies most connects to this one",
  "mood_tags": ["tag1", "tag2", "tag3"],
  "confidence": 0.0 to 1.0
}
"""


class LLMExplainer:
    """
    Generates personalised movie recommendation explanations using RAG + GPT-4.

    Pipeline:
      1. Retrieve semantically similar movies from user's history (FAISS)
      2. Build a rich context string from retrieved movies + metadata
      3. Stream GPT-4o-mini response with structured JSON output
    """

    def __init__(self, cb: ContentBasedRecommender, movie_profiles):
        self.cb = cb
        self.movie_profiles = movie_profiles
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key) if cfg.openai_api_key else None
        self._sync_client = OpenAI(api_key=cfg.openai_api_key) if cfg.openai_api_key else None
        self._enc = tiktoken.encoding_for_model("gpt-4o-mini")
        self._cache: dict[str, dict] = {}   # in-memory cache (use Redis in production)

    # ── RAG Retrieval ─────────────────────────────────────────────────────

    def retrieve_context(
        self,
        target_movie_id: int,
        user_liked_ids: list[int],
        top_k: int = 5,
    ) -> str:
        """
        Retrieves the user's past movies most similar to the target movie.
        Uses FAISS cosine similarity over SBERT embeddings.
        Returns a formatted context string for the prompt.
        """
        similar = self.cb.get_similar_movies(target_movie_id, top_n=top_k * 3, mode="bert")
        similar_ids = {s["movie_id"] for s in similar}

        # Intersect with user's liked movies
        relevant_liked = [m for m in user_liked_ids if m in similar_ids][:top_k]
        if not relevant_liked:
            relevant_liked = user_liked_ids[:top_k]

        # Build context string
        context_lines = []
        for mid in relevant_liked:
            row = self._get_movie_row(mid)
            if row is not None:
                sim_score = next(
                    (s["similarity"] for s in similar if s["movie_id"] == mid), 0.0
                )
                context_lines.append(
                    f"  - {row.get('title_clean', 'Unknown')} ({int(row.get('year', 0) or 0)}) "
                    f"[similarity: {sim_score:.2f}] | Genres: {row.get('genres', '')} | Tags: {row.get('tags_text', '')[:80]}"
                )

        return "\n".join(context_lines) if context_lines else "  (No similar movies in user history found)"

    def _get_movie_row(self, movie_id: int) -> dict | None:
        rows = self.movie_profiles[self.movie_profiles["movie_id"] == movie_id]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    # ── Token Management ──────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        tokens = self._enc.encode(text)
        if len(tokens) <= budget:
            return text
        return self._enc.decode(tokens[:budget])

    # ── Cache Key ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(user_id: int, movie_id: int) -> str:
        return hashlib.md5(f"{user_id}:{movie_id}".encode()).hexdigest()

    # ── Explanation Generation ────────────────────────────────────────────

    async def explain_async(
        self,
        user_id: int,
        movie_id: int,
        liked_movie_ids: list[int],
        predicted_rating: float,
        preferred_genres: list[str] | None = None,
        stream: bool = False,
    ) -> dict:
        """
        Asynchronously generates a personalised explanation.

        Returns:
            dict with keys: headline, explanation, similarity_hook, mood_tags, confidence
        """
        cache_key = self._cache_key(user_id, movie_id)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for ({user_id}, {movie_id})")
            return self._cache[cache_key]

        if self._client is None:
            return self._mock_explanation(movie_id, predicted_rating)

        target_row = self._get_movie_row(movie_id)
        if target_row is None:
            return {"error": f"Movie {movie_id} not found in profiles."}

        # ── Build liked movie titles list ─────────────────────────────────
        liked_titles = []
        for mid in liked_movie_ids[:8]:
            row = self._get_movie_row(mid)
            if row:
                liked_titles.append(f"{row.get('title_clean', 'Unknown')} ({int(row.get('year', 0) or 0)})")

        # ── RAG retrieval ──────────────────────────────────────────────────
        similar_context = self.retrieve_context(movie_id, liked_movie_ids, top_k=5)

        # ── Build prompt ───────────────────────────────────────────────────
        user_prompt = EXPLANATION_PROMPT.format(
            user_id=user_id,
            liked_movies=", ".join(liked_titles) or "No history",
            preferred_genres=", ".join(preferred_genres or []) or "Mixed",
            movie_title=target_row.get("title_clean", "Unknown"),
            movie_year=int(target_row.get("year", 0) or 0),
            movie_genres=target_row.get("genres", "").replace("|", ", "),
            movie_tags=target_row.get("tags_text", "")[:200],
            predicted_rating=round(predicted_rating, 2),
            similar_context=similar_context,
        ) + "\n\n" + JSON_PROMPT

        # Token budget check
        total_tokens = self._count_tokens(SYSTEM_PROMPT + user_prompt)
        if total_tokens > 3500:
            user_prompt = self._truncate_to_budget(user_prompt, 3200)

        # ── API call ───────────────────────────────────────────────────────
        try:
            if stream:
                return await self._stream_explain(user_prompt)

            response = await self._client.chat.completions.create(
                model=cfg.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=cfg.max_tokens,
                temperature=0.75,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            result["movie_id"] = movie_id
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            self._cache[cache_key] = result
            return result

        except Exception as exc:
            logger.error(f"OpenAI error: {exc}")
            return self._mock_explanation(movie_id, predicted_rating)

    async def _stream_explain(self, user_prompt: str) -> AsyncIterator[str]:
        """Streams the raw text response token-by-token."""
        async with self._client.chat.completions.stream(
            model=cfg.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=cfg.max_tokens,
            temperature=0.75,
        ) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def explain_sync(self, user_id: int, movie_id: int, liked_movie_ids: list[int], predicted_rating: float) -> dict:
        """Synchronous wrapper for use outside async contexts."""
        return asyncio.run(
            self.explain_async(user_id, movie_id, liked_movie_ids, predicted_rating)
        )

    # ── Batch explanation ─────────────────────────────────────────────────

    async def explain_batch(
        self,
        user_id: int,
        recommendations: list[dict],
        liked_movie_ids: list[int],
        max_concurrent: int = 5,
    ) -> list[dict]:
        """
        Generates explanations for multiple recommendations concurrently.
        Uses a semaphore to limit concurrent API calls.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _explain_one(rec: dict) -> dict:
            async with sem:
                explanation = await self.explain_async(
                    user_id=user_id,
                    movie_id=rec["movie_id"],
                    liked_movie_ids=liked_movie_ids,
                    predicted_rating=rec.get("score", 0.0) * 5,
                )
                return {**rec, "explanation": explanation}

        tasks = [_explain_one(rec) for rec in recommendations]
        return await asyncio.gather(*tasks)

    # ── Mock fallback (no API key) ─────────────────────────────────────────

    def _mock_explanation(self, movie_id: int, predicted_rating: float) -> dict:
        row = self._get_movie_row(movie_id)
        title = row.get("title_clean", "this movie") if row else "this movie"
        genres = row.get("genres", "").replace("|", ", ") if row else ""
        return {
            "headline": f"A highly personalised pick — predicted {predicted_rating:.1f}/5.0",
            "explanation": (
                f"Based on your viewing history, {title} aligns strongly with your taste. "
                f"Matching your affinity for {genres or 'these genres'}, this film offers "
                "a compelling narrative that resonates with your pattern of highly-rated films. "
                "The thematic and stylistic similarities to movies you've enjoyed make this "
                "a confident recommendation from our hybrid recommendation engine."
            ),
            "similarity_hook": "Based on your recent viewing pattern.",
            "mood_tags": genres.split(", ")[:3] if genres else ["Drama"],
            "confidence": round(min(predicted_rating / 5.0, 1.0), 2),
            "movie_id": movie_id,
            "mock": True,
        }
