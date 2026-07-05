"""Query Embedding pool: embed query text -> ANN search against LanceDB text tables.

`InstructedQueryEmbedder` composes a doc-distribution-
aligned query string from prior music turns (`title: ..., artist: ...,
album: ...` lowercase, talkpl `entity_str` format) and embeds the raw text
via Ollama Qwen3-Embedding (no instructed-query prefix — matches doc
distribution).

Pairs best with the `qemb_metadata_rich` vector_type (rich entity_str doc
with year/popularity/tags inline). Embedding model: `MYMODULE_EMB_MODEL`
(default `qwen3-embedding:0.6b`).
"""

from __future__ import annotations

from loguru import logger

from mymodule.feature.ollama_embed import InstructedQueryEmbedder
from mymodule.feature.store import EmbeddingStore
from mymodule.strategies.pool.base import BasePool
from mymodule.utils.seen import extract_session_seen_tracks, take_unseen_pairs


class QueryEmbedPool(BasePool):
    """Embed user_query with Ollama -> search a text-embedding LanceDB table.

    Unlike VectorDBPool (item-to-item), this pool uses the raw user query text
    as the search vector. Particularly useful for turn 1 where no previous
    tracks exist for item-to-item retrieval.
    """

    def __init__(
        self,
        vector_type: str = "metadata-qwen3_embedding_0.6b",
        n_candidates: int = 100,
        **kwargs,
    ) -> None:
        self.vector_type = vector_type
        self.n_candidates = n_candidates
        self.embedder = InstructedQueryEmbedder()
        self.store = EmbeddingStore.open_or_build()
        logger.info(f"QueryEmbedPool: vector_type={vector_type}, n_candidates={n_candidates}, embedder={self.embedder}")

    def _retrieve_with_scores(self, user_query: str, chat_history: list[dict]) -> list[tuple[str, float]]:
        query_vec = self.embedder.embed_query(user_query, chat_history=chat_history)
        # Search with headroom so the post-filter result still yields
        # n_candidates unseen track_ids (the session-seen filter is lossless:
        # GT never repeats within a session).
        seen = extract_session_seen_tracks(chat_history)
        raw = self.store.search_with_scores(self.vector_type, query_vec, self.n_candidates + len(seen))
        return take_unseen_pairs(raw, seen, self.n_candidates)

    def generate(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[str]:
        return [tid for tid, _ in self._retrieve_with_scores(user_query, chat_history)]

    def generate_with_scores(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Expose native cosine-similarity scores (1 - distance)."""
        return self._retrieve_with_scores(user_query, chat_history)
