"""Base interface for reranking (stage 2)."""

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Common interface for stage-2 reranking.

    To add a reranker: create a .py file under strategies/rerank/, subclass
    this class and implement rerank(); the file name becomes the reranker name.
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialization (model loading, etc.)."""

    @abstractmethod
    def rerank(
        self,
        candidates: list[str],
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
        pool_rankings: dict[str, list[tuple[str, float]]] | None = None,
        fused_scores: dict[str, float] | None = None,
        pool_names: list[str] | None = None,
    ) -> list[str]:
        """Rerank candidates and return them.

        Args:
            candidates: post-fusion track ID list.
            pool_rankings: pool name → ``[(track_id, score), ...]`` (descending).
                Used by learned rerankers (e.g. GBM) for per-pool rank/score
                features; ignored by passthrough.
            fused_scores: track_id → RRF score, paired with ``pool_rankings``.
            pool_names: pool names of the current strategy (TID pool part),
                used to route to the matching trained checkpoint.

        Returns:
            Reranked track ID list (top N).
        """
