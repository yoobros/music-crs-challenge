"""Passthrough reranker: no reranking, return candidates as-is."""

from mymodule.strategies.rerank.base import BaseReranker


class PassthroughReranker(BaseReranker):
    """Reranker that returns candidates unchanged.

    Used when no rerank stage is needed; final top_k truncation happens in
    BaseStrategy.predict().
    """

    def __init__(self, **kwargs) -> None:
        pass

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
        # pool_rankings / fused_scores / pool_names are for learned rerankers;
        # passthrough ignores them and returns the fused result as-is.
        return candidates
