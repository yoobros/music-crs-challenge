"""Base interface for candidate pool generation (stage 1)."""

from abc import ABC, abstractmethod


class BasePool(ABC):
    """Common interface for first-stage candidate generation.

    To add a pool: create a new .py file under strategies/pool/, subclass
    this, and implement generate(). The file name becomes the pool name
    (auto-registered).
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize indexes / embeddings."""

    @abstractmethod
    def generate(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[str]:
        """Return candidate track IDs (tens to hundreds, rank-ordered)."""

    def generate_with_scores(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Return (track_id, score) pairs, higher score = better.

        Default wraps generate() with a rank-derived score `1/(rank+1)`.
        Pools with native scores should override to expose them.
        """
        ids = self.generate(
            user_query=user_query,
            chat_history=chat_history,
            user_id=user_id,
            conversation_goal=conversation_goal,
            goal_progress=goal_progress,
            user_profile=user_profile,
        )
        return [(tid, 1.0 / (rank + 1)) for rank, tid in enumerate(ids)]
