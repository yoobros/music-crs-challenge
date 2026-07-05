"""Base interface for response generation (stage 3, after retrieve + rerank)."""

from abc import ABC, abstractmethod


class BaseResponseGenerator(ABC):
    """Common interface for natural-language response generators.

    Pipeline position:
        stage 1 (pool retrieve) → stage 2 (fusion + rerank) → **stage 3 (response)**
    Takes the top-k reranked track_ids and returns the user-facing response.

    To add a new generator, create a .py file under strategies/response/ that
    subclasses this class and implements generate(); the filename becomes the
    generator name and is auto-registered.
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Initialize model/client resources."""

    @abstractmethod
    def generate(
        self,
        track_ids: list[str],
        user_query: str,
        chat_history: list[dict],
        user_profile: dict | None = None,
        conversation_goal: dict | None = None,
    ) -> str:
        """Return a natural-language response for the top-k recommendations.

        Args:
            track_ids: top-k track_ids from the reranker (usually 20). The
                generator may look up metadata for only as many as it needs.
            user_query: the user's last utterance.
            chat_history: prior turns [{turn_number, role, content}].
            user_profile: user profile dict (age_group, country, preferred_language, ...).
            conversation_goal: evaluation-set metadata.

        Returns:
            Response string. Empty string "" is schema-valid (fail-soft fallback).
        """
