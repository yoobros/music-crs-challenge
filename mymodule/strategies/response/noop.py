"""No-op response generator: returns an empty string."""

from mymodule.strategies.response.base import BaseResponseGenerator


class NoopResponseGenerator(BaseResponseGenerator):
    """Returns an empty response (schema-valid for submissions).

    Useful for local iteration and for regression tests with response
    generation disabled.
    """

    def __init__(self, **kwargs) -> None:
        pass

    def generate(
        self,
        track_ids: list[str],
        user_query: str,
        chat_history: list[dict],
        user_profile: dict | None = None,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_id: str | None = None,
        **kwargs,
    ) -> str:
        return ""
