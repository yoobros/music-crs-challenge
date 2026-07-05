"""Three-stage DSPy module orchestrating the `auto` response pipeline.

Stages
------
1. `PersonalizationAnalysis`  (ChainOfThought) — extract user-specific anchors
2. `TrackExplanationPlan`      (ChainOfThought) — per-track WHY grounded in anchors
3. `ResponseComposition`       (Predict)        — judge-optimized final reply

Rationale for the decomposition: each Gemini LLM-Judge sub-dimension
(Personalization, Explanation Quality) gets its own dedicated reasoning
pass, producing explicit intermediate artifacts (anchors, track highlights,
bridging theme) that the final compose stage can weave into prose without
re-doing the reasoning under token pressure.

Rate limiting
-------------
The parent `DspyResponseGenerator` wraps its single predictor with
`RateLimitedPredictor` when `MYMODULE_LLM_MIN_INTERVAL > 0`. That wrapper
covers only the OUTER call — a Module's internal sub-Predicts bypass it.
`AutoResponseGenerator._build_predictor` therefore returns this Module
*without* the external wrapper; we invoke `apply_rate_limit(provider)`
here before each stage so every LM call participates in the global rate
limit (and honors `MYMODULE_LLM_WORKERS` scaling semantics).
"""

from __future__ import annotations

from typing import Any

from mymodule.strategies.response.auto.signature import (
    PersonalizationAnalysis,
    ResponseComposition,
    TrackExplanationPlan,
)
from mymodule.utils.common_dspy import Provider, apply_rate_limit

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


class AutoResponseModule(dspy.Module if dspy is not None else object):
    """Orchestrates PersonalizationAnalysis → TrackExplanationPlan → ResponseComposition."""

    def __init__(self, provider: Provider) -> None:
        if dspy is None:
            raise RuntimeError("dspy is required for AutoResponseModule.")
        super().__init__()
        self.provider: Provider = provider
        # ChainOfThought on the reasoning-heavy stages; Predict on compose (we
        # want all tokens for the final response, not for a rationale field).
        self.personalize = dspy.ChainOfThought(PersonalizationAnalysis)
        self.explain = dspy.ChainOfThought(TrackExplanationPlan)
        self.compose = dspy.Predict(ResponseComposition)

    def forward(self, **kwargs: Any) -> Any:
        # Stage 1 — personalization anchors
        apply_rate_limit(self.provider)
        p = self.personalize(
            user_query=kwargs["user_query"],
            chat_history=kwargs["chat_history"],
            user_profile=kwargs["user_profile"],
            listener_goal=kwargs["listener_goal"],
            recommended_tracks_overview=kwargs["recommended_tracks_overview"],
        )

        # Stage 2 — per-track explanations grounded in anchors
        apply_rate_limit(self.provider)
        e = self.explain(
            primary_axis=p.primary_axis,
            anchors=p.anchors,
            recommended_tracks_detailed=kwargs["recommended_tracks_detailed"],
            recommended_tracks_overview=kwargs["recommended_tracks_overview"],
            query_similarity_hints=kwargs.get("query_similarity_hints", "") or "",
            track_similarity_hints=kwargs.get("track_similarity_hints", "") or "",
        )

        # Stage 3 — compose final judge-optimized reply
        apply_rate_limit(self.provider)
        out = self.compose(
            user_query=kwargs["user_query"],
            opener_hook_seed=p.opener_hook_seed,
            anchors=p.anchors,
            track_highlights=e.track_highlights,
            bridging_theme=e.bridging_theme,
            tail_one_liner=e.tail_one_liner,
            recommended_tracks_detailed=kwargs["recommended_tracks_detailed"],
        )

        # Attach intermediate stages for post-process / debug access.
        # Avoid shadowing output fields by using underscore-prefixed attrs.
        try:
            setattr(out, "_personalization", p)
            setattr(out, "_explanation_plan", e)
        except Exception:
            pass
        return out
