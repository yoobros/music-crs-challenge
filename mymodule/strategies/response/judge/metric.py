"""DSPy metric adapter for the in-house LLM-as-Judge.

`BootstrapFewShot` and friends call `metric(example, prediction, trace=None)
-> float` for each candidate trace. We wrap `Judge.score(...)` so that the
optimizer sees the same composite-aligned scalar the official judge would
contribute (post-normalization, before the 0.30 weight):

    metric_value = mean( (P_score - 1) / 4, (E_score - 1) / 4 )  ∈ [0, 1]

Bigger = better. The optimizer maximizes by selecting demonstrations whose
metric value clears `BootstrapFewShot.metric_threshold` (default 0).
"""

from __future__ import annotations

from typing import Any

from mymodule.strategies.response.judge.judge import get_default_judge


def _attr(example: Any, name: str, default: str = "") -> str:
    """Read a string field from a `dspy.Example` (which behaves like a Namespace)."""
    val = getattr(example, name, default)
    if val is None:
        return default
    return str(val)


def judge_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score one (example, prediction) pair via the singleton judge.

    Returns a float in [0, 1]. Empty `predicted_response` → 0.0.

    Parameters
    ----------
    example : dspy.Example
        Carries the inputs the generator was conditioned on, plus the GT
        response in `example.response` (built by `_trainset.build_dspy_trainset`).
    prediction : dspy.Prediction
        The generator's output. `prediction.response` is the candidate text.
    trace : optional
        Unused — DSPy passes per-step traces here when the optimizer wants
        token-level signal.
    """
    candidate = str(getattr(prediction, "response", "") or "").strip()
    if not candidate:
        return 0.0

    judge = get_default_judge()
    score = judge.score(
        user_query=_attr(example, "user_query"),
        chat_history=_attr(example, "chat_history"),
        user_profile=_attr(example, "user_profile"),
        listener_goal=_attr(example, "listener_goal"),
        recommended_tracks_detailed=_attr(example, "recommended_tracks_detailed"),
        predicted_response=candidate,
        gt_response=_attr(example, "response"),
    )
    return score.mean_normalized


__all__ = ["judge_metric"]
