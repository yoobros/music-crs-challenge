"""DSPy signature for the in-house LLM-as-Judge.

Mirrors the public-facing description of the official Gemini judge from
`assets/eval_rule.png`:

    "A large language model evaluates a sampled subset of sessions across
     multiple dimensions, each scored on a 1-5 integer scale. The judge
     evaluates two text-only dimensions: Personalization and Explanation
     Quality. These dimensions evaluate the written response independently
     from recommendation accuracy."

Compositing (also from the figure):

    score_norm = (score - 1) / 4
    LLM-Judge contribution = 0.30 * mean(P_norm, E_norm)

Detailed criteria are not disclosed by the organizers, so we re-derive them
from the rubric we use to *generate* responses (`auto/signature.py`) — same
two axes, same hard constraints, with the rationale that an aligned grader
gives a less noisy training signal for `BootstrapFewShot`.
"""

from __future__ import annotations

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    class JudgeRubric(dspy.Signature):
        """Score the written music-recommendation response only.

        Official alignment target: Gemini usually gives score 4 to a
        non-broken response that is clearly tailored, cites valid tracks, and
        gives concrete musical or lyrical reasons. It drops when the prose is
        generic, malformed, or explains extra picks weakly. Track retrieval
        accuracy is scored elsewhere; ignore whether the track is the hidden
        ground truth.
        nDCG already scores retrieval, so do not penalize a cited title only
        because it is not listed in recommended_tracks_detailed. When that
        field is provided, use it only as optional attribute context.
        Do not use external music knowledge to fact-check album, catalog, or artist claims.
        Penalize only contradictions visible in the inputs.

        Do not write analysis. Fill only the requested output fields.
        Reasons must be <= 18 words.

        Score both dimensions with this anchor:
        5 = exceptional, all checks pass plus a distinctive user-specific
            insight or emotional connection.
        4 = strong, all baseline checks pass; this is the default for a
            coherent, concrete, personalized response.
        3 = adequate, exactly one important check fails.
        2 = weak, two or more checks fail, or generic phrasing dominates.
        1 = empty, trivial, malformed beyond reading, or could fit anyone.

        Personalization checks:
        - Uses a concrete token from the current user request.
        - If chat history exists, refers to a prior track, artist, mood, or
          stated preference; empty history auto-passes this check.
        - Matches listener_goal specificity: committed for tight goals,
          exploratory for loose goals.
        - Avoids demographic stereotypes.

        Explanation checks:
        - Every cited pick has a concrete attribute: production, vocal,
          instrumentation, lyric image, groove, tempo, dynamics, or era.
        - The why connects that attribute to the request, history, or goal.
        - Multiple cited picks explain different dimensions.
        - No stock verdict phrases dominate: strong start, strong candidate,
          close match, might be it, hits the mark, perfect for, great for,
          you'll love, check out, this song, this track.

        Assess holistically, not as a regex checklist. Treat these as evidence
        for lower quality only when they materially weaken personalization or
        explanation:
        - malformed citation or duplicated artist/title quote hurts readability.
        - a second cited pick with no distinct reason weakens Explanation.
        - numeric metadata alone is weaker than perceptual musical detail.
        - generic bridges ("same vibe", "keeps momentum") need concrete support.
        - missing request/history anchors weakens Personalization.
        """

        # ---- inputs ----------------------------------------------------------
        user_query: str = dspy.InputField(desc="Current-turn user message.")
        chat_history: str = dspy.InputField(desc="Prior role:content lines (user / assistant / music). May be empty.")
        user_profile: str = dspy.InputField(desc="Demographics — age, country, preferred language, culture.")
        listener_goal: str = dspy.InputField(desc="Stated listening goal and specificity code (HH/HL/LH/LL).")
        recommended_tracks_detailed: str = dspy.InputField(
            desc="Optional attribute context. Do not penalize titles absent from this list; nDCG handles retrieval."
        )
        predicted_response: str = dspy.InputField(desc="The candidate response being graded. May be empty (assign 1).")
        gt_response: str = dspy.InputField(
            desc="Optional reference / ground-truth response. Empty string when unavailable. "
            "Use only as calibration anchor — do NOT require verbatim match."
        )

        # ---- outputs ---------------------------------------------------------
        personalization_score: int = dspy.OutputField(desc="Integer in {1,2,3,4,5} for Personalization dimension.")
        personalization_reason: str = dspy.OutputField(desc="<=18 words. Cite one concrete pass/fail feature.")
        explanation_quality_score: int = dspy.OutputField(
            desc="Integer in {1,2,3,4,5} for Explanation Quality dimension."
        )
        explanation_quality_reason: str = dspy.OutputField(desc="<=18 words. Cite one concrete pass/fail feature.")


__all__ = ["JudgeRubric"]
