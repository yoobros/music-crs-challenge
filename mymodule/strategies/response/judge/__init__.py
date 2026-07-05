"""In-house LLM-as-Judge mimicking the official Gemini judge for offline use.

Background
----------
The official Music-CRS evaluator (`assets/eval_rule.png`) defines the composite:

    Score = 0.50·nDCG@20 + 0.10·CatalogDiversity + 0.10·LexicalDiversity + 0.30·LLM-Judge

Where the LLM-Judge term comes from a Gemini model that scores each sampled
session-turn on **two text-only dimensions** (1-5 integer):

    - Personalization
    - Explanation Quality

These dimensions evaluate the **written response independently from
recommendation accuracy** (track-correctness is already counted in nDCG).
Each score is normalized via `(s - 1) / 4` and averaged.

This module provides a local re-implementation so we can:
    1. Use it as a `dspy.BootstrapFewShot` metric to compile prompt-optimized
       generators (`auto/optimize.py`).
    2. Compare variants offline before spending blind submissions.

Caveat
------
Gemini's exact prompt is not public. We mimic via the same rubric encoded in
`auto/signature.py`. Internal-judge ranks are trusted only as a *relative*
ordering across variants — absolute scores must always be calibrated against
blind composite deltas.
"""

from mymodule.strategies.response.judge.judge import Judge, JudgeScore
from mymodule.strategies.response.judge.metric import judge_metric

__all__ = ["Judge", "JudgeScore", "judge_metric"]
