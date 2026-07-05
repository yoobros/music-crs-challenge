"""Compile a few-shot demo set for the PAS response generator.

The compile runs the LLM on training examples, scores each candidate response with
the in-house LLM-judge, and saves up to `--n-boot` demos that pass `--metric-threshold`
plus a length floor (`--min-words` / `--max-words`).

Usage (current default):
    uv run python -m mymodule.strategies.response.pas.optimize \\
        --response-provider openai \\
        --train-n 100 \\
        --n-boot 5 \\
        --metric-threshold 0.5 \\
        --parallel-workers 8 \\
        --seed 42 \\
        --llm-temperature 0.0

`--seed` + `--llm-temperature 0.0` give a deterministic compile on backends
that honor `seed` (vLLM / SGLang / OpenAI). Same flags → same demos.

Active runtime knobs (env, all defaulted to the winner config):
    MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT=selective    # default; off-topic prior music turns dropped
    MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_MD=0.55
    MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_BPR=0.15

Outputs (3 sidecar files in `mymodule/strategies/response/ckpt/`; gitignored):
    pas_{provider}.json           — canonical DSPy ckpt (loaded by `dspy.Predict.load`)
    pas_{provider}.buckets.json   — DemoRouter sidecar (per-demo spec/turn/score metadata)
    pas_{provider}.meta.json      — compile provenance (seed, llm_temperature, threshold,
                                    judge_model, git_sha, signature_hash) for reproducibility

Two compile paths — pick via `--parallel-workers`:
    1. `--parallel-workers 1` (default): DSPy's sequential `BootstrapFewShot`. Honors
       `--n-label > 0` for labeled GT demos and DSPy's `max_rounds`. Slowest but stable.
    2. `--parallel-workers N` (N > 1): custom ThreadPoolExecutor parallel bootstrap.
       Same first-N-passing semantics but ~5-6x faster on rate-limit-free endpoints.
       Ignores `--n-label` (always 0 — train GT is too short for PAS docstring target
       250-380 words). Recommended for high-throughput OpenAI-compatible endpoints.
       Set `--candidate-source rewrite_gt` to compare each original train assistant
       response against a PAS-generated synthetic rewrite, then keep the synthetic
       demo only when it scores better under the in-house judge.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import threading
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

from mymodule.strategies.response.pas.config import load_rules
from mymodule.strategies.response.pas.generator import (
    _LEXDIV_STOPWORDS,
    _WORD_TOKEN_RE,
)

# Checkpoints live under response/ckpt/ (one level up from pas/).
CKPT_DIR = Path(__file__).parent.parent / "ckpt"


_UNKNOWN_RESPONSE = "Unknown message"  # known training noise (5.65% of GT responses)
_TITLE_CITATION_RE = re.compile(r'["“]([A-Za-z0-9][^"”\n]{1,60})["”]')


class LexDiversityTracker:
    """Accepted-demo vocab tracker for compile-time lexical-diversity reward.

    Final compile metric is an additive blend:

        combined = (1 - w) * judge + w * novelty

    where `judge ∈ [0, 1]` comes from the LLM-as-a-Judge and `novelty ∈ [0, 1]`
    is the fraction of content tokens in the candidate response that are NOT yet
    present in `_vocab` (the set of content tokens from accepted demos so far).

    Word matching reuses generator-side filters so compile- and inference-time
    diversity definitions stay aligned:
    - `_WORD_TOKEN_RE` — ≥4-letter [A-Za-z] runs (skips most stopwords).
    - `_LEXDIV_STOPWORDS` — function-word fillers + music-grounding tokens that
      should NOT count as "overused" (title, artist, album, …).

    Thread-safe: parallel bootstrap reads `combine(...)` from N workers and
    `register_accepted(...)` from the acceptance loop concurrently.
    """

    def __init__(self, weight: float):
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"lexdiv weight must be in [0, 1], got {weight}")
        self._vocab: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._weight = float(weight)
        self.accepted_count = 0  # for logging only

    @property
    def weight(self) -> float:
        return self._weight

    def _tokens(self, text: str) -> set[str]:
        if not text:
            return set()
        return {m.lower() for m in _WORD_TOKEN_RE.findall(text)} - _LEXDIV_STOPWORDS

    def novelty(self, text: str) -> float:
        """Fraction of content tokens not yet seen across accepted demos.

        Returns 1.0 when no demo has been accepted yet (no anchor to compare
        against) and when the candidate text has no content tokens — neither
        case should be penalized.
        """
        if self._weight <= 0.0:
            return 1.0
        tokens = self._tokens(text)
        if not tokens:
            return 1.0
        with self._lock:
            if not self._vocab:
                return 1.0
            overlap = sum(1 for t in tokens if self._vocab[t] > 0)
        return 1.0 - (overlap / len(tokens))

    def combine(self, judge_score: float, text: str) -> float:
        """Blend `judge_score` with novelty using configured weight."""
        if self._weight <= 0.0:
            return float(judge_score)
        novelty = self.novelty(text)
        return (1.0 - self._weight) * float(judge_score) + self._weight * novelty

    def register_accepted(self, text: str) -> None:
        """Update vocab after a demo is officially accepted. Idempotent on empty text."""
        if self._weight <= 0.0:
            return
        tokens = self._tokens(text)
        if not tokens:
            return
        with self._lock:
            self._vocab.update(tokens)
            self.accepted_count += 1


# PAS signature InputField keys (kept here so we can fingerprint Examples without
# importing dspy at module-import time — keeps test footprint small).
_PAS_INPUT_KEYS = (
    "user_query",
    "listener_goal",
    "response_style_notes",
    "chat_history",
    "user_profile",
    "recommended_tracks_overview",
    "recommended_tracks_detailed",
    "recommended_titles_pool",
    "intent_groups",
    "track_similarity_hints",
    "query_similarity_hints",
    "lyric_similarity_hints",
    "overused_words",
    "overused_bigrams",
)
_PAS_OUTPUT_KEYS = (
    "personalization_anchors",
    "track_roles",
    "themes",
    "themes_excluded_patterns",
    "cited_titles",
    "response",
)


def _extract_cited_titles(gt_response: str) -> str:
    """Pull every double-/smart-quoted span from the GT response.

    One per line, de-duplicated while preserving order. Returned as a newline-joined
    string so it fits the DSPy `cited_titles` OutputField shape.
    """
    seen: list[str] = []
    for m in _TITLE_CITATION_RE.finditer(gt_response or ""):
        t = m.group(1).strip()
        if t and t not in seen:
            seen.append(t)
    return "\n".join(seen)


def _turn_position(turn_number: int) -> str:
    """Coarse turn-position bucket: first / mid / final. Mirrors compile_router."""
    if turn_number <= 1:
        return "first"
    if turn_number >= 8:
        return "final"
    return "mid"


def _example_fingerprint(inputs: dict[str, Any]) -> str:
    """Stable hash of the InputField subset of a dspy.Example.

    Used to look up per-example metadata after BootstrapFewShot has potentially
    constructed a new Example object for bootstrapped demos: the input fields
    are preserved across that transformation, so the fingerprint matches the
    source training example.
    """
    payload = {k: str(inputs.get(k, "")) for k in _PAS_INPUT_KEYS}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _stratify_round_robin(
    examples: list,
    meta_by_fp: dict[str, dict[str, Any]],
    key_fn,
) -> list:
    """Reorder a flat trainset so adjacent examples cycle through buckets.

    BootstrapFewShot iterates trainset linearly to find demos passing the metric
    threshold. With sequential ordering, the first ~N examples can all share one
    bucket (commonly category=B, specificity=HL on this dataset), producing
    monoculture demos. Stratifying first guarantees that even if bootstrap stops
    at n_boot accepts, those accepts come from different buckets.

    `key_fn(meta_dict) -> bucket_key`. Stable order within each bucket preserves
    `random.seed` determinism.
    """
    from collections import defaultdict, deque

    buckets: dict[Any, deque] = defaultdict(deque)
    for ex in examples:
        # Mirror the input fingerprint logic so we can look up meta.
        try:
            inputs_view = ex.inputs().toDict() if hasattr(ex.inputs(), "toDict") else dict(ex.inputs())
        except Exception:
            inputs_view = {k: getattr(ex, k, "") for k in _PAS_INPUT_KEYS}
        fp = _example_fingerprint(inputs_view)
        meta = meta_by_fp.get(fp, {})
        buckets[key_fn(meta)].append(ex)

    interleaved: list = []
    while any(buckets.values()):
        for k in list(buckets.keys()):
            if buckets[k]:
                interleaved.append(buckets[k].popleft())
            else:
                buckets.pop(k, None)
    return interleaved


def _coverage_key(meta: dict[str, Any], coverage_by: str) -> tuple[str, ...]:
    """Return the synthetic-demo coverage key for one training example.

    Coverage selection is intentionally compile-time only. Runtime sees the same
    sidecar shape, with `category` / `specificity` / `turn_position` preserved in
    `meta` for the DemoRouter.
    """
    category = str(meta.get("category", "") or "unknown")
    specificity = str(meta.get("specificity", "") or "unknown")
    turn_position = str(meta.get("turn_position", "") or "unknown")
    if coverage_by == "none":
        return ("global",)
    if coverage_by == "specificity":
        return (specificity,)
    if coverage_by == "specificity_turn":
        return (specificity, turn_position)
    if coverage_by == "category_specificity":
        return (category, specificity)
    if coverage_by == "category_specificity_turn":
        return (category, specificity, turn_position)
    raise ValueError(f"unknown coverage_by={coverage_by!r}")


def _coverage_key_label(key: tuple[str, ...]) -> str:
    return "|".join(str(part) for part in key)


def _normalize_title_key(text: str) -> str:
    """Normalize a title-like span for compile-time citation checks."""
    text = (text or "").strip().lower()
    text = re.sub(r"^[\s\"'“”‘’`*\-•\d.)]+", "", text)
    text = re.sub(r"[\s\"'“”‘’`]+$", "", text)
    text = re.sub(r"\s+by\s+.+$", "", text)
    text = re.sub(r"\s+—\s+.+$", "", text)
    text = re.sub(r"\s+-\s+.+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_title_pool_ranks(title_pool: str) -> dict[str, int]:
    """Map normalized title text to its 1-based rank in `recommended_titles_pool`."""
    ranks: dict[str, int] = {}
    for line in (title_pool or "").splitlines():
        m = re.match(r"\s*(\d+)\.\s*[\"“](.*?)[\"”]\s+[—-]\s+.+\s*$", line)
        if not m:
            continue
        key = _normalize_title_key(m.group(2))
        if key and key not in ranks:
            ranks[key] = int(m.group(1))
    return ranks


def _extract_cited_title_ranks(pred: Any, inputs_view: dict[str, Any]) -> list[int]:
    """Return sorted unique title-pool ranks cited by a generated demo candidate."""
    pool_ranks = _parse_title_pool_ranks(str(inputs_view.get("recommended_titles_pool", "") or ""))
    if not pool_ranks:
        return []
    seen: set[int] = set()

    cited_titles = str(getattr(pred, "cited_titles", "") or "")
    for raw in cited_titles.splitlines():
        key = _normalize_title_key(raw)
        rank = pool_ranks.get(key)
        if rank is not None:
            seen.add(rank)

    response = str(getattr(pred, "response", "") or "")
    for m in _TITLE_CITATION_RE.finditer(response):
        key = _normalize_title_key(m.group(1))
        rank = pool_ranks.get(key)
        if rank is not None:
            seen.add(rank)

    return sorted(seen)


def _score_dim(score: Any | None, attr: str) -> float:
    if score is None:
        return 0.0
    try:
        return float(getattr(score, attr, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _passes_rewrite_gate(
    *,
    synthetic_judge_score: float,
    gt_judge_score: float,
    threshold: float,
    rewrite_margin: float,
    synthetic_judge_detail: Any | None = None,
    min_p_score: float = 0.0,
    min_e_score: float = 0.0,
    require_rank1_citation: bool = False,
    cited_ranks: list[int] | None = None,
) -> bool:
    """Return whether a synthetic rewrite is worth using as a few-shot demo.

    The gate is intentionally judge-first: synthetic demos must clear the normal
    compile threshold AND improve over the original train assistant response.
    This prevents the few-shot bank from inheriting examples whose style or
    personalization/explainability level is weaker than the training response.
    """
    if synthetic_judge_score < threshold:
        return False
    if synthetic_judge_score < gt_judge_score + rewrite_margin:
        return False
    if min_p_score > 0 and _score_dim(synthetic_judge_detail, "p_score") < min_p_score:
        return False
    if min_e_score > 0 and _score_dim(synthetic_judge_detail, "e_score") < min_e_score:
        return False
    if require_rank1_citation and 1 not in (cited_ranks or []):
        return False
    return True


def _format_rank1_citation_feedback(pred: Any, inputs_view: dict[str, Any]) -> str:
    pool_ranks = _parse_title_pool_ranks(str(inputs_view.get("recommended_titles_pool", "") or ""))
    rank1_title = next((title for title, rank in pool_ranks.items() if rank == 1), "")
    cited_ranks = _extract_cited_title_ranks(pred, inputs_view)
    if 1 in cited_ranks:
        return ""
    if rank1_title:
        return (
            "Grounding failure: this train example injects the GT track at rank 1, "
            f"so the synthetic demo must cite the rank-1 title ({rank1_title!r}) as the core pick. "
            f"Current cited ranks: {cited_ranks or 'none'}."
        )
    return "Grounding failure: cite the rank-1 title from the provided title pool as the core pick."


def _score_detail_meta(prefix: str, score: Any | None) -> dict[str, Any]:
    if score is None:
        return {
            f"{prefix}_p_score": 0.0,
            f"{prefix}_e_score": 0.0,
        }
    return {
        f"{prefix}_p_score": _score_dim(score, "p_score"),
        f"{prefix}_e_score": _score_dim(score, "e_score"),
    }


def _candidate_gate_meta(
    pred: Any,
    inputs_view: dict[str, Any],
    *,
    require_rank1_citation: bool,
) -> dict[str, Any]:
    cited_ranks = _extract_cited_title_ranks(pred, inputs_view)
    return {
        "cited_title_ranks": cited_ranks,
        "primary_cited_rank": min(cited_ranks) if cited_ranks else None,
        "rank1_title_cited": 1 in cited_ranks,
        "require_rank1_citation": require_rank1_citation,
    }


def _has_candidate_grounding(
    pred: Any,
    inputs_view: dict[str, Any],
    *,
    require_rank1_citation: bool,
) -> bool:
    if not require_rank1_citation:
        return True
    return 1 in _extract_cited_title_ranks(pred, inputs_view)


def _rank1_gate_fail_message(
    pred: Any,
    inputs_view: dict[str, Any],
    *,
    require_rank1_citation: bool,
) -> str:
    if not require_rank1_citation or _has_candidate_grounding(
        pred,
        inputs_view,
        require_rank1_citation=require_rank1_citation,
    ):
        return ""
    return _format_rank1_citation_feedback(pred, inputs_view)


def _select_synthetic_demo_bank(
    candidates: list[tuple[Any, Any, float, dict[str, Any]]],
    *,
    meta_by_fp: dict[str, dict[str, Any]],
    score_log: dict[str, float],
    n_boot: int,
    coverage_by: str,
    min_per_case: int,
) -> list[tuple[Any, Any, float, dict[str, Any]]]:
    """Pick a quality-first but coverage-aware synthetic demo bank.

    Candidate generation and scoring are separated from final demo selection:
    every candidate already passed the in-house judge threshold, then this
    selector ensures sparse but important cases (specificity/category/turn) get
    represented before filling remaining slots by score.
    """
    if n_boot <= 0 or not candidates:
        return []

    records: list[dict[str, Any]] = []
    for cand in candidates:
        _ex, _pred, score, inputs_view = cand
        fp = _example_fingerprint({k: inputs_view.get(k, "") for k in _PAS_INPUT_KEYS})
        meta = meta_by_fp.get(fp, {})
        key = _coverage_key(meta, coverage_by)
        records.append(
            {
                "candidate": cand,
                "fp": fp,
                "score": float(score),
                "key": key,
            }
        )
        score_log[fp] = max(score_log.get(fp, 0.0), float(score))

    if coverage_by == "none" or min_per_case <= 0:
        selected = sorted(records, key=lambda r: (-r["score"], r["fp"]))[:n_boot]
    else:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            grouped[rec["key"]].append(rec)
        for key in grouped:
            grouped[key].sort(key=lambda r: (-r["score"], r["fp"]))

        selected = []
        selected_fps: set[str] = set()
        ordered_keys = sorted(grouped.keys(), key=lambda k: (-len(grouped[k]), _coverage_key_label(k)))
        # Pass 1: take up to `min_per_case` high-scoring demos from each case.
        for _round_idx in range(max(1, min_per_case)):
            for key in ordered_keys:
                if len(selected) >= n_boot:
                    break
                bucket = grouped[key]
                while bucket and bucket[0]["fp"] in selected_fps:
                    bucket.pop(0)
                if not bucket:
                    continue
                rec = bucket.pop(0)
                selected.append(rec)
                selected_fps.add(rec["fp"])
            if len(selected) >= n_boot:
                break
        # Pass 2: fill the rest globally by score.
        if len(selected) < n_boot:
            remaining = [rec for rec in records if rec["fp"] not in selected_fps]
            remaining.sort(key=lambda r: (-r["score"], r["fp"]))
            for rec in remaining:
                selected.append(rec)
                selected_fps.add(rec["fp"])
                if len(selected) >= n_boot:
                    break

    for rank, rec in enumerate(selected, start=1):
        meta = meta_by_fp.setdefault(rec["fp"], {})
        meta["synthetic_demo"] = True
        meta.setdefault("demo_origin", "llm_bootstrap")
        meta.setdefault("synthetic_source", "llm_bootstrap_judge_selected")
        meta["coverage_by"] = coverage_by
        meta["coverage_key"] = _coverage_key_label(rec["key"])
        meta["selection_rank"] = rank
    return [rec["candidate"] for rec in selected]


def _build_examples(
    train_n: int,
    kv: object,
    stratify_by: str = "none",
    *,
    keep_raw_for_embed: bool = False,
) -> tuple[list, dict[str, dict[str, Any]]]:
    """Sample train sessions and build dspy.Example objects with ground-truth responses.

    Returns (examples, meta_by_fingerprint) where meta carries the per-Example
    `(category, specificity, turn_position)` triple for downstream bucket sidecar.

    `stratify_by`:
      - "none"          : sequential by sampled index (legacy)
      - "specificity"   : round-robin across HH/HL/LH/LL — gives even spec coverage
      - "bucket"        : round-robin across (specificity, turn_position) — finest

    When `keep_raw_for_embed=True`, each meta entry also carries underscore-prefixed
    compile-time-only fields (`_chat_history_raw`, `_user_query_raw`,
    `_conversation_goal_raw`, `_user_profile_raw`, `_goal_progress_raw`). These are
    consumed by `_embed_demo_pool` and stripped before the sidecar is written, so
    they never leak to the runtime sidecar JSON.
    """
    import dspy
    from datasets import load_dataset
    from tqdm import tqdm as _tqdm

    from mymodule.strategies.response.base_dspy import (
        fmt_recommended_tracks,
        fmt_tracks_overview,
    )
    from mymodule.strategies.response.pas.helpers import (
        fmt_lyric_similarity_hints,
        fmt_query_similarity_hints,
        fmt_track_similarity_hints,
    )
    from mymodule.strategies.response.pas.propose import (
        classify_intents,
        format_intent_groups_for_prompt,
    )
    from mymodule.utils.common_dspy import (
        fmt_chat_history,
        fmt_conversation_goal,
        fmt_user_profile,
    )

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    indices = random.sample(range(len(ds)), min(train_n * 4, len(ds)))  # oversample to tolerate filtering
    examples: list[dspy.Example] = []
    meta_by_fp: dict[str, dict[str, Any]] = {}

    # Compile-side candidate pool inflation — runtime sees top-20 retrieved
    # tracks, but compile by default sees only the GT music turn (typically
    # 1 track). To make the demo's track_ids more representative of inference,
    # we expand each GT track into a top-20 pool by metadata-rich-qwen3
    # cosine NN search via the LanceDB EmbeddingStore. GT is injected at
    # rank 1 if not already in the NN top-K, so the demo can still ground
    # its WHY on the dataset's labelled pick. Falls back to GT-only when the
    # store is not available (CI / partial env / no NN index).
    try:
        from mymodule.feature.store import EmbeddingStore as _EmbStore

        _emb_store = _EmbStore.open_or_build()
    except Exception as e:
        logger.warning(
            f"[pas-compile] EmbeddingStore open failed ({type(e).__name__}: {e}); "
            "skipping candidate pool inflation — demos will see GT-only track_ids."
        )
        _emb_store = None
    _POOL_VECTOR_TYPE = "metadata_rich-qwen3_embedding_0.6b"
    _POOL_SIZE = 20

    def _inflate_compile_track_pool(gt_track_ids: list[str]) -> list[str]:
        """Expand `[gt_track_id, ...]` to a ~20-track pool via NN around GT.

        - Anchor on the FIRST GT track (most representative of the turn).
        - Pull metadata_rich-qwen3 embedding from KV, search LanceDB for top-K.
        - GT track_ids always come first (preserves dataset ordering for the
          demo's `cited_titles`), then unique NNs to fill up to `_POOL_SIZE`.
        - On any failure (no embedding / store down) → return GT-only.
        """
        if not gt_track_ids or _emb_store is None:
            return list(gt_track_ids)
        anchor_tid = gt_track_ids[0]
        try:
            anchor_vec = kv.get_track_embedding(anchor_tid, _POOL_VECTOR_TYPE)
        except Exception:
            anchor_vec = None
        if anchor_vec is None:
            return list(gt_track_ids)
        try:
            nn_tids = _emb_store.search(_POOL_VECTOR_TYPE, anchor_vec, top_k=_POOL_SIZE)
        except Exception as e:
            logger.warning(f"[pas-compile] NN search for {anchor_tid[:8]}… failed ({type(e).__name__}: {e}).")
            return list(gt_track_ids)
        # GT-first ordering: keep GT track_ids in dataset order, then NN
        # extras that aren't already in the GT set, up to _POOL_SIZE.
        seen: set[str] = set()
        pool: list[str] = []
        for tid in gt_track_ids:
            if tid not in seen:
                pool.append(tid)
                seen.add(tid)
        for tid in nn_tids:
            if len(pool) >= _POOL_SIZE:
                break
            if tid not in seen:
                pool.append(tid)
                seen.add(tid)
        return pool

    # Progress over the oversampled index pool so the user can watch how fast
    # we are converging on `train_n`. Each iteration runs PROPOSE / similarity
    # hints / (optional) chat-selective gate per turn, so latency is dominated
    # by KV + embedder calls — not negligible for n=500.
    pbar = _tqdm(
        total=len(indices),
        desc=f"building examples (target n={train_n})",
        unit="idx",
    )
    pbar.set_postfix(collected=0)

    for idx in indices:
        pbar.update(1)
        item = ds[idx]
        convs = item["conversations"]
        music_turns = {m["turn_number"] for m in convs if m["role"] == "music"}
        assistant_turns = {m["turn_number"]: m["content"] for m in convs if m["role"] == "assistant"}

        for turn_num in sorted(music_turns & set(assistant_turns.keys())):
            gt_response = (assistant_turns[turn_num] or "").strip()
            if not gt_response or gt_response == _UNKNOWN_RESPONSE:
                continue

            chat_history = [m for m in convs if m["turn_number"] < turn_num]
            user_msgs = [m for m in convs if m["role"] == "user" and m["turn_number"] == turn_num]
            if not user_msgs:
                continue
            user_query = user_msgs[0]["content"]

            music_msgs = [m for m in convs if m["role"] == "music" and m["turn_number"] == turn_num]
            gt_track_ids = [m["content"] for m in music_msgs if m.get("content")]
            if not gt_track_ids:
                continue
            # Inflate GT-only ids into a runtime-like top-20 candidate pool
            # via metadata-rich cosine NN. Downstream formatters (`overview`,
            # `detailed`, `titles_pool`, `intent_groups`, similarity hints)
            # all consume this `track_ids` so the demo prompt shape now
            # matches what `PasResponseGenerator.generate(track_ids=[20])`
            # passes at inference.
            track_ids = _inflate_compile_track_pool(gt_track_ids)

            # Pass conversation context through so query-embedding helpers
            # hit the same cache key that qemb retrieval pool would write
            # (see strategies/pool/qemb.py for the parallel hardcoding of
            # `thought=None`).
            emb_ctx = dict(
                user_profile=item.get("user_profile"),
                conversation_goal=item.get("conversation_goal"),
                goal_progress_assessments=item.get("goal_progress_assessments"),
                thought=None,
            )
            groups = classify_intents(track_ids, user_query, chat_history, kv, top_n=20, **emb_ctx)

            # Selective chat-expand parity with runtime — only query-relevant
            # prior tracks get expanded to metadata; the rest are masked as
            # `[omitted]` (see fmt_chat_history). Mirrors PasResponseGenerator
            # default. Override via env `MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT=off`.
            chat_mode = os.getenv("MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT", "selective").strip().lower()
            selective_relevant_ids = None
            if chat_mode == "selective":
                from mymodule.strategies.response.pas.generator import (
                    _compute_relevant_priors,
                )

                selective_relevant_ids = _compute_relevant_priors(
                    user_query=user_query,
                    chat_history=chat_history,
                    candidate_track_ids=track_ids[:5],
                    kv=kv,
                    threshold_md=float(os.getenv("MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_MD", "0.55")),
                    threshold_bpr=float(os.getenv("MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_BPR", "0.15")),
                    user_profile=item.get("user_profile"),
                    conversation_goal=item.get("conversation_goal"),
                    goal_progress_assessments=item.get("goal_progress_assessments"),
                    thought=None,
                )

            # Pipeline parity with runtime: enrich `user_profile` with prior
            # listens. Train-side leakage cut — `filter_before_date=session_date`
            # drops any session on/after this example's date (strict `<`),
            # `exclude_session_id=session_id` drops self even on the same date.
            # Defensive `.get()` so a missing field falls back to demographics-only
            # rather than crashing compile.
            from mymodule.strategies.response.pas.generator import (
                _format_overused_bigrams_for_prompt,
                _format_response_style_notes,
                _format_user_profile_with_history,
                _user_history_enabled_from_env,
                _user_history_sim_threshold_from_env,
                _user_history_top_k_from_env,
            )

            hf_user_id = item.get("user_id") or ""
            hf_session_id = item.get("session_id") or ""
            hf_session_date = item.get("session_date") or ""

            if hf_user_id and hf_session_id and hf_session_date:
                user_profile_str = _format_user_profile_with_history(
                    item.get("user_profile"),
                    hf_user_id,
                    kv,
                    user_query,
                    enabled=_user_history_enabled_from_env(),
                    top_k=_user_history_top_k_from_env(),
                    sim_threshold=_user_history_sim_threshold_from_env(),
                    filter_before_date=hf_session_date,
                    exclude_session_id=hf_session_id,
                )
            else:
                if not getattr(_build_examples, "_warned_missing_keys", False):
                    logger.warning(
                        f"[pas-compile] item missing user_id/session_id/session_date "
                        f"(uid={hf_user_id!r} sid={hf_session_id!r} sd={hf_session_date!r}); "
                        "falling back to demographics-only user_profile."
                    )
                    _build_examples._warned_missing_keys = True
                user_profile_str = fmt_user_profile(item.get("user_profile"))

            # `with_crawl` is gated by the same env that runtime PAS uses
            # (`MYMODULE_LLM_RECOMMENDED_WITH_CRAWL`, default ON). Mirroring
            # ensures the compile demos carry the same `{caption; lyric; key;
            # tempo}` segments the LLM sees at inference — without this,
            # demos look thinner than the live prompt and DSPy fewshot
            # learns to ignore the crawl block.
            _with_crawl = os.getenv("MYMODULE_LLM_RECOMMENDED_WITH_CRAWL", "1").strip().lower() not in (
                "0",
                "false",
                "off",
                "",
            )
            # Lazy import inside the loop body — `fmt_recommended_titles_pool`
            # is consumed only here, so ruff does not strip it as unused at
            # the top-level import block.
            from mymodule.strategies.response.base_dspy import (
                fmt_recommended_titles_pool as _fmt_recommended_titles_pool,
            )

            inputs = dict(
                user_query=user_query,
                listener_goal=fmt_conversation_goal(item.get("conversation_goal")),
                response_style_notes=_format_response_style_notes(
                    user_query,
                    chat_history,
                    item.get("conversation_goal"),
                ),
                chat_history=fmt_chat_history(chat_history, kv=kv, selective_relevant_ids=selective_relevant_ids),
                user_profile=user_profile_str,
                recommended_tracks_overview=fmt_tracks_overview(track_ids, 20, kv),
                recommended_tracks_detailed=fmt_recommended_tracks(track_ids, 5, kv, with_crawl=_with_crawl),
                # Closed-set citation grounding — runtime InputField. Without
                # it the LLM at compile sees no "must cite from this pool"
                # constraint, so demos can normalize free-form citations and
                # train fewshot in a direction REPAIR has to undo at inference.
                recommended_titles_pool=_fmt_recommended_titles_pool(track_ids, 20, kv),
                intent_groups=format_intent_groups_for_prompt(groups),
                # All three similarity hints use `graded=True` so the LLM sees a
                # 3-tier band label ("strong" / "moderate" / "weak") instead of
                # a numeric score. Lowered thresholds (track 0.45 / query 0.30 /
                # lyric 0.25) replace the prior conservative defaults that left
                # most turn-1 demos with empty hint fields.
                track_similarity_hints=fmt_track_similarity_hints(track_ids, 5, chat_history, kv, graded=True),
                query_similarity_hints=fmt_query_similarity_hints(
                    track_ids,
                    5,
                    user_query,
                    kv,
                    graded=True,
                    chat_history=chat_history,
                    **emb_ctx,
                ),
                lyric_similarity_hints=fmt_lyric_similarity_hints(
                    track_ids, 5, user_query, kv, graded=True, chat_history=chat_history, **emb_ctx
                ),
                # `overused_words` is process-wide at runtime; compile has no
                # accepted-response counter yet. Seed `overused_bigrams` the
                # same way runtime does so demos learn the first-batch lexical
                # diversity contract instead of treating the field as absent.
                overused_words="",
                overused_bigrams=_format_overused_bigrams_for_prompt([]),
            )

            ex = dspy.Example(
                **inputs,
                # `personalization_anchors` and `track_roles` are filled by
                # the LLM during bootstrap (not from GT, which doesn't carry
                # per-signal mapping or pre-committed indices). Empty
                # placeholders keep DSPy's example shape well-formed while
                # bootstrap traces produce real values.
                personalization_anchors="",
                track_roles="",
                themes="",
                themes_excluded_patterns="",
                cited_titles=_extract_cited_titles(gt_response),
                response=gt_response,
            ).with_inputs(*_PAS_INPUT_KEYS)

            cg = item.get("conversation_goal") or {}
            fp = _example_fingerprint(inputs)
            meta_by_fp[fp] = {
                "category": cg.get("category", ""),
                "specificity": cg.get("specificity", ""),
                "turn_position": _turn_position(turn_num),
                "turn_number": turn_num,
                "gt_word_count": len(gt_response.split()),
            }
            if keep_raw_for_embed:
                # Compile-only context for KNN demo embedding — stripped before
                # the sidecar is written (see `_embed_demo_pool`).
                meta_by_fp[fp]["_user_query_raw"] = user_query
                meta_by_fp[fp]["_chat_history_raw"] = chat_history
                meta_by_fp[fp]["_conversation_goal_raw"] = item.get("conversation_goal")
                meta_by_fp[fp]["_user_profile_raw"] = item.get("user_profile")
                meta_by_fp[fp]["_goal_progress_raw"] = item.get("goal_progress_assessments")
            examples.append(ex)
            pbar.set_postfix(collected=len(examples))
            if len(examples) >= train_n:
                break
        if len(examples) >= train_n:
            break
    pbar.close()

    if stratify_by == "specificity":
        examples = _stratify_round_robin(examples, meta_by_fp, key_fn=lambda m: m.get("specificity", ""))
    elif stratify_by == "bucket":
        examples = _stratify_round_robin(
            examples,
            meta_by_fp,
            key_fn=lambda m: (m.get("specificity", ""), m.get("turn_position", "")),
        )

    return examples, meta_by_fp


def _make_simple_length_aware_metric(
    min_words: int,
    max_words: int,
    tracker: LexDiversityTracker | None = None,
):
    """Same length-floor + judge wrapper but WITHOUT tqdm side-effect or score_log.

    Used by `_parallel_bootstrap_demos` which manages its own progress bar +
    score_log to avoid double counter updates.

    When `tracker` is provided and its weight > 0, the returned score blends
    judge with `tracker.novelty(response)`. The parallel acceptance loop owns
    `tracker.register_accepted(...)` so the metric stays side-effect-free here.
    """
    from mymodule.strategies.response.judge.judge import Judge

    judge_tls = threading.local()

    def _attr(obj: Any, name: str, default: str = "") -> str:
        val = getattr(obj, name, default)
        if val is None:
            return default
        return str(val)

    def _thread_judge() -> Judge:
        judge = getattr(judge_tls, "judge", None)
        if judge is None:
            judge = Judge()
            judge_tls.judge = judge
        return judge

    def _wrapped(ex: Any, pred: Any, trace: Any = None) -> float:
        text = (getattr(pred, "response", "") or "").strip()
        if not text:
            return 0.0
        wc = len(text.split())
        if wc < min_words or wc > max_words:
            return 0.0
        score = _thread_judge().score(
            user_query=_attr(ex, "user_query"),
            chat_history=_attr(ex, "chat_history"),
            user_profile=_attr(ex, "user_profile"),
            listener_goal=_attr(ex, "listener_goal"),
            recommended_tracks_detailed=_attr(ex, "recommended_tracks_detailed"),
            predicted_response=text,
            gt_response=_attr(ex, "response"),
        )
        judge = float(score.mean_normalized)
        if tracker is None:
            return judge
        return float(tracker.combine(judge, text))

    return _wrapped


def _resolve_max_pending_futures(num_workers: int) -> int:
    """Bound concurrent submitted futures so LM call artifacts are released promptly."""
    default = max(1, num_workers * 2)
    raw = os.getenv("MYMODULE_PAS_COMPILE_MAX_PENDING", "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            f"Invalid MYMODULE_PAS_COMPILE_MAX_PENDING={raw!r}; using default {default}."
        )
        return default
    return max(1, parsed)


def _format_reflection_feedback(
    *,
    score: Any,
    threshold: float,
    gt_score: float,
    rewrite_margin: float,
    min_p_score: float = 0.0,
    min_e_score: float = 0.0,
    citation_feedback: str = "",
) -> str:
    """Compact judge feedback used by the compile-only reflection prompt."""
    if score is None:
        p_score = e_score = 1
        p_reason = e_reason = "candidate was empty, too short, too long, or judge failed"
        mean = 0.0
    else:
        p_score = getattr(score, "p_score", 1)
        e_score = getattr(score, "e_score", 1)
        p_reason = str(getattr(score, "p_reason", "") or "").strip()
        e_reason = str(getattr(score, "e_reason", "") or "").strip()
        mean = float(getattr(score, "mean_normalized", 0.0))
    target = max(float(threshold), float(gt_score) + float(rewrite_margin))
    dim_targets: list[str] = []
    if min_p_score > 0:
        dim_targets.append(f"Personalization must be >= {min_p_score:g}/5")
    if min_e_score > 0:
        dim_targets.append(f"Explanation must be >= {min_e_score:g}/5")
    dim_target_text = f"\nAdditional hard targets: {'; '.join(dim_targets)}." if dim_targets else ""
    citation_text = f"\n{citation_feedback}" if citation_feedback else ""
    return (
        f"Current normalized judge score: {mean:.3f}. Required: >= {target:.3f} "
        f"(threshold={threshold:.3f}, gt={gt_score:.3f}, margin={rewrite_margin:.3f})."
        f"{dim_target_text}{citation_text}\n"
        f"Personalization: {p_score}/5 - {p_reason or 'no reason'}.\n"
        f"Explanation: {e_score}/5 - {e_reason or 'no reason'}.\n"
        "Revise for higher Personalization and Explanation Quality: anchor on the exact user request, "
        "use one concrete prior/history/profile signal when available, and tie each cited pick to a "
        "specific musical or lyrical attribute. Keep citations inside the provided title pool. "
        "Avoid stock verdicts, unsupported exact-match claims, and weak second-pick explanations."
    )


def _make_reflection_program(dspy_mod: Any) -> Any:
    """Build a compile-only PAS reflection predictor.

    This signature is intentionally not imported by runtime PAS generation, so
    zero-shot and normal few-shot inference keep the same field contract.
    """

    class CRSResponseReflection(dspy_mod.Signature):
        """Revise a PAS response candidate using judge feedback.

        The draft was generated for the same recommendation context. Improve it
        only where the feedback points to weak Personalization or Explanation
        Quality. Preserve valid grounded choices, but rewrite any generic,
        hedged, or unsupported parts. Return the same PAS output fields as the
        runtime response signature.
        """

        user_query: str = dspy_mod.InputField(desc="Current-turn user message.")
        listener_goal: str = dspy_mod.InputField(desc="Goal and specificity code.")
        response_style_notes: str = dspy_mod.InputField(desc="Style/length notes for this turn.")
        chat_history: str = dspy_mod.InputField(desc="Prior conversation and music turns.")
        user_profile: str = dspy_mod.InputField(desc="Profile and prior-listen context.")
        recommended_tracks_overview: str = dspy_mod.InputField(desc="Top-20 recommendation overview.")
        recommended_tracks_detailed: str = dspy_mod.InputField(desc="Rich metadata for top recommendations.")
        recommended_titles_pool: str = dspy_mod.InputField(desc="Closed citation set. Cite only titles from here.")
        intent_groups: str = dspy_mod.InputField(desc="Evidence groups for candidate tracks.")
        track_similarity_hints: str = dspy_mod.InputField(desc="Track-to-history similarity hints.")
        query_similarity_hints: str = dspy_mod.InputField(desc="Query-to-track similarity hints.")
        lyric_similarity_hints: str = dspy_mod.InputField(desc="Lyric-to-query similarity hints.")
        overused_words: str = dspy_mod.InputField(desc="Avoid these repeated words.")
        overused_bigrams: str = dspy_mod.InputField(desc="Avoid these repeated bigrams.")
        draft_response: str = dspy_mod.InputField(desc="Previous candidate response to improve.")
        judge_feedback: str = dspy_mod.InputField(desc="In-house P/E judge feedback and target score.")
        gt_response: str = dspy_mod.InputField(desc="Original train assistant response, for calibration only.")

        personalization_anchors: str = dspy_mod.OutputField(desc="1-3 grounded signal-to-pick mappings.")
        track_roles: str = dspy_mod.OutputField(desc="Committed cited track roles by top-20 index.")
        themes: str = dspy_mod.OutputField(desc="Personalization anchor and pick rationale trace.")
        themes_excluded_patterns: str = dspy_mod.OutputField(desc="Patterns intentionally avoided.")
        cited_titles: str = dspy_mod.OutputField(desc="One cited title per line, from the title pool.")
        response: str = dspy_mod.OutputField(desc="Final natural-language recommendation reply.")

    return dspy_mod.Predict(CRSResponseReflection)


def _parallel_bootstrap_demos(
    program: Any,
    trainset: list,
    metric_fn: Any,
    threshold: float,
    n_boot: int,
    num_workers: int,
    score_log: dict[str, float],
    tracker: LexDiversityTracker | None = None,
    *,
    meta_by_fp: dict[str, dict[str, Any]] | None = None,
    coverage_by: str = "none",
    min_per_case: int = 1,
    candidate_source: str = "bootstrap",
    rewrite_metric_fn: Any | None = None,
    rewrite_margin: float = 0.0,
    reflection_rounds: int = 0,
    rewrite_min_personalization_score: float = 0.0,
    rewrite_min_explanation_score: float = 0.0,
    require_rank1_citation: bool = False,
    min_words: int = 15,
    max_words: int = 130,
) -> list:
    """Custom parallel bootstrap demo collection.

    Replaces `dspy.BootstrapFewShot.compile` which is sequential. Runs
    `program(**inputs)` on trainset examples in parallel via ThreadPoolExecutor.
    In the default `bootstrap` mode, generated responses are scored by
    `metric_fn`. In `rewrite_gt` mode, the generated PAS response is compared
    against the original train assistant response with the same in-house judge,
    and only synthetic rewrites that beat GT by `rewrite_margin` become demos.

    Maintains its own tqdm bar with live `acc/rej_len/rej_score/err` postfix.
    Populates `score_log` with the per-example fingerprint score so the bucket
    sidecar carries demo scores like the sequential path.

    When `tracker` is provided, every demo actually appended to `accepted` (i.e.,
    a final, capped acceptance) contributes its response vocab back to the
    tracker so subsequent metric calls see an updated novelty signal. The N
    metric calls already in-flight cannot benefit (their score is fixed), but
    later workers compute against the freshly-updated vocab.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    import dspy
    from tqdm import tqdm as _tqdm

    coverage_active = coverage_by != "none" and meta_by_fp is not None
    accepted: list = []
    accepted_candidates: list[tuple[Any, Any, float, dict[str, Any]]] = []
    accepted_lock = threading.Lock()
    counters_lock = threading.Lock()
    counters = {"acc": 0, "rej_len": 0, "rej_score": 0, "errors": 0}
    stop_flag = threading.Event()

    def _attr(obj: Any, name: str, default: str = "") -> str:
        val = getattr(obj, name, default)
        if val is None:
            return default
        return str(val)

    def _thread_rewrite_judge() -> Any:
        judge = getattr(worker_tls, "rewrite_judge", None)
        if judge is None:
            from mymodule.strategies.response.judge.judge import Judge

            judge = Judge()
            worker_tls.rewrite_judge = judge
        return judge

    def _score_for_rewrite(ex: Any, pred: Any) -> tuple[float, Any | None]:
        text = (getattr(pred, "response", "") or "").strip()
        if not text:
            return 0.0, None
        wc = len(text.split())
        if wc < min_words or wc > max_words:
            return 0.0, None
        score = _thread_rewrite_judge().score(
            user_query=_attr(ex, "user_query"),
            chat_history=_attr(ex, "chat_history"),
            user_profile=_attr(ex, "user_profile"),
            listener_goal=_attr(ex, "listener_goal"),
            recommended_tracks_detailed=_attr(ex, "recommended_tracks_detailed"),
            predicted_response=text,
            gt_response=_attr(ex, "response"),
        )
        return float(score.mean_normalized), score

    def _thread_reflection_program() -> Any:
        reflection_program = getattr(worker_tls, "reflection_program", None)
        if reflection_program is None:
            reflection_program = _make_reflection_program(dspy)
            worker_tls.reflection_program = reflection_program
        return reflection_program

    def _maybe_reflect(
        ex: Any,
        pred: Any,
        inputs_view: dict[str, Any],
        synthetic_judge_score: float,
        synthetic_judge_detail: Any | None,
        gt_judge_score: float,
    ) -> tuple[Any, float, Any | None, dict[str, Any]]:
        best_pred = pred
        best_score = synthetic_judge_score
        best_detail = synthetic_judge_detail
        meta = {"reflection_attempted": False, "reflection_rounds_used": 0}
        if reflection_rounds <= 0:
            return best_pred, best_score, best_detail, meta

        for round_idx in range(1, reflection_rounds + 1):
            if _passes_rewrite_gate(
                synthetic_judge_score=best_score,
                gt_judge_score=gt_judge_score,
                threshold=threshold,
                rewrite_margin=rewrite_margin,
                synthetic_judge_detail=best_detail,
                min_p_score=rewrite_min_personalization_score,
                min_e_score=rewrite_min_explanation_score,
                require_rank1_citation=require_rank1_citation,
                cited_ranks=_extract_cited_title_ranks(best_pred, inputs_view),
            ):
                break
            meta["reflection_attempted"] = True
            citation_feedback = _rank1_gate_fail_message(
                best_pred,
                inputs_view,
                require_rank1_citation=require_rank1_citation,
            )
            feedback = _format_reflection_feedback(
                score=best_detail,
                threshold=threshold,
                gt_score=gt_judge_score,
                rewrite_margin=rewrite_margin,
                min_p_score=rewrite_min_personalization_score,
                min_e_score=rewrite_min_explanation_score,
                citation_feedback=citation_feedback,
            )
            reflected = _thread_reflection_program()(
                **inputs_view,
                draft_response=str(getattr(best_pred, "response", "") or ""),
                judge_feedback=feedback,
                gt_response=str(getattr(ex, "response", "") or ""),
            )
            reflected_score, reflected_detail = _score_for_rewrite(ex, reflected)
            if reflected_score >= best_score:
                best_pred = reflected
                best_score = reflected_score
                best_detail = reflected_detail
                meta.update(
                    {
                        "reflection_rounds_used": round_idx,
                        "reflection_last_feedback": feedback,
                    }
                )
        return best_pred, best_score, best_detail, meta

    def _process(ex: Any) -> tuple | None:
        if stop_flag.is_set():
            return None
        try:
            inputs_view = ex.inputs().toDict() if hasattr(ex.inputs(), "toDict") else dict(ex.inputs())
            local_program = getattr(worker_tls, "program", None)
            if local_program is None:
                signature = getattr(program, "signature", None)
                local_program = dspy.Predict(signature) if signature is not None else program
                worker_tls.program = local_program
            pred = local_program(**inputs_view)
            if candidate_source == "rewrite_gt":
                if rewrite_metric_fn is None and reflection_rounds <= 0:
                    raise ValueError("rewrite_metric_fn is required when candidate_source='rewrite_gt'")
                gt_pred = SimpleNamespace(response=str(getattr(ex, "response", "") or ""))
                if reflection_rounds > 0:
                    synthetic_judge_score, synthetic_judge_detail = _score_for_rewrite(ex, pred)
                    gt_judge_score, _gt_judge_detail = _score_for_rewrite(ex, gt_pred)
                    pred, synthetic_judge_score, synthetic_judge_detail, reflection_meta = _maybe_reflect(
                        ex,
                        pred,
                        inputs_view,
                        synthetic_judge_score,
                        synthetic_judge_detail,
                        gt_judge_score,
                    )
                else:
                    synthetic_judge_score = float(rewrite_metric_fn(ex, pred, None))
                    gt_judge_score = float(rewrite_metric_fn(ex, gt_pred, None))
                    synthetic_judge_detail = None
                    _gt_judge_detail = None
                    reflection_meta = {"reflection_attempted": False, "reflection_rounds_used": 0}
                score_delta = synthetic_judge_score - gt_judge_score
                citation_meta = _candidate_gate_meta(
                    pred,
                    inputs_view,
                    require_rank1_citation=require_rank1_citation,
                )
                if not _passes_rewrite_gate(
                    synthetic_judge_score=synthetic_judge_score,
                    gt_judge_score=gt_judge_score,
                    threshold=threshold,
                    rewrite_margin=rewrite_margin,
                    synthetic_judge_detail=synthetic_judge_detail,
                    min_p_score=rewrite_min_personalization_score,
                    min_e_score=rewrite_min_explanation_score,
                    require_rank1_citation=require_rank1_citation,
                    cited_ranks=citation_meta["cited_title_ranks"],
                ):
                    score = 0.0 if synthetic_judge_score == 0.0 else min(synthetic_judge_score, threshold - 1e-9)
                else:
                    score = synthetic_judge_score
                return (
                    ex,
                    pred,
                    score,
                    inputs_view,
                    {
                        "demo_origin": "synthetic_rewrite",
                        "synthetic_source": "llm_rewrite_gt_judge_better",
                        "gt_judge_score": gt_judge_score,
                        "synthetic_judge_score": synthetic_judge_score,
                        "score_delta_vs_gt": score_delta,
                        "rewrite_margin": rewrite_margin,
                        "rewrite_min_personalization_score": rewrite_min_personalization_score,
                        "rewrite_min_explanation_score": rewrite_min_explanation_score,
                        "reflection_rounds_configured": reflection_rounds,
                        **_score_detail_meta("gt", _gt_judge_detail),
                        **_score_detail_meta("synthetic", synthetic_judge_detail),
                        **citation_meta,
                        **reflection_meta,
                    },
                )
            score = float(metric_fn(ex, pred, None))
            return (ex, pred, score, inputs_view, {})
        except Exception as e:
            with counters_lock:
                counters["errors"] += 1
            logger.debug(f"[parallel-bootstrap] {type(e).__name__}: {e}")
            return None

    worker_tls = threading.local()
    pbar = _tqdm(total=len(trainset), desc=f"parallel bootstrap (workers={num_workers})")
    max_pending = _resolve_max_pending_futures(num_workers)
    logger.info(
        f"[parallel-bootstrap] bounded future window: max_pending={max_pending} "
        f"(workers={num_workers})"
    )

    def _update_pbar() -> None:
        acc_key = "pass" if coverage_active else "acc"
        acc_val = str(counters["acc"]) if coverage_active else f"{counters['acc']}/{n_boot}"
        pbar.set_postfix(
            {
                acc_key: acc_val,
                "rej_len": counters["rej_len"],
                "rej_score": counters["rej_score"],
                "err": counters["errors"],
            }
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        train_iter = iter(trainset)
        pending: set[Any] = set()
        exhausted = False
        processed = 0

        def _submit_until_window_full() -> None:
            nonlocal exhausted
            while not exhausted and not stop_flag.is_set() and len(pending) < max_pending:
                try:
                    ex = next(train_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(_process, ex))

        _submit_until_window_full()
        try:
            while pending:
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    processed += 1
                    pbar.update(1)
                    if fut.cancelled():
                        _update_pbar()
                        continue
                    result = None
                    try:
                        result = fut.result()
                    except Exception as e:
                        with counters_lock:
                            counters["errors"] += 1
                        logger.debug(f"[parallel-bootstrap] future failed: {type(e).__name__}: {e}")
                        _update_pbar()
                        continue
                    if stop_flag.is_set():
                        _update_pbar()
                        continue
                    if result is None:
                        _update_pbar()
                        continue
                    ex, pred, score, inputs_view, meta_extra = result
                    if score == 0.0:  # length-floor reject (metric returned 0)
                        with counters_lock:
                            counters["rej_len"] += 1
                    elif score >= threshold:
                        if coverage_active:
                            accepted_candidates.append((ex, pred, score, inputs_view))
                            with counters_lock:
                                counters["acc"] += 1
                            try:
                                fp = _example_fingerprint({k: inputs_view.get(k, "") for k in _PAS_INPUT_KEYS})
                                if meta_extra and meta_by_fp is not None:
                                    meta_by_fp.setdefault(fp, {}).update(meta_extra)
                                score_log[fp] = max(score_log.get(fp, 0.0), float(score))
                            except Exception:
                                pass
                        else:
                            with accepted_lock:
                                if len(accepted) < n_boot:
                                    accepted.append((ex, pred, score, inputs_view))
                                    with counters_lock:
                                        counters["acc"] += 1
                                    # populate score_log for bucket sidecar
                                    try:
                                        fp = _example_fingerprint(
                                            {k: inputs_view.get(k, "") for k in _PAS_INPUT_KEYS}
                                        )
                                        if meta_extra and meta_by_fp is not None:
                                            meta_by_fp.setdefault(fp, {}).update(meta_extra)
                                        score_log[fp] = max(score_log.get(fp, 0.0), float(score))
                                    except Exception:
                                        pass
                                    # Register vocab so downstream workers see the
                                    # accepted demo's tokens as "seen" → novelty drops.
                                    if tracker is not None:
                                        tracker.register_accepted((getattr(pred, "response", "") or "").strip())
                                    if len(accepted) >= n_boot:
                                        stop_flag.set()
                    else:
                        with counters_lock:
                            counters["rej_score"] += 1
                    _update_pbar()
                    del result
                    if processed % max(16, max_pending * 2) == 0:
                        gc.collect()
                    if stop_flag.is_set():
                        for fut in pending:
                            fut.cancel()
                        pending.clear()
                        break
                if stop_flag.is_set():
                    break
                _submit_until_window_full()
        finally:
            _update_pbar()
            pbar.close()

    if coverage_active:
        accepted = _select_synthetic_demo_bank(
            accepted_candidates,
            meta_by_fp=meta_by_fp or {},
            score_log=score_log,
            n_boot=n_boot,
            coverage_by=coverage_by,
            min_per_case=min_per_case,
        )
        if tracker is not None:
            for _ex, pred, _score, _inputs_view in accepted:
                tracker.register_accepted((getattr(pred, "response", "") or "").strip())
        coverage_counts = Counter()
        for _ex, _pred, _score, inputs_view in accepted:
            fp = _example_fingerprint({k: inputs_view.get(k, "") for k in _PAS_INPUT_KEYS})
            meta = (meta_by_fp or {}).get(fp, {})
            coverage_counts[_coverage_key_label(_coverage_key(meta, coverage_by))] += 1
        logger.info(
            f"[synthetic-bank] selected {len(accepted)}/{len(accepted_candidates)} passing candidates "
            f"by coverage={coverage_by}, min_per_case={min_per_case}; "
            f"coverage_counts={dict(sorted(coverage_counts.items()))}"
        )

    # Build dspy.Example demos from accepted (input + output fields).
    demos: list = []
    for ex, pred, score, inputs_view in accepted[:n_boot]:
        try:
            inputs_dict = {k: inputs_view.get(k, "") for k in _PAS_INPUT_KEYS}
            outputs_dict = {k: str(getattr(pred, k, "") or "") for k in _PAS_OUTPUT_KEYS}
            demo = dspy.Example(**inputs_dict, **outputs_dict).with_inputs(*_PAS_INPUT_KEYS)
            demos.append(demo)
        except Exception as e:
            logger.warning(f"[parallel-bootstrap] demo build failed: {type(e).__name__}: {e}")
    logger.info(
        f"[parallel-bootstrap] collected {len(demos)} demos "
        f"(acc={counters['acc']}, rej_len={counters['rej_len']}, "
        f"rej_score={counters['rej_score']}, err={counters['errors']})"
    )
    return demos


def _make_length_aware_judge_metric(
    min_words: int,
    max_words: int,
    score_log: dict[str, float],
    threshold: float,
    n_boot: int,
    tracker: LexDiversityTracker | None = None,
):
    """Wrap `judge_metric` with a hard length floor + ceiling and side-channel score logging.

    DSPy `BootstrapFewShot` passes per-trial (example, prediction, trace) to the
    metric; we record fingerprint → score so the bucket sidecar can rank demos by
    judge score without re-evaluating after compile.

    Updates the active tqdm progress bar's postfix with `accepted=N/n_boot` and
    rejection counts so compile progress is visible in real time.

    Returns 0.0 if the candidate text is outside [min_words, max_words] — this
    rejects both train-GT short demos (median ~37 words) and runaway long-form
    LLM generations that would blow the token budget.

    When `tracker` is provided and its weight > 0, the candidate's response text
    is blended with the accepted-demo vocab novelty signal before the threshold
    check. The combined score is also what gets logged and what gates
    `tracker.register_accepted(...)`. BootstrapFewShot runs the metric
    sequentially per example, so updating vocab on threshold-pass here is an
    acceptable approximation of "vocab of accepted demos so far". Repeat metric
    calls on the same trace (max_rounds) over-count Counter values but novelty
    only checks `t in vocab`, so the signal is stable.
    """
    from mymodule.strategies.response.judge.metric import judge_metric

    counters = {"acc": 0, "rej_len": 0, "rej_score": 0}

    def _update_tqdm() -> None:
        try:
            import tqdm as _tqdm_mod  # noqa: PLC0415

            for bar in list(_tqdm_mod.tqdm._instances):
                bar.set_postfix(
                    {
                        "acc": f"{counters['acc']}/{n_boot}",
                        "rej_len": counters["rej_len"],
                        "rej_score": counters["rej_score"],
                    },
                    refresh=True,
                )
        except Exception:
            pass

    def _wrapped(ex: Any, pred: Any, trace: Any = None) -> float:
        text = (getattr(pred, "response", "") or "").strip()
        if not text:
            counters["rej_len"] += 1
            _update_tqdm()
            return 0.0
        wc = len(text.split())
        if wc < min_words or wc > max_words:
            counters["rej_len"] += 1
            _update_tqdm()
            return 0.0
        judge = float(judge_metric(ex, pred, trace))
        score = float(tracker.combine(judge, text)) if tracker is not None else judge
        if score >= threshold:
            counters["acc"] += 1
            if tracker is not None:
                tracker.register_accepted(text)
        else:
            counters["rej_score"] += 1
        _update_tqdm()
        # Side-channel: stash the score keyed by the source example fingerprint so
        # the bucket sidecar can carry per-demo scores. We use the example's input
        # fingerprint; for a successful trace the demo we save will share it.
        try:
            inputs = ex.inputs().toDict() if hasattr(ex, "inputs") else dict(ex)
            fp = _example_fingerprint({k: inputs.get(k, "") for k in _PAS_INPUT_KEYS})
            # Keep best score per fingerprint (multiple bootstrap trials per example).
            score_log[fp] = max(score_log.get(fp, 0.0), float(score))
        except Exception as e:  # never let logging crash compile
            logger.debug(f"[pas-optimize] score-log skip: {type(e).__name__}: {e}")
        return float(score)

    return _wrapped


def _demo_to_dict(demo: Any, meta_by_fp: dict[str, dict[str, Any]], score_log: dict[str, float]) -> dict[str, Any]:
    """Serialize a `dspy.Example` demo into the bucket-sidecar shape."""
    if hasattr(demo, "inputs") and callable(getattr(demo, "inputs", None)):
        try:
            inputs_view = demo.inputs().toDict() if hasattr(demo.inputs(), "toDict") else dict(demo.inputs())
        except Exception:
            inputs_view = {k: getattr(demo, k, "") for k in _PAS_INPUT_KEYS}
    else:
        inputs_view = {k: getattr(demo, k, "") for k in _PAS_INPUT_KEYS}
    inputs_dict = {k: str(inputs_view.get(k, "") or "") for k in _PAS_INPUT_KEYS}
    outputs_dict = {k: str(getattr(demo, k, "") or "") for k in _PAS_OUTPUT_KEYS}
    fp = _example_fingerprint(inputs_dict)
    meta = dict(meta_by_fp.get(fp, {}))
    if fp in score_log:
        meta["score"] = score_log[fp]
    return {"inputs": inputs_dict, "outputs": outputs_dict, "meta": meta}


_EMBED_META_LEAK_KEYS = (
    "_user_query_raw",
    "_chat_history_raw",
    "_conversation_goal_raw",
    "_user_profile_raw",
    "_goal_progress_raw",
)


def _embed_demo_pool(bucket_demos: list[dict[str, Any]], meta_by_fp: dict[str, dict[str, Any]]) -> int:
    """Embed each accepted demo's query and stamp into `demo['meta']`.

    Uses the same `InstructedQueryEmbedder.embed_query(user_query, chat_history=...)`
    signature that the qemb retrieval pool / inference-time generator use, so
    KNN demo retrieval at runtime can call the identical embedder and get
    aligned vectors. Per-call cache hits land in the shared KV namespace.

    Mutates `bucket_demos` in-place. Strips compile-only `_*_raw` keys from
    both `demo['meta']` and `meta_by_fp` so the sidecar never leaks them.

    Returns the number of demos that successfully received an embedding. A
    failure for any single demo logs a warning and continues — the demo just
    won't be eligible for the KNN path (router auto-falls back to bucket).
    """
    from mymodule.feature.ollama_embed import get_shared_instructed_embedder

    embedder = get_shared_instructed_embedder()
    model_id = embedder.raw.model_id
    n_ok = 0
    for d in bucket_demos:
        meta = d.get("meta") or {}
        # Look up raw context by fingerprint — the same one _demo_to_dict computed.
        inputs_dict = d.get("inputs") or {}
        fp = _example_fingerprint(inputs_dict)
        raw_meta = meta_by_fp.get(fp) or {}
        user_query = raw_meta.get("_user_query_raw") or inputs_dict.get("user_query") or ""
        chat_history = raw_meta.get("_chat_history_raw")
        conversation_goal = raw_meta.get("_conversation_goal_raw")
        user_profile = raw_meta.get("_user_profile_raw")
        goal_progress = raw_meta.get("_goal_progress_raw")
        try:
            vec = embedder.embed_query(
                user_query,
                chat_history=chat_history,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress_assessments=goal_progress,
                thought=None,
            )
            meta["query_embedding"] = [float(x) for x in vec.tolist()]
            meta["embedding_model"] = model_id
            n_ok += 1
        except Exception as e:
            logger.warning(
                f"[pas-optimize] embed_query failed for fp={fp[:8]} "
                f"({type(e).__name__}: {e}) — demo will be bucket-only at runtime."
            )
        # Strip compile-only fields from BOTH locations.
        for k in _EMBED_META_LEAK_KEYS:
            meta.pop(k, None)
            raw_meta.pop(k, None)
        d["meta"] = meta
    return n_ok


def _save_bucket_sidecar(out_path: Path, demos: list[dict[str, Any]], compile_args: dict[str, Any]) -> None:
    """Write the DemoRouter-friendly sidecar."""
    payload = {
        "version": 1,
        "schema": "pas_demo_buckets@1",
        "compiled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "compile_args": compile_args,
        "demos": demos,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _signature_hash() -> str:
    """SHA-1 of the live PAS signature docstring + field descs.

    Used by `pas_{provider}.meta.json` so we can later detect ckpt staleness when
    `signature.py` is edited (Tier D / follow-up PR will add an active staleness
    check at load time).
    """
    from mymodule.strategies.response.pas.signature import CRSResponse

    parts: list[str] = [(CRSResponse.__doc__ or "").strip()]
    for name, field in {**CRSResponse.input_fields, **CRSResponse.output_fields}.items():
        desc = ""
        try:
            desc = (field.json_schema_extra or {}).get("desc", "") or ""
        except Exception:
            pass
        parts.append(f"{name}::{desc}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _save_metadata_sidecar(out_path: Path, args: argparse.Namespace, n_demos: int, bucket_counts: dict) -> None:
    payload = {
        "version": 1,
        "compiled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "signature_hash": _signature_hash(),
        "args": {
            "response_provider": args.response_provider,
            "seed": args.seed,
            "llm_temperature": getattr(args, "llm_temperature", None),
            "train_n": args.train_n,
            "n_label": args.n_label,
            "n_boot": args.n_boot,
            "n_pool": getattr(args, "n_pool", 0),
            "metric_threshold": args.metric_threshold,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "lexdiv_weight": getattr(args, "lexdiv_weight", 0.0),
            "synthetic_bank": getattr(args, "synthetic_bank", False),
            "candidate_source": getattr(args, "candidate_source", "bootstrap"),
            "rewrite_margin": getattr(args, "rewrite_margin", 0.0),
            "rewrite_min_personalization_score": getattr(args, "rewrite_min_personalization_score", 0.0),
            "rewrite_min_explanation_score": getattr(args, "rewrite_min_explanation_score", 0.0),
            "require_rank1_citation": getattr(args, "require_rank1_citation", False),
            "reflection_rounds": getattr(args, "reflection_rounds", 0),
            "coverage_by": getattr(args, "coverage_by", "none"),
            "min_demos_per_case": getattr(args, "min_demos_per_case", 0),
        },
        "judge": {
            "provider": os.getenv("MYMODULE_JUDGE_PROVIDER", ""),
            "model": os.getenv("MYMODULE_JUDGE_MODEL", ""),
        },
        "generator": {
            "provider": args.response_provider,
            "model": (
                os.getenv("MYMODULE_LLM_OPENAI_MODEL", "")
                if args.response_provider == "openai"
                else os.getenv("MYMODULE_LLM_OLLAMA_MODEL", "")
            ),
            "temperature": getattr(args, "llm_temperature", None),
        },
        "n_demos_total": n_demos,
        "bucket_counts": {f"{k[0]}|{k[1]}": v for k, v in bucket_counts.items()},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the argparse parser used by `main()`.

    Extracted so tests under `tests/strategies/response/test_pas_rules.py`
    can introspect default values without invoking the full compile.
    """
    parser = argparse.ArgumentParser(description="Compile DSPy few-shot program for PAS response generation.")
    parser.add_argument(
        "--response-provider",
        choices=["ollama", "openai"],
        default="openai",
        help="LM provider for the PAS signature compilation (default: openai for judge alignment).",
    )
    parser.add_argument("--train-n", type=int, default=int(os.getenv("MYMODULE_LLM_OPTIM_TRAIN_N", "100")))
    parser.add_argument(
        "--n-label",
        type=int,
        default=0,
        help="max_labeled_demos for BootstrapFewShot. Default 0 because train GT median word "
        "count is ~37, which is *within* the post-PR-#80 spec word budget (HH/HL 25-45w, "
        "LH/LL 35-55w) but labeling would still skew toward GT's terser register — bootstrapped "
        "demos (--n-boot) cover this gap by sampling LLM responses that pass length+judge filter.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=int(os.getenv("MYMODULE_LLM_FEW_SHOT_N", "5")),
        help="max_bootstrapped_demos for BootstrapFewShot. LLM is run on training inputs and "
        "demos are taken from traces whose response passes (length floor + judge threshold).",
    )
    # Rule defaults are sourced from `mymodule/strategies/response/pas/config/rule.yaml`
    # so signature / compile / runtime stay aligned. CLI flags override on demand.
    _rules = load_rules()
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=float(os.getenv("MYMODULE_LLM_METRIC_THRESHOLD", str(_rules.compile_metric.threshold))),
        help="Minimum judge mean_normalized score (0-1) for a bootstrap trace to qualify "
        "as a demo. Default from `config/rule.yaml::compile_metric.threshold` "
        "(currently 0.5 ≈ average dim score 3 'Adequate').",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=_rules.response_length.compile_floor_words,
        help="Length floor (words). Below this the metric returns 0.0 regardless of judge "
        "score. Default from `config/rule.yaml::response_length.compile_floor_words` "
        "(15 = ~10w margin under HH lower-bound 25w). Edit rule.yaml to change project-wide.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=_rules.response_length.compile_ceiling_words,
        help="Upper guard (words). Above this the metric returns 0.0. Default from "
        "`config/rule.yaml::response_length.compile_ceiling_words` (130 = ~10w margin over "
        "the signature's NEVER-exceed-120w hard cap). Edit rule.yaml to change project-wide.",
    )
    parser.add_argument(
        "--stratify-by",
        choices=["none", "specificity", "bucket"],
        default="none",
        help="Re-order trainset so adjacent examples cycle through buckets, before passing "
        "to BootstrapFewShot. 'specificity' rotates HH/HL/LH/LL; 'bucket' rotates "
        "(specificity, turn_position). Use when n_boot demos are converging on a single "
        "bucket (HF dataset's natural ordering favors category=B / specificity=HL).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="If > 1, use a custom parallel bootstrap (ThreadPoolExecutor) instead of "
        "DSPy's sequential BootstrapFewShot. Each worker runs program(input) + metric "
        "concurrently. Stops as soon as n_boot demos pass threshold. Recommended 4-8 "
        "for endpoints without rate limits — 5-6x speedup. Set to 1 (default) for legacy "
        "BootstrapFewShot path with multi-round support.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed: drives random.seed for sampling AND is forwarded to the "
        "compile-time LM (via dspy.LM seed=) so provider-supported backends "
        "(vLLM, SGLang, OpenAI) produce deterministic demo bootstrap. Different "
        "--seed values give different but reproducible compile realizations.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Compile-time LM temperature (default 0.0 for deterministic). "
        "Combined with --seed, gives reproducible demos. Set to 0.7 to match "
        "the runtime LM (varied responses) at the cost of reproducibility.",
    )
    parser.add_argument(
        "--query-composition",
        choices=["legacy", "rich"],
        default=None,
        help="Override query composition for retrieval helpers / selective gate. "
        "`legacy` (default) = title/artist/album from prior music turns; `rich` = "
        "also year/popularity/tags (experimental). "
        "When unset, falls back to env `MYMODULE_QEMB_QUERY_COMPOSITION` (default `legacy`).",
    )
    parser.add_argument(
        "--lexdiv-weight",
        type=float,
        default=0.25,
        help="Lexical-diversity component weight in the compile metric. The final "
        "score is `(1 - w) * judge + w * novelty`, where novelty is the fraction "
        "of ≥4-letter content tokens in the candidate response that are NOT yet "
        "present in the vocab of accepted demos (stopwords / music-grounding "
        "tokens filtered, same as inference-time governor). Default 0.25 gives "
        "a 3:1 judge:lexical blend — judge still anchors quality while demos "
        "are nudged toward distinct vocabulary. Set to 0.0 to fall back to "
        "pure judge.",
    )
    parser.add_argument(
        "--synthetic-bank",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MYMODULE_PAS_SYNTHETIC_BANK", "1").strip().lower() not in ("0", "false", "off", ""),
        help="When enabled with --parallel-workers > 1, score all generated bootstrap candidates first, "
        "then select the final few-shot bank by judge score plus coverage quotas instead of taking the "
        "first N threshold-passing traces. This is the recommended path for PAS few-shot.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=["bootstrap", "rewrite_gt"],
        default=os.getenv("MYMODULE_PAS_SYNTHETIC_CANDIDATE_SOURCE", "bootstrap"),
        help="How to build few-shot demo candidates in the parallel compile path. "
        "`bootstrap` keeps the existing behavior: generate PAS responses from train inputs and "
        "gate them by judge threshold. `rewrite_gt` additionally scores the original train "
        "assistant response and accepts a synthetic PAS rewrite only when it beats that GT "
        "response by --rewrite-margin under the same in-house judge.",
    )
    parser.add_argument(
        "--rewrite-margin",
        type=float,
        default=float(os.getenv("MYMODULE_PAS_REWRITE_MARGIN", "0.02")),
        help="Minimum judge-score improvement required for --candidate-source rewrite_gt. "
        "Scores are normalized to [0,1], so 0.02 is roughly 0.1 point on a 1-5 judge scale.",
    )
    parser.add_argument(
        "--reflection-rounds",
        type=int,
        default=int(os.getenv("MYMODULE_PAS_REFLECTION_ROUNDS", "0")),
        help="Compile-only self-reflection rounds for --candidate-source rewrite_gt. "
        "When a synthetic rewrite fails the judge threshold or GT-improvement gate, "
        "feed its GLM P/E feedback back into a reflection prompt and re-score the revision. "
        "Use this with a higher --llm-temperature for diverse demo drafts, while keeping "
        "blind inference temperature low.",
    )
    parser.add_argument(
        "--rewrite-min-personalization-score",
        type=float,
        default=float(os.getenv("MYMODULE_PAS_REWRITE_MIN_PERSONALIZATION_SCORE", "0")),
        help="Optional hard floor on the synthetic demo judge Personalization dimension (1-5 scale). "
        "Use with --candidate-source rewrite_gt to avoid demos that pass by Explanation only.",
    )
    parser.add_argument(
        "--rewrite-min-explanation-score",
        type=float,
        default=float(os.getenv("MYMODULE_PAS_REWRITE_MIN_EXPLANATION_SCORE", "0")),
        help="Optional hard floor on the synthetic demo judge Explanation dimension (1-5 scale). "
        "Use with --candidate-source rewrite_gt to keep P/E balanced.",
    )
    parser.add_argument(
        "--require-rank1-citation",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MYMODULE_PAS_REQUIRE_RANK1_CITATION", "0").strip().lower()
        in ("1", "true", "yes", "on"),
        help="For rewrite_gt synthetic demos, require the generated demo to cite the rank-1 title. "
        "Compile examples inject the train GT track at rank 1; this prevents demos from teaching "
        "the model to abandon the strongest retrieved/GT candidate for another pool item.",
    )
    parser.add_argument(
        "--coverage-by",
        choices=[
            "none",
            "specificity",
            "specificity_turn",
            "category_specificity",
            "category_specificity_turn",
        ],
        default=os.getenv("MYMODULE_PAS_SYNTHETIC_COVERAGE_BY", "category_specificity_turn"),
        help="Coverage axis for --synthetic-bank final demo selection. "
        "category_specificity_turn gives the richest bank; use specificity_turn if train_n/n_pool is small.",
    )
    parser.add_argument(
        "--min-demos-per-case",
        type=int,
        default=int(os.getenv("MYMODULE_PAS_SYNTHETIC_MIN_DEMOS_PER_CASE", "1")),
        help="Target minimum demos per coverage case before filling by global score. Best-effort: "
        "sparse cases with no judge-passing candidates are skipped.",
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=0,
        help="Pool size for KNN demo retrieval. When >0, this OVERRIDES --n-boot — "
        "the compile collects N pool-size demos (still gated by length floor + "
        "metric threshold), then computes a query embedding for each one and "
        "stores it in the bucket sidecar. At inference, "
        "MYMODULE_PAS_DEMO_SELECTION=knn switches the router to per-call kNN "
        "retrieval that returns the demos most relevant to the current "
        "session's user_query, then re-ranks within the kNN window by judge / "
        "lexdiv score. Compile cost scales linearly with --n-pool. Default 0 "
        "keeps the legacy bucket-routing behavior (no embeddings stored).",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.query_composition is not None:
        os.environ["MYMODULE_QEMB_QUERY_COMPOSITION"] = args.query_composition

    # --n-pool overrides --n-boot when set (KNN demo pool mode). Keep the
    # original n_boot around for the meta sidecar so the provenance is honest.
    knn_pool_mode = args.n_pool > 0
    if knn_pool_mode:
        if args.n_boot != args.n_pool:
            logger.info(f"--n-pool={args.n_pool} overrides --n-boot={args.n_boot} for KNN demo pool compile.")
        args.n_boot = args.n_pool

    random.seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CKPT_DIR / f"pas_{args.response_provider}.json"
    bucket_path = CKPT_DIR / f"pas_{args.response_provider}.buckets.json"
    meta_path = CKPT_DIR / f"pas_{args.response_provider}.meta.json"

    import dspy

    from mymodule.strategies.response.base_dspy import try_open_kvstore
    from mymodule.strategies.response.pas.signature import CRSResponse
    from mymodule.utils.common_dspy import ensure_lm_configured

    ensure_lm_configured(args.response_provider, seed=args.seed, temperature=args.llm_temperature)
    logger.info(f"LM configured with seed={args.seed}, temperature={args.llm_temperature}")

    kv = try_open_kvstore()
    chat_mode = os.getenv("MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT", "selective").strip().lower()
    if chat_mode == "selective":
        sel = (
            f"md={os.getenv('MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_MD', '0.55')},"
            f"bpr={os.getenv('MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_BPR', '0.15')}"
        )
    else:
        sel = "n/a"
    logger.info(
        f"Building up to {args.train_n} training examples "
        f"(stratify_by={args.stratify_by}, chat_expand={chat_mode}, sel_thresholds=[{sel}]) …"
    )
    trainset, meta_by_fp = _build_examples(
        args.train_n,
        kv,
        stratify_by=args.stratify_by,
        keep_raw_for_embed=knn_pool_mode,
    )
    logger.info(f"Built {len(trainset)} examples (after filtering 'Unknown message' / empty turns).")

    if args.n_boot <= 0 and args.n_label <= 0:
        raise SystemExit("Both --n-label and --n-boot are 0 — nothing to compile.")
    if len(trainset) < max(args.n_label, args.n_boot):
        logger.warning(f"only {len(trainset)} examples, fewer than max(n_label={args.n_label}, n_boot={args.n_boot})")
    if args.parallel_workers <= 1 and args.candidate_source != "bootstrap":
        raise SystemExit("--candidate-source rewrite_gt requires --parallel-workers > 1")

    # PAS reasoning happens silently inside the LLM; use Predict so compile targets
    # match runtime shape (no CoT rationale field).
    program = dspy.Predict(CRSResponse)

    score_log: dict[str, float] = {}
    tracker_weight = 0.0 if args.candidate_source == "rewrite_gt" else args.lexdiv_weight
    tracker = LexDiversityTracker(weight=tracker_weight)
    if args.candidate_source == "rewrite_gt" and args.lexdiv_weight > 0:
        logger.info(
            "rewrite_gt candidate source uses pure judge scores for GT-vs-synthetic comparison; "
            f"ignoring lexdiv_weight={args.lexdiv_weight:.3f} during compile scoring."
        )
    if tracker.weight > 0:
        logger.info(
            f"Lexical-diversity reward enabled: weight={tracker.weight:.3f} "
            f"(judge:lexical ≈ {1 - tracker.weight:.2f}:{tracker.weight:.2f})"
        )

    if args.parallel_workers > 1:
        # Custom parallel path — bypasses DSPy's sequential BootstrapFewShot.
        simple_metric = _make_simple_length_aware_metric(args.min_words, args.max_words, tracker=tracker)
        rewrite_metric = (
            _make_simple_length_aware_metric(args.min_words, args.max_words, tracker=None)
            if args.candidate_source == "rewrite_gt"
            else None
        )
        coverage_by = args.coverage_by if args.synthetic_bank else "none"
        logger.info(
            f"Compiling: parallel_bootstrap(workers={args.parallel_workers}, n_boot={args.n_boot}, "
            f"threshold={args.metric_threshold}, len=[{args.min_words},{args.max_words}], "
            f"lexdiv_weight={tracker.weight:.3f}, synthetic_bank={args.synthetic_bank}, "
            f"coverage_by={coverage_by}, min_per_case={args.min_demos_per_case}, "
            f"candidate_source={args.candidate_source}, rewrite_margin={args.rewrite_margin:.3f}, "
            f"rewrite_min_p={args.rewrite_min_personalization_score:.1f}, "
            f"rewrite_min_e={args.rewrite_min_explanation_score:.1f}, "
            f"require_rank1_citation={args.require_rank1_citation}, "
            f"reflection_rounds={args.reflection_rounds})"
        )
        demos = _parallel_bootstrap_demos(
            program=program,
            trainset=trainset,
            metric_fn=simple_metric,
            threshold=args.metric_threshold,
            n_boot=args.n_boot,
            num_workers=args.parallel_workers,
            score_log=score_log,
            tracker=tracker,
            meta_by_fp=meta_by_fp,
            coverage_by=coverage_by,
            min_per_case=max(0, args.min_demos_per_case),
            candidate_source=args.candidate_source,
            rewrite_metric_fn=rewrite_metric,
            rewrite_margin=max(0.0, args.rewrite_margin),
            reflection_rounds=max(0, args.reflection_rounds),
            rewrite_min_personalization_score=max(0.0, args.rewrite_min_personalization_score),
            rewrite_min_explanation_score=max(0.0, args.rewrite_min_explanation_score),
            require_rank1_citation=bool(args.require_rank1_citation),
            min_words=args.min_words,
            max_words=args.max_words,
        )
        compiled = dspy.Predict(CRSResponse)
        compiled.demos = demos
    else:
        metric = _make_length_aware_judge_metric(
            args.min_words,
            args.max_words,
            score_log,
            args.metric_threshold,
            args.n_boot,
            tracker=tracker,
        )

        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_labeled_demos=args.n_label,
            max_bootstrapped_demos=args.n_boot,
            metric_threshold=args.metric_threshold,
        )
        logger.info(
            f"Compiling: BootstrapFewShot(metric=length_aware_judge, n_label={args.n_label}, "
            f"n_boot={args.n_boot}, threshold={args.metric_threshold}, "
            f"len=[{args.min_words},{args.max_words}], lexdiv_weight={tracker.weight:.3f})"
        )
        compiled = optimizer.compile(program, trainset=trainset)

    compiled.save(str(out_path))
    logger.success(f"Saved compiled program to {out_path}")

    # Bucket sidecar — DemoRouter at runtime reads this.
    demos_raw = list(getattr(compiled, "demos", []) or [])
    if not demos_raw:
        logger.warning(
            "compiled.demos is empty — no candidates passed (length floor + judge threshold). "
            "Try lowering --metric-threshold or --min-words, or raising --train-n."
        )
    bucket_demos = [_demo_to_dict(d, meta_by_fp, score_log) for d in demos_raw]

    embedding_model_recorded = ""
    if knn_pool_mode and bucket_demos:
        logger.info(f"Embedding {len(bucket_demos)} demos for KNN retrieval (pool mode) …")
        n_emb_ok = _embed_demo_pool(bucket_demos, meta_by_fp)
        if n_emb_ok > 0:
            embedding_model_recorded = bucket_demos[0]["meta"].get("embedding_model", "")
        logger.success(
            f"Embedded {n_emb_ok}/{len(bucket_demos)} demos "
            f"(model={embedding_model_recorded or '?'}); "
            "set MYMODULE_PAS_DEMO_SELECTION=knn at inference to enable kNN routing."
        )

    compile_args = {
        "n_label": args.n_label,
        "n_boot": args.n_boot,
        "n_pool": args.n_pool,
        "metric_threshold": args.metric_threshold,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "seed": args.seed,
        "train_n": args.train_n,
        "response_provider": args.response_provider,
        "generator_model": (
            os.getenv("MYMODULE_LLM_OPENAI_MODEL", "")
            if args.response_provider == "openai"
            else os.getenv("MYMODULE_LLM_OLLAMA_MODEL", "")
        ),
        "llm_temperature": args.llm_temperature,
        "judge_model": os.getenv("MYMODULE_JUDGE_MODEL", ""),
        "lexdiv_weight": tracker.weight,
        "synthetic_bank": args.synthetic_bank,
        "candidate_source": args.candidate_source,
        "rewrite_margin": args.rewrite_margin,
        "rewrite_min_personalization_score": args.rewrite_min_personalization_score,
        "rewrite_min_explanation_score": args.rewrite_min_explanation_score,
        "require_rank1_citation": args.require_rank1_citation,
        "reflection_rounds": args.reflection_rounds,
        "coverage_by": args.coverage_by if args.synthetic_bank else "none",
        "min_demos_per_case": args.min_demos_per_case,
        "embedding_model": embedding_model_recorded,
    }
    _save_bucket_sidecar(bucket_path, bucket_demos, compile_args)
    logger.success(f"Saved bucket sidecar with {len(bucket_demos)} demos → {bucket_path}")

    # Metadata sidecar — provenance + signature hash for staleness detection.
    bucket_counts: dict[tuple[str, str], int] = {}
    for d in bucket_demos:
        meta = d.get("meta") or {}
        key = (str(meta.get("specificity", "") or ""), str(meta.get("turn_position", "") or ""))
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
    _save_metadata_sidecar(meta_path, args, n_demos=len(bucket_demos), bucket_counts=bucket_counts)
    logger.success(f"Saved metadata sidecar → {meta_path}")
    logger.info(f"Bucket counts: {sorted(bucket_counts.items())}")

    # Verify the standard ckpt loads — same shape (Predict) the runtime generator uses.
    loaded = dspy.Predict(CRSResponse)
    loaded.load(str(out_path))
    logger.success("Load verification: OK")


if __name__ == "__main__":
    main()
