"""Devset inference script.

Generates 1,000 sessions x 8 turns = 8,000 predictions in the evaluator
output format. Each turn receives prior-turn ground truth as history
(teacher forcing).

Usage:
    uv run python -m mymodule.run_inference_devset --tid <tid>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ruff: noqa: E402


def _preload_lightgbm_for_gbm_tid(argv: list[str]) -> None:
    """Load LightGBM before pandas/datasets for GBM inference on macOS."""
    if not any("__gbm" in arg for arg in argv):
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import lightgbm  # noqa: F401


_preload_lightgbm_for_gbm_tid(sys.argv[1:])

import pandas as pd
from datasets import load_dataset
from loguru import logger

from mymodule.strategies import get_strategy
from mymodule.utils.inference_parallel import resolve_workers, run_parallel_predictions
from mymodule.utils.path import inference_path, save_inference

# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------


def parse_devset_turn(
    conversations: list[dict],
    target_turn_number: int,
    conversation_goal: dict | None = None,
    goal_progress_assessments: list[dict] | None = None,
    user_profile: dict | None = None,
) -> dict[str, Any]:
    """Build model inputs for target_turn_number within one session.

    Returns a dict with user_query, chat_history (prior-turn messages),
    conversation_goal, goal_progress (assessments before this turn), and
    user_profile.
    """
    df = pd.DataFrame(conversations)

    # History = messages from turns before the target.
    df_history = df[df["turn_number"] < target_turn_number]
    chat_history = df_history[["turn_number", "role", "content"]].to_dict(orient="records")

    # Current-turn user utterance (first message in the turn, role=user).
    df_current = df[df["turn_number"] == target_turn_number]
    user_query = df_current.iloc[0]["content"]

    # goal_progress: only assessments before the current turn.
    goal_progress = None
    if goal_progress_assessments:
        goal_progress = [g for g in goal_progress_assessments if g["turn_number"] < target_turn_number]

    return {
        "user_query": user_query,
        "chat_history": chat_history,
        "conversation_goal": conversation_goal,
        "goal_progress": goal_progress,
        "user_profile": user_profile,
    }


# ---------------------------------------------------------------------------
# Per-session processing (parallelizable)
# ---------------------------------------------------------------------------


# Module-level rewrite map (set by main when --rewrite-map is used)
_REWRITE_MAP: dict[str, str] | None = None
# Pool names that should receive rewritten query (None = all pools)
_REWRITE_POOLS: set[str] | None = None


def _predict_session(strategy: Any, item: dict, top_k: int) -> list[dict]:
    """Process one session (all 8 turns) and return the results. Thread-safe."""
    results = []
    session_id = item["session_id"]
    user_id = item["user_id"]

    for target_turn in range(1, 9):
        parsed = parse_devset_turn(
            conversations=item["conversations"],
            target_turn_number=target_turn,
            conversation_goal=item.get("conversation_goal"),
            goal_progress_assessments=item.get("goal_progress_assessments"),
            user_profile=item.get("user_profile"),
        )

        # Prepare rewritten_query when a rewrite map is loaded.
        user_query = parsed["user_query"]
        rewritten_query = None
        if _REWRITE_MAP is not None:
            key = f"{session_id}:{target_turn}"
            if key in _REWRITE_MAP:
                rewritten_query = _REWRITE_MAP[key]

        # _REWRITE_POOLS None = replace for all pools; a set = per-pool routing.
        effective_query = user_query
        if rewritten_query is not None and _REWRITE_POOLS is None:
            effective_query = rewritten_query

        track_ids, response = strategy.predict(
            user_query=effective_query,
            rewritten_query=rewritten_query if _REWRITE_POOLS is not None else None,
            rewrite_pools=_REWRITE_POOLS,
            chat_history=parsed["chat_history"],
            user_id=user_id,
            conversation_goal=parsed["conversation_goal"],
            goal_progress=parsed["goal_progress"],
            user_profile=parsed["user_profile"],
            top_k=top_k,
        )

        results.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": target_turn,
                "predicted_track_ids": track_ids,
                "predicted_response": response,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    if args.help_tids:
        from mymodule.strategies import show_available

        print(show_available())
        return

    # Propagate refresh flag into the process env so InstructedQueryEmbedder
    dataset = "devset" if args.top_k == 20 else f"devset_topk{args.top_k}"

    strategy_kwargs = {}
    if args.fallback:
        strategy_kwargs["fallback"] = args.fallback
    if args.ensemble:
        strategy_kwargs["ensemble"] = args.ensemble
    if args.response_gen:
        strategy_kwargs["responder_name"] = args.response_gen
    if args.response_provider:
        strategy_kwargs["responder_provider"] = args.response_provider
    if args.turn1_pool:
        strategy_kwargs["turn1_pool"] = args.turn1_pool
    if args.turn1_specificity:
        if not args.turn1_pool:
            logger.warning(
                "[turn1-specificity] --turn1-specificity is set but --turn1-pool is not — "
                "the filter has no effect. Ignoring."
            )
        else:
            strategy_kwargs["turn1_specificity"] = args.turn1_specificity
    strategy = get_strategy(args.tid, **strategy_kwargs)
    # Save path strictly follows `{strategy}__{pools}__{reranker}` naming —
    # the turn1-pool override is a runtime knob, not a new pool combination,
    # so it is NOT encoded in the saved filename. If you need to keep runs
    # with/without the override side-by-side for comparison, rename the
    # artifact manually or stash it before re-running.
    effective_tid = args.tid

    # Load query rewrite map.
    global _REWRITE_MAP
    if args.rewrite_map:
        import json as _json

        rewrite_path = Path(args.rewrite_map)
        if not rewrite_path.exists():
            logger.error(f"Rewrite map not found: {rewrite_path}")
            return
        with open(rewrite_path, encoding="utf-8") as f:
            _REWRITE_MAP = _json.load(f)
        logger.info(f"Loaded rewrite map: {len(_REWRITE_MAP)} entries from {rewrite_path}")
        if args.rewrite_pools:
            _REWRITE_POOLS = {p.strip() for p in args.rewrite_pools.split(",")}
            logger.info(f"Rewrite pools: {_REWRITE_POOLS} (others get original query)")
        # Prefix TID with qrw_ to keep rewritten runs separate.
        parts = effective_tid.split("__", 1)
        if len(parts) == 2:
            effective_tid = f"{parts[0]}__qrw_{parts[1]}"
        else:
            effective_tid = f"qrw_{effective_tid}"
        logger.info(f"Effective TID with rewrite: {effective_tid}")

    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    items = list(db)
    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]
        logger.info(f"Loaded {len(db)} sessions; truncated to first {len(items)} (--limit).")
        # Smoke runs (partial dataset) should not clobber the canonical artifact —
        # write into a sibling `_limit{N}` dataset path instead.
        dataset = f"{dataset}_limit{args.limit}"
    else:
        logger.info(f"Loaded {len(db)} sessions, will produce {len(db) * 8} predictions")

    # Incremental JSONL write: live progress inspection + crash resilience.
    jsonl_path = inference_path(effective_tid, dataset).with_suffix(".jsonl")
    workers = resolve_workers(args.response_gen)

    inference_results = run_parallel_predictions(
        strategy,
        items,
        _predict_session,
        args.top_k,
        workers=workers,
        desc="Sessions",
        jsonl_path=jsonl_path,
        resume=args.resume,
    )

    save_inference(inference_results, effective_tid, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Devset inference (mymodule)")
    parser.add_argument("--tid", type=str, required=False, help="Task ID: {strategy}__{pools}__{reranker}")
    parser.add_argument(
        "--fallback",
        type=str,
        default=None,
        help="Fallback pool list ('-' separated, e.g. 'bm25' or 'popularity'). Default: DEFAULT_FALLBACK.",
    )
    parser.add_argument("--ensemble", type=str, default=None, help="Ensemble method (default: rrf)")
    parser.add_argument(
        "--turn1-pool",
        type=str,
        default=None,
        help=(
            "Pool spec (pool-list syntax) used ONLY on cold-start turns (chat_history has no "
            "role=music message). Other turns take the primary `--tid` path. Output is saved "
            "under the plain `{tid}.json` path — this override is a runtime knob, not part of "
            "the TID grammar."
        ),
    )
    parser.add_argument(
        "--turn1-specificity",
        type=str,
        default=None,
        help=(
            "Comma-separated listener specificity filter for `--turn1-pool` (e.g. 'LH,LL'). "
            "When set, the turn1-pool override fires only for cold-start sessions whose "
            "`conversation_goal.specificity` is in this set. Default None = apply to every T1. "
            "No effect when --turn1-pool is unset."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top predictions to save (default: 20). >20 is stored under devset_topk{k}.",
    )
    parser.add_argument(
        "--response-gen",
        type=str,
        choices=["noop", "pas", "auto"],
        default=None,
        help=(
            "Response generator (noop / pas / auto). Devset defaults to noop for fast local "
            "iteration; env MYMODULE_RESPONSE_GEN overrides. The blindset runner defaults to pas. "
            "'auto' runs a 3-stage LLM-judge optimized pipeline (~3x LM calls)."
        ),
    )
    parser.add_argument(
        "--response-provider",
        type=str,
        choices=["ollama", "openai"],
        default=None,
        help=(
            "LM provider for the DSPy responder (ollama / openai); irrelevant for noop. "
            "Default ollama; env MYMODULE_CHAT_PROVIDER overrides."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit the number of sessions (smoke test / partial run). Results go to "
            "`devset_limit{N}/{tid}.json` so the canonical path is untouched."
        ),
    )
    parser.add_argument(
        "--rewrite-map",
        type=str,
        default=None,
        help=(
            "Path to a query rewrite map JSON ({session_id}:{turn_number} → rewritten_query). "
            "When set, user_query is replaced with the rewritten query and the TID gets a "
            "qrw_ prefix."
        ),
    )
    parser.add_argument(
        "--rewrite-pools",
        type=str,
        default=None,
        help=(
            "Comma-separated pool names that should receive the rewritten query; other pools "
            "keep the original. Unset = apply the rewrite to every pool."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sessions already present in the existing JSONL and continue (crash recovery).",
    )
    parser.add_argument("--help-tids", action="store_true", help="Show available strategies, pools, rerankers")
    parser.add_argument(
        "--query-composition",
        choices=["legacy", "rich"],
        default=None,
        help="Override query composition. `legacy` (default) = title/artist/album from prior "
        "music turns; `rich` = also year/popularity/tags (experimental). When unset, falls "
        "back to env `MYMODULE_QEMB_QUERY_COMPOSITION` (default `legacy`).",
    )
    args = parser.parse_args()
    if args.query_composition is not None:
        import os

        os.environ["MYMODULE_QEMB_QUERY_COMPOSITION"] = args.query_composition
    if not args.tid and not args.help_tids:
        parser.error("--tid is required (or use --help-tids)")
    main(args)
