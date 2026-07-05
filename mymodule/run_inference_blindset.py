"""Blindset inference script.

Blind sessions are truncated mid-conversation with the last message always
role=user; one prediction is generated per session. With top-k=20 (default),
`prediction.json` and a flat `submission.zip` are built automatically after
inference; `{tid}.json` is kept for tracking.

Usage:
    uv run python -m mymodule.run_inference_blindset --tid <tid> --eval_dataset blindset_B
"""

import argparse
import os
import sys

# ruff: noqa: E402


def _preload_lightgbm_for_gbm_tid(argv: list[str]) -> None:
    """Load LightGBM before datasets/strategy imports for GBM inference."""
    if not any("__gbm" in arg for arg in argv):
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import lightgbm  # noqa: F401


_preload_lightgbm_for_gbm_tid(sys.argv[1:])

from datasets import load_dataset
from loguru import logger

from mymodule.strategies import get_strategy
from mymodule.utils.inference_parallel import resolve_workers, run_parallel_predictions
from mymodule.utils.path import inference_path, save_inference
from mymodule.utils.submission import (
    SubmissionValidationError,
    build_submission_zip,
    print_upload_guide,
)

# eval_dataset key → HuggingFace dataset path.
# Blind-B is a generalization split — half of the sessions are cold-start
# (no user_id / user_profile) and several optional fields are absent;
# parse_blindset_item reads them with .get() so both splits share one parser.
BLINDSET_DATASETS: dict[str, str] = {
    "blindset_A": "talkpl-ai/TalkPlayData-Challenge-Blind-A",
    "blindset_B": "talkpl-ai/TalkPlayData-Challenge-Blind-B",
}


def parse_blindset_item(item: dict) -> dict:
    """Convert one blindset session into model inputs.

    The last message (role=user) is the current query; everything before it
    is chat history. goal_progress keeps only assessments from earlier turns.
    """
    conversations = item["conversations"]

    user_query = conversations[-1]["content"]
    turn_number = conversations[-1]["turn_number"]

    chat_history = [
        {"turn_number": m["turn_number"], "role": m["role"], "content": m["content"]} for m in conversations[:-1]
    ]

    goal_progress = None
    gpa = item.get("goal_progress_assessments")
    if gpa:
        goal_progress = [g for g in gpa if g["turn_number"] < turn_number]

    return {
        "user_query": user_query,
        "chat_history": chat_history,
        "turn_number": turn_number,
        "conversation_goal": item.get("conversation_goal"),
        "goal_progress": goal_progress,
        "user_profile": item.get("user_profile"),
    }


def _predict_one(strategy, item: dict, top_k: int) -> dict:
    """Predict a single session; side-effect free so it can run in a thread pool."""
    parsed = parse_blindset_item(item)
    # Cold-start sessions may have an empty/missing user_id; retrieval is
    # content-based so it still works without user signals.
    user_id = item.get("user_id") or ""
    track_ids, response = strategy.predict(
        user_query=parsed["user_query"],
        chat_history=parsed["chat_history"],
        user_id=user_id,
        conversation_goal=parsed["conversation_goal"],
        goal_progress=parsed["goal_progress"],
        user_profile=parsed["user_profile"],
        top_k=top_k,
    )
    return {
        "session_id": item["session_id"],
        "user_id": user_id,
        "turn_number": parsed["turn_number"],
        "predicted_track_ids": track_ids,
        "predicted_response": response,
    }


def main(args: argparse.Namespace) -> None:
    if args.help_tids:
        from mymodule.strategies import show_available

        print(show_available())
        return

    if args.eval_dataset not in BLINDSET_DATASETS:
        available = sorted(BLINDSET_DATASETS.keys())
        raise SystemExit(f"Unknown --eval_dataset {args.eval_dataset!r}. Available: {available}.")

    strategy_kwargs = {
        "responder_name": args.response_gen,
        "responder_provider": args.response_provider,
    }
    if args.fallback:
        strategy_kwargs["fallback"] = args.fallback
    if args.ensemble:
        strategy_kwargs["ensemble"] = args.ensemble
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
    # so it is NOT encoded in the saved filename.
    effective_tid = args.tid

    # top-k > 20 is outside the submission format; store under a separate subdir.
    is_submittable_topk = args.top_k == 20
    is_partial = args.limit is not None and args.limit > 0
    dataset_name = args.eval_dataset if is_submittable_topk else f"{args.eval_dataset}_topk{args.top_k}"
    if is_partial:
        # Partial runs go to a sibling subdir so they don't clobber the canonical
        # submission artifact (prediction.json / submission.zip).
        dataset_name = f"{dataset_name}_limit{args.limit}"

    hf_path = BLINDSET_DATASETS[args.eval_dataset]
    db = load_dataset(hf_path, split="test")
    items = list(db)
    if is_partial:
        items = items[: args.limit]
        logger.info(f"Loaded {len(db)} sessions from {hf_path}; truncated to first {len(items)} (--limit).")
    else:
        logger.info(f"Loaded {len(db)} sessions from {hf_path}")

    # Incremental JSONL write: live progress inspection + crash resilience.
    jsonl_path = inference_path(effective_tid, dataset_name).with_suffix(".jsonl")
    workers = resolve_workers(args.response_gen)

    results = run_parallel_predictions(
        strategy,
        items,
        _predict_one,
        args.top_k,
        workers=workers,
        desc="Sessions",
        jsonl_path=jsonl_path,
    )

    n_empty = sum(1 for r in results if not r.get("predicted_response"))
    if args.response_gen != "noop" and n_empty:
        logger.warning(f"[response-gen] {n_empty}/{len(results)} sessions produced empty responses.")

    save_inference(results, effective_tid, dataset_name)

    # Blindset + top-k=20 → automatic packaging; otherwise skip.
    if not is_submittable_topk:
        logger.info(f"[skip packaging] top-k={args.top_k} (≠20) is not the submission format.")
        return
    if is_partial:
        logger.info(f"[skip packaging] --limit {args.limit} is a partial run, not submittable.")
        return
    if args.no_package:
        logger.info("[skip packaging] --no-package given.")
        return

    try:
        zip_path = build_submission_zip(
            tid=effective_tid,
            eval_dataset=args.eval_dataset,
            validate_catalog=not args.skip_catalog_check,
        )
    except SubmissionValidationError as e:
        logger.error(f"[packaging failed] submission format validation failed: {e}")
        logger.error("  → {tid}.json was saved but prediction.json / submission.zip were not created.")
        raise SystemExit(1) from e

    print_upload_guide(zip_path, args.eval_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blindset inference (mymodule)")
    parser.add_argument("--tid", type=str, required=False, help="Task ID: {strategy}__{pools}__{reranker}")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="blindset_A",
        choices=sorted(BLINDSET_DATASETS.keys()),
        help="Evaluation dataset (default: blindset_A). blindset_B = generalization split with cold-start sessions.",
    )
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
        "--response-gen",
        type=str,
        choices=["noop", "pas", "auto"],
        default="pas",
        help=(
            "Response generator (default: pas). 'auto' runs a 3-stage LLM-judge optimized "
            "pipeline (~3x LM calls). Env MYMODULE_RESPONSE_GEN overrides."
        ),
    )
    parser.add_argument(
        "--response-provider",
        type=str,
        choices=["ollama", "openai"],
        default="ollama",
        help=(
            "LM provider for the DSPy responder (default: ollama); irrelevant for noop. "
            "Env MYMODULE_CHAT_PROVIDER overrides; parallelism via MYMODULE_LLM_WORKERS (default 4)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top predictions to save (default: 20). >20 is stored under {dataset}_topk{k}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit the number of sessions (smoke test / partial run). Results go to "
            "`{eval_dataset}_limit{N}/{tid}.json`; packaging is skipped."
        ),
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Skip automatic packaging (prediction.json / submission.zip). Only meaningful at top-k=20.",
    )
    parser.add_argument(
        "--skip-catalog-check",
        action="store_true",
        help="Skip track_id catalog validation (avoids loading Track-Metadata; debug only).",
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
        os.environ["MYMODULE_QEMB_QUERY_COMPOSITION"] = args.query_composition
    if not args.tid and not args.help_tids:
        parser.error("--tid is required (or use --help-tids)")
    main(args)
