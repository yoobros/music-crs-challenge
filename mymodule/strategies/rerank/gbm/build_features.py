"""CLI: build the OOF feature parquet for a TID.

Usage:
    uv run python -m mymodule.strategies.rerank.gbm.build_features --tid <tid> \\
        [--top-k-rrf 100] [--num-folds 5] [--split-seed 42] [--workers 4]

Output: ``mymodule/strategies/rerank/gbm/oof/{pool_signature}.parquet``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from mymodule.strategies.rerank.gbm._specs import get_pool_specs, pool_signature
from mymodule.strategies.rerank.gbm.oof_features import (
    DEFAULT_TOP_K_RRF,
    MAX_TURNS_TRAIN,
    build_features,
)
from mymodule.utils.tid import parse_tid

OOF_ROOT = Path(__file__).parent / "oof"


def main() -> None:
    p = argparse.ArgumentParser(description="GBM reranker — OOF feature parquet builder")
    p.add_argument("--tid", required=True, help="TID whose pool part defines the feature set")
    p.add_argument("--top-k-rrf", type=int, default=DEFAULT_TOP_K_RRF)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS_TRAIN)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(debug) process only the first N train sessions; all when unset.",
    )
    p.add_argument("--out", type=str, default=None, help="override the output parquet path")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing jsonl progress and restart from scratch (default: resume)",
    )
    args = p.parse_args()

    _strategy, pool_names, reranker = parse_tid(args.tid)
    if reranker != "gbm":
        logger.warning(f"TID reranker part is '{reranker}', not 'gbm'; using the pool part only.")
    specs = get_pool_specs(pool_names)
    sig = pool_signature(pool_names)
    # Layout: ``{sig}/`` directory (turn_number-partitioned parquet root); the
    # intermediate JSONL lives next to it as ``{sig}.jsonl``.
    out_dir = Path(args.out) if args.out else (OOF_ROOT / sig)

    sessions = None
    fold_assignment = None
    if args.limit:
        from datasets import load_dataset

        from mymodule.strategies.rerank.gbm.oof_features import compute_fold_assignment

        # Compute fold assignment over the full train universe so sub-samples
        # keep the same fold idx as full training (leakage-safe).
        logger.info(f"--limit={args.limit} → using first {args.limit} train sessions (fold map from full universe)")
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
        all_session_ids = [s["session_id"] for s in ds]
        fold_assignment = compute_fold_assignment(all_session_ids, num_folds=args.num_folds, split_seed=args.split_seed)
        sessions = list(ds.select(range(args.limit)))

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Incremental jsonl write — progress survives aborts; auto-resume unless --no-resume.
    jsonl_path = out_dir.parent / f"{sig}.jsonl"
    build_features(
        specs,
        num_folds=args.num_folds,
        split_seed=args.split_seed,
        top_k_rrf=args.top_k_rrf,
        sessions=sessions,
        fold_assignment=fold_assignment,
        max_turns=args.max_turns,
        workers=args.workers,
        jsonl_path=str(jsonl_path),
        parquet_dir=str(out_dir),
        resume=(not args.no_resume),
    )
    logger.success(f"Saved OOF features → {out_dir}/ (partitioned by turn_number); intermediate jsonl: {jsonl_path}")


if __name__ == "__main__":
    main()
