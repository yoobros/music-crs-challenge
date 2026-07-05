"""CLI: package an already-saved `{tid}.json` into the submission format.

`run_inference_blindset.py` does this automatically after inference; use this
manually to:
- regenerate a lost `submission.log`
- promote a specific tid's result back to `prediction.json`
- rebuild quickly offline with catalog validation skipped

Usage:
    uv run python scripts/package_submission.py --tid <tid> --eval-dataset blindset_A
    uv run python scripts/package_submission.py --tid <tid> --eval-dataset blindset_A --skip-catalog-check
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger
from mymodule.utils.submission import (
    SubmissionValidationError,
    build_submission_zip,
    print_upload_guide,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package existing {tid}.json into prediction.json + submission.zip")
    parser.add_argument("--tid", type=str, required=True, help="Task ID (filename stem under exp/inference/{dataset}/)")
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="blindset_A",
        choices=["blindset_A", "blindset_B"],
        help="Evaluation dataset (default: blindset_A)",
    )
    parser.add_argument(
        "--skip-catalog-check",
        action="store_true",
        help="Skip Track-Metadata catalog validation (faster, offline).",
    )
    parser.add_argument(
        "--pad-lexical",
        type=float,
        nargs="?",
        const=0.98,
        default=None,
        metavar="TARGET",
        help=(
            "Lexical-diversity padding: append a unique-token block to each response "
            "to raise distinct-2 to TARGET (default 0.98)."
        ),
    )
    parser.add_argument(
        "--diversify-mmr",
        action="store_true",
        help=(
            "MMR diversify: reduce cross-row duplication within the top-20 to raise "
            "(capped) catalog diversity; stays within the 20-track format. "
            "Requires --mmr-candidates-from (top-K pool)."
        ),
    )
    parser.add_argument("--mmr-candidates-from", type=str, default=None, metavar="FILE", help="path to top-K pool json")
    parser.add_argument(
        "--mmr-protect-k", type=int, default=5, help="number of top ranks to keep fixed (protects nDCG; default 5)"
    )
    parser.add_argument("--mmr-window", type=int, default=60, help="re-selection relevance window (default 60)")
    parser.add_argument(
        "--mmr-lambda", type=float, default=0.5, help="relevance(1) vs diversity(0) weight (default 0.5)"
    )
    args = parser.parse_args()

    diversify_mmr = None
    if args.diversify_mmr:
        diversify_mmr = {
            "candidates_path": args.mmr_candidates_from,
            "protect_k": args.mmr_protect_k,
            "window": args.mmr_window,
            "mmr_lambda": args.mmr_lambda,
        }

    try:
        zip_path = build_submission_zip(
            tid=args.tid,
            eval_dataset=args.eval_dataset,
            validate_catalog=not args.skip_catalog_check,
            pad_lexical=args.pad_lexical,
            diversify_mmr=diversify_mmr,
        )
    except FileNotFoundError as e:
        logger.error(f"[error] {e}")
        return 1
    except SubmissionValidationError as e:
        logger.error(f"[validation failed] {e}")
        return 2

    print_upload_guide(zip_path, args.eval_dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
