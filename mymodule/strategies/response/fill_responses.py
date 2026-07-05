"""Fill response field for existing inference results (track predictions only).

Usage:
    uv run python -m mymodule.strategies.response.fill_responses \
        --tid <tid> \
        [--response-gen pas|noop] \
        [--response-provider ollama|openai] \
        [--dataset devset|devset_topk100]

Reads existing {tid}.json, generates responses for each turn, and saves back.
Uses ThreadPoolExecutor for parallel response generation + JSONL streaming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from mymodule.strategies.response import get_response_generator
from mymodule.strategies.response.base import BaseResponseGenerator


def load_predictions(tid: str, dataset: str = "devset") -> list:
    """Load existing predictions from devset/{tid}.json or devset_topk{k}/{tid}.json."""
    base_dir = Path(__file__).parent.parent.parent / "exp" / "inference"

    if "topk" in dataset:
        pred_file = base_dir / f"{dataset}" / f"{tid}.json"
    else:
        pred_file = base_dir / dataset / f"{tid}.json"

    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")

    with open(pred_file) as f:
        return json.load(f)


_BLINDSET_HF_REPOS: dict[str, str] = {
    "blindset_A": "talkpl-ai/TalkPlayData-Challenge-Blind-A",
    "blindset_B": "talkpl-ai/TalkPlayData-Challenge-Blind-B",
}


def _build_devset_context_index(dataset: str = "devset") -> dict[tuple[str, int], dict]:
    """Build a `(session_id, turn_number) → context` map from HF dataset.

    Supports devset (HF test split) and blindset_A (separate HF repo).
    Predictions JSONs only persist `predicted_track_ids` (no conversation
    context), so the responder needs us to re-fetch the per-turn `user_query`,
    `chat_history`, `user_profile`, `conversation_goal`, and
    `goal_progress` from the HF dataset.
    """
    from datasets import load_dataset

    if dataset in _BLINDSET_HF_REPOS:
        repo = _BLINDSET_HF_REPOS[dataset]
        logger.info(f"Loading HF blindset ({repo}) for context join …")
        ds = load_dataset(repo, split="test")
    else:
        logger.info("Loading HF devset split for context join …")
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")

    index: dict[tuple[str, int], dict] = {}
    for item in ds:
        session_id = item["session_id"]
        convs = item["conversations"]
        all_turns = sorted({c.get("turn_number") for c in convs if c.get("turn_number") is not None})
        for turn_num in all_turns:
            user_msgs = [c for c in convs if c.get("turn_number") == turn_num and c.get("role") == "user"]
            if not user_msgs:
                continue
            chat_history = [c for c in convs if (c.get("turn_number") or 0) < turn_num]
            index[(session_id, turn_num)] = {
                "user_query": user_msgs[0].get("content", ""),
                "chat_history": chat_history,
                "user_profile": item.get("user_profile") or {},
                "conversation_goal": item.get("conversation_goal") or {},
                "goal_progress": item.get("goal_progress_assessments") or [],
            }
    logger.info(f"Devset context index size: {len(index)}")
    return index


def resolve_workers(response_gen: str | None) -> int:
    """Determine worker count based on responder type.

    noop → 1 worker (sequential)
    LLM-backed (pas, ...) → MYMODULE_LLM_WORKERS (default 4)
    """
    if response_gen in (None, "noop"):
        return 1
    return int(os.environ.get("MYMODULE_LLM_WORKERS", "4") or 4)


class JsonlWriter:
    """Thread-safe JSONL writer for streaming responses."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "w", encoding="utf-8")

    def write(self, obj: dict) -> None:
        with self._lock:
            self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()


def _generate_response_worker(
    pred: dict,
    responder: BaseResponseGenerator,
    ctx: dict | None = None,
) -> dict:
    """Worker function: generate response for a single prediction.

    `ctx` overlays conversation context (user_query / chat_history /
    user_profile / conversation_goal / goal_progress) onto the prediction
    when the prediction JSON itself lacks those fields. Devset/Blind-A
    inference outputs only persist `predicted_track_ids` so the HF-joined
    context fed via `ctx` is what makes a real personalized response
    possible at fill_responses time.
    """
    try:
        merged: dict = {**(ctx or {}), **pred}  # pred wins when both present
        response = responder.generate(
            track_ids=merged.get("predicted_track_ids", []),
            user_query=merged.get("user_query", "") or "",
            chat_history=merged.get("chat_history", []) or [],
            user_profile=merged.get("user_profile"),
            conversation_goal=merged.get("conversation_goal"),
            goal_progress=merged.get("goal_progress"),
        )
        pred["predicted_response"] = response
        return pred
    except Exception as e:
        logger.warning(f"[response-gen] failed for {pred.get('session_id', '?')}: {e}")
        pred["predicted_response"] = ""
        return pred


def fill_responses(
    tid: str,
    dataset: str = "devset",
    response_gen: str = "pas",
    response_provider: str = "ollama",
    *,
    output_tid: str | None = None,
    join_devset_context: bool = True,
    max_records: int | None = None,
) -> None:
    """Fill response field for predictions using parallel ThreadPoolExecutor.

    Streams responses to JSONL during generation, then converts to final JSON.

    `output_tid` lets callers redirect the output to a sibling file so
    multiple variants can be compared side-by-side without overwriting.
    `join_devset_context=True` (default) joins HF devset context per
    `(session_id, turn_number)` so the responder gets real conversation
    state — needed because predictions JSONs only persist track ids.
    `max_records` truncates the predictions list to its first N entries
    (sorted by `(session_id, turn_number)` so the truncation is stable
    across variants). Useful for cheap variant smoke comparisons.
    """
    predictions = load_predictions(tid, dataset)

    if not isinstance(predictions, list):
        raise ValueError(f"Expected predictions to be a list, got {type(predictions)}")

    if max_records is not None and max_records > 0 and max_records < len(predictions):
        predictions_sorted = sorted(predictions, key=lambda p: (p.get("session_id", ""), p.get("turn_number", 0)))
        predictions = predictions_sorted[:max_records]
        logger.info(
            f"Truncated to first {max_records}/{len(predictions_sorted)} predictions "
            f"(sorted by session+turn for stable variant comparison)."
        )

    ctx_index: dict[tuple[str, int], dict] = {}
    if join_devset_context:
        try:
            ctx_index = _build_devset_context_index(dataset)
        except Exception as e:
            logger.warning(f"[fill_responses] HF context join unavailable ({e}); proceeding without.")

    logger.info(f"Loading '{response_gen}' responder (provider={response_provider})...")
    responder = get_response_generator(response_gen, provider=response_provider)

    base_dir = Path(__file__).parent.parent.parent / "exp" / "inference"
    out_tid = output_tid or tid
    if "topk" in dataset:
        output_file = base_dir / f"{dataset}" / f"{out_tid}.json"
    else:
        output_file = base_dir / dataset / f"{out_tid}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_file.with_suffix(".jsonl")

    total = len(predictions)
    workers = resolve_workers(response_gen)
    writer = JsonlWriter(jsonl_file)

    logger.info(f"Generating {total} responses using {workers} workers (output: {output_file.name})...")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_response_worker,
                pred,
                responder,
                ctx_index.get((pred.get("session_id"), pred.get("turn_number"))),
            ): i
            for i, pred in enumerate(predictions)
        }

        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  {completed}/{total} responses generated")

            pred_with_response = future.result()
            writer.write(pred_with_response)

    logger.info(f"  {total}/{total} responses generated")
    writer.close()

    logger.info(f"Merging {jsonl_file.name} → {output_file.name}...")
    with open(jsonl_file) as inf, open(output_file, "w") as outf:
        merged = [json.loads(line) for line in inf]
        json.dump(merged, outf, indent=2)

    jsonl_file.unlink()
    logger.success(f"Responses saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill response field for existing inference results.")
    parser.add_argument("--tid", required=True, help="Task ID")
    parser.add_argument(
        "--response-gen",
        choices=["noop", "pas", "auto"],
        default="pas",
        help="Response generator (default: pas). 'auto' = multi-stage judge-optimized pipeline.",
    )
    parser.add_argument(
        "--response-provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LM provider for DSPy-backed generators (default: ollama). Ignored for noop.",
    )
    parser.add_argument(
        "--dataset",
        default="devset",
        help="Dataset (default: devset). e.g. devset, devset_topk100, devset_limit50, blindset_A, blindset_B.",
    )
    parser.add_argument(
        "--output-tid",
        default=None,
        help="Optional output TID to write under instead of overwriting `--tid`. "
        "Useful for variant sweeps (e.g. `<tid>__auto_rich`).",
    )
    parser.add_argument(
        "--no-context-join",
        action="store_true",
        help="Skip the HF devset join — predictions JSON would need to already carry "
        "user_query / chat_history for personalization to work. Default off.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Truncate predictions to first N (sorted by session+turn) for cheap "
        "variant smoke runs. Default: process all.",
    )

    args = parser.parse_args()

    try:
        fill_responses(
            tid=args.tid,
            dataset=args.dataset,
            response_gen=args.response_gen,
            response_provider=args.response_provider,
            output_tid=args.output_tid,
            join_devset_context=not args.no_context_join,
            max_records=args.max_records,
        )
    except FileNotFoundError as e:
        logger.error(f"{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
