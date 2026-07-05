"""Shared parallel inference runner with JSONL incremental write.

Session-level parallel execution utility shared by the devset/blindset scripts.

  - `resolve_workers(response_gen)`: worker count per response-gen kind —
    LLM-backed (e.g. pas) uses `MYMODULE_LLM_WORKERS` (default 4), noop uses 1.
  - `run_parallel_predictions(...)`: ThreadPoolExecutor-based session parallelism.
    With jsonl_path, each finished prediction is appended to a JSONL immediately
    (live progress via `tail -f`, completed work survives crashes).

JSONL policy: overwritten at run start, flushed per write, kept after completion
for debugging; the evaluation pipeline only reads the final JSON.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Union

from tqdm import tqdm

# predict_fn(strategy, item, top_k) -> dict (one per session) or list[dict] (N per session)
PredictFn = Callable[[Any, dict, int], Union[list[dict], dict]]


def resolve_workers(response_gen: str | None) -> int:
    """Parallel only when response-gen is LLM-backed; noop stays sequential.

    - noop / None → workers=1 (avoids native-lib concurrency issues, e.g. rocksdb)
    - LLM-backed (e.g. pas) → `MYMODULE_LLM_WORKERS` env (default 4)
    """
    if response_gen in (None, "noop"):
        return 1
    return int(os.environ.get("MYMODULE_LLM_WORKERS", "4") or 4)


class _JsonlWriter:
    """Thread-safe JSONL append writer.

    Progress can be watched with `tail -f <path>` during parallel inference.
    Flushes after each write so completed work survives crashes.
    """

    def __init__(self, path: Path, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            path.unlink(missing_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "a" if append else "w", encoding="utf-8")

    def write_all(self, objs: list[dict]) -> None:
        with self._lock:
            for obj in objs:
                self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()


def _load_completed_sessions(jsonl_path: Path) -> tuple[set[str], list[dict]]:
    """Load the completed session_id set and prior results from an existing JSONL."""
    completed: set[str] = set()
    results: list[dict] = []
    if not jsonl_path.exists():
        return completed, results
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            completed.add(obj["session_id"])
            results.append(obj)
    return completed, results


def run_parallel_predictions(
    strategy: Any,
    items: list[dict],
    predict_fn: PredictFn,
    top_k: int,
    workers: int = 1,
    desc: str = "Sessions",
    jsonl_path: Path | None = None,
    resume: bool = False,
) -> list[dict]:
    """Session-level parallel inference runner.

    Args:
        strategy: strategy object (assumed thread-safe; `dspy.Predict` is stateless).
        items: list of sessions/items.
        predict_fn: `(strategy, item, top_k) -> dict | list[dict]`, side-effect free.
        top_k: number of top predictions to save.
        workers: 1 = sequential, ≥2 = ThreadPoolExecutor.
        desc: tqdm bar description.
        jsonl_path: if set, append each prediction to the JSONL immediately.
        resume: if True, skip sessions already present in the existing JSONL.

    Returns:
        Accumulated predictions, flattened in input item order. The incremental
        JSONL is still written as each worker finishes so progress survives
        crashes, but the final JSON stays deterministic.
    """
    results: list[dict] = []

    if resume and jsonl_path and jsonl_path.exists():
        completed_sids, prior_results = _load_completed_sessions(jsonl_path)
        results.extend(prior_results)
        before = len(items)
        items = [it for it in items if it["session_id"] not in completed_sids]
        from loguru import logger

        logger.info(
            f"Resume: {len(completed_sids)} sessions loaded, {before - len(items)} skipped, {len(items)} remaining"
        )
        if not items:
            logger.info("All sessions already completed.")
            return results

    writer = _JsonlWriter(jsonl_path, append=resume) if jsonl_path else None

    def _as_list(r: list[dict] | dict) -> list[dict]:
        return r if isinstance(r, list) else [r]

    def _collect(r: list[dict] | dict) -> None:
        new = _as_list(r)
        results.extend(new)
        if writer:
            writer.write_all(new)

    try:
        if workers == 1:
            for item in tqdm(items, desc=desc):
                _collect(predict_fn(strategy, item, top_k))
        else:
            completed: dict[int, list[dict]] = {}
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="session") as ex:
                futures = {ex.submit(predict_fn, strategy, item, top_k): i for i, item in enumerate(items)}
                with tqdm(total=len(futures), desc=f"{desc} (workers={workers})") as pbar:
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        new = _as_list(fut.result())
                        completed[idx] = new
                        if writer:
                            writer.write_all(new)
                        pbar.update(1)
            results.extend(obj for idx in sorted(completed) for obj in completed[idx])
    finally:
        if writer:
            writer.close()

    return results
