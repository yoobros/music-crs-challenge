"""OOF (out-of-fold) feature builder for the GBM reranker.

For each (session, turn) of the HF ``train`` split, call the registered pools
and emit per-pool rank/score + RRF rank/score + categorical features over the
post-RRF top-K candidates as a DataFrame.

Leakage control:
- ``compute_fold_assignment`` splits train sessions into K group-aware folds.
- ``fold_aware=True`` pools score fold-i sessions only with the ckpt instance
  where fold i is held out.
- ``fold_aware=False`` pools (e.g. BM25, QEmb) reuse a single instance; static
  retrievers carry no leakage risk.

GT is extracted inline in the same pass (no separate GT file): for each turn,
row index 1 (role=music) of the conversations DataFrame holds the GT track_id.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from mymodule.cv import CVConfig, make_folds
from mymodule.strategies.pool.base import BasePool
from mymodule.strategies.rerank.gbm.pool_spec import PoolSpec
from mymodule.utils.fusion import rrf_score_map

# Max turns to attempt per train session. Actual train max is ~24; we try up
# to 32 and break when a turn is missing.
MAX_TURNS_TRAIN = 32

# Candidate rows per (session, turn). The reranker operates on top of the RRF
# result; pre-RRF per-pool rank/score are features attached to those candidates.
# The top-K used in training is stored in ckpt meta as ``top_k_rerank`` and
# reused at inference to keep the distributions aligned.
DEFAULT_TOP_K_RRF = 100


def compute_fold_assignment(
    session_ids: list[str],
    num_folds: int = 5,
    split_seed: int = 42,
) -> dict[str, int]:
    """Build a ``{session_id: fold_idx}`` map via ``mymodule.cv.make_folds``.

    Must use the same ``(num_folds, split_seed)`` as fold-aware pool training so
    the held-out sets of the fold ckpts match the OOF routing.
    """
    cfg = CVConfig(num_folds=num_folds, fold_idx=None, split_seed=split_seed)
    folds = make_folds(session_ids, group_fn=lambda s: s, config=cfg)
    out: dict[str, int] = {}
    for fold_i, (_train_idx, val_idx) in enumerate(folds):
        for j in val_idx:
            out[session_ids[j]] = fold_i
    if len(out) != len(session_ids):
        raise RuntimeError(f"fold assignment incomplete: covered {len(out)}/{len(session_ids)} sessions")
    return out


# ---------------------------------------------------------------------------
# Per-turn helpers
# ---------------------------------------------------------------------------


def _parse_train_turn(
    conversations: list[dict],
    target_turn_number: int,
    conversation_goal: dict | None,
    goal_progress_assessments: list[dict] | None,
    user_profile: dict | None,
) -> dict[str, Any] | None:
    """Extract (chat_history, user_query, gt_track_id) for one train-split turn.

    Same logic as devset turn parsing, plus inline GT extraction.
    Returns None when the turn does not exist.
    """
    df = pd.DataFrame(conversations)
    df_current = df[df["turn_number"] == target_turn_number]
    if df_current.empty:
        return None
    df_history = df[df["turn_number"] < target_turn_number]
    chat_history = df_history[["turn_number", "role", "content"]].to_dict(orient="records")
    # Convention (devset & train): row 0 = user, row 1 = music (GT), row 2 = assistant
    if len(df_current) < 2:
        # Malformed turn — skip
        return None
    user_query = df_current.iloc[0]["content"]
    music_row = df_current.iloc[1]
    if music_row.get("role") != "music":
        # Schema mismatch — skip
        return None
    gt_track_id = music_row["content"]
    goal_progress = None
    if goal_progress_assessments:
        goal_progress = [g for g in goal_progress_assessments if g["turn_number"] < target_turn_number]
    return {
        "user_query": user_query,
        "chat_history": chat_history,
        "conversation_goal": conversation_goal,
        "goal_progress": goal_progress,
        "user_profile": user_profile,
        "gt_track_id": gt_track_id,
    }


def _instantiate_pool_pool(specs: list[PoolSpec], num_folds: int) -> dict[tuple[str, int | None], BasePool]:
    """Pre-build pool instances — K per fold-aware spec, 1 otherwise."""
    out: dict[tuple[str, int | None], BasePool] = {}
    for spec in specs:
        if spec.fold_aware:
            for f in range(num_folds):
                logger.info(f"OOF: instantiating {spec.name} fold {f}…")
                out[(spec.name, f)] = spec.instantiate(f)
        else:
            logger.info(f"OOF: instantiating {spec.name} (fold-unaware)…")
            out[(spec.name, None)] = spec.instantiate(None)
    return out


def _pool_for(pool_inst: dict[tuple[str, int | None], BasePool], spec: PoolSpec, fold_idx: int) -> BasePool:
    """Return the instance for the given spec + fold idx."""
    return pool_inst[(spec.name, fold_idx if spec.fold_aware else None)]


# ---------------------------------------------------------------------------
# Feature row builder for one session-turn
# ---------------------------------------------------------------------------


def _build_rows_for_turn(
    *,
    session_id: str,
    user_id: str,
    turn_number: int,
    parsed: dict[str, Any],
    pool_results: dict[str, list[tuple[str, float]]],
    top_k_rrf: int,
) -> list[dict[str, Any]]:
    """One (session, turn) → feature rows (RRF top-K of them)."""
    rrf_map = rrf_score_map([[t for t, _ in scored] for scored in pool_results.values()])
    if not rrf_map:
        return []
    top_k = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)[:top_k_rrf]
    rrf_rank = {tid: r for r, (tid, _) in enumerate(top_k)}

    # Per-pool track_id → (rank, score) dict for fast lookup
    pool_idx: dict[str, dict[str, tuple[int, float]]] = {
        name: {tid: (r, s) for r, (tid, s) in enumerate(scored)} for name, scored in pool_results.items()
    }

    cg = parsed.get("conversation_goal") or {}
    category = cg.get("category", "<unk>")
    specificity = cg.get("specificity", "<unk>")
    gt_tid = parsed.get("gt_track_id")

    rows: list[dict[str, Any]] = []
    for tid, rrf_score in top_k:
        row: dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": int(turn_number),
            "track_id": tid,
            "rank_rrf": rrf_rank[tid] + 1,
            "score_rrf": float(rrf_score),
            "category": str(category),
            "specificity": str(specificity),
            "label": int(tid == gt_tid),
        }
        for name, idx in pool_idx.items():
            if tid in idx:
                r, s = idx[tid]
                row[f"rank_{name}"] = int(r) + 1
                row[f"score_{name}"] = float(s)
            else:
                row[f"rank_{name}"] = None  # NaN — pandas converts automatically
                row[f"score_{name}"] = None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_features(
    pool_specs: list[PoolSpec],
    *,
    num_folds: int = 5,
    split_seed: int = 42,
    top_k_rrf: int = DEFAULT_TOP_K_RRF,
    sessions: Iterable[dict] | None = None,
    fold_assignment: dict[str, int] | None = None,
    max_turns: int = MAX_TURNS_TRAIN,
    workers: int = 4,
    jsonl_path: "str | None" = None,
    parquet_dir: "str | None" = None,
    parquet_chunksize: int = 100_000,
    resume: bool = True,
) -> pd.DataFrame:
    """Build OOF features.

    Two output modes:
    - ``parquet_dir`` set: JSONL → turn_number-partitioned parquet, streamed;
      no in-memory DataFrame accumulation (safe for 5GB+). Returns an empty
      DataFrame for compatibility.
    - unset (legacy): JSONL → in-memory DataFrame.

    Args:
        pool_specs: result of ``get_pool_specs``; one spec per logical pool.
        num_folds, split_seed: must match fold-aware pool training (5, 42).
        top_k_rrf: emit rows only for the RRF top-K candidates per (session, turn).
        sessions: HF train split iterable (mockable in tests, sub-sample with
            ``--limit``). None loads the full train split.
        fold_assignment: ``{session_id: fold_idx}`` map — precompute it over the
            full train universe so sub-samples stay fold-leakage-free. None
            computes it from ``sessions`` (safe only when they match the full
            universe).
        max_turns: max turns to attempt per session.
        workers: per-turn pool call parallelism (ThreadPoolExecutor).
        jsonl_path: when set, rows are appended as NDJSON per (session, turn) —
            progress survives aborts. None accumulates in memory.
        parquet_dir: when set, convert the JSONL to a turn_number-partitioned
            parquet directory via streaming.
        parquet_chunksize: streaming JSONL chunk size in rows (default 100k).
        resume: when ``jsonl_path`` exists, skip already-recorded session_ids
            and continue; False overwrites.

    Returns:
        Long-format DataFrame; empty in ``parquet_dir`` mode (write only).
    """
    import json
    from pathlib import Path as _Path

    if sessions is None:
        from datasets import load_dataset

        sessions = list(load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train"))
    sessions_list: list[dict] = list(sessions)
    sub_session_ids = [s["session_id"] for s in sessions_list]
    if fold_assignment is None:
        fold_of = compute_fold_assignment(sub_session_ids, num_folds=num_folds, split_seed=split_seed)
    else:
        # Externally computed map — must contain every session_id of the sub-sample.
        missing = [s for s in sub_session_ids if s not in fold_assignment]
        if missing:
            raise KeyError(
                f"fold_assignment missing {len(missing)} session ids (sample: {missing[:3]}). "
                "Make sure compute_fold_assignment was called over the full train universe."
            )
        fold_of = fold_assignment

    pool_inst = _instantiate_pool_pool(pool_specs, num_folds)

    # Incremental write mode — abort-safe and low memory.
    jsonl_p: "_Path | None" = _Path(jsonl_path) if jsonl_path else None
    done_sessions: set[str] = set()
    rows_in_memory: list[dict[str, Any]] = []
    write_mode = "w"
    if jsonl_p is not None:
        jsonl_p.parent.mkdir(parents=True, exist_ok=True)
        if resume and jsonl_p.exists():
            with jsonl_p.open() as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        done_sessions.add(rec["session_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            write_mode = "a"
            logger.info(f"OOF resume: skipping {len(done_sessions)} sessions already in {jsonl_p}")
        else:
            jsonl_p.unlink(missing_ok=True)

    fout = jsonl_p.open(write_mode) if jsonl_p is not None else None
    try:
        pbar = tqdm(sessions_list, desc="OOF feature build", unit="session")
        for session in pbar:
            session_id = session["session_id"]
            if session_id in done_sessions:
                continue
            user_id = session["user_id"]
            f = int(fold_of[session_id])
            for turn_number in range(1, max_turns + 1):
                parsed = _parse_train_turn(
                    conversations=session["conversations"],
                    target_turn_number=turn_number,
                    conversation_goal=session.get("conversation_goal"),
                    goal_progress_assessments=session.get("goal_progress_assessments"),
                    user_profile=session.get("user_profile"),
                )
                if parsed is None:
                    break
                kwargs = {
                    "user_query": parsed["user_query"],
                    "chat_history": parsed["chat_history"],
                    "user_id": user_id,
                    "conversation_goal": parsed["conversation_goal"],
                    "goal_progress": parsed["goal_progress"],
                    "user_profile": parsed["user_profile"],
                }
                # Call pools in parallel — only fold_aware specs route by fold idx.
                if len(pool_specs) == 1:
                    spec = pool_specs[0]
                    pool_results = {spec.name: _pool_for(pool_inst, spec, f).generate_with_scores(**kwargs)}
                else:
                    with ThreadPoolExecutor(max_workers=min(workers, len(pool_specs))) as ex:
                        futures = {
                            spec.name: ex.submit(_pool_for(pool_inst, spec, f).generate_with_scores, **kwargs)
                            for spec in pool_specs
                        }
                        pool_results = {name: fut.result() for name, fut in futures.items()}
                new_rows = _build_rows_for_turn(
                    session_id=session_id,
                    user_id=user_id,
                    turn_number=turn_number,
                    parsed=parsed,
                    pool_results=pool_results,
                    top_k_rrf=top_k_rrf,
                )
                if fout is not None:
                    for row in new_rows:
                        fout.write(json.dumps(row) + "\n")
                else:
                    rows_in_memory.extend(new_rows)
            if fout is not None:
                fout.flush()  # per-session flush — an abort loses only that session
    finally:
        if fout is not None:
            fout.close()

    # ------------------------------------------------------------------
    # Output: streaming partitioned parquet (preferred) OR in-memory DataFrame
    # ------------------------------------------------------------------
    if parquet_dir is not None:
        if jsonl_p is None:
            raise ValueError("parquet_dir requires jsonl_path to be set as well.")
        return _stream_jsonl_to_partitioned_parquet(
            jsonl_p, _Path(parquet_dir), chunksize=parquet_chunksize, num_sessions=len(sessions_list)
        )

    if jsonl_p is not None:
        # Legacy: JSONL → in-memory DataFrame (chunked read).
        chunks = []
        for chunk in pd.read_json(jsonl_p, lines=True, chunksize=100_000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame(rows_in_memory)
    logger.success(
        f"OOF: built {len(df)} rows from {len(sessions_list)} sessions "
        f"({df['label'].sum()} positives, label coverage rate "
        f"{df.groupby(['session_id', 'turn_number'])['label'].max().mean():.3f})"
    )
    return df


def _stream_jsonl_to_partitioned_parquet(
    jsonl_p,
    parquet_dir,
    *,
    chunksize: int,
    num_sessions: int,
) -> pd.DataFrame:
    """Stream JSONL to a ``turn_number``-partitioned parquet directory.

    Each chunk becomes a pyarrow Table written via ``pq.write_to_dataset``; the
    full DataFrame is never held in memory (safe for 5GB+). Returns an empty
    DataFrame for compatibility.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Reset directory if it exists (re-build) — partition writes append, so
    # mixing with old partition files would leave stale rows.
    if any(parquet_dir.iterdir()):
        import shutil

        logger.info(f"OOF: resetting existing parquet dir {parquet_dir}")
        shutil.rmtree(parquet_dir)
        parquet_dir.mkdir(parents=True)

    total_rows = 0
    total_pos = 0
    coverage_keys: set[tuple[str, int]] = set()
    coverage_hits: set[tuple[str, int]] = set()

    for chunk in pd.read_json(jsonl_p, lines=True, chunksize=chunksize):
        # Stat tracking (group-level label coverage)
        total_rows += len(chunk)
        total_pos += int(chunk["label"].sum())
        grouped = chunk.groupby(["session_id", "turn_number"])["label"].max()
        for (sid, turn), has_pos in grouped.items():
            coverage_keys.add((sid, int(turn)))
            if has_pos:
                coverage_hits.add((sid, int(turn)))

        # Partition write
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(parquet_dir),
            partition_cols=["turn_number"],
            existing_data_behavior="overwrite_or_ignore",
        )

    cov_rate = (len(coverage_hits) / len(coverage_keys)) if coverage_keys else 0.0
    logger.success(
        f"OOF: wrote {total_rows} rows ({total_pos} positives, label coverage "
        f"{cov_rate:.3f}) → partitioned parquet at {parquet_dir} (from {num_sessions} sessions)"
    )
    return pd.DataFrame()  # empty placeholder
