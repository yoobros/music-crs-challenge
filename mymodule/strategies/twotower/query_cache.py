"""Encode train / devset / blindset query bodies with a LoRA adapter (or base)
and persist to a `.npz` cache keyed by `(session_id, turn_number)`.

This is the **query-side** twin of `doc_cache.py` (which caches the 47K track
doc vectors). Together they let a CV-trained two-tower fold be scored offline
without re-encoding: doc_cache holds `track_id -> doc_vec`, query_cache holds
`(session_id, turn_number) -> query_vec`, both produced by the SAME fold adapter
so their cosine lives in one vector space.

Motivation — OOF feature building: a GBM reranker needs leakage-free
`twotower` rank/score features over the full train set. With K fold adapters,
that means K × (encode every train query) + K × (encode 47K docs). Caching the
query side per (fold, dataset) removes the dominant repeated cost: the consumer
just loads `query_cache_<adapter>_fold{i}_<dataset>.npz` and `doc_cache_<adapter>_fold{i}.npz`
and does a single matmul per fold.

Query bodies are composed via `TwoTowerPool.compose_query_body` (load_docs=False,
so no doc cache is required at build time) — byte-for-byte identical to what the
online retrieval pool produces, guaranteeing cached vectors match inference.

Cache layout (.npz):
    session_ids   : np.ndarray[str]      (N,)
    turn_numbers  : np.ndarray[int]      (N,)
    query_vecs    : np.ndarray[float32]  (N, D), L2-normalized
    meta          : json string {base_model, adapter, dataset, dim, n, max_len}

Adapter binding: `meta.adapter` records the adapter path. The consumer must
sanity-check it against the runtime adapter (mismatch → mixed vector space),
exactly as `doc_cache._load_doc_cache` does. `load_query_cache` enforces it.

Usage:
    # fold adapter, devset queries
    uv run python -m mymodule.strategies.twotower.query_cache \\
        --adapter mymodule/strategies/twotower/ckpt/qwen3emb06b_lora_full_fold0 \\
        --dataset devset \\
        --out mymodule/strategies/twotower/data/query_cache_qwen3emb06b_lora_full_fold0_devset.npz

    # same fold, train queries (the OOF target set)
    uv run python -m mymodule.strategies.twotower.query_cache \\
        --adapter mymodule/strategies/twotower/ckpt/qwen3emb06b_lora_full_fold0 \\
        --dataset train \\
        --out mymodule/strategies/twotower/data/query_cache_qwen3emb06b_lora_full_fold0_train.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from mymodule.strategies.twotower.encoder import DEFAULT_MAX_LEN

# dataset key → (HuggingFace path, split). Mirrors run_inference_devset /
# run_inference_blindset so the cached queries match the eval harness exactly.
DATASET_HF: dict[str, tuple[str, str]] = {
    "train": ("talkpl-ai/TalkPlayData-Challenge-Dataset", "train"),
    "devset": ("talkpl-ai/TalkPlayData-Challenge-Dataset", "test"),
    "blindset_A": ("talkpl-ai/TalkPlayData-Challenge-Blind-A", "test"),
    "blindset_B": ("talkpl-ai/TalkPlayData-Challenge-Blind-B", "test"),
}


def load_query_cache(path: Path, adapter: Path | None) -> tuple[list[str], list[int], np.ndarray]:
    """Load `(session_ids, turn_numbers, query_vecs)` and sanity-check adapter binding.

    Symmetric to `doc_cache._load_doc_cache`. Warns (does not raise) on an
    adapter mismatch so a caller can still inspect a cache built by a different
    fold, but the warning flags that retrieval would run in a mixed space.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Query cache not found: {path}. Build it with "
            f"`python -m mymodule.strategies.twotower.query_cache --adapter {adapter} --dataset <ds> --out {path}`."
        )
    z = np.load(path, allow_pickle=True)
    sids = z["session_ids"].astype(str).tolist()
    turns = z["turn_numbers"].astype(int).tolist()
    vecs = z["query_vecs"].astype(np.float32)
    try:
        meta = json.loads(str(z["meta"].item())) if "meta" in z.files else {}
    except Exception:
        meta = {}
    cached_adapter = meta.get("adapter")
    if adapter is not None:
        if cached_adapter is None:
            logger.warning(
                f"[twotower-query-cache] {path} was built without an adapter (meta.adapter is null) "
                f"but a runtime adapter ({adapter}) was given — mixed vector space."
            )
        elif Path(cached_adapter).resolve() != Path(adapter).resolve():
            logger.warning(
                f"[twotower-query-cache] cache adapter ({cached_adapter}) ≠ runtime adapter ({adapter}). "
                "Query vectors live in a mismatched space — rebuild the cache."
            )
    logger.info(f"[twotower-query-cache] loaded {path} ({vecs.shape[0]} queries × {vecs.shape[1]} dim)")
    return sids, turns, vecs


def _iter_queries(dataset: str, limit: int | None):
    """Yield `(session_id, turn_number, user_query, chat_history, conversation_goal)`.

    Reuses the exact turn-parsing the inference runners use so query bodies match:
    `parse_devset_turn` for train/devset (each session → every turn 1..N), and
    `parse_blindset_item` for blindset (one cut-off turn per session).
    """
    hf_path, split = DATASET_HF[dataset]
    logger.info(f"[twotower-query-cache] loading {hf_path} (split={split}) for dataset={dataset}")
    db = load_dataset(hf_path, split=split)
    items = list(db)
    if limit is not None and limit > 0:
        items = items[:limit]
        logger.info(f"[twotower-query-cache] truncated to first {len(items)} sessions (--limit)")

    if dataset in ("train", "devset"):
        from mymodule.run_inference_devset import parse_devset_turn

        for item in items:
            session_id = item["session_id"]
            conversations = item["conversations"]
            turns = sorted({m["turn_number"] for m in conversations})
            for turn in turns:
                parsed = parse_devset_turn(
                    conversations=conversations,
                    target_turn_number=turn,
                    conversation_goal=item.get("conversation_goal"),
                    goal_progress_assessments=item.get("goal_progress_assessments"),
                    user_profile=item.get("user_profile"),
                )
                yield session_id, turn, parsed["user_query"], parsed["chat_history"], parsed["conversation_goal"]
    else:  # blindset_A / blindset_B
        from mymodule.run_inference_blindset import parse_blindset_item

        for item in items:
            parsed = parse_blindset_item(item)
            yield (
                item["session_id"],
                parsed["turn_number"],
                parsed["user_query"],
                parsed["chat_history"],
                parsed["conversation_goal"],
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the query-side (.npz) cache for a two-tower adapter.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", choices=sorted(DATASET_HF.keys()), required=True)
    p.add_argument("--adapter", type=Path, default=None, help="PEFT adapter dir (omit → base model, rarely useful).")
    p.add_argument("--base-model", default=None, help="Override base model (default: encoder.DEFAULT_BASE_MODEL).")
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype. float16 ~2x faster on Apple MPS; bfloat16 best on CUDA.",
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: autodetect).")
    p.add_argument("--limit", type=int, default=None, help="Cap sessions for a smoke run.")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Query-only pool: no doc cache needed at build time (the fold doc cache is a
    # separate artifact). compose_query_body + encode_query_bodies are byte-for-byte
    # identical to the online retrieval path.
    from mymodule.strategies.twotower.pool import TwoTowerPool

    pool_kwargs: dict = dict(
        adapter_path=args.adapter,
        load_docs=False,
        max_len=args.max_len,
        dtype=args.dtype,
        encode_batch=args.encode_batch,
        device=args.device,
    )
    if args.base_model:
        pool_kwargs["base_model"] = args.base_model
    pool = TwoTowerPool(**pool_kwargs)

    session_ids: list[str] = []
    turn_numbers: list[int] = []
    bodies: list[str] = []
    for sid, turn, user_query, chat_history, goal in tqdm(
        _iter_queries(args.dataset, args.limit), desc="compose_query"
    ):
        session_ids.append(str(sid))
        turn_numbers.append(int(turn))
        bodies.append(pool.compose_query_body(user_query, chat_history, goal))

    if not bodies:
        raise SystemExit(f"No queries produced for dataset={args.dataset} (limit={args.limit}).")

    logger.info(f"[twotower-query-cache] encoding {len(bodies)} query bodies (adapter={args.adapter})")
    vecs = pool.encode_query_bodies(bodies, batch_size=args.encode_batch, normalize=True)
    logger.success(f"[twotower-query-cache] vecs shape={vecs.shape} dtype={vecs.dtype}")

    # Body-hash key → lets the serving pool look up a query vec by the composed
    # body (sha256) without loading the encoder. compose_query_body is byte-for-byte
    # deterministic (same tokenizer + query_summaries), so the serve-time hash matches.
    body_hashes = [hashlib.sha256(b.encode("utf-8")).hexdigest() for b in bodies]

    np.savez(
        args.out,
        session_ids=np.array(session_ids, dtype=object),
        turn_numbers=np.array(turn_numbers, dtype=np.int64),
        query_vecs=vecs.astype(np.float32),
        query_body_hashes=np.array(body_hashes, dtype=object),
        meta=json.dumps(
            {
                "base_model": args.base_model or "default",
                "adapter": str(args.adapter) if args.adapter else None,
                "dataset": args.dataset,
                "dim": int(vecs.shape[1]),
                "n": int(vecs.shape[0]),
                "max_len": args.max_len,
            }
        ),
    )
    logger.success(f"[twotower-query-cache] saved → {args.out}")


if __name__ == "__main__":
    main()
