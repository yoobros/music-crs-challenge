"""Retrieval eval: base Qwen3-Embedding-0.6B vs one or more LoRA adapters.

Encodes 47k tracks (rich metadata doc text, same recipe as
`qemb_metadata_rich`) and devset (= talkpl `test` split) user queries,
then for each query computes top-K cosine similarity. Prints R@K and
nDCG@K side by side (base vs each adapter).

Speed-up: when `--doc-cache` is provided, doc encoding is skipped and the
.npz file is loaded directly. Build the cache once with
`build_doc_cache.py`, then re-use across many adapter evals.

For "is per-epoch overfit?" workflow:
    --adapter ckpt/run_epoch1 ckpt/run_epoch2 ckpt/run_epoch3 ...

Each adapter has its own doc projection, so the **base** doc cache cannot be
re-used for adapter evals (query+doc must live in the same vector space).
Provide adapter-specific caches via `--doc-cache-adapter` (positional w.r.t.
`--adapter`) — build each one with `build_doc_cache.py --adapter ...`.

Usage:
    # base + one LoRA, doc caches for both
    uv run python -m mymodule.strategies.twotower.eval \\
        --adapter mymodule/strategies/twotower/ckpt/qwen3emb06b_lora_turn1_inbatch \\
        --doc-cache-base mymodule/strategies/twotower/data/doc_cache_base.npz \\
        --doc-cache-adapter mymodule/strategies/twotower/data/doc_cache_qwen3emb06b_lora_turn1_inbatch.npz \\
        --turns 1 --top-k 20 --encode-batch 32

    # multiple adapters with parallel caches
    uv run python -m mymodule.strategies.twotower.eval \\
        --adapter ckpt/run_epoch1 ckpt/run_epoch2 ckpt/run_epoch3 \\
        --doc-cache-base mymodule/strategies/twotower/data/doc_cache_base.npz \\
        --doc-cache-adapter mymodule/strategies/twotower/data/cache_epoch1.npz \\
                            mymodule/strategies/twotower/data/cache_epoch2.npz \\
                            mymodule/strategies/twotower/data/cache_epoch3.npz
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from mymodule.feature.kvdb import _compose_track_metadata_rich_text
from mymodule.feature.ollama_embed import _compose_talkpl_metadata_rich_query
from mymodule.strategies.twotower.encoder import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_LEN,
    QWEN3_QUERY_INSTRUCTION,
    QwenEmbedderTorch,
)
from mymodule.strategies.twotower.pool import build_query_body
from mymodule.strategies.twotower.text_compose import ALL_ABLATIONS, build_tag_freq, compose_doc


def _resolve_max_body_tokens(tokenizer, max_len: int) -> int:
    """max_len minus the Instruct/Query prefix length, measured via tokenizer."""
    prefix = f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery: "
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    return max(32, max_len - prefix_tokens - 1)


def load_track_corpus(
    limit: int | None = None,
    compose_mode: str = "qd",
    ablate: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (track_ids, doc_texts) for the full track catalog.

    In ``qd`` mode the doc is built via ``compose_doc`` with KV crawl metadata
    and (when available) the JSONL-cached synth use-cases / mood / themes
    enrichment from ``mymodule.strategies.twotower.synth_doc``. Missing crawl
    or synth entries cause those lines to be skipped per-track.
    """
    logger.info("[twotower-eval] loading talkpl-ai Track-Metadata (split=all_tracks)")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    n = len(ds) if limit is None else min(limit, len(ds))
    tids: list[str] = []
    docs: list[str] = []
    if compose_mode == "qd":
        logger.info("[twotower-eval] mode=qd doc — building tag_freq + composing docs")
        freq_map = build_tag_freq(ds)

        kv = None
        try:
            from mymodule.feature.kvdb import KVStore

            kv = KVStore.open(read_only=True)
            logger.info(f"[twotower-eval] KV opened (crawl_count={kv.crawl_count()})")
        except Exception as e:
            logger.warning(f"[twotower-eval] KV unavailable ({type(e).__name__}: {e}); crawl fields empty.")

        from mymodule.strategies.twotower.synth_doc import DEFAULT_OUT as SYNTH_OUT
        from mymodule.strategies.twotower.synth_doc import load_synth_use_cases

        synth_map: dict[str, dict] = {}
        if SYNTH_OUT.exists():
            synth_map = load_synth_use_cases(SYNTH_OUT)
            logger.info(f"[twotower-eval] loaded synth use-cases for {len(synth_map)} tracks")
        else:
            logger.warning(
                f"[twotower-eval] synth use-cases cache absent at {SYNTH_OUT}. "
                "Docs will miss the EOS 'Suitable for:' anchor."
            )

        for i in tqdm(range(n), desc="compose_doc"):
            row = ds[int(i)]
            tid = row["track_id"]
            tids.append(tid)
            crawl = None
            if kv is not None:
                try:
                    crawl = kv.get_track_crawl(tid)
                except Exception:
                    crawl = None
            docs.append(compose_doc(row, freq_map, crawl=crawl, synth=synth_map.get(tid), ablate=ablate))
    else:
        logger.info("[twotower-eval] mode=prod doc — production entity_str")
        for i in tqdm(range(n), desc="compose_doc_prod"):
            row = ds[int(i)]
            tids.append(row["track_id"])
            docs.append(_compose_track_metadata_rich_text(row))
    return tids, docs


def parse_devset_queries(
    turns: Iterable[int],
    compose_mode: str = "qd",
    tokenizer=None,
    max_body_tokens: int = 240,
    ablate: set[str] | None = None,
) -> list[dict]:
    """Build (query_text, gt_track_ids) pairs from devset (talkpl `test` split)."""
    logger.info("[lora-eval] loading devset (split=test)")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    out: list[dict] = []
    turns_set = set(int(t) for t in turns)

    try:
        from mymodule.feature.kvdb import KVStore

        kv = KVStore.open(read_only=True)
    except Exception as e:
        logger.warning(f"[lora-eval] KV unavailable ({type(e).__name__}: {e}); using raw user_query.")
        kv = None

    freq_map: dict[str, int] = {}
    if compose_mode == "qd":
        from mymodule.strategies.twotower.text_compose import load_tag_freq

        try:
            freq_map = load_tag_freq()
            logger.info(f"[lora-eval] mode=qd query — loaded tag_freq ({len(freq_map)} tags)")
        except (FileNotFoundError, RuntimeError):
            logger.info("[lora-eval] mode=qd query — tag_freq cache missing, building from track metadata")
            track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
            freq_map = build_tag_freq(track_ds)

    # Optional pre-baked LLM session summaries — drives the `Session so far:` line.
    summary_map: dict[str, str] = {}
    if compose_mode == "qd":
        from mymodule.strategies.twotower.query_summary import (
            DEFAULT_OUT as SUMMARY_OUT,
        )
        from mymodule.strategies.twotower.query_summary import (
            load_query_summaries,
        )

        if SUMMARY_OUT.exists():
            summary_map = load_query_summaries(SUMMARY_OUT)
            logger.info(f"[twotower-eval] loaded {len(summary_map)} session summaries")
        else:
            logger.warning(
                f"[twotower-eval] session summaries cache absent at {SUMMARY_OUT}. "
                "Pre-bake via `python -m mymodule.strategies.twotower.query_summary "
                "--dataset devset` to populate the 'Session so far:' line."
            )

    for item in tqdm(ds, desc="parse_devset"):
        goal = item.get("conversation_goal")
        conv = item["conversations"]
        by_turn: dict[int, list[dict]] = {}
        for row in conv:
            by_turn.setdefault(int(row["turn_number"]), []).append(row)
        for tn in sorted(by_turn.keys()):
            if tn not in turns_set:
                continue
            cur = by_turn[tn]
            user_msg = next((m for m in cur if m.get("role") == "user"), None)
            if user_msg is None:
                continue
            gt = [m["content"] for m in cur if m.get("role") == "music" and m.get("content")]
            if not gt:
                continue
            chat_history = [m for m in conv if int(m.get("turn_number", 0)) < tn]
            if compose_mode == "qd":
                # Shared with serving (pool) + in-training eval — single source of truth.
                q_text = build_query_body(
                    user_msg["content"],
                    chat_history,
                    goal,
                    kv=kv,
                    tag_freq=freq_map,
                    tokenizer=tokenizer,
                    max_body_tokens=max_body_tokens,
                    summary_map=summary_map,
                    ablate=ablate,
                )
            else:
                q_text = _compose_talkpl_metadata_rich_query(user_msg["content"], chat_history, kv)
            out.append(
                {
                    "session_id": item.get("session_id"),
                    "turn": tn,
                    "query_text": q_text,
                    "gt": gt,
                    "specificity": (item.get("conversation_goal") or {}).get("specificity") or "unknown",
                    "category": (item.get("conversation_goal") or {}).get("category") or "unknown",
                }
            )
    return out


def recall_at_k(pred: list[str], gt: list[str], k: int) -> float:
    if not gt:
        return 0.0
    truth = set(gt)
    hits = sum(1 for t in pred[:k] if t in truth)
    return hits / min(k, len(truth))


def ndcg_at_k(pred: list[str], gt: list[str], k: int) -> float:
    if not gt:
        return 0.0
    truth = set(gt)
    dcg = 0.0
    for i, t in enumerate(pred[:k]):
        if t in truth:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _autodetect_eval_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def topk_search(q_vecs: np.ndarray, d_vecs: np.ndarray, k: int, batch: int = 64) -> np.ndarray:
    nq = q_vecs.shape[0]
    out = np.zeros((nq, k), dtype=np.int64)
    qt = torch.from_numpy(q_vecs).to(_autodetect_eval_device())
    dt = torch.from_numpy(d_vecs).to(qt.device)
    for i in tqdm(range(0, nq, batch), desc="topk"):
        sims = qt[i : i + batch] @ dt.T
        top = sims.topk(k, dim=-1).indices.cpu().numpy()
        out[i : i + batch] = top
    return out


def summarize(label: str, queries: list[dict], tids: list[str], top_idx: np.ndarray, k_list: list[int]) -> dict:
    """Compute R@k / nDCG@k for every k in `k_list` from a single top-K index matrix.

    `top_idx` must have width >= max(k_list); each k slices `top_idx[:, :k]`.
    Output keeps the same nested shape as before (by_turn / by_specificity /
    by_category / by_turn_specificity) but each leaf carries `R@k`/`nDCG@k`
    pairs for every k.
    """
    ks = sorted(set(k_list))
    max_k = max(ks)
    if top_idx.shape[1] < max_k:
        raise ValueError(f"top_idx width {top_idx.shape[1]} < max(k_list)={max_k}")

    # Per-query metrics for every k: shape (n_queries, len(ks), 2) — last dim = (R, nDCG)
    per_q: dict[int, list[tuple[float, float]]] = {k: [] for k in ks}
    by_turn: dict[int, dict[int, list[tuple[float, float]]]] = {}
    by_spec: dict[str, dict[int, list[tuple[float, float]]]] = {}
    by_cat: dict[str, dict[int, list[tuple[float, float]]]] = {}
    by_turn_spec: dict[int, dict[str, dict[int, list[tuple[float, float]]]]] = {}

    for qi, q in enumerate(queries):
        full_pred = [tids[int(j)] for j in top_idx[qi]]
        t = q.get("turn", 0)
        s = q.get("specificity", "unknown")
        c = q.get("category", "unknown")
        for k in ks:
            pred = full_pred[:k]
            r = recall_at_k(pred, q["gt"], k)
            n = ndcg_at_k(pred, q["gt"], k)
            per_q[k].append((r, n))
            by_turn.setdefault(t, {}).setdefault(k, []).append((r, n))
            by_spec.setdefault(s, {}).setdefault(k, []).append((r, n))
            by_cat.setdefault(c, {}).setdefault(k, []).append((r, n))
            by_turn_spec.setdefault(t, {}).setdefault(s, {}).setdefault(k, []).append((r, n))

    def _avg(per_k: dict[int, list[tuple[float, float]]]) -> dict:
        any_pairs = next(iter(per_k.values()))
        out = {"n": len(any_pairs)}
        for k in ks:
            pairs = per_k[k]
            out[f"R@{k}"] = sum(x[0] for x in pairs) / max(1, len(pairs))
            out[f"nDCG@{k}"] = sum(x[1] for x in pairs) / max(1, len(pairs))
        return out

    summary: dict = {"label": label, "n_queries": len(queries)}
    for k in ks:
        summary[f"R@{k}"] = sum(x[0] for x in per_q[k]) / max(1, len(per_q[k]))
        summary[f"nDCG@{k}"] = sum(x[1] for x in per_q[k]) / max(1, len(per_q[k]))
    summary["by_turn"] = {t: _avg(per_k) for t, per_k in sorted(by_turn.items())}
    summary["by_specificity"] = {s: _avg(per_k) for s, per_k in sorted(by_spec.items())}
    summary["by_category"] = {c: _avg(per_k) for c, per_k in sorted(by_cat.items())}
    summary["by_turn_specificity"] = {
        t: {s: _avg(per_k) for s, per_k in sorted(d.items())} for t, d in sorted(by_turn_spec.items())
    }
    return summary


def encode_or_load(
    embedder: QwenEmbedderTorch,
    tids: list[str],
    docs: list[str],
    queries: list[dict],
    encode_batch: int,
    cache_path: Path | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        logger.info(f"[lora-eval/{label}] loading doc cache {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        cached_tids = z["track_ids"].astype(str).tolist()
        if cached_tids != tids:
            raise RuntimeError("doc cache track_ids order mismatch — rebuild with build_doc_cache.py")
        d_vecs = z["doc_vecs"].astype(np.float32)
    else:
        logger.info(f"[lora-eval/{label}] encoding {len(docs)} docs (batch={encode_batch})")
        d_vecs = embedder.encode_docs(docs, batch_size=encode_batch, normalize=True)

    logger.info(f"[lora-eval/{label}] encoding {len(queries)} queries (batch={encode_batch})")
    q_vecs = embedder.encode_queries([q["query_text"] for q in queries], batch_size=encode_batch, normalize=True)
    return q_vecs, d_vecs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        type=Path,
        nargs="+",
        required=False,
        default=[],
        help="One or more PEFT adapter dirs to evaluate (in addition to base unless --skip-base).",
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--turns", type=int, nargs="+", default=[1])
    p.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=[20, 100],
        help="List of K values to compute R@K and nDCG@K for. Encoding cost = max(K). Default: 20 100.",
    )
    p.add_argument("--corpus-limit", type=int, default=None)
    p.add_argument("--query-limit", type=int, default=None)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--skip-base", action="store_true")
    p.add_argument(
        "--doc-cache-base",
        type=Path,
        default=None,
        help="Pre-encoded base doc cache (.npz from build_doc_cache.py). Speeds up the base eval.",
    )
    p.add_argument(
        "--doc-cache-adapter",
        type=Path,
        nargs="+",
        default=[],
        help=(
            "Pre-encoded adapter doc caches (.npz from "
            "`build_doc_cache.py --adapter ...`). Order must match --adapter; "
            "use an empty / non-existent path to fall back to on-the-fly encoding "
            "for a specific slot. Each cache is bound to its own adapter — caches "
            "are NOT interchangeable across adapters."
        ),
    )
    p.add_argument(
        "--compose",
        choices=["prod", "qd"],
        default="qd",
        help=(
            "Compose mode for query+doc. "
            "'prod' = production entity_str (mymodule.feature), "
            "'qd' = mymodule.strategies.twotower.text_compose.compose_doc + compose_query."
        ),
    )
    p.add_argument("--out", type=Path, default=Path("mymodule/strategies/twotower/data/eval_summary.json"))
    p.add_argument(
        "--ablate",
        nargs="*",
        default=[],
        choices=sorted(ALL_ABLATIONS) + ["none"],
        help=(
            "Compose ablation flags. Pass one or more of "
            f"{sorted(ALL_ABLATIONS)} to drop the named component from compose. "
            "`none` = no ablation (default)."
        ),
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_base and not args.adapter:
        raise SystemExit("Nothing to evaluate: --skip-base and no --adapter.")

    tokenizer = None
    max_body_tokens = args.max_len
    if args.compose == "qd":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        max_body_tokens = _resolve_max_body_tokens(tokenizer, args.max_len)
        logger.info(f"[lora-eval] max_body_tokens={max_body_tokens} (max_len={args.max_len})")

    ablate_set: set[str] = {a for a in args.ablate if a != "none"}
    if ablate_set:
        logger.info(f"[twotower-eval] ablation flags active: {sorted(ablate_set)}")
    tids, docs = load_track_corpus(limit=args.corpus_limit, compose_mode=args.compose, ablate=ablate_set)
    queries = parse_devset_queries(
        args.turns,
        compose_mode=args.compose,
        tokenizer=tokenizer,
        max_body_tokens=max_body_tokens,
        ablate=ablate_set,
    )
    if args.query_limit:
        queries = queries[: args.query_limit]
    logger.info(f"[lora-eval] {len(queries)} queries (turns={args.turns}) vs {len(tids)} tracks")

    summaries: list[dict] = []

    max_k = max(args.top_ks)
    if not args.skip_base:
        emb = QwenEmbedderTorch(base_model=args.base_model, adapter_path=None, max_len=args.max_len)
        q_v, d_v = encode_or_load(emb, tids, docs, queries, args.encode_batch, args.doc_cache_base, "base")
        top_idx = topk_search(q_v, d_v, max_k)
        summaries.append(summarize("base", queries, tids, top_idx, args.top_ks))
        del emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.doc_cache_adapter and len(args.doc_cache_adapter) != len(args.adapter):
        raise SystemExit(
            f"--doc-cache-adapter ({len(args.doc_cache_adapter)}) must match --adapter count "
            f"({len(args.adapter)}); use a non-existent placeholder path to skip a slot."
        )

    for i, adapter in enumerate(args.adapter):
        label = adapter.name
        # adapter changes the doc-side projection — only an adapter-specific cache is valid.
        cache_path = args.doc_cache_adapter[i] if args.doc_cache_adapter else None
        emb = QwenEmbedderTorch(base_model=args.base_model, adapter_path=adapter, max_len=args.max_len)
        q_v, d_v = encode_or_load(emb, tids, docs, queries, args.encode_batch, cache_path, label)
        top_idx = topk_search(q_v, d_v, max_k)
        summaries.append(summarize(label, queries, tids, top_idx, args.top_ks))
        del emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(summaries, indent=2))

    # pretty table — one column pair (R@k, nDCG@k) per k in args.top_ks
    ks = sorted(set(args.top_ks))
    header_metric_cols = "  ".join(f"{'R@' + str(k):>10}  {'nDCG@' + str(k):>12}" for k in ks)
    table_width = 40 + 6 + 2 + len(header_metric_cols)
    print()
    print("=" * table_width)
    print(f"{'label':<40} {'n':>6}  {header_metric_cols}")
    print("-" * table_width)
    base_metrics: dict[int, tuple[float, float]] = {}
    for s in summaries:
        if s["label"] == "base":
            base_metrics = {k: (s[f"R@{k}"], s[f"nDCG@{k}"]) for k in ks}
        row = f"{s['label']:<40} {s['n_queries']:>6}"
        for k in ks:
            r = s[f"R@{k}"]
            n = s[f"nDCG@{k}"]
            cell = f"  {r:>10.4f}  {n:>12.4f}"
            if s["label"] != "base" and k in base_metrics:
                br, bn = base_metrics[k]
                cell += f" (Δ {r - br:+.4f} / {n - bn:+.4f})"
            row += cell
        print(row)
    print("=" * table_width)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
