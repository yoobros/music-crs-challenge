"""Extract (query_text, track_doc) training pairs from talkpl `train` split.

⚠️  IMPORTANT: this reads `talkpl-ai/TalkPlayData-Challenge-Dataset` split
**`train`** (15,199 sessions). It is *NOT* the devset — the devset is the
`test` split (1,000 sessions) used by `eval.py` and never appears in any
training run.

For each (session, turn) where music tracks are present, we build one JSONL
row per GT track:

| field             | description                                              |
|-------------------|----------------------------------------------------------|
| session_id        | for session-aware train/val splits                       |
| turn_number       | 1..8                                                     |
| track_id          | GT track                                                 |
| query             | composed via `compose_query` (qd) or production (prod)   |
| doc_text          | composed via `compose_doc` (qd) or production (prod)     |
| specificity       | conversation_goal.specificity (HH/HL/LH/LL/unknown)      |
| track_artist      | GT track's artist (hard-neg pool key)                    |
| track_album       | GT track's album (hard-neg pool key)                     |
| track_top_tags    | GT track's top-3 cleaned tags (hard-neg pool key)        |
| track_pop_bucket  | GT track's 10-pt popularity bucket (e.g. "p60-69")       |
| bm25_top_tids     | top-N BM25 hits over catalog (excluding GTs of this turn)|

Side files emitted alongside the JSONL (always, unless --no-side-files):
- `doc_text_map_{compose}.jsonl` — `{tid: doc_text}` for all 47k tracks.
  train.py loads this to convert hard-neg track_ids → doc text.
- `track_meta_map_{compose}.json` — `{tid: {artist, album, top_tags, pop_bucket}}`
  for all 47k tracks. train.py inverts this into HardNegPools.

BM25 negatives use `mymodule.strategies.pool.bm25.BM25Pool` directly (same
spaCy + noun_tags + music_only config as production), so the hard-negs the
LoRA model sees during training are the **exact same** mistakes BM25 would
make at inference. Index cache: `mymodule/feature/.bm25_cache/`.

Usage:
    uv run python -m mymodule.strategies.twotower.extract_pairs --split train --n 1000000 \\
        --compose qd --bm25-topk 20 --out mymodule/strategies/twotower/data/train_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from mymodule.feature.kvdb import _compose_track_metadata_rich_text
from mymodule.feature.ollama_embed import _compose_talkpl_metadata_rich_query
from mymodule.strategies.twotower.encoder import DEFAULT_BASE_MODEL, DEFAULT_MAX_LEN, QWEN3_QUERY_INSTRUCTION
from mymodule.strategies.twotower.pool import build_query_body
from mymodule.strategies.twotower.text_compose import (
    ALL_ABLATIONS,
    _bucket_pop_str,
    _first_str,
    _popularity,
    build_tag_freq,
    clean_tags,
    compose_doc,
)
from mymodule.utils.seen import extract_session_seen_tracks


def _resolve_max_body_tokens(tokenizer, max_len: int) -> int:
    """max_len minus the Instruct/Query prefix length, measured via tokenizer."""
    prefix = f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery: "
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    return max(32, max_len - prefix_tokens - 1)


def _extract_track_meta(track_row: dict, freq_map: dict[str, int]) -> dict[str, Any]:
    """Per-track {artist, album, top_tags, pop_bucket} for hard-neg pool keys.

    `top_tags` runs through the same `clean_tags` pipeline as compose_doc so
    doc tags and hard-neg pool keys stay aligned.
    """
    artist = _first_str(track_row.get("artist_name"), "")
    album = _first_str(track_row.get("album_name"), "")
    top_tags = clean_tags(
        track_row.get("tag_list"),
        freq_map,
        min_freq=5,
        top_n=3,
        normalize_british=True,
        substring_dedup=True,
        preserve_input_order=True,
    )
    pop_bucket = _bucket_pop_str(_popularity(track_row.get("popularity")))
    return {"artist": artist, "album": album, "top_tags": top_tags, "pop_bucket": pop_bucket}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--turns", type=int, nargs="+", default=None, help="Restrict to specific turns (default: all 1..8).")
    p.add_argument("--out", type=Path, required=True)
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
    p.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Used to load the Qwen3 tokenizer for token-budget-aware query truncation.",
    )
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument(
        "--bm25-topk",
        type=int,
        default=20,
        help="Number of BM25 hits to pre-bake per (session,turn) for hard-neg pool. "
        "0 disables BM25 mining (bm25_top_tids will be empty lists).",
    )
    p.add_argument(
        "--no-side-files",
        action="store_true",
        help="Skip emitting doc_text_map_{compose}.jsonl and track_meta_map_{compose}.json. "
        "Use when you only need the training jsonl and side files already exist.",
    )
    p.add_argument(
        "--synth-use-cases",
        type=Path,
        default=None,
        help=(
            "Path to synth use-cases JSONL "
            "(default: mymodule/strategies/twotower/data/synth_use_cases.jsonl). "
            "Drives the Mood / Themes / Suitable-for doc lines used as training-side doc text."
        ),
    )
    p.add_argument(
        "--ablate",
        nargs="*",
        default=[],
        choices=sorted(ALL_ABLATIONS) + ["none"],
        help=(
            "Compose ablation flags applied to BOTH compose_doc and compose_query during "
            "extract. Keep this consistent with eval.py --ablate so train/inference share "
            "the same text shape. Use 'no-cat-tail' to drop the Looking-for cat tail."
        ),
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ablate_set: set[str] = {a for a in args.ablate if a != "none"}
    if ablate_set:
        logger.info(f"[extract] ablation flags active: {sorted(ablate_set)}")

    logger.info(f"[extract] loading talkpl Challenge-Dataset split={args.split}")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=args.split)

    logger.info("[extract] loading talkpl Track-Metadata (split=all_tracks) for doc lookup")
    track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")

    logger.info("[extract] building tag_freq (used by clean_tags + hard-neg top_tags)")
    freq_map = build_tag_freq(track_ds)

    # KV + synth side-channel — qd mode only (prod uses entity_str standalone).
    kv = None
    synth_map: dict[str, dict] = {}
    summary_map: dict[str, str] = {}
    if args.compose == "qd":
        try:
            from mymodule.feature.kvdb import KVStore

            kv = KVStore.open(read_only=True)
            logger.info(f"[extract] KV opened (crawl_count={kv.crawl_count()})")
        except Exception as e:
            logger.warning(f"[extract] KV unavailable ({type(e).__name__}: {e}); crawl fields empty.")
        from mymodule.strategies.twotower.synth_doc import DEFAULT_OUT as SYNTH_OUT
        from mymodule.strategies.twotower.synth_doc import load_synth_use_cases

        synth_path = args.synth_use_cases or SYNTH_OUT
        if synth_path.exists():
            synth_map = load_synth_use_cases(synth_path)
            logger.info(f"[extract] loaded synth use-cases for {len(synth_map)} tracks from {synth_path}")
        else:
            logger.warning(
                f"[extract] synth use-cases cache not found at {synth_path}. "
                "Docs will miss the EOS 'Suitable for:' anchor — run "
                "`python -m mymodule.strategies.twotower.synth_doc` to populate."
            )

        # Query-side 'Session so far:' summaries — mirror inference (pool.py
        # compose_query_body) so the LoRA trains on the SAME query distribution
        # it sees at inference. Keyed by summary_key(chat_history, user_msg).
        from mymodule.strategies.twotower.query_summary import load_query_summaries

        summary_map = load_query_summaries()
        if summary_map:
            logger.info(f"[extract] loaded {len(summary_map)} query summaries ('Session so far:' injection)")
        else:
            logger.warning(
                "[extract] query_summaries cache empty — train queries will OMIT the 'Session so far:' "
                "line while inference injects it (train/inference skew). Run "
                "`python -m mymodule.strategies.twotower.query_summary --dataset train` first."
            )

    tokenizer = None
    max_body_tokens = args.max_len
    if args.compose == "qd":
        logger.info(f"[extract] mode=qd — composing docs (base={args.base_model})")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        max_body_tokens = _resolve_max_body_tokens(tokenizer, args.max_len)
        logger.info(f"[extract] max_body_tokens={max_body_tokens} (max_len={args.max_len})")
        tid_to_doc: dict[str, str] = {}
        for row in tqdm(track_ds, desc="compose_doc"):
            tid = row["track_id"]
            crawl = None
            if kv is not None:
                try:
                    crawl = kv.get_track_crawl(tid)
                except Exception:
                    crawl = None
            tid_to_doc[tid] = compose_doc(row, freq_map, crawl=crawl, synth=synth_map.get(tid), ablate=ablate_set)
    else:
        logger.info("[extract] mode=prod — production entity_str")
        tid_to_doc = {}
        for row in tqdm(track_ds, desc="compose_doc_prod"):
            tid_to_doc[row["track_id"]] = _compose_track_metadata_rich_text(row)

    logger.info("[extract] extracting per-track artist/album/top_tags/pop_bucket for hard-neg pools")
    tid_to_meta: dict[str, dict[str, Any]] = {}
    for row in tqdm(track_ds, desc="extract_meta"):
        tid_to_meta[row["track_id"]] = _extract_track_meta(row, freq_map)

    # BM25 mining pool — reuse production's BM25Pool (cached index in
    # `mymodule/feature/.bm25_cache`). We over-request (topk+10) so the
    # GT-filter below still yields topk negatives even when GT lands in the
    # BM25 top hits (it usually does at turn>=2).
    bm25_pool = None
    if args.bm25_topk > 0:
        logger.info(f"[extract] initializing BM25Pool (top-{args.bm25_topk} per turn)")
        from mymodule.strategies.pool.bm25 import BM25Pool

        bm25_pool = BM25Pool(n_candidates=args.bm25_topk + 10)

    try:
        from mymodule.feature.kvdb import KVStore

        kv = KVStore.open(read_only=True)
    except Exception as e:
        logger.warning(f"[extract] KV unavailable ({type(e).__name__}: {e}); chat history will be empty.")
        kv = None

    turns_set = set(args.turns) if args.turns else set(range(1, 9))
    pairs: list[dict] = []
    for item in tqdm(ds, desc="parse_sessions"):
        sid = item.get("session_id") or item.get("id")
        uid = item.get("user_id") or ""
        goal = item.get("conversation_goal")
        specificity = ((goal or {}).get("specificity") or "unknown").strip().upper() or "unknown"
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
            seen_tids = sorted(extract_session_seen_tracks(chat_history))
            if args.compose == "qd":
                # Shared with serving (pool) + train/devset eval — single source of truth
                # (budget + summary injection + overflow guard).
                q_text = build_query_body(
                    user_msg["content"],
                    chat_history,
                    goal,
                    kv=kv,
                    tag_freq=freq_map,
                    tokenizer=tokenizer,
                    max_body_tokens=max_body_tokens,
                    summary_map=summary_map,
                    ablate=ablate_set,
                )
            else:
                q_text = _compose_talkpl_metadata_rich_query(user_msg["content"], chat_history, kv)

            # BM25 hits — one call per (session, turn). Filter out current
            # turn's GT (BM25 happily returns it since GT isn't "seen" yet
            # at this turn).
            bm25_topk: list[str] = []
            if bm25_pool is not None:
                try:
                    bm25_hits = bm25_pool.generate(
                        user_msg["content"],
                        chat_history,
                        uid,
                        conversation_goal=goal,
                    )
                    gt_set = set(gt)
                    bm25_topk = [t for t in bm25_hits if t not in gt_set][: args.bm25_topk]
                except Exception as e:  # robustness — log and keep going
                    logger.warning(f"[extract] BM25 retrieve failed (session={sid}, turn={tn}): {e}")

            for tid in gt:
                doc = tid_to_doc.get(tid)
                if not doc:
                    continue
                meta = tid_to_meta.get(tid, {})
                pairs.append(
                    {
                        "session_id": sid,
                        "turn_number": tn,
                        "track_id": tid,
                        "query": q_text,
                        "doc_text": doc,
                        "specificity": specificity,
                        "track_artist": meta.get("artist", ""),
                        "track_album": meta.get("album", ""),
                        "track_top_tags": meta.get("top_tags", []),
                        "track_pop_bucket": meta.get("pop_bucket", ""),
                        "bm25_top_tids": bm25_topk,
                        "seen": seen_tids,
                    }
                )

    logger.info(f"[extract] collected {len(pairs)} (query, doc) pairs")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    sampled = pairs[: args.n]
    with args.out.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success(f"[extract] wrote {len(sampled)} → {args.out}")

    if args.no_side_files:
        logger.info("[extract] --no-side-files set; skipping side file emission")
        return

    doc_map_path = args.out.parent / f"doc_text_map_{args.compose}.jsonl"
    with doc_map_path.open("w") as f:
        for tid, doc_text in tid_to_doc.items():
            f.write(json.dumps({"tid": tid, "doc_text": doc_text}, ensure_ascii=False) + "\n")
    logger.success(f"[extract] wrote doc_text_map ({len(tid_to_doc)} tids) → {doc_map_path}")

    meta_map_path = args.out.parent / f"track_meta_map_{args.compose}.json"
    with meta_map_path.open("w") as f:
        json.dump(tid_to_meta, f, ensure_ascii=False)
    logger.success(f"[extract] wrote track_meta_map ({len(tid_to_meta)} tids) → {meta_map_path}")


if __name__ == "__main__":
    main()
