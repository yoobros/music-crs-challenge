"""Encode all 47k tracks once with base Qwen3-Embedding-0.6B (or a LoRA adapter)
and persist to a `.npz` cache so downstream steps (hard-neg mining, eval) can
re-use without re-encoding.

Cache layout:
    npz file with keys:
        track_ids : np.ndarray[str]      (N,)
        doc_vecs  : np.ndarray[float32]  (N, D), L2-normalized
        meta      : json string with model + adapter + dim + compose mode

Usage:
    # base
    uv run python -m mymodule.strategies.twotower.doc_cache \\
        --out mymodule/strategies/twotower/data/doc_cache_base.npz --encode-batch 32

    # adapter
    uv run python -m mymodule.strategies.twotower.doc_cache \\
        --adapter mymodule/strategies/twotower/ckpt/qwen3emb06b_lora_train1_full \\
        --out mymodule/strategies/twotower/data/doc_cache_lora.npz --encode-batch 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from mymodule.feature.kvdb import _compose_track_metadata_rich_text
from mymodule.strategies.twotower.encoder import DEFAULT_BASE_MODEL, DEFAULT_MAX_LEN, QwenEmbedderTorch
from mymodule.strategies.twotower.text_compose import ALL_ABLATIONS, build_tag_freq, compose_doc

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", type=Path, default=None, help="Optional PEFT adapter dir.")
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument(
        "--ablate",
        nargs="*",
        default=[],
        choices=sorted(ALL_ABLATIONS) + ["none"],
        help="Compose ablation flags (must match what eval.py will use against this cache).",
    )
    p.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="float32",
        help="Model dtype. float16 ~2x faster on Apple MPS; bfloat16 best on CUDA.",
    )
    p.add_argument(
        "--compose",
        choices=["prod", "qd"],
        default="qd",
        help=(
            "Doc composition mode. "
            "'prod' = mymodule.feature._compose_track_metadata_rich_text (production entity_str), "
            "'qd' = mymodule.strategies.twotower.text_compose.compose_doc (natural-language prose + "
            "crawl meta + LLM-generated 'Suitable for:' EOS anchor)."
        ),
    )
    p.add_argument(
        "--synth-use-cases",
        type=Path,
        default=None,
        help=(
            "Path to synth use-cases JSONL "
            "(default: mymodule/strategies/twotower/data/synth_use_cases.jsonl). "
            "Drives the Mood / Themes / Suitable-for doc lines. Missing → those lines are skipped."
        ),
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ablate_set: set[str] = {a for a in args.ablate if a != "none"}
    if ablate_set:
        logger.info(f"[doc-cache] ablation flags active: {sorted(ablate_set)}")

    logger.info("[doc-cache] loading talkpl Track-Metadata (split=all_tracks)")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")

    # KV + synth side-channel are only meaningful in qd mode (prod uses
    # production entity_str only).
    kv = None
    synth_map: dict[str, dict] = {}
    if args.compose == "qd":
        try:
            from mymodule.feature.kvdb import KVStore

            kv = KVStore.open(read_only=True)
            logger.info(f"[doc-cache] KV opened (crawl_count={kv.crawl_count()})")
        except Exception as e:
            logger.warning(f"[doc-cache] KV unavailable ({type(e).__name__}: {e}); crawl fields will be empty.")
        from mymodule.strategies.twotower.synth_doc import DEFAULT_OUT as SYNTH_OUT
        from mymodule.strategies.twotower.synth_doc import load_synth_use_cases

        synth_path = args.synth_use_cases or SYNTH_OUT
        if synth_path.exists():
            synth_map = load_synth_use_cases(synth_path)
            logger.info(f"[doc-cache] loaded synth use-cases for {len(synth_map)} tracks from {synth_path}")
        else:
            logger.warning(
                f"[doc-cache] synth use-cases cache not found at {synth_path}. "
                "Run `python -m mymodule.strategies.twotower.synth_doc` to populate, "
                "or pass --synth-use-cases <path>. Docs will be missing the EOS 'Suitable for:' anchor."
            )

    tids: list[str] = []
    docs: list[str] = []
    if args.compose == "qd":
        logger.info("[doc-cache] mode=qd — building tag_freq + composing docs")
        freq_map = build_tag_freq(ds)
        for row in tqdm(ds, desc="compose_doc"):
            tid = row["track_id"]
            tids.append(tid)
            crawl = None
            if kv is not None:
                try:
                    crawl = kv.get_track_crawl(tid)
                except Exception:
                    crawl = None
            docs.append(compose_doc(row, freq_map, crawl=crawl, synth=synth_map.get(tid), ablate=ablate_set))
    else:
        logger.info("[doc-cache] mode=prod — production entity_str docs")
        for row in tqdm(ds, desc="compose_doc_prod"):
            tids.append(row["track_id"])
            docs.append(_compose_track_metadata_rich_text(row))

    logger.info(f"[doc-cache] encoding {len(docs)} docs (adapter={args.adapter}, dtype={args.dtype})")
    embedder = QwenEmbedderTorch(
        base_model=args.base_model,
        adapter_path=args.adapter,
        max_len=args.max_len,
        dtype=DTYPE_MAP[args.dtype],
    )
    vecs = embedder.encode_docs(docs, batch_size=args.encode_batch, normalize=True)
    logger.success(f"[doc-cache] vecs shape={vecs.shape} dtype={vecs.dtype}")

    np.savez(
        args.out,
        track_ids=np.array(tids, dtype=object),
        doc_vecs=vecs.astype(np.float32),
        meta=json.dumps(
            {
                "base_model": args.base_model,
                "adapter": str(args.adapter) if args.adapter else None,
                "dim": int(vecs.shape[1]),
                "n": int(vecs.shape[0]),
                "max_len": args.max_len,
                "compose": args.compose,
                "ablate": sorted(ablate_set),
            }
        ),
    )
    logger.success(f"[doc-cache] saved → {args.out}")


if __name__ == "__main__":
    main()
