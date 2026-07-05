"""Offline devset eval for a saved two-tower LoRA adapter.

Decouples evaluation from training: load a base model + a saved LoRA adapter
dir and run the SAME full-catalog devset retrieval the training loop runs
(``evaluate_retrieval_texts`` over ``_load_devset_queries`` + ``_load_catalog``),
so a checkpoint's devset nDCG@20 / R@20 / R@100 / MRR@20 can be measured after
the fact — e.g. when per-epoch eval was skipped, or to compare archived ckpts.

Usage:
    uv run python -m mymodule.strategies.twotower.eval_ckpt \
        --base-model Qwen/Qwen3-Embedding-4B \
        --adapter mymodule/strategies/twotower/ckpt/qwen3emb4b_bf16lora_epoch1 \
        --compose qd --max-len 256 --batch-size 16 --bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from mymodule.strategies.twotower.train import (
    _load_catalog,
    _load_devset_queries,
    evaluate_retrieval_texts,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="HF base model id (must match training).")
    p.add_argument("--adapter", type=Path, required=True, help="Saved LoRA adapter dir to evaluate.")
    p.add_argument("--compose", default="qd", help="Query/doc compose recipe (must match training).")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("mymodule/strategies/twotower/data"),
        help="Dir with the catalog/doc/track maps (must match training).",
    )
    p.add_argument("--bf16", action="store_true", help="Load base in bf16 (match training dtype).")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON path to write metrics to.")
    args = p.parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(f"adapter dir not found: {args.adapter}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    logger.info(f"[eval-ckpt] base={args.base_model} adapter={args.adapter} device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(
        args.base_model, padding_side="left", truncation_side="left", trust_remote_code=True
    )
    base = AutoModel.from_pretrained(
        args.base_model, dtype=dtype, attn_implementation="sdpa", trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()

    catalog_tids, catalog_doc_texts = _load_catalog(args.meta_dir, args.compose)
    logger.info(f"[eval-ckpt] catalog loaded: {len(catalog_tids)} tracks")

    devset_queries = _load_devset_queries(args.compose, args.max_len, tok)
    logger.info(f"[eval-ckpt] devset queries: {len(devset_queries)}")

    metrics = evaluate_retrieval_texts(
        model,
        tok,
        query_texts=[q["query_text"] for q in devset_queries],
        gt_sets=[q["gt"] for q in devset_queries],
        catalog_tids=catalog_tids,
        catalog_doc_texts=catalog_doc_texts,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=device,
        seen_sets=[q["seen"] for q in devset_queries],
    )
    logger.success(
        f"[eval-ckpt] {args.adapter.name} devset nDCG@20={metrics['ndcg@20']:.4f} "
        f"R@20={metrics['recall@20']:.4f} R@100={metrics['recall@100']:.4f} "
        f"MRR@20={metrics['mrr@20']:.4f} (n_eval={int(metrics['n_eval_queries'])})"
    )
    if args.out:
        args.out.write_text(json.dumps({"adapter": str(args.adapter), **metrics}, indent=2))
        logger.info(f"[eval-ckpt] wrote {args.out}")


if __name__ == "__main__":
    main()
