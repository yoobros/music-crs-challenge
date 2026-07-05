"""Two-tower retrieval pool: Qwen3-Embedding-0.6B fine-tuned with LoRA.

Pipeline (`mymodule.strategies.twotower`):

- `text_compose` — query / doc text composition (tag normalization,
  decade/popularity buckets, token-budget truncation).
- `encoder` — HF backbone + optional PEFT LoRA adapter, last-token pooling.
- `hard_negative` — specificity-aware hard-negative sampler used at training.
- `extract_pairs` — CLI to emit (query, doc) JSONL from talkpl `train` split.
- `doc_cache` — CLI to encode all 47k tracks once → `.npz` for ANN.
- `train` — InfoNCE multi-positive contrastive trainer.
- `eval` — base vs adapter R@K / nDCG side-by-side on devset.
- `pool.TwoTowerPool` — `BasePool` implementation for inference; registered
  as the `twotower` pool via `mymodule/strategies/pool/twotower.py`.

Default checkpoint / cache live alongside the module so `TwoTowerPool()` works
without arguments once `train.py` + `doc_cache.py` have been run:

    mymodule/strategies/twotower/ckpt/twotower/       (PEFT adapter dir)
    mymodule/strategies/twotower/data/doc_cache.npz   (adapter-encoded tracks)
"""
