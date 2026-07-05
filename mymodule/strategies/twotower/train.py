"""LoRA-tune Qwen3-Embedding-0.6B on (query, GT-tracks) groups with
multi-positive InfoNCE.

Input: JSONL produced by `mymodule/strategies/twotower/extract_pairs.py`. Each row =
`{session_id, turn_number, track_id, query, doc_text}`.

Pipeline
--------
1. **Group by query text.** Rows sharing the same `query` collapse into a
   single training instance with `gt_docs: list[str]`. This naturally
   handles two cases:
   - Single (session, turn) yielding multiple GT tracks (one row per track,
     same query) → all GTs become positives for that query.
   - Different sessions reusing the same query text → same as above.
2. **Session-aware train/val split.** A fraction of *sessions* (not rows)
   is held out for val so multi-turn rows from the same session never
   straddle the split.
3. **Multi-positive InfoNCE.**
   - Each batch carries B unique queries and a flat doc bank D of
     `sum_i K_i` rendered positive docs (K_i = #positives for query i).
   - logits = (Q @ D.T) / temperature, shape (B, sum_K).
   - For query i, **all** doc indices belonging to query i are positives;
     all other doc indices (= other queries' positives) are negatives.
   - Loss_i = -[logsumexp(logits[i, pos_i]) - logsumexp(logits[i, :])],
     averaged across queries. This is the standard supervised-contrastive /
     multi-positive InfoNCE form.
   - Symmetric d→q is **not** applied (D bank is asymmetric in size).

Pooling = last-token (Qwen3-Embedding official). Both branches share the
same LoRA-injected backbone — single set of LoRA weights applied to
query- and doc-side text identically.

Typical recipe (Option B: full multi-turn, multi-positive)
----------------------------------------------------------
    uv run python -m mymodule.strategies.twotower.train \\
      --data mymodule/strategies/twotower/data/train_full.jsonl \\
      --run-name qwen3emb06b_lora_full_mp \\
      --batch-size 16 --epochs 3 --lr 5e-5 --temperature 0.05 \\
      --bf16 --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 --max-len 256

Outputs:
    mymodule/strategies/twotower/ckpt/{run_name}/             best ckpt
        (highest val_ndcg@20 — full catalog retrieval; falls back to lowest
         val_loss when --val-retrieval-disable is set)
    mymodule/strategies/twotower/ckpt/{run_name}_epoch{N}/    each epoch (for per-epoch eval)
    mymodule/strategies/twotower/ckpt/{run_name}_final/       final epoch
    mymodule/strategies/twotower/ckpt/{run_name}/history.json per-epoch metrics
        (val_loss/acc/pos_cos/max_neg_cos + val_ndcg@20/recall@{20,100}/mrr@20
         when retrieval validation runs that epoch)
    mymodule/strategies/twotower/ckpt/{run_name}/train.log
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False

from mymodule.cv import CVConfig, split_items
from mymodule.strategies.twotower.encoder import (
    DEFAULT_BASE_MODEL,
    QWEN3_QUERY_INSTRUCTION,
    assert_torch_cuda_or_warn,
    ensure_dirs,
    format_doc,
    format_query,
    get_lora_target_modules,
    last_token_pool,
    resolve_max_body_tokens,
)
from mymodule.strategies.twotower.hard_negative import (
    HardNegPools,
    get_distribution,
    sample_hard_negs,
)

# DDP collective timeout. The per-epoch full-catalog val + devset eval (rank0
# metric compute, two 47K-catalog encodes) + ckpt save can hold
# non-main ranks at the end-of-epoch barrier far longer than NCCL's default
# (10 min) → they SIGABRT on timeout. A 4B model on L4 encodes the 47K catalog
# at ~1.15 s/it (~56 min each), so val+devset alone can exceed 2h. Raise it
# generously so long rank0 work never trips the barrier (4B/L4 crash fix).
_DDP_TIMEOUT = timedelta(hours=6)


class GroupedDataset(Dataset):
    """Each item = one unique query and its list of GT doc_texts.

    Rows are loaded from JSONL and grouped by `query` text. `session_id` of
    each contributing row is collected for session-aware splitting. `track_id`
    is also kept (parallel to docs) so the trainer can apply per-doc logQ
    correction without re-reading the source JSONL.

    Hard-negative metadata (specificity, first-positive artist/album/top_tags/
    pop_bucket, bm25_top_tids) is also captured when present in the JSONL —
    `collate_factory` uses it for `--smart-negative` injection. When the
    fields are missing, defaults are empty strings / lists so legacy JSONLs
    continue to work with `--smart-negative off`.
    """

    def __init__(self, path: Path) -> None:
        self.queries: list[str] = []
        self.docs: list[list[str]] = []
        self.track_ids: list[list[str]] = []  # parallel to docs; "" if missing
        self.sessions: list[set[str]] = []
        self.seen: list[set[str]] = []  # session-seen tracks per query (for serving-aligned val seen-filter)
        # Hard-neg metadata (per query, taken from the first contributing row).
        self.specificity: list[str] = []
        self.first_artist: list[str] = []
        self.first_album: list[str] = []
        self.first_top_tags: list[list[str]] = []
        self.first_pop_bucket: list[str] = []
        self.bm25_top_tids: list[list[str]] = []
        # Group by query text.
        agg: dict[str, dict] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = (obj.get("query") or "").strip()
                d = (obj.get("doc_text") or "").strip()
                tid = obj.get("track_id") or ""
                if not (q and d):
                    continue
                slot = agg.setdefault(
                    q,
                    {
                        "docs": [],
                        "track_ids": [],
                        "sessions": set(),
                        "seen": set(obj.get("seen") or []),
                        "specificity": (obj.get("specificity") or "unknown").strip().upper() or "unknown",
                        "first_artist": obj.get("track_artist") or "",
                        "first_album": obj.get("track_album") or "",
                        "first_top_tags": list(obj.get("track_top_tags") or []),
                        "first_pop_bucket": obj.get("track_pop_bucket") or "",
                        "bm25_top_tids": list(obj.get("bm25_top_tids") or []),
                    },
                )
                slot["docs"].append(d)
                slot["track_ids"].append(tid)
                if obj.get("session_id"):
                    slot["sessions"].add(obj["session_id"])
        for q, slot in agg.items():
            self.queries.append(q)
            self.docs.append(slot["docs"])
            self.track_ids.append(slot["track_ids"])
            self.sessions.append(slot["sessions"])
            self.seen.append(slot["seen"])
            self.specificity.append(slot["specificity"])
            self.first_artist.append(slot["first_artist"])
            self.first_album.append(slot["first_album"])
            self.first_top_tags.append(slot["first_top_tags"])
            self.first_pop_bucket.append(slot["first_pop_bucket"])
            self.bm25_top_tids.append(slot["bm25_top_tids"])
        if not self.queries:
            raise RuntimeError(f"no rows in {path}")
        n_total_pairs = sum(len(d) for d in self.docs)
        n_multi = sum(1 for d in self.docs if len(d) > 1)
        n_with_spec = sum(1 for s in self.specificity if s != "unknown")
        n_with_bm25 = sum(1 for b in self.bm25_top_tids if b)
        logger.info(
            f"[lora-train] grouped {n_total_pairs} pairs into {len(self.queries)} "
            f"unique queries ({n_multi} multi-positive, {n_with_spec} with specificity, "
            f"{n_with_bm25} with bm25 hits)"
        )

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "query": format_query(self.queries[idx], QWEN3_QUERY_INSTRUCTION),
            "docs": [format_doc(d) for d in self.docs[idx]],
            "track_ids": list(self.track_ids[idx]),
            "specificity": self.specificity[idx],
            "first_artist": self.first_artist[idx],
            "first_album": self.first_album[idx],
            "first_top_tags": list(self.first_top_tags[idx]),
            "first_pop_bucket": self.first_pop_bucket[idx],
            "bm25_top_tids": list(self.bm25_top_tids[idx]),
        }


def compute_doc_log_freq(dataset: GroupedDataset) -> tuple[dict[str, float], float]:
    """Build `track_id -> log(freq_smoothed)` for logQ correction.

    Frequency = number of (query, GT) pairs in which the track appears as
    positive. Laplace-smoothed: `log((c + 1) / (n_total + n_unique))`.
    Returns (log_p_dict, default_log_p) where `default_log_p` is used for
    track_ids unseen in the train pool (treated as singleton).
    """
    import math
    from collections import Counter

    counter: Counter[str] = Counter()
    n_total = 0
    for tid_list in dataset.track_ids:
        for tid in tid_list:
            if tid:
                counter[tid] += 1
                n_total += 1
    n_unique = len(counter)
    denom = n_total + n_unique
    log_p = {tid: math.log((c + 1) / denom) for tid, c in counter.items()}
    default_log_p = math.log(1 / denom) if denom > 0 else 0.0
    logger.info(f"[lora-train] doc_log_freq built: {n_unique} unique tracks, {n_total} pair occurrences")
    return log_p, default_log_p


def collate_factory(
    tokenizer,
    max_len: int,
    hard_neg_ctx: dict | None = None,
    rng: random.Random | None = None,
):
    """Tokenize B queries and the flat doc bank.

    Returns dict with:
        q_enc        : tokenizer output for B queries
        d_enc        : tokenizer output for sum(K_i) + n_hard_neg docs
        pos_offsets  : list[(start, end)] — for query i, positives sit at
                       d[start:end]. Length B. Hard-negs live AFTER the
                       positive slab and are NOT in any pos_offsets range,
                       so they automatically become negatives in the loss.
        track_ids    : flat list of track_ids parallel to d (for logQ).
        n_hard_neg   : total hard-negs added this batch (0 if disabled).

    When `hard_neg_ctx` is set (`--smart-negative != off`), each batch item's
    metadata drives `sample_hard_negs` and the resulting doc texts are
    appended to the doc bank.
    """
    rng = rng or random.Random()

    def _collate(batch: list[dict]) -> dict:
        qs = [item["query"] for item in batch]
        flat_docs: list[str] = []
        flat_tids: list[str] = []
        pos_offsets: list[tuple[int, int]] = []
        for item in batch:
            doc_list = item["docs"]
            tid_list = item["track_ids"] if item.get("track_ids") else [""] * len(doc_list)
            start = len(flat_docs)
            flat_docs.extend(doc_list)
            flat_tids.extend(tid_list)
            pos_offsets.append((start, len(flat_docs)))

        n_hard_neg = 0
        if hard_neg_ctx is not None:
            pools: HardNegPools = hard_neg_ctx["pools"]
            doc_text_map: dict[str, str] = hard_neg_ctx["doc_text_map"]
            distribution_table = hard_neg_ctx["distribution_table"]
            k = hard_neg_ctx["k"]

            # Collect every positive track_id in the batch — hard-negs must
            # not collide with any other query's positives.
            batch_pos_tids: set[str] = set()
            for item in batch:
                for t in item["track_ids"]:
                    if t:
                        batch_pos_tids.add(t)

            for item in batch:
                pos_tid = (item["track_ids"][0] if item["track_ids"] else "") or ""
                hn_tids = sample_hard_negs(
                    item["specificity"],
                    pos_track_id=pos_tid,
                    pos_artist=item["first_artist"],
                    pos_album=item["first_album"],
                    pos_top_tags=item["first_top_tags"],
                    pos_pop_bucket=item["first_pop_bucket"],
                    pos_bm25_top_tids=item["bm25_top_tids"],
                    pools=pools,
                    k=k,
                    rng=rng,
                    exclude=batch_pos_tids,
                    distribution_table=distribution_table,
                )
                for tid in hn_tids:
                    doc_text = doc_text_map.get(tid)
                    if not doc_text:
                        continue
                    flat_docs.append(format_doc(doc_text))
                    flat_tids.append(tid)
                    n_hard_neg += 1

        q_enc = tokenizer(qs, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        d_enc = tokenizer(flat_docs, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {
            "q": q_enc,
            "d": d_enc,
            "pos_offsets": pos_offsets,
            "track_ids": flat_tids,
            "n_hard_neg": n_hard_neg,
        }

    return _collate


def encode_with_grad(model, enc: dict, device: str) -> torch.Tensor:
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc).last_hidden_state
    vec = last_token_pool(out, enc["attention_mask"])
    return F.normalize(vec, dim=-1)


def _load_catalog(meta_dir: Path, compose: str) -> tuple[list[str], list[str]]:
    """Load full track catalog (tids, doc_texts) for retrieval validation.

    Uses the same `doc_text_map_{compose}.jsonl` that `extract_pairs.py` writes
    — guarantees the val ground-truth docs sit inside this candidate pool
    (a strict super-set when --compose matches the train data).
    """
    doc_map_path = meta_dir / f"doc_text_map_{compose}.jsonl"
    if not doc_map_path.exists():
        raise SystemExit(
            f"Full-catalog retrieval validation requires {doc_map_path}. "
            f"Run extract_pairs.py with --compose {compose} to generate it, or pass "
            f"--val-retrieval-disable to fall back to val_loss as the best-ckpt criterion."
        )
    tids: list[str] = []
    docs: list[str] = []
    with doc_map_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("tid")
            d = obj.get("doc_text")
            if t and d:
                tids.append(t)
                docs.append(d)
    return tids, docs


def _assert_catalog_matches_training(
    catalog_tids: list[str],
    catalog_doc_texts: list[str],
    full: "GroupedDataset",
    compose: str,
    sample_size: int = 300,
) -> None:
    """Guard against eval-catalog drift (train/eval doc-composition skew).

    The model trains on the positive `doc_text` stored in `train_full.jsonl`, but
    retrieval eval (val + devset) encodes the catalog from `doc_text_map_{compose}.jsonl`.
    Both are supposed to be the *same* `compose_doc` output per track. If a different
    experiment overwrote `doc_text_map` with another composition, training still
    succeeds yet eval silently scores against mismatched docs → misleading nDCG that
    looks like a "reproduction failure" (observed 2026-06: a "relaxing"-flavoured
    doc_text_map dropped devset from ~0.19 to ~0.15 with identical weights).

    Cross-check a sample of training `(tid -> doc_text)` against the catalog and fail
    loudly on real divergence. Override intentionally with `--allow-catalog-mismatch`.
    """
    cat = dict(zip(catalog_tids, catalog_doc_texts))
    checked = mismatch = 0
    examples: list[tuple[str, str, str]] = []
    for tids, docs in zip(full.track_ids, full.docs):
        for tid, doc in zip(tids, docs):
            if not tid:
                continue
            cdoc = cat.get(tid)
            if cdoc is None:
                continue
            checked += 1
            if cdoc.strip() != doc.strip():
                mismatch += 1
                if len(examples) < 3:
                    examples.append((tid, doc.strip()[:120], cdoc.strip()[:120]))
            if checked >= sample_size:
                break
        if checked >= sample_size:
            break
    if checked == 0:
        logger.warning(
            "[catalog-check] no tid overlap between training data and catalog — cannot verify doc composition match."
        )
        return
    frac = mismatch / checked
    if frac > 0.02:  # >2% = genuinely different composition, not stripping/encoding noise
        lines = [
            f"[catalog-check] doc_text_map_{compose}.jsonl does NOT match the training doc "
            f"composition: {mismatch}/{checked} ({frac:.0%}) sampled tids have different doc_text.",
        ]
        for tid, tdoc, cdoc in examples:
            lines.append(f"  tid={tid}\n    train  : {tdoc!r}\n    catalog: {cdoc!r}")
        lines.append(
            "Eval would score against mismatched docs (train/eval skew → misleading nDCG). "
            f"Rebuild doc_text_map with the SAME compose as the training data "
            f"(extract_pairs.py --compose {compose}), or pass --allow-catalog-mismatch to override."
        )
        raise SystemExit("\n".join(lines))
    logger.info(
        f"[catalog-check] OK — training doc_text matches catalog "
        f"({checked} tids sampled, {mismatch} mismatch, {frac:.1%})."
    )


@torch.no_grad()
def _encode_texts(
    model,
    tok,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: str,
    desc: str,
) -> torch.Tensor:
    """Chunked encode → (N, dim) on `device`, fp32 for stable matmul/topk."""
    model.eval()
    vecs: list[torch.Tensor] = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=desc,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
    ):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        v = last_token_pool(out, enc["attention_mask"])
        v = F.normalize(v, dim=-1).to(torch.float32)
        vecs.append(v)
    return torch.cat(vecs, dim=0)


def _ndcg_at_k(hit_positions: list[int], k: int, n_relevant: int) -> float:
    """nDCG@k with binary relevance; supports multi-positive via IDCG = sum_{i<min(R,k)} 1/log2(i+2)."""
    dcg = 0.0
    for pos in hit_positions:
        if pos < k:
            dcg += 1.0 / math.log2(pos + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate_retrieval(
    model,
    tok,
    val_idx: list[int],
    full: GroupedDataset,
    catalog_tids: list[str],
    catalog_doc_texts: list[str],
    max_len: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Full-catalog cosine retrieval on val queries.

    Encodes all catalog docs + val queries with the current LoRA state, runs a
    single GPU matmul (Q × D, both L2-normalized → cosine), and computes
    nDCG@20 / R@20 / R@100 / MRR@20 against each query's ground-truth
    track_ids. This is the closest in-loop proxy to the production retrieval
    metric — same candidate pool size (47K) the inference pipeline sees.
    """
    catalog_size = len(catalog_tids)
    doc_inputs = [format_doc(d) for d in catalog_doc_texts]
    doc_vecs = _encode_texts(model, tok, doc_inputs, max_len, batch_size, device, desc="val-ret enc catalog")

    query_inputs = [format_query(full.queries[i], QWEN3_QUERY_INSTRUCTION) for i in val_idx]
    q_vecs = _encode_texts(model, tok, query_inputs, max_len, batch_size, device, desc="val-ret enc queries")

    gt_per_query: list[set[str]] = [{t for t in full.track_ids[i] if t} for i in val_idx]
    seen_per_query: list[set[str]] = [full.seen[i] for i in val_idx]

    from mymodule.utils.seen import take_unseen

    top_k = 100
    # Over-fetch by the largest seen-set so post-filter top_k stays full (mirrors
    # serving pool + devset eval). Chunk over queries to bound peak memory.
    max_seen = max((len(s) for s in seen_per_query), default=0)
    fetch_k = min(top_k + max_seen, catalog_size)
    q_chunk_size = max(1, min(512, q_vecs.shape[0]))
    top_idx_chunks: list[torch.Tensor] = []
    for s in range(0, q_vecs.shape[0], q_chunk_size):
        sims = q_vecs[s : s + q_chunk_size] @ doc_vecs.T  # (q, D)
        top_idx_chunks.append(sims.topk(k=fetch_k, dim=-1).indices)
    top_idx = torch.cat(top_idx_chunks, dim=0).cpu().tolist()  # (Q, fetch_k)

    ndcg20 = 0.0
    r20 = 0.0
    r100 = 0.0
    mrr20 = 0.0
    n_eval = 0
    for qi, gt in enumerate(gt_per_query):
        if not gt:
            continue
        n_eval += 1
        n_rel = len(gt)
        # Drop session-seen tracks via the SAME take_unseen helper as serving.
        ranked = take_unseen([catalog_tids[int(j)] for j in top_idx[qi]], seen_per_query[qi], top_k)
        hit_positions = [rank for rank, tid in enumerate(ranked) if tid in gt]
        ndcg20 += _ndcg_at_k(hit_positions, k=20, n_relevant=n_rel)
        r20 += sum(1 for p in hit_positions if p < 20) / n_rel
        r100 += sum(1 for p in hit_positions if p < 100) / n_rel
        if hit_positions and hit_positions[0] < 20:
            mrr20 += 1.0 / (hit_positions[0] + 1)

    denom = max(1, n_eval)
    return {
        "ndcg@20": ndcg20 / denom,
        "recall@20": r20 / denom,
        "recall@100": r100 / denom,
        "mrr@20": mrr20 / denom,
        "n_eval_queries": float(n_eval),
        "catalog_size": float(catalog_size),
    }


def info_nce_multipos(
    q: torch.Tensor,
    d_flat: torch.Tensor,
    pos_offsets: list[tuple[int, int]],
    temperature: float,
    log_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Multi-positive InfoNCE, with optional sampling-bias logQ correction.

    For each query i, positives are docs in `d_flat[pos_offsets[i][0]:pos_offsets[i][1]]`;
    negatives are all other docs. Loss_i = -[logsumexp(pos) - logsumexp(all)].

    `log_p`, when provided, is the per-doc empirical log-probability vector
    of being sampled as in-batch negative (estimated from training pair
    frequency). It is subtracted from each column's logit BEFORE softmax —
    the standard logQ correction (Yi et al 2019) for biased in-batch
    negative sampling. Both positive and negative logits get the same
    column-wise offset, so the relative scores between positives and
    negatives within a single query are preserved up to the correction.
    """
    b = q.shape[0]
    n_d = d_flat.shape[0]
    device = q.device
    raw_logits = q @ d_flat.T / temperature  # (B, N)
    # logQ correction: subtract log-prob from each column (broadcast over rows).
    if log_p is not None:
        logits = raw_logits - log_p[None, :]
    else:
        logits = raw_logits
    # build pos mask
    pos_mask = torch.zeros(b, n_d, dtype=torch.bool, device=device)
    for i, (s, e) in enumerate(pos_offsets):
        pos_mask[i, s:e] = True
    neg_inf = torch.tensor(float("-inf"), device=device)
    # For each query: -[logsumexp(pos) - logsumexp(all)]
    pos_logits = logits.masked_fill(~pos_mask, neg_inf)
    pos_lse = torch.logsumexp(pos_logits, dim=-1)
    all_lse = torch.logsumexp(logits, dim=-1)
    loss = (all_lse - pos_lse).mean()

    with torch.no_grad():
        # acc, pos_cos, max_neg_cos use the **uncorrected** scores so they
        # remain comparable across logq-on / logq-off runs.
        top1 = logits.argmax(dim=-1)
        in_pos = pos_mask[torch.arange(b, device=device), top1]
        acc = in_pos.float().mean().item()
        sims = raw_logits * temperature
        pos_cos_per_q = (sims.masked_fill(~pos_mask, 0.0).sum(-1) / pos_mask.sum(-1).clamp(min=1)).mean().item()
        neg_logits = sims.masked_fill(pos_mask, -1.0)
        max_neg = neg_logits.max(dim=-1).values.mean().item()
    return loss, {"acc": acc, "pos_cos": pos_cos_per_q, "max_neg_cos": max_neg}


def _maybe_init_wandb(args: argparse.Namespace, out_dir: Path) -> Any | None:
    """Initialize wandb if ``WANDB_API_KEY`` is set; otherwise return ``None``.

    The training loop is wandb-aware but never depends on it — we never fail
    the run because of wandb. Behavior:

    - ``WANDB_API_KEY`` unset → return ``None``, all wandb calls become no-ops.
    - ``WANDB_API_KEY`` set but ``wandb`` not importable / init fails → log a
      warning and return ``None``.
    - Otherwise → return the imported ``wandb`` module (the live run is
      accessible via ``wandb.run`` and you call ``wandb.log({...})`` /
      ``wandb.finish()`` directly).

    Environment knobs (all optional):
      ``WANDB_PROJECT``   (default ``recsys-challenge-2026``)
      ``WANDB_ENTITY``    (default unset → personal account)
      ``WANDB_MODE``      (e.g., ``offline`` / ``disabled``) — passed through
    """
    import os

    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("[wandb] WANDB_API_KEY set but `wandb` not importable — skipping wandb logging")
        return None
    project = os.getenv("WANDB_PROJECT", "recsys-challenge-2026")
    entity = os.getenv("WANDB_ENTITY")
    # Path values aren't JSON-serializable by every wandb backend; stringify them.
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    try:
        wandb.init(
            project=project,
            entity=entity,
            name=out_dir.name,
            config=cfg,
            dir=str(out_dir),
            reinit=True,
        )
    except Exception as e:
        logger.warning(f"[wandb] init failed ({type(e).__name__}: {e}) — skipping wandb logging")
        return None
    logger.info(f"[wandb] init project={project} entity={entity or '(default)'} run={args.run_name}")
    return wandb


@torch.no_grad()
def evaluate_retrieval_texts(
    model,
    tok,
    query_texts: list[str],
    gt_sets: list[set[str]],
    catalog_tids: list[str],
    catalog_doc_texts: list[str],
    max_len: int,
    batch_size: int,
    device: str,
    seen_sets: list[set[str]] | None = None,
) -> dict[str, float]:
    """Full-catalog retrieval eval given arbitrary pre-formatted query texts.

    Mirrors evaluate_retrieval but accepts query texts + gt sets directly,
    so we can evaluate on the actual devset (not just the train val-split).

    ``seen_sets`` (one per query) drops session-seen tracks via the SAME
    ``take_unseen`` helper the serving pool uses (``pool._retrieve_with_scores``),
    so the per-epoch devset metric matches production retrieval exactly.
    """
    model.eval()
    catalog_size = len(catalog_tids)
    doc_inputs = [format_doc(d) for d in catalog_doc_texts]
    doc_vecs = _encode_texts(model, tok, doc_inputs, max_len, batch_size, device, desc="devset-ret enc catalog")
    q_vecs = _encode_texts(model, tok, query_texts, max_len, batch_size, device, desc="devset-ret enc queries")

    from mymodule.utils.seen import take_unseen

    top_k = 100
    # Over-fetch by the largest seen-set so that, after dropping session-seen
    # tracks, every query still has a full top_k unseen ranking (mirrors the
    # pool: `topk(n_candidates + len(seen))` then `take_unseen`).
    max_seen = max((len(s) for s in seen_sets), default=0) if seen_sets else 0
    fetch_k = min(top_k + max_seen, catalog_size)
    q_chunk_size = max(1, min(512, q_vecs.shape[0]))
    top_idx_chunks: list[torch.Tensor] = []
    for s in range(0, q_vecs.shape[0], q_chunk_size):
        sims = q_vecs[s : s + q_chunk_size] @ doc_vecs.T
        top_idx_chunks.append(sims.topk(k=fetch_k, dim=-1).indices)
    top_idx = torch.cat(top_idx_chunks, dim=0).cpu().tolist()

    ndcg20 = r20 = r100 = mrr20 = 0.0
    n_eval = 0
    for qi, gt in enumerate(gt_sets):
        if not gt:
            continue
        n_eval += 1
        n_rel = len(gt)
        ranked = [catalog_tids[int(j)] for j in top_idx[qi]]
        ranked = take_unseen(ranked, seen_sets[qi], top_k) if seen_sets is not None else ranked[:top_k]
        hit_positions = [rank for rank, tid in enumerate(ranked) if tid in gt]
        ndcg20 += _ndcg_at_k(hit_positions, k=20, n_relevant=n_rel)
        r20 += sum(1 for p in hit_positions if p < 20) / n_rel
        r100 += sum(1 for p in hit_positions if p < 100) / n_rel
        if hit_positions and hit_positions[0] < 20:
            mrr20 += 1.0 / (hit_positions[0] + 1)

    denom = max(1, n_eval)
    return {
        "ndcg@20": ndcg20 / denom,
        "recall@20": r20 / denom,
        "recall@100": r100 / denom,
        "mrr@20": mrr20 / denom,
        "n_eval_queries": float(n_eval),
    }


def _load_devset_queries(compose: str, max_len: int, tok) -> list[dict]:
    """Load HF devset (test split) and compose query texts for retrieval eval."""
    from datasets import load_dataset

    from mymodule.strategies.twotower.pool import build_query_body
    from mymodule.strategies.twotower.text_compose import build_tag_freq
    from mymodule.utils.seen import extract_session_seen_tracks

    try:
        from mymodule.strategies.twotower.text_compose import load_tag_freq

        _load_tag_freq = load_tag_freq
    except ImportError:
        _load_tag_freq = None

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")

    kv = None
    freq_map: dict = {}
    if compose == "qd":
        try:
            from mymodule.feature.kvdb import KVStore

            kv = KVStore.open(read_only=True)
        except Exception as e:
            logger.warning(f"[devset-eval] KV unavailable ({type(e).__name__}); crawl fields empty")
        try:
            freq_map = _load_tag_freq() if _load_tag_freq else {}
        except Exception:
            track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
            freq_map = build_tag_freq(track_ds)

    summary_map: dict = {}
    try:
        from mymodule.strategies.twotower.query_summary import DEFAULT_OUT as SUMMARY_OUT
        from mymodule.strategies.twotower.query_summary import load_query_summaries

        if SUMMARY_OUT.exists():
            summary_map = load_query_summaries(SUMMARY_OUT)
            logger.info(f"[devset-eval] loaded {len(summary_map)} query summaries")
    except Exception as e:
        logger.warning(f"[devset-eval] query summaries unavailable: {e}")

    # Must match extract_pairs (training queries) and pool.py (serving): the
    # prefix `Instruct: …\nQuery: ` is ~44 tokens, so the body budget is
    # resolve_max_body_tokens (= max_len - prefix - 1). The old `max_len - 20`
    # left body+prefix > max_len → the encoder left-truncated into the
    # Instruction, skewing devset-eval below the true (serving) score.
    max_body_tokens = resolve_max_body_tokens(tok, max_len)
    results: list[dict] = []
    for item in ds:
        goal = item.get("conversation_goal")
        conv = item["conversations"]
        by_turn: dict[int, list] = {}
        for row in conv:
            by_turn.setdefault(int(row["turn_number"]), []).append(row)
        for tn in sorted(by_turn.keys()):
            cur = by_turn[tn]
            user_msg = next((m for m in cur if m.get("role") == "user"), None)
            if user_msg is None:
                continue
            gt = {m["content"] for m in cur if m.get("role") == "music" and m.get("content")}
            if not gt:
                continue
            chat_history = [m for m in conv if int(m.get("turn_number", 0)) < tn]
            if compose == "qd":
                # Shared with serving (pool.TwoTowerPool.compose_query_body) so
                # train-mid devset eval and production render byte-identical queries.
                q_body = build_query_body(
                    user_msg["content"],
                    chat_history,
                    goal,
                    kv=kv,
                    tag_freq=freq_map,
                    tokenizer=tok,
                    max_body_tokens=max_body_tokens,
                    summary_map=summary_map,
                )
            else:
                from mymodule.feature.ollama_embed import _compose_talkpl_metadata_rich_query

                q_body = _compose_talkpl_metadata_rich_query(user_msg["content"], chat_history, kv)
            results.append(
                {
                    "query_text": format_query(q_body, QWEN3_QUERY_INSTRUCTION),
                    "gt": gt,
                    "turn": tn,
                    "seen": extract_session_seen_tracks(chat_history),
                }
            )
    logger.info(f"[devset-eval] loaded {len(results)} (session, turn) pairs from HF devset")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("mymodule/strategies/twotower/data/train_full.jsonl"))
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-root", type=Path, default=Path("mymodule/strategies/twotower/ckpt"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16, help="# unique queries per batch.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--val-ratio", type=float, default=0.05, help="Session-level val fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--qlora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="QLoRA: load the base model in 4-bit NF4 (double-quant, bf16 compute) and inject "
        "LoRA on top. Cuts base-weight VRAM ~4x so larger backbones (e.g. Qwen3-Embedding-4B/8B) "
        "fit a single 24GB GPU with a much larger batch. Implies bf16 compute; requires "
        "bitsandbytes. Single-GPU only (combine with --no-ddp).",
    )
    p.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade compute for memory.",
    )
    p.add_argument(
        "--logq-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract log P(d) from each column's logit before softmax (Yi et al 2019). "
        "Mitigates in-batch negative sampling bias when popular tracks are over-represented. "
        "Default ON — REPORT.md v3 confirmed turn-1 R@20 +0.05 lift; effect grows with "
        "hard-neg injection where bm25/metadata strategies skew toward popular tracks. "
        "Use --no-logq-correction to disable.",
    )
    p.add_argument(
        "--smart-negative",
        choices=["off", "specificity-aware", "bm25"],
        default="off",
        help="Hard-negative injection. 'off' = in-batch random only (default). "
        "'specificity-aware' = weighted distribution per conversation_goal.specificity "
        "(HH=album/artist/bm25, HL=tag/bm25/pop, LH=artist/bm25/album, LL=off). "
        "'bm25' = bm25-topk only for non-LL.",
    )
    p.add_argument(
        "--hard-neg-k",
        type=int,
        default=2,
        help="Hard-negatives injected per query (when --smart-negative != off).",
    )
    p.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("mymodule/strategies/twotower/data"),
        help="Directory holding doc_text_map_{compose}.jsonl and track_meta_map_{compose}.json.",
    )
    p.add_argument(
        "--compose",
        choices=["prod", "qd"],
        default="qd",
        help="Compose mode tag used to locate doc_text_map / track_meta_map side files.",
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Resume from epoch N: loads _epoch{N-1} checkpoint and continues from epoch N.",
    )
    # ---- DDP / multi-GPU ----
    p.add_argument("--no-ddp", action="store_true", help="Disable Accelerate DDP even with multiple GPUs.")
    # ---- devset eval ----
    p.add_argument(
        "--devset-eval-every",
        type=int,
        default=0,
        help="Run actual devset nDCG@20 (HF test split) every N epochs (0 = disabled). "
        "Heavy: re-encodes full catalog + all devset queries each time.",
    )
    # ---- full-catalog retrieval validation (nDCG@20 / R@20 / R@100 / MRR@20) ----
    p.add_argument(
        "--val-retrieval-every",
        type=int,
        default=1,
        help="Run full-catalog retrieval validation every N epochs (default 1 = every epoch). "
        "Encodes the full catalog (~47K tracks from doc_text_map_{compose}.jsonl) + val "
        "queries, runs Q×D cosine, reports val_ndcg@20 / val_recall@{20,100} / val_mrr@20. "
        "Bump to 2/3 if encoding cost dominates wall time.",
    )
    p.add_argument(
        "--val-retrieval-batch",
        type=int,
        default=64,
        help="Batch size for catalog/query encoding inside retrieval validation.",
    )
    p.add_argument(
        "--val-retrieval-disable",
        action="store_true",
        help="Disable full-catalog retrieval validation. Best ckpt then falls back to "
        "val_loss (lowest) instead of val_ndcg@20 (highest).",
    )
    p.add_argument(
        "--allow-catalog-mismatch",
        action="store_true",
        help="Skip the train/eval doc-composition consistency check. By default training "
        "aborts if doc_text_map_{compose}.jsonl does not match the doc_text the model "
        "trains on (prevents silent train/eval skew from a drifted catalog).",
    )
    # --- Cross-validation (optional) ---
    # Exposes --num-folds / --fold-idx / --split-seed. When --fold-idx is set the
    # session-aware split becomes a K-fold slice (fold i held out) and the run_name
    # gets a `_fold{i}` suffix so each fold's adapter lands in its own ckpt dir.
    CVConfig.add_cli_args(p)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Accelerate DDP (multi-GPU) ---
    use_ddp = not args.no_ddp and _HAS_ACCELERATE and torch.cuda.device_count() > 1
    if args.qlora and use_ddp:
        raise SystemExit(
            "--qlora is single-GPU only (bitsandbytes 4-bit + DDP is unsupported here). "
            "Pass --no-ddp, or set CUDA_VISIBLE_DEVICES to a single GPU."
        )
    if args.qlora and not args.bf16:
        logger.warning("[lora-train] --qlora implies bf16 compute; forcing --bf16 on.")
        args.bf16 = True
    if use_ddp:
        # Long collective timeout so rank0 per-epoch val + ckpt save
        # never trips the end-of-epoch barrier (see _DDP_TIMEOUT).
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=_DDP_TIMEOUT)])
        device = str(accelerator.device)
        is_main = accelerator.is_main_process
        logger.info(
            f"[ddp] Accelerate DDP: {accelerator.num_processes} processes, device={device}, pg_timeout={_DDP_TIMEOUT}"
        )
    else:
        accelerator = None
        device = assert_torch_cuda_or_warn()
        is_main = True
    # CV-aware run name: `_fold{i}` suffix when --fold-idx is set, empty otherwise.
    cv_cfg = CVConfig.from_cli(args)
    run_name = f"{args.run_name}{cv_cfg.run_suffix()}"
    out_dir = args.out_root / run_name
    ensure_dirs(out_dir)
    log_path = out_dir / "train.log"
    logger.add(str(log_path), level="INFO")

    logger.info(f"[lora-train] args = {vars(args)}")

    # ---- wandb (optional, no-op when WANDB_API_KEY is unset) ----
    wb = _maybe_init_wandb(args, out_dir)

    # ---- data ----
    full = GroupedDataset(args.data)
    # Group-aware, CV-aware split via the shared mymodule.cv infra (no model-local
    # re-implementation; see .claude/rules/cv.md). Group key per grouped-query item =
    # a representative session id (sorted-first for determinism); queries with no
    # session_id get a unique synthetic key so they never collapse into one fold.
    # split_items branches on cv_cfg.is_active: K-fold slice (fold_idx held out) when
    # active, group-aware random val_ratio split otherwise.
    group_keys = [(min(s) if s else f"__nosession_{i}") for i, s in enumerate(full.sessions)]
    train_idx, val_idx = split_items(
        group_keys,
        group_fn=lambda g: g,
        config=cv_cfg,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )
    if cv_cfg.is_active:
        logger.info(
            f"[lora-train] CV fold {cv_cfg.fold_idx}/{cv_cfg.num_folds} (split_seed={cv_cfg.split_seed}): "
            f"train={len(train_idx)} queries, val={len(val_idx)} queries → {out_dir.name}"
        )
    else:
        logger.info(f"[lora-train] session-aware split: train={len(train_idx)} queries, val={len(val_idx)} queries")

    # ---- catalog for full-retrieval validation + devset eval (static, encoded
    # each epoch) ----
    # Both val-retrieval AND devset eval encode this catalog, so load it whenever
    # EITHER is active. (Previously gated on val-retrieval only, so
    # --val-retrieval-disable silently skipped devset eval too — its `do_devset`
    # guard requires a non-empty catalog.)
    catalog_tids: list[str] = []
    catalog_doc_texts: list[str] = []
    needs_catalog = (not args.val_retrieval_disable) or args.devset_eval_every > 0
    if needs_catalog:
        catalog_tids, catalog_doc_texts = _load_catalog(args.meta_dir, args.compose)
        logger.info(
            f"[catalog] loaded: {len(catalog_tids)} tracks "
            f"(val-retrieval={'off' if args.val_retrieval_disable else f'every {args.val_retrieval_every}ep'}, "
            f"devset-eval={'off' if args.devset_eval_every <= 0 else f'every {args.devset_eval_every}ep'})"
        )
        if not args.allow_catalog_mismatch:
            _assert_catalog_matches_training(catalog_tids, catalog_doc_texts, full, args.compose)

    class _Sub(Dataset):
        def __init__(self, indices, source: GroupedDataset):
            self.indices = indices
            self.source = source

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.source[self.indices[i]]

    # ---- model + LoRA ----
    logger.info(f"[lora-train] tokenizer + base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(
        args.base_model, padding_side="left", truncation_side="left", trust_remote_code=True
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    quant_cfg = None
    if args.qlora:
        # 4-bit NF4 + double quant; bf16 compute. Shrinks base-weight VRAM ~4x so a 4B/8B
        # backbone fits one 24GB GPU with a much larger batch. bnb requires the quantized
        # weights placed on a GPU at load (device_map) — `.to(device)` is invalid afterwards.
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info("[lora-train] QLoRA ON: 4-bit NF4 base (double-quant, bf16 compute)")
    base = AutoModel.from_pretrained(
        args.base_model,
        dtype=dtype,
        quantization_config=quant_cfg,
        # device_map pins the (quantized) base on the target GPU; with QLoRA we must NOT call
        # `.to(device)` afterwards. Without quantization we keep the explicit `.to(device)` below.
        device_map={"": device} if args.qlora else None,
        # SDPA picks the memory-efficient kernel automatically (flash if avail, else efficient).
        # Compared to "eager", attention activations stop at O(N) instead of O(N²) per layer
        # — necessary at max_len ≥ 192 with batch ≥ 16 on 16GB.
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    if not args.qlora:
        base = base.to(device)
    else:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        # Casts layernorms to fp32 and freezes the 4-bit base for stable adapter training.
        # Gradient checkpointing is wired separately below (args.grad_checkpoint), so disable
        # it here to keep that toggle the single source of truth.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=get_lora_target_modules(),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    if args.start_epoch > 1:
        from peft import PeftModel  # noqa: PLC0415

        resume_ckpt = out_dir.with_name(out_dir.name + f"_epoch{args.start_epoch - 1}")
        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f"[lora-train] --start-epoch {args.start_epoch} requested but checkpoint not found: {resume_ckpt}"
            )
        logger.info(f"[lora-train] resuming from checkpoint: {resume_ckpt}")
        model = PeftModel.from_pretrained(base, str(resume_ckpt), is_trainable=True)
    else:
        model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    if args.bf16:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

    # Gradient checkpointing: trades ~1.3-1.5× compute for ~3-4× activation
    # memory savings — necessary at max_len ≥ 256 with batch ≥ 12 on 16GB.
    # `enable_input_require_grads` is required for PEFT (frozen embeddings).
    if args.grad_checkpoint:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logger.info("[lora-train] gradient checkpointing ON")

    # ---- hard-neg context ----
    hard_neg_ctx: dict | None = None
    if args.smart_negative != "off":
        meta_path = args.meta_dir / f"track_meta_map_{args.compose}.json"
        doc_map_path = args.meta_dir / f"doc_text_map_{args.compose}.jsonl"
        if not meta_path.exists() or not doc_map_path.exists():
            raise SystemExit(
                f"--smart-negative={args.smart_negative} requires side files: "
                f"{meta_path} and {doc_map_path}. Run extract_pairs.py "
                f"with --compose {args.compose} to generate them."
            )
        hn_pools = HardNegPools.from_track_meta_map(meta_path)
        doc_text_map: dict[str, str] = {}
        with doc_map_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tid = obj.get("tid")
                txt = obj.get("doc_text")
                if tid and txt:
                    doc_text_map[tid] = txt
        hard_neg_ctx = {
            "pools": hn_pools,
            "doc_text_map": doc_text_map,
            "distribution_table": get_distribution(args.smart_negative),
            "k": args.hard_neg_k,
        }
        logger.info(f"[hard-neg] mode={args.smart_negative} k={args.hard_neg_k} docs_loaded={len(doc_text_map)}")

    # ---- optim ----
    train_loader = DataLoader(
        _Sub(train_idx, full),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_factory(tok, args.max_len, hard_neg_ctx=hard_neg_ctx, rng=random.Random(args.seed)),
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        _Sub(val_idx, full),
        batch_size=args.batch_size,
        shuffle=False,
        # Val collate also injects hard-negs so val/train signals stay
        # comparable. Use a deterministic rng so val numbers are stable.
        collate_fn=collate_factory(tok, args.max_len, hard_neg_ctx=hard_neg_ctx, rng=random.Random(args.seed + 1)),
        drop_last=False,
        num_workers=0,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ---- Accelerate DDP prepare ----
    if accelerator is not None:
        model, optim, train_loader, val_loader, sched = accelerator.prepare(
            model, optim, train_loader, val_loader, sched
        )
        # total_steps uses the prepared loader length (DDP-adjusted)
        total_steps = max(1, len(train_loader) * args.epochs)

    # ---- resume: fast-forward scheduler to the correct LR position ----
    if args.start_epoch > 1:
        steps_done = (args.start_epoch - 1) * len(train_loader)
        for _ in range(steps_done):
            sched.step()
        logger.info(f"[lora-train] scheduler fast-forwarded {steps_done} steps (epoch {args.start_epoch - 1} done)")

    # ---- logQ correction setup ----
    doc_log_p: dict[str, float] = {}
    default_log_p: float = 0.0
    if args.logq_correction:
        # Compute log-prob from the *full* dataset (train+val pool) so the
        # correction reflects the true sampling distribution. We don't restrict
        # to the train split because that would add tiny bias against val
        # leak — the val pool is held out at the query level, not doc level.
        doc_log_p, default_log_p = compute_doc_log_freq(full)
        logger.info("[lora-train] logQ correction ON")

    def _step(batch: dict) -> tuple[torch.Tensor, dict]:
        q_vec = encode_with_grad(model, batch["q"], device)
        d_vec = encode_with_grad(model, batch["d"], device)
        log_p_tensor: torch.Tensor | None = None
        if args.logq_correction:
            tids = batch.get("track_ids") or []
            log_p_tensor = torch.tensor(
                [doc_log_p.get(t, default_log_p) for t in tids],
                dtype=q_vec.dtype,
                device=device,
            )
        return info_nce_multipos(q_vec, d_vec, batch["pos_offsets"], args.temperature, log_p=log_p_tensor)

    history: list[dict] = []
    step = (args.start_epoch - 1) * len(train_loader)
    use_retrieval_best = not args.val_retrieval_disable
    devset_queries: list[dict] | None = None  # lazy-loaded on first devset eval epoch
    # primary_metric = "val_ndcg@20" (higher is better) when retrieval is on,
    # otherwise fall back to "val_loss" (lower is better).
    best_score = -math.inf if use_retrieval_best else math.inf
    logger.info(
        f"[lora-train] best-ckpt criterion = "
        f"{'val_ndcg@20 ↑ (full-catalog retrieval)' if use_retrieval_best else 'val_loss ↓ (in-batch surrogate)'}"
    )
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        # Progress bar: shows step/total, it/s → ETA, and live loss/acc/pos/neg/lr.
        # `dynamic_ncols=True` lets the bar reflow in narrow terminals; `mininterval=0.5`
        # caps redraw rate so postfix updates don't dominate step time.
        pbar = tqdm(
            train_loader,
            desc=f"train ep {epoch}/{args.epochs}",
            total=len(train_loader),
            dynamic_ncols=True,
            mininterval=0.5,
            leave=False,
        )
        for batch in pbar:
            loss, stats = _step(batch)
            optim.zero_grad(set_to_none=True)
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            params_to_clip = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            optim.step()
            sched.step()
            step += 1
            lr_now = sched.get_last_lr()[0]
            if is_main:
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{stats['acc']:.3f}",
                    "pos": f"{stats['pos_cos']:.3f}",
                    "neg": f"{stats['max_neg_cos']:.3f}",
                    "lr": f"{lr_now:.1e}",
                }
                if hard_neg_ctx is not None:
                    postfix["hn"] = batch.get("n_hard_neg", 0)
                pbar.set_postfix(postfix)
            if is_main and (step % args.log_every == 0 or step == 1):
                hn_part = f" hn={batch.get('n_hard_neg', 0)}" if hard_neg_ctx is not None else ""
                tqdm.write(
                    f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                    f"acc={stats['acc']:.3f} pos={stats['pos_cos']:.3f} "
                    f"neg={stats['max_neg_cos']:.3f}{hn_part} lr={lr_now:.2e}"
                )
                logger.info(
                    f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                    f"acc={stats['acc']:.3f} pos={stats['pos_cos']:.3f} "
                    f"neg={stats['max_neg_cos']:.3f}{hn_part} lr={lr_now:.2e}"
                )
                if wb is not None:
                    payload = {
                        "train/loss": loss.item(),
                        "train/acc": stats["acc"],
                        "train/pos_cos": stats["pos_cos"],
                        "train/max_neg_cos": stats["max_neg_cos"],
                        "train/lr": lr_now,
                        "epoch": epoch,
                    }
                    if hard_neg_ctx is not None:
                        payload["train/n_hard_neg"] = batch.get("n_hard_neg", 0)
                    wb.log(payload, step=step)
        pbar.close()

        model.eval()
        with torch.no_grad():
            agg = defaultdict(list)
            for batch in tqdm(
                val_loader,
                desc=f"val ep {epoch}/{args.epochs}",
                total=len(val_loader),
                dynamic_ncols=True,
                mininterval=0.5,
                leave=False,
                disable=not is_main,
            ):
                loss, stats = _step(batch)
                # gather loss across DDP processes for a global average
                if accelerator is not None:
                    loss_val = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
                else:
                    loss_val = loss.item()
                agg["loss"].append(loss_val)
                for k in ("acc", "pos_cos", "max_neg_cos"):
                    agg[k].append(stats[k])

        rec = {
            "epoch": epoch,
            "val_loss": sum(agg["loss"]) / max(1, len(agg["loss"])),
            "val_acc": sum(agg["acc"]) / max(1, len(agg["acc"])),
            "val_pos_cos": sum(agg["pos_cos"]) / max(1, len(agg["pos_cos"])),
            "val_max_neg_cos": sum(agg["max_neg_cos"]) / max(1, len(agg["max_neg_cos"])),
        }
        if is_main:
            logger.info(
                f"[val] epoch={epoch} loss={rec['val_loss']:.4f} acc={rec['val_acc']:.3f} "
                f"pos={rec['val_pos_cos']:.3f} neg={rec['val_max_neg_cos']:.3f}"
            )
            if wb is not None:
                wb.log(
                    {
                        "val/loss": rec["val_loss"],
                        "val/acc": rec["val_acc"],
                        "val/pos_cos": rec["val_pos_cos"],
                        "val/max_neg_cos": rec["val_max_neg_cos"],
                        "epoch": epoch,
                    },
                    step=step,
                )

        # ---- save epoch checkpoint FIRST (main process only) ----
        # Save the epoch weights *before* the long full-catalog
        # eval below. On multi-GPU the non-main ranks block at the end-of-epoch
        # barrier while rank0 runs eval; if that eval ever outlasts the NCCL
        # timeout, the process group is torn down. Saving first guarantees the
        # epoch weights (the resume anchor) survive a crash during eval.
        if is_main:
            epoch_dir = out_dir.with_name(out_dir.name + f"_epoch{epoch}")
            ensure_dirs(epoch_dir)
            _unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
            _unwrapped.save_pretrained(str(epoch_dir))
            tok.save_pretrained(str(epoch_dir))

        # ---- full-catalog retrieval validation (real metric, not surrogate) ----
        # Only runs on main process; non-main processes wait at the barrier below.
        do_retrieval = (
            is_main
            and not args.val_retrieval_disable
            and args.val_retrieval_every > 0
            and epoch % args.val_retrieval_every == 0
            and len(val_idx) > 0
        )
        if do_retrieval:
            _model = accelerator.unwrap_model(model) if accelerator is not None else model
            ret_metrics = evaluate_retrieval(
                _model,
                tok,
                val_idx=val_idx,
                full=full,
                catalog_tids=catalog_tids,
                catalog_doc_texts=catalog_doc_texts,
                max_len=args.max_len,
                batch_size=args.val_retrieval_batch,
                device=device,
            )
            rec.update({f"val_{k}": v for k, v in ret_metrics.items()})
            logger.info(
                f"[val-retrieval] epoch={epoch} ndcg@20={ret_metrics['ndcg@20']:.4f} "
                f"R@20={ret_metrics['recall@20']:.4f} R@100={ret_metrics['recall@100']:.4f} "
                f"MRR@20={ret_metrics['mrr@20']:.4f} "
                f"(n_eval={int(ret_metrics['n_eval_queries'])}, catalog={int(ret_metrics['catalog_size'])})"
            )
            if wb is not None:
                wb.log(
                    {
                        "val/ndcg@20": ret_metrics["ndcg@20"],
                        "val/recall@20": ret_metrics["recall@20"],
                        "val/recall@100": ret_metrics["recall@100"],
                        "val/mrr@20": ret_metrics["mrr@20"],
                        "epoch": epoch,
                    },
                    step=step,
                )

        # ---- actual devset eval (HF test split) ----
        do_devset = (
            is_main
            and args.devset_eval_every > 0
            and epoch % args.devset_eval_every == 0
            and catalog_tids  # catalog must be loaded
        )
        devset_metrics: dict = {}
        if do_devset:
            if devset_queries is None:
                devset_queries = _load_devset_queries(args.compose, args.max_len, tok)
            _model = accelerator.unwrap_model(model) if accelerator is not None else model
            devset_q_texts = [q["query_text"] for q in devset_queries]
            devset_gt_sets = [q["gt"] for q in devset_queries]
            devset_metrics = evaluate_retrieval_texts(
                _model,
                tok,
                query_texts=devset_q_texts,
                gt_sets=devset_gt_sets,
                catalog_tids=catalog_tids,
                catalog_doc_texts=catalog_doc_texts,
                max_len=args.max_len,
                batch_size=args.val_retrieval_batch,
                device=device,
                seen_sets=[q["seen"] for q in devset_queries],
            )
            rec.update({f"devset_{k}": v for k, v in devset_metrics.items()})
            logger.success(
                f"[devset-eval] epoch={epoch} nDCG@20={devset_metrics['ndcg@20']:.4f} "
                f"R@20={devset_metrics['recall@20']:.4f} R@100={devset_metrics['recall@100']:.4f} "
                f"MRR@20={devset_metrics['mrr@20']:.4f}"
            )

        if is_main:
            history.append(rec)

        # ---- best ckpt selection (epoch weights already saved above) ----
        if is_main:
            # ---- best ckpt: val_ndcg@20 ↑ (retrieval on) | val_loss ↓ (fallback) ----
            if use_retrieval_best:
                score = rec.get("val_ndcg@20")
                if score is not None and score > best_score:
                    best_score = score
                    (out_dir / "best.json").write_text(json.dumps(rec, indent=2))
                    _unwrapped.save_pretrained(str(out_dir))
                    tok.save_pretrained(str(out_dir))
                    logger.success(f"[lora-train] best (val_ndcg@20={score:.4f}) → {out_dir}")
            else:
                score = rec["val_loss"]
                if score < best_score:
                    best_score = score
                    (out_dir / "best.json").write_text(json.dumps(rec, indent=2))
                    _unwrapped.save_pretrained(str(out_dir))
                    tok.save_pretrained(str(out_dir))
                    logger.success(f"[lora-train] best (val_loss={score:.4f}) → {out_dir}")

        # DDP barrier — non-main processes wait while main saves/evals
        if accelerator is not None:
            accelerator.wait_for_everyone()

    if is_main:
        final_dir = out_dir.with_name(out_dir.name + "_final")
        ensure_dirs(final_dir)
        _unwrapped_final = accelerator.unwrap_model(model) if accelerator is not None else model
        _unwrapped_final.save_pretrained(str(final_dir))
        tok.save_pretrained(str(final_dir))
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.success(f"[lora-train] done. best→{out_dir}, final→{final_dir}, history→{out_dir / 'history.json'}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
