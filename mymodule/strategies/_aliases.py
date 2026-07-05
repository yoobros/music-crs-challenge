"""Pool alias & builder for the `ensemble` strategy.

EnsembleStrategy receives pool-name strings from the TID (e.g. "bm25",
"qemb_metadata_rich", "qemb_twotower_8b") and assembles the matching BasePool
instances. This module owns that mapping and alias expansion
(`qemb_twotower_8b_all` → fold names).

The underscore prefix excludes this module from strategy auto-discovery.

Fallback recursion policy: `build_fallback_pool()` always forces
`fallback="popularity"` on sub-pools, so the fallback chain recurses at most
one level and converges to popularity.
"""

from __future__ import annotations

import os
import re

import torch
from loguru import logger

from mymodule.strategies.pool import get_pool
from mymodule.strategies.pool.base import BasePool

_DEVICE_ENV_NAMES = ("MYMODULE_TORCH_DEVICE",)
_VALID_DEVICES = {"cpu", "cuda", "mps"}


def _get_device(*specific_env_names: str) -> str:
    """Resolve torch device: explicit env override, then cuda > mps > cpu."""
    for env_name in (*specific_env_names, *_DEVICE_ENV_NAMES):
        value = os.getenv(env_name)
        if not value:
            continue
        device = value.strip().lower()
        if device not in _VALID_DEVICES:
            raise ValueError(f"{env_name} must be one of {sorted(_VALID_DEVICES)}, got {value!r}")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"{env_name}=cuda was requested but CUDA is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(f"{env_name}=mps was requested but MPS is not available")
        logger.info(f"Using torch device override from {env_name}: {device}")
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# qemb_<variant> → text-embedding table (Ollama embedded query → ANN search)
QEMB_VARIANTS: dict[str, str] = {
    "qemb_metadata": "metadata-qwen3_embedding_0.6b",
    "qemb_lyrics": "lyrics-qwen3_embedding_0.6b",
    "qemb_attributes": "attributes-qwen3_embedding_0.6b",
    # Locally built rich variant: doc input includes year/popularity/tags
    # alongside the talkpl entity_str. See mymodule.feature.store --build-metadata-rich.
    "qemb_metadata_rich": "metadata_rich-qwen3_embedding_0.6b",
}

TWOTOWER_FOLDS_DEFAULT = 5

# 8B 5-fold bag (epoch2 checkpoints). The base model is Qwen3-Embedding-8B and
# must be injected explicitly (encoder default is 0.6B).
# - adapter dir : ckpt/qwen3emb8b_qlora_fold{i}_epoch2  (adapter has the _epoch2 suffix)
# - doc cache   : data/doc_cache_qwen3emb8b_qlora_fold{i}.npz  (no _epoch2 in cache names)
# qemb_twotower_8b / _all → qemb_twotower_8b_fold{0..4} 5-way RRF (expand_pool_names).
QEMB_TWOTOWER_8B_ADAPTER = "qwen3emb8b_qlora"
QEMB_TWOTOWER_8B_BASE = "Qwen/Qwen3-Embedding-8B"
_QEMB_TWOTOWER_8B_FOLD_PATTERN = re.compile(r"^qemb_twotower_8b_fold(\d+)$")

# twotower_<variant> → adapter dir + doc cache pair under mymodule/strategies/twotower/
# Default `twotower` reads ckpt/twotower/ + data/doc_cache.npz; variants read
# ckpt/twotower_<variant>/ + data/twotower_<variant>_doc_cache.npz.
_TWOTOWER_VARIANT_PATTERN = re.compile(r"^twotower_([a-z][a-z_0-9]*)$")


def expand_pool_names(names: list[str]) -> list[str]:
    """Expand aliases like `qemb_twotower_8b(_all)` into fold names; order-preserving, deduped.

    >>> expand_pool_names(["qemb_twotower_8b_all"])
    ['qemb_twotower_8b_fold0', ..., 'qemb_twotower_8b_fold4']
    """
    expanded: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in ("qemb_twotower_8b", "qemb_twotower_8b_all"):
            # 8B 5-fold bag (qwen3emb8b_qlora_fold{i}_epoch2).
            targets = [f"qemb_twotower_8b_fold{i}" for i in range(TWOTOWER_FOLDS_DEFAULT)]
        else:
            targets = [name]
        for t in targets:
            if t not in seen:
                seen.add(t)
                expanded.append(t)
    return expanded


def build_pool(name: str, *, n_candidates: int = 100, fallback: str = "bm25") -> BasePool:
    """Build a BasePool instance from a pool alias.

    - qemb_<variant>          → QueryEmbedPool(vector_type=...)
    - bm25 / bm25_*           → BM25Pool (production RRF or single-variant configs)
    - twotower                → TwoTowerPool (LoRA Qwen3-Embedding, default adapter)
    - twotower_<variant>      → TwoTowerPool with adapter `ckpt/twotower_<variant>/`
    - qemb_twotower_8b(_all)  → 8B 5-fold bag (RRF) of QLoRA adapters
    - qemb_twotower_8b_fold<N> → single 8B fold adapter + fold doc cache
    - bm25_qmr                → EnsemblePool(bm25, qemb_metadata_rich) exposed as a
                                single pool.
    - anything else           → get_pool(name, n_candidates=...) directly
    """
    if name in QEMB_VARIANTS:
        return get_pool("qemb", vector_type=QEMB_VARIANTS[name], n_candidates=n_candidates)
    # bm25 production default = nested RRF(bm25_simple + bm25_tags_crawl_kw).
    if name == "bm25":
        from mymodule.strategies.pool.ensemble import EnsemblePool

        _bm25_simple = get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags"],
            spacy_filter=True,
        )
        _bm25_tags_crawl = build_pool("bm25_tags_crawl_kw", n_candidates=n_candidates)
        return EnsemblePool(
            pools=[_bm25_simple, _bm25_tags_crawl],
            method="rrf",
            n_candidates=n_candidates,
            ignore_failures=True,
        )
    if name == "bm25_simple":
        # Legacy bm25 — music_only + noun_tags + spaCy.
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags"],
            spacy_filter=True,
        )
    if name == "bm25_music_only":
        # query_mode only (corpus = 4-field default, no spaCy).
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date"],
            spacy_filter=False,
        )
    if name == "bm25_history_noun_tags_spacy":
        # query_mode = history + doc-side noun_tags + query spaCy.
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="history",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags"],
            spacy_filter=True,
        )
    if name == "bm25_crawl":
        # BM25 with ALL crawled metadata fields (lyrics, mb_tags, label, country, etc.)
        from mymodule.strategies.pool.bm25 import CRAWL_ALL_FIELDS

        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags"] + CRAWL_ALL_FIELDS,
            spacy_filter=True,
        )
    if name == "bm25_crawl_kw":
        # BM25 with crawl keyword fields ONLY (no lyrics/caption — avoids token dilution)
        from mymodule.strategies.pool.bm25 import CRAWL_FIELDS

        _kw_fields = [k for k in CRAWL_FIELDS if k not in ("lyrics_crawl", "caption_crawl")]
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags"] + _kw_fields,
            spacy_filter=True,
        )
    if name == "bm25_crawl_lyrics":
        # BM25 with lyrics ONLY from crawl (isolating lyrics effect)
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=[
                "track_name",
                "artist_name",
                "album_name",
                "release_date",
                "noun_tags",
                "lyrics_crawl",
            ],
            spacy_filter=True,
        )
    if name == "bm25_tags_raw":
        # BM25 with raw tag_list from HF dataset (no cleaning)
        from mymodule.strategies.pool.bm25 import TAG_LIST_RAW_FIELD

        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags", TAG_LIST_RAW_FIELD],
            spacy_filter=True,
        )
    if name == "bm25_tags":
        # BM25 with cleaned tag_list (lowercase, dedup, noise filter)
        from mymodule.strategies.pool.bm25 import CRAWL_FIELDS, TAG_LIST_CLEAN_FIELD

        _kw_fields = [k for k in CRAWL_FIELDS if k not in ("lyrics_crawl", "caption_crawl")]
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags", TAG_LIST_CLEAN_FIELD],
            spacy_filter=True,
        )
    if name == "bm25_tags_crawl_kw":
        # BM25 with cleaned tag_list + crawl keyword fields
        from mymodule.strategies.pool.bm25 import CRAWL_FIELDS, TAG_LIST_CLEAN_FIELD

        _kw_fields = [k for k in CRAWL_FIELDS if k not in ("lyrics_crawl", "caption_crawl")]
        return get_pool(
            "bm25",
            n_candidates=n_candidates,
            query_mode="music_only",
            corpus_types=["track_name", "artist_name", "album_name", "release_date", "noun_tags", TAG_LIST_CLEAN_FIELD]
            + _kw_fields,
            spacy_filter=True,
        )
    if name == "twotower":
        # Lazy import keeps torch/peft out of the module-load path.
        return get_pool("twotower", n_candidates=n_candidates, device=_get_device("MYMODULE_TWOTOWER_DEVICE"))
    if name in ("qemb_twotower_8b", "qemb_twotower_8b_all"):
        # 8B 5-fold bag (RRF) — qwen3emb8b_qlora_fold{0..4}_epoch2. expand_pool_names
        # expands folds at TID parse time; direct build_pool calls get an EnsemblePool.
        from mymodule.strategies.pool.ensemble import EnsemblePool

        fold_pools = [
            build_pool(f"qemb_twotower_8b_fold{i}", n_candidates=n_candidates, fallback=fallback)
            for i in range(TWOTOWER_FOLDS_DEFAULT)
        ]
        return EnsemblePool(
            pools=fold_pools,
            method="rrf",
            n_candidates=n_candidates,
            ignore_failures=True,
        )
    m = _QEMB_TWOTOWER_8B_FOLD_PATTERN.match(name)
    if m:
        # qemb_twotower_8b_fold{i} → 8B fold adapter (_epoch2) + fold doc cache.
        # base_model is set to 8B (encoder default is 0.6B); no _epoch2 in cache names.
        from mymodule.strategies.twotower.pool import _MODULE_DIR

        fold_idx = int(m.group(1))
        adapter_path = _MODULE_DIR / "ckpt" / f"{QEMB_TWOTOWER_8B_ADAPTER}_fold{fold_idx}_epoch2"
        doc_cache_path = _MODULE_DIR / "data" / f"doc_cache_{QEMB_TWOTOWER_8B_ADAPTER}_fold{fold_idx}.npz"
        # query-cache-read serving: if per-fold query caches exist, the pool reads
        # query vecs by body-hash and NEVER loads the 8B encoder → a 5-fold bag
        # serves on one GPU without OOM. Absent (e.g. during OOF build) → online encode.
        import glob as _glob

        qc_glob = str(_MODULE_DIR / "data" / f"query_cache_{QEMB_TWOTOWER_8B_ADAPTER}_fold{fold_idx}_*.npz")
        query_cache_path = qc_glob if _glob.glob(qc_glob) else None
        return get_pool(
            "twotower",
            n_candidates=n_candidates,
            adapter_path=adapter_path,
            doc_cache_path=doc_cache_path,
            base_model=QEMB_TWOTOWER_8B_BASE,
            query_cache_path=query_cache_path,
            device=_get_device("MYMODULE_TWOTOWER_DEVICE"),
        )
    if name == "bm25_qmr":
        # bm25_qmr → 2-way RRF(bm25, qemb_metadata_rich) exposed as a single pool,
        # so downstream TIDs can compose against it 1:1 without flat-RRF dilution.
        from mymodule.strategies.pool.ensemble import EnsemblePool

        _bm25 = build_pool("bm25", n_candidates=n_candidates, fallback=fallback)
        _qmr = build_pool("qemb_metadata_rich", n_candidates=n_candidates, fallback=fallback)
        return EnsemblePool(
            pools=[_bm25, _qmr],
            method="rrf",
            n_candidates=n_candidates,
            ignore_failures=True,
        )
    m = _TWOTOWER_VARIANT_PATTERN.match(name)
    if m:
        from mymodule.strategies.twotower.pool import DEFAULT_ADAPTER, DEFAULT_DOC_CACHE

        variant = m.group(1)
        adapter_path = DEFAULT_ADAPTER.parent / f"twotower_{variant}"
        doc_cache_path = DEFAULT_DOC_CACHE.parent / f"twotower_{variant}_doc_cache.npz"
        return get_pool(
            "twotower",
            n_candidates=n_candidates,
            adapter_path=adapter_path,
            doc_cache_path=doc_cache_path,
            device=_get_device("MYMODULE_TWOTOWER_DEVICE"),
        )
    # Fallthrough: try building a registered pool without extra kwargs.
    return get_pool(name, n_candidates=n_candidates)


def available_pool_names() -> list[str]:
    """All pool names usable in a TID (for help/docs)."""
    return [
        *QEMB_VARIANTS.keys(),
        "qemb_twotower_8b",
        "qemb_twotower_8b_all",
        *[f"qemb_twotower_8b_fold{i}" for i in range(TWOTOWER_FOLDS_DEFAULT)],
        "bm25",
        "bm25_qmr",
        "twotower",
    ]


def build_fallback_pool(fallback: str, *, n_candidates: int = 100) -> BasePool | None:
    """Parse a fallback string into a BasePool (or None for popularity).

    Same '-'-separated syntax as the TID pool list:

      - "popularity"                → None (caller uses store.get_popular_tracks)
      - "bm25"                      → single BM25Pool
      - "bm25-qemb_metadata_rich"   → EnsemblePool(RRF, ignore_failures) (current default)

    Sub-pools are forced to `fallback="popularity"` so recursion stops after one level.
    """
    if fallback == "popularity":
        return None

    names = fallback.split("-")
    pools = [build_pool(n, n_candidates=n_candidates, fallback="popularity") for n in names]
    if len(pools) == 1:
        return pools[0]

    # Lazy import: pool/ensemble.py imports BasePool, avoid a cycle.
    from mymodule.strategies.pool.ensemble import EnsemblePool

    return EnsemblePool(
        pools=pools,
        method="rrf",
        n_candidates=n_candidates,
        ignore_failures=True,
    )


DEFAULT_FALLBACK = "bm25-qemb_metadata_rich"
