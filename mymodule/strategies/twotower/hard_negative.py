"""Hard-negative sampling for two-tower LoRA training.

The training loop uses `info_nce_multipos` with in-batch random negatives only.
This module supplies a specificity-aware hard-negative sampler that the
collate function can use to inject additional negatives into each batch.

Specificity → weighted distribution over strategies
---------------------------------------------------
Each specificity bucket maps to a **weighted distribution** over strategies.
For each hard-negative position the sampler draws a strategy from that
distribution and tries it; if the pool is empty/insufficient for that query,
it falls back through the remaining strategies in weight-descending order.

Strategies implemented:
- `bm25-topk`       per-query top-K BM25 hits over the doc corpus, pre-baked
                    at extract time and carried in `pos_bm25_top_tids`.
                    The most "directly hard" neg — mimics what BM25 retrieval
                    would mistakenly rank high.
- `same-artist`     same artist_name, different track_id
- `same-album`      same album_name, different track_id
- `same-tag`        any of the positive's top tags overlaps a candidate's
- `same-pop-bucket` same 10-point popularity bucket (e.g. "p60-69")
- `off`             explicit no-op (terminates the chain)

Default distributions (see `SPECIFICITY_DISTRIBUTION`):
- HH (specific goal + specific track): album / artist / bm25 heavy
- HL (specific goal, vague track): tag / bm25 heavy
- LH (vague goal, specific track): artist / bm25 / album
- LL (vague goal, vague track): off
- unknown / None: off

This module is *pure* — no model, no loss, no tokenizer.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from loguru import logger

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

# Weight values are relative — they get normalized at sampling time.
SPECIFICITY_DISTRIBUTION: dict[str, list[tuple[str, float]]] = {
    "HH": [
        ("bm25-topk", 0.3),
        ("same-album", 0.3),
        ("same-artist", 0.3),
        ("same-tag", 0.1),
    ],
    "HL": [
        ("bm25-topk", 0.3),
        ("same-tag", 0.4),
        ("same-pop-bucket", 0.2),
        ("same-album", 0.1),
    ],
    "LH": [
        ("bm25-topk", 0.3),
        ("same-artist", 0.3),
        ("same-album", 0.2),
        ("same-tag", 0.2),
    ],
    "LL": [("off", 1.0)],
}

# Standalone mode "bm25": for non-LL queries, use bm25-topk only (no metadata
# strategy). Provided for ablation — see `make_distribution` below.
_BM25_ONLY_DISTRIBUTION: dict[str, list[tuple[str, float]]] = {
    "HH": [("bm25-topk", 1.0)],
    "HL": [("bm25-topk", 1.0)],
    "LH": [("bm25-topk", 1.0)],
    "LL": [("off", 1.0)],
}

# All known strategy names (validated at module load time).
KNOWN_STRATEGIES: frozenset[str] = frozenset(
    {"bm25-topk", "same-artist", "same-album", "same-tag", "same-pop-bucket", "off"}
)


def _validate_distributions() -> None:
    for table_name, table in (
        ("SPECIFICITY_DISTRIBUTION", SPECIFICITY_DISTRIBUTION),
        ("_BM25_ONLY_DISTRIBUTION", _BM25_ONLY_DISTRIBUTION),
    ):
        for spec, dist in table.items():
            for strategy, weight in dist:
                if strategy not in KNOWN_STRATEGIES:
                    raise ValueError(f"unknown strategy {strategy!r} in {table_name}[{spec!r}]")
                if weight < 0:
                    raise ValueError(f"negative weight {weight} for {strategy!r} in {spec!r}")


_validate_distributions()


def get_distribution(mode: str) -> dict[str, list[tuple[str, float]]]:
    """Return the distribution table for a given --smart-negative mode.

    Modes:
    - "specificity-aware": default mixed distributions per specificity
    - "bm25":              bm25-topk only (non-LL)
    """
    if mode == "specificity-aware":
        return SPECIFICITY_DISTRIBUTION
    if mode == "bm25":
        return _BM25_ONLY_DISTRIBUTION
    raise ValueError(f"unknown smart-negative mode {mode!r}")


# ---------------------------------------------------------------------------
# Pools — indexed for fast lookup at collate time
# ---------------------------------------------------------------------------


@dataclass
class HardNegPools:
    """Inverted indices over the full track catalog.

    Each map: lookup key → list of track_ids sharing that key. Build once at
    training startup from `track_meta_map_{compose}.json` and hold in memory.
    Total footprint for 47k tracks ≈ a few MB.
    """

    artist_to_tids: dict[str, list[str]] = field(default_factory=dict)
    album_to_tids: dict[str, list[str]] = field(default_factory=dict)
    top_tag_to_tids: dict[str, list[str]] = field(default_factory=dict)
    pop_bucket_to_tids: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_track_meta_map(cls, path: Path) -> "HardNegPools":
        """Load and invert {tid: {artist, album, top_tags, pop_bucket}}."""
        with path.open() as f:
            tid_to_meta: dict[str, dict] = json.load(f)

        artist_to_tids: dict[str, list[str]] = defaultdict(list)
        album_to_tids: dict[str, list[str]] = defaultdict(list)
        top_tag_to_tids: dict[str, list[str]] = defaultdict(list)
        pop_bucket_to_tids: dict[str, list[str]] = defaultdict(list)
        for tid, meta in tid_to_meta.items():
            artist = (meta.get("artist") or "").strip()
            if artist:
                artist_to_tids[artist].append(tid)
            album = (meta.get("album") or "").strip()
            if album:
                album_to_tids[album].append(tid)
            for tag in meta.get("top_tags") or []:
                if tag:
                    top_tag_to_tids[tag].append(tid)
            pop_bucket = (meta.get("pop_bucket") or "").strip()
            if pop_bucket:
                pop_bucket_to_tids[pop_bucket].append(tid)
        logger.info(
            f"[hard-neg] pools built from {path}: "
            f"artists={len(artist_to_tids)} albums={len(album_to_tids)} "
            f"top-tags={len(top_tag_to_tids)} pop-buckets={len(pop_bucket_to_tids)}"
        )
        return cls(
            artist_to_tids=dict(artist_to_tids),
            album_to_tids=dict(album_to_tids),
            top_tag_to_tids=dict(top_tag_to_tids),
            pop_bucket_to_tids=dict(pop_bucket_to_tids),
        )


# ---------------------------------------------------------------------------
# Per-strategy sampling primitives
# ---------------------------------------------------------------------------


def _pick_one(
    pool: Iterable[str],
    rng: random.Random,
    block: set[str],
) -> str | None:
    """Random pick of one tid from pool, skipping anything in block."""
    candidates = [t for t in pool if t not in block]
    if not candidates:
        return None
    return rng.choice(candidates)


@dataclass
class _PositiveKeys:
    """The positive track's signature, used to look up matching pools."""

    artist: str
    album: str
    top_tags: list[str]
    pop_bucket: str
    bm25_top_tids: list[str]


def _strategy_pool(strategy: str, keys: _PositiveKeys, pools: HardNegPools) -> list[str]:
    """Resolve the candidate pool (caller filters via `block`)."""
    if strategy == "off":
        return []
    if strategy == "bm25-topk":
        return keys.bm25_top_tids
    if strategy == "same-artist":
        return pools.artist_to_tids.get(keys.artist, [])
    if strategy == "same-album":
        return pools.album_to_tids.get(keys.album, [])
    if strategy == "same-pop-bucket":
        return pools.pop_bucket_to_tids.get(keys.pop_bucket, [])
    if strategy == "same-tag":
        # Union of all top-tag pools, dedup preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for tag in keys.top_tags or []:
            for tid in pools.top_tag_to_tids.get(tag, []):
                if tid in seen:
                    continue
                seen.add(tid)
                out.append(tid)
        return out
    raise ValueError(f"unknown strategy {strategy!r}")


def _weighted_choice(
    distribution: list[tuple[str, float]],
    rng: random.Random,
) -> str:
    """Sample one strategy name from (name, weight) pairs."""
    total = sum(max(0.0, w) for _, w in distribution)
    if total <= 0 or not distribution:
        return rng.choice([s for s, _ in distribution]) if distribution else "off"
    r = rng.random() * total
    cum = 0.0
    for strategy, w in distribution:
        cum += max(0.0, w)
        if r < cum:
            return strategy
    return distribution[-1][0]


# ---------------------------------------------------------------------------
# Public sampler
# ---------------------------------------------------------------------------


def sample_hard_negs(
    specificity: str | None,
    pos_track_id: str,
    pos_artist: str,
    pos_album: str,
    pos_top_tags: list[str],
    pos_pop_bucket: str,
    pos_bm25_top_tids: list[str],
    pools: HardNegPools,
    k: int,
    rng: random.Random,
    exclude: set[str],
    distribution_table: dict[str, list[tuple[str, float]]] | None = None,
) -> list[str]:
    """Return up to K hard-neg track_ids for one query.

    Algorithm
    ---------
    For each of K hard-neg slots:
        1. Draw a strategy from `distribution_table[specificity]`.
        2. Pick one tid from that strategy's pool (excluding `exclude`,
           `pos_track_id`, and already-collected tids).
        3. If empty, fall through the remaining strategies in the same
           distribution, in **weight-descending** order, picking the first
           that yields a tid.
        4. If all strategies exhaust, stop early — result may be shorter
           than K.

    `exclude` should include every positive track_id in the current batch
    so that cross-query positive collisions don't break the loss.

    `distribution_table` defaults to `SPECIFICITY_DISTRIBUTION`; pass
    `_BM25_ONLY_DISTRIBUTION` (via `get_distribution("bm25")`) for the
    bm25-only ablation.
    """
    if k <= 0:
        return []
    table = distribution_table if distribution_table is not None else SPECIFICITY_DISTRIBUTION
    distribution = table.get(specificity or "", None)
    if not distribution or all(strategy == "off" for strategy, _ in distribution):
        return []

    keys = _PositiveKeys(
        artist=pos_artist or "",
        album=pos_album or "",
        top_tags=pos_top_tags or [],
        pop_bucket=pos_pop_bucket or "",
        bm25_top_tids=pos_bm25_top_tids or [],
    )

    block: set[str] = set(exclude) | {pos_track_id}
    fallback_order = [s for s, _ in sorted(distribution, key=lambda x: -x[1])]

    collected: list[str] = []
    for _ in range(k):
        primary = _weighted_choice(distribution, rng)
        attempt_order: list[str] = []
        if primary != "off":
            attempt_order.append(primary)
        for s in fallback_order:
            if s != primary and s != "off" and s not in attempt_order:
                attempt_order.append(s)

        pick: str | None = None
        for strategy in attempt_order:
            pool = _strategy_pool(strategy, keys, pools)
            pick = _pick_one(pool, rng, block)
            if pick is not None:
                break
        if pick is None:
            break
        collected.append(pick)
        block.add(pick)
    return collected


__all__ = [
    "SPECIFICITY_DISTRIBUTION",
    "KNOWN_STRATEGIES",
    "HardNegPools",
    "get_distribution",
    "sample_hard_negs",
]
