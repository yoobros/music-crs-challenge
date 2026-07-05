"""Ensemble strategy: a thin pool-assembly strategy.

TID: `ensemble__<pool-name-list>__<reranker>`

The pool-name list is '-'-separated; see `_aliases.available_pool_names()`
for the valid names (bm25 variants, qemb_<variant>, twotower, the 8B
two-tower fold bag, etc.).

Fallback / ensemble method are NOT part of the TID; they are CLI kwargs:
  - `--fallback <pool-list>` / `--fallback popularity`
  - `--ensemble rrf` (default) / `--ensemble rbc`

BaseStrategy.predict() dispatches self.pools in parallel on a shared
ThreadPoolExecutor.
"""

from __future__ import annotations

from mymodule.strategies._aliases import DEFAULT_FALLBACK, build_fallback_pool, build_pool, expand_pool_names
from mymodule.strategies.base import BaseStrategy
from mymodule.strategies.rerank import get_reranker
from mymodule.strategies.response import get_response_generator


class EnsembleStrategy(BaseStrategy):
    def __init__(
        self,
        *,
        tid: str,
        pool_names: list[str],
        reranker_name: str = "passthrough",
        responder_name: str = "noop",
        responder_provider: str = "ollama",
        fallback: str | None = None,
        ensemble: str = "rrf",
        turn1_pool: str | None = None,
        turn1_specificity: str | None = None,
        **kwargs,
    ) -> None:
        fb = fallback or DEFAULT_FALLBACK
        self.ensemble = ensemble
        # pool_names after alias expansion (e.g. a fold-bag alias → fold names);
        # self.pools and self.pool_names stay index-aligned 1:1.
        self.pool_names = expand_pool_names(pool_names)
        self.pools = [build_pool(name, n_candidates=100, fallback=fb) for name in self.pool_names]
        self.reranker = get_reranker(reranker_name, top_k=20, tid=tid)
        self.responder = get_response_generator(responder_name, provider=responder_provider)
        # Reuses fallback pool-list syntax ("bm25-qemb_metadata"). None → no
        # turn-1 override (BaseStrategy.predict keeps its default path).
        self.turn1_pool = build_fallback_pool(turn1_pool, n_candidates=100) if turn1_pool else None
        # Comma-separated specificity filter (e.g. "LH,LL"). When set, the
        # turn1_pool override only fires on cold-start turns whose listener
        # specificity is in this set; other T1 sessions fall through to the
        # primary path. Whitespace and case insensitive; ignored when
        # turn1_pool is None (with a warning emitted at CLI parse-time).
        spec_set: frozenset[str] | None = None
        if turn1_specificity:
            tokens = [t.strip().upper() for t in turn1_specificity.split(",") if t.strip()]
            spec_set = frozenset(tokens) if tokens else None
        self.turn1_specificity = spec_set
