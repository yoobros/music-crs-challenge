"""Ensemble pool: compose multiple pools and fuse their rankings.

Reuses the `fuse()` utility and pool registry to expose a combination of
pools as one composite pool conforming to the BasePool contract, so callers
mixing several retrieval signals (e.g. BM25 + qemb RRF for cold-start
fallback) only call `generate()` once.

ignore_failures=True skips a sub-pool's ranking when it raises and fuses the
rest; if all fail, returns an empty list. RRF over a single ranking preserves
the original order, so it degrades gracefully to the surviving pool.
"""

from __future__ import annotations

from loguru import logger

from mymodule.strategies.pool.base import BasePool
from mymodule.utils.fusion import fuse


class EnsemblePool(BasePool):
    def __init__(
        self,
        pools: list[BasePool],
        method: str = "rrf",
        n_candidates: int = 100,
        weights: list[float] | None = None,
        ignore_failures: bool = True,
        **kwargs,
    ) -> None:
        if not pools:
            raise ValueError("EnsemblePool requires at least one sub-pool")
        self.pools = pools
        self.method = method
        self.n_candidates = n_candidates
        self.weights = weights
        self.ignore_failures = ignore_failures
        logger.info(
            f"EnsemblePool: {len(pools)} pools, method={method}, "
            f"n_candidates={n_candidates}, ignore_failures={ignore_failures}"
        )

    def generate(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[str]:
        # NOTE: sequential on purpose. BaseStrategy.predict already dispatches
        # pools in parallel on a shared executor; nesting parallel submits here
        # causes executor/Ollama contention and slows things down.
        rankings: list[list[str]] = []
        for pool in self.pools:
            try:
                rankings.append(
                    pool.generate(
                        user_query=user_query,
                        chat_history=chat_history,
                        user_id=user_id,
                        conversation_goal=conversation_goal,
                        goal_progress=goal_progress,
                        user_profile=user_profile,
                    )
                )
            except Exception as e:
                if not self.ignore_failures:
                    raise
                logger.warning(f"EnsemblePool: sub-pool {type(pool).__name__} failed ({e}); skipping")

        if not rankings:
            return []
        return fuse(rankings, method=self.method, top_n=self.n_candidates, weights=self.weights)
