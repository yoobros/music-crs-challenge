"""Expose a trained GBM reranked ensemble as a retrieval pool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mymodule.strategies.pool.base import BasePool
from mymodule.utils.fusion import fuse, rrf_score_map


class GBMEnsemblePool(BasePool):
    """Run inner pools, GBM-rerank their fused candidates, and expose as one pool."""

    def __init__(
        self,
        pool_names: list[str],
        *,
        n_candidates: int = 100,
        fallback: str = "bm25",
        ckpt_root: str | Path | None = None,
        pools: list[BasePool] | None = None,
        reranker: Any | None = None,
        **kwargs,
    ) -> None:
        self.pool_names = list(pool_names)
        self.n_candidates = n_candidates
        self.fallback = fallback
        if pools is None:
            from mymodule.strategies._aliases import build_pool

            self.pools = [build_pool(name, n_candidates=n_candidates, fallback=fallback) for name in self.pool_names]
        else:
            self.pools = pools
        if reranker is None:
            from mymodule.strategies.rerank.gbm import GBMReranker

            reranker = GBMReranker(pool_names=self.pool_names, ckpt_root=ckpt_root)
        self.reranker = reranker

    def generate(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[str]:
        return [
            tid
            for tid, _score in self.generate_with_scores(
                user_query=user_query,
                chat_history=chat_history,
                user_id=user_id,
                conversation_goal=conversation_goal,
                goal_progress=goal_progress,
                user_profile=user_profile,
            )
        ]

    def generate_with_scores(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[tuple[str, float]]:
        kwargs = dict(
            user_query=user_query,
            chat_history=chat_history,
            user_id=user_id,
            conversation_goal=conversation_goal,
            goal_progress=goal_progress,
            user_profile=user_profile,
        )
        rankings_scored = [pool.generate_with_scores(**kwargs) for pool in self.pools]
        rankings = [[tid for tid, _score in scored] for scored in rankings_scored]
        if not rankings:
            return []

        candidates = fuse(rankings, method="rrf", top_n=self.n_candidates)
        fused_scores = rrf_score_map(rankings)
        pool_rankings = dict(zip(self.pool_names, rankings_scored))
        ranked = self.reranker.rerank(
            candidates=candidates,
            pool_rankings=pool_rankings,
            fused_scores=fused_scores,
            pool_names=self.pool_names,
            **kwargs,
        )[: self.n_candidates]
        return [(tid, 1.0 / (rank + 1)) for rank, tid in enumerate(ranked)]
