"""Base strategy interface for inference."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from mymodule.strategies.pool.base import BasePool
from mymodule.strategies.rerank.base import BaseReranker
from mymodule.strategies.response.base import BaseResponseGenerator
from mymodule.utils.fusion import fuse, rrf_score_map

# Cached thread pool for per-prediction multi-pool parallelism. Sized once on
# first use; reused across all predictions (also by EnsemblePool for intra-pool
# sub-pool parallelism). Threading is appropriate here because pool work is
# dominated by LanceDB C++ search (releases GIL), BM25 C-backed retrieval, and
# HTTP (Ollama embeddings) — GIL isn't the bottleneck.
_POOL_EXECUTOR: ThreadPoolExecutor | None = None


def get_pool_executor(n_pools: int) -> ThreadPoolExecutor:
    """Shared ThreadPoolExecutor for dispatching multiple pool.generate() calls.

    Used by BaseStrategy.predict (ensemble of strategy-level pools) and
    EnsemblePool.generate (composite pool's sub-pools). `MYMODULE_POOL_WORKERS`
    env overrides max_workers (default min(8, n_pools)).
    """
    global _POOL_EXECUTOR
    if _POOL_EXECUTOR is None:
        env_max = int(os.environ.get("MYMODULE_POOL_WORKERS", "0") or 0)
        # default: cap at 8 (typical ensemble width) and don't exceed n_pools
        max_workers = env_max if env_max > 0 else max(1, min(8, n_pools))
        _POOL_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pool")
    return _POOL_EXECUTOR


class BaseStrategy(ABC):
    """Common interface for per-tid model strategies.

    To add a strategy: create a new .py file under strategies/, subclass this,
    and map pool names to pool instances in __init__. The file name becomes
    the strategy name (auto-registered).
    """

    pools: list[BasePool]
    pool_names: list[str]
    reranker: BaseReranker
    responder: BaseResponseGenerator
    ensemble: str
    # Optional: when set, cold-start turns (no prior role=music message in
    # chat_history) dispatch to this pool instead of `self.pools`. Other turns
    # take the normal path. Built from a pool-list spec by the concrete
    # strategy (reuses `_aliases.build_fallback_pool` parsing).
    turn1_pool: BasePool | None = None
    # Optional: when non-None, restrict `turn1_pool` to sessions whose
    # `conversation_goal['specificity']` is in this set. Other T1 sessions
    # (and all T>1) take the primary path. None = "apply to every cold-start
    # turn" (backward-compatible default). Parsed from comma-separated CLI
    # value (e.g., "LH,LL") in the concrete strategy.
    turn1_specificity: frozenset[str] | None = None

    @abstractmethod
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
        **kwargs,
    ) -> None:
        """Subclasses set pool_names → self.pools, reranker_name → self.reranker,
        and (responder_name, responder_provider) → self.responder.

        responder_provider selects the LM provider for DSPy-backed responders
        ("ollama" | "openai"); ignored by noop. fallback=None uses
        `mymodule.strategies._aliases.DEFAULT_FALLBACK` (pool-list syntax)."""

    def predict(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
        top_k: int = 20,
        rewritten_query: str | None = None,
        rewrite_pools: set[str] | None = None,
    ) -> tuple[list[str], str]:
        """Default pipeline: pools → ensemble → rerank → response.

        Multiple pools run in parallel via ThreadPoolExecutor (LanceDB, bm25s,
        rocksdict calls release the GIL); a single pool is called directly.
        `MYMODULE_POOL_WORKERS` overrides max_workers (default min(8, len(pools))).

        Stage 3 (response): the reranked top-k is passed to
        `self.responder.generate(...)`; the default noop responder returns an
        empty string. top_k is the final return count (default 20).
        """
        kwargs = dict(
            user_query=user_query,
            chat_history=chat_history,
            user_id=user_id,
            conversation_goal=conversation_goal,
            goal_progress=goal_progress,
            user_profile=user_profile,
        )

        # Turn-1 cold-start routing. When `--turn1-pool` is set and the
        # session has no prior `role=music` turn, dispatch to that pool
        # instead of the primary ensemble. Optionally narrowed by
        # `--turn1-specificity` — when set, only sessions whose
        # `conversation_goal['specificity']` is in the filter set get the
        # override; other T1 sessions fall through to the primary path. All
        # T>1 take the normal path unchanged.
        is_turn1 = not any((m or {}).get("role") == "music" for m in (chat_history or []))
        spec_ok = self.turn1_specificity is None or (
            isinstance(conversation_goal, dict)
            and (conversation_goal.get("specificity") or "") in self.turn1_specificity
        )
        active_pools: list[BasePool]
        active_pool_names: list[str]
        if self.turn1_pool is not None and is_turn1 and spec_ok:
            active_pools = [self.turn1_pool]
            active_pool_names = ["turn1_fallback"]
        else:
            active_pools = list(self.pools)
            active_pool_names = list(self.pool_names)

        # Call every pool via generate_with_scores to preserve native scores.
        # Per-pool query routing: when rewrite_pools is set, only those pools
        # receive the rewritten query.
        def _pool_kwargs(pool_name: str) -> dict:
            if rewritten_query is not None and rewrite_pools is not None and pool_name in rewrite_pools:
                return {**kwargs, "user_query": rewritten_query}
            return kwargs

        # Single pool → direct call; multiple pools → thread-pool parallel.
        if len(active_pools) == 1:
            rankings_scored = [active_pools[0].generate_with_scores(**_pool_kwargs(active_pool_names[0]))]
        else:
            exec_ = get_pool_executor(len(active_pools))
            futures = [
                exec_.submit(p.generate_with_scores, **_pool_kwargs(name))
                for p, name in zip(active_pools, active_pool_names)
            ]
            rankings_scored = [f.result() for f in futures]

        rankings = [[tid for tid, _ in scored] for scored in rankings_scored]

        if len(rankings) > 1:
            candidates = fuse(rankings, method=self.ensemble, top_n=100)
            fused_scores = rrf_score_map(rankings)
        else:
            candidates = rankings[0] if rankings else []
            # With a single pool its own scores are the fused scores.
            fused_scores = {tid: score for tid, score in rankings_scored[0]} if rankings_scored else {}

        pool_rankings = dict(zip(active_pool_names, rankings_scored))

        # Session-level seen-track filtering is handled inside each pool
        # (see `mymodule/utils/seen.py`) so fusion/rerank here see only
        # unseen candidates. No strategy-level filter required.

        ranked = self.reranker.rerank(
            candidates=candidates,
            pool_rankings=pool_rankings,
            fused_scores=fused_scores,
            pool_names=active_pool_names,
            **kwargs,
        )
        top = ranked[:top_k]

        response = self.responder.generate(
            track_ids=top,
            user_query=user_query,
            chat_history=chat_history,
            user_profile=user_profile,
            conversation_goal=conversation_goal,
            goal_progress=goal_progress,
            user_id=user_id,
        )
        return top, response
