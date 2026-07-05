"""Rank-ensemble functions that combine multiple ranking lists.

Shared helpers usable across strategies, pools, and vector types.
"""

from __future__ import annotations


def rrf_score_map(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Return the RRF score dict only (no sorting).

    Used when a reranker consumes RRF scores as raw features. For the same input,
    `sorted(rrf_score_map(...), key=..., reverse=True)` yields the same order as
    ``rrf_fusion``.

    score(item) = Σ weight_i / (k + rank_i(item))
    """
    if not rankings:
        return {}

    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)
    return scores


def rrf_fusion(
    rankings: list[list[str]],
    k: int = 60,
    top_n: int | None = None,
    weights: list[float] | None = None,
) -> list[str]:
    """Combine multiple ranking lists with Reciprocal Rank Fusion.

    score(item) = Σ weight_i / (k + rank_i(item))

    Hyperbolic decay: contribution falls off gently with rank.

    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
    individual Rank Learning Methods" (SIGIR 2009)
    """
    if not rankings:
        return []

    scores = rrf_score_map(rankings, k=k, weights=weights)
    sorted_items = sorted(scores, key=scores.__getitem__, reverse=True)

    if top_n is not None:
        return sorted_items[:top_n]
    return sorted_items


def rbc_fusion(
    rankings: list[list[str]],
    phi: float = 0.9,
    top_n: int | None = None,
    weights: list[float] | None = None,
) -> list[str]:
    """Combine multiple ranking lists with Rank Biased Centroids (RBC).

    score(item) = Σ weight_i * φ^rank_i(item)

    Exponential decay: more sensitive to top-rank differences than RRF.
    φ tunes the curve — closer to 1 weighs deep ranks, closer to 0 only the top.

    Reference: Bailey et al., inspired by Rank-Biased Overlap (RBO).
    """
    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + weight * (phi**rank)

    sorted_items = sorted(scores, key=scores.__getitem__, reverse=True)

    if top_n is not None:
        return sorted_items[:top_n]
    return sorted_items


# ensemble name → function mapping
ENSEMBLE_METHODS = {
    "rrf": rrf_fusion,
    "rbc": rbc_fusion,
}


def fuse(
    rankings: list[list[str]],
    method: str = "rrf",
    top_n: int | None = None,
    weights: list[float] | None = None,
    **kwargs,
) -> list[str]:
    """Dispatch to a fusion function by ensemble name.

    Args:
        method: "rrf" or "rbc"
        **kwargs: method-specific extras (k=60 for rrf, phi=0.9 for rbc)
    """
    if method not in ENSEMBLE_METHODS:
        available = ", ".join(sorted(ENSEMBLE_METHODS.keys()))
        raise ValueError(f"Unknown ensemble method '{method}'. Available: [{available}]")
    return ENSEMBLE_METHODS[method](rankings, top_n=top_n, weights=weights, **kwargs)
