"""Lightweight numpy helpers for cosine-similarity-based retrieval.

Existing similarity calculations are scattered (ad-hoc `np.dot` after manual
normalization in `generator.py`, `analyze_similarity_thresholds.py`). This
module is the single re-usable surface so all callers agree on
normalization, dtype, and tie-breaking semantics.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return `vec` rescaled to unit L2 norm. Returns the input unchanged
    when the norm is below `eps` (rather than dividing by zero)."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr
    return arr / norm


def stack_normalized(vecs: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Stack a list of 1-D vectors into a `(N, D)` float32 matrix with each row
    L2-normalized. Raises `ValueError` for empty input or shape mismatch."""
    if isinstance(vecs, np.ndarray):
        if vecs.ndim != 2:
            raise ValueError(f"expected 2-D matrix, got shape {vecs.shape}")
        rows = [vecs[i] for i in range(vecs.shape[0])]
    else:
        rows = list(vecs)
    if not rows:
        raise ValueError("stack_normalized: empty input")
    arr = np.stack([l2_normalize(r) for r in rows], axis=0).astype(np.float32, copy=False)
    return arr


def cosine_topk(query: np.ndarray, matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `(indices, similarities)` for the top-`top_k` rows of `matrix`
    by cosine similarity against `query`.

    Both `query` and the rows of `matrix` are assumed to be L2-normalized
    already (caller's responsibility — `stack_normalized` is the standard
    builder). Cosine then reduces to dot product. The returned arrays are
    sorted by similarity descending.

    Edge cases:
    - `top_k <= 0` or `matrix` empty → returns `(array([]), array([]))`.
    - `top_k >= len(matrix)` → returns all rows, still sorted by similarity.
    """
    if top_k <= 0 or matrix.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    sims = matrix.astype(np.float32, copy=False) @ q  # (N,)
    k = min(top_k, sims.shape[0])
    # argpartition gives an unordered top-k → sort within for final order.
    part = np.argpartition(-sims, kth=k - 1)[:k]
    order = part[np.argsort(-sims[part])]
    return order.astype(np.int64, copy=False), sims[order].astype(np.float32, copy=False)


__all__ = ["cosine_topk", "l2_normalize", "stack_normalized"]
