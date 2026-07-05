"""Pool fold-awareness abstraction.

Lets the OOF feature build treat every pool through one entry point.
- ``fold_aware=False`` (static retrievers like BM25/QEmb): single reused instance.
- ``fold_aware=True`` (session-trained pools): fold-i sessions are scored only
  by the ckpt where fold i is held out; ``instantiate(fold_idx)`` returns that
  fold-specific instance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mymodule.strategies.pool.base import BasePool


@dataclass(frozen=True)
class PoolSpec:
    """Pool instantiation spec.

    Attributes:
        name: feature-column prefix (e.g. ``"bm25"``), used for the
            ``rank_<name>`` / ``score_<name>`` columns in the OOF parquet.
        fold_aware: True means a separate instance is needed per fold idx.
        instantiate: builds a ``BasePool`` from ``fold_idx`` (None | int);
            called with fold_idx=None when fold_aware=False.
    """

    name: str
    fold_aware: bool
    instantiate: Callable[[int | None], BasePool]
