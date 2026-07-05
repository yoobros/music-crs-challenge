"""Automatic mapping from TID pool names to :class:`PoolSpec`.

Single source of truth is ``mymodule.strategies._aliases.build_pool``; this
module only adds GBM-side metadata (fold_aware, logical name) on top.

Rules:
- ``qemb_twotower_8b`` / ``_all`` → fold-aware (leakage-free OOF), logical name ``qemb_twotower_8b``
- everything else (bm25/qemb/bm25_qmr/...) → fold-unaware static, delegated to
  ``build_pool``; unregistered names raise KeyError at call time.

``build_pool`` pulls in ``torch``, so it is imported lazily inside the
instantiators to avoid OpenMP conflicts with lightgbm on macOS.
"""

from __future__ import annotations

from mymodule.strategies.rerank.gbm.pool_spec import PoolSpec


def _instantiate_static(pool_name: str):
    """Lazy instantiator factory for fold-unaware pools."""

    def _make(_fold_idx):
        from mymodule.strategies._aliases import build_pool  # lazy: avoid torch at import time

        return build_pool(pool_name)

    return _make


def _instantiate_twotower_8b_fold():
    """Lazy instantiator for the fold-aware 8B two-tower pool (leakage-free OOF).

    The LoRA adapters were trained on train sessions, so OOF rows in fold i must
    be scored only by the fold-i held-out adapter; a static bag would leak
    (4 of 5 adapters have seen the session).
    """

    def _make(fold_idx):
        from mymodule.strategies._aliases import build_pool

        return build_pool(f"qemb_twotower_8b_fold{fold_idx}")

    return _make


def _resolve_pool_spec(name: str) -> PoolSpec:
    """Resolve a pool name to a :class:`PoolSpec`.

    ``qemb_twotower_8b`` is fold-aware; everything else is static. The ``_all``
    suffix is dropped during logical-name normalization.
    """
    # 8B two-tower bag: OOF rows must be scored by the fold-i held-out adapter
    # to stay leakage-free.
    if name in {"qemb_twotower_8b", "qemb_twotower_8b_all"}:
        return PoolSpec(name="qemb_twotower_8b", fold_aware=True, instantiate=_instantiate_twotower_8b_fold())

    # Everything else: delegate to build_pool (bm25 / bm25_qmr / qemb_* ...).
    return PoolSpec(name=name, fold_aware=False, instantiate=_instantiate_static(name))


def get_pool_specs(pool_names: list[str]) -> list[PoolSpec]:
    """Map pool names to specs; unknown names raise KeyError when build_pool is called."""
    out: list[PoolSpec] = []
    seen_logical_names: set[str] = set()
    for n in pool_names:
        spec = _resolve_pool_spec(n)
        # Keep only the first spec per logical name (idempotency).
        if spec.name in seen_logical_names:
            continue
        seen_logical_names.add(spec.name)
        out.append(spec)
    return out


def pool_signature(pool_names: list[str]) -> str:
    """Stable signature for OOF parquet / ckpt directory names.

    Sorted so it is order-independent (``[bm25, qemb_twotower_8b]`` and
    ``[qemb_twotower_8b, bm25]`` share one signature).
    """
    specs = get_pool_specs(pool_names)
    return "-".join(sorted(s.name for s in specs))
