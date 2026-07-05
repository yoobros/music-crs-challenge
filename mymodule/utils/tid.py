"""TID parsing utilities.

TID format: {strategy}__{pool1-pool2-...}__{reranker}
  - three parts separated by `__`
  - reranker defaults to "passthrough" when omitted
  - pools are combined with `-` (RRF fusion)

Recommended strategy = `ensemble` (pure pool composition).
Pool names include `bm25`, `qemb_<variant>`, `qemb_twotower`.
"""

from __future__ import annotations


def parse_tid(tid: str) -> tuple[str, list[str], str]:
    """Parse a tid into (strategy_name, pool_names, reranker_name).

    Examples:
        "ensemble__bm25-qemb_metadata"          → ("ensemble", ["bm25", "qemb_metadata"], "passthrough")
        "ensemble__bm25_qmr-qemb_twotower__gbm" → ("ensemble", ["bm25_qmr", "qemb_twotower"], "gbm")
    """
    parts = tid.split("__")

    strategy = parts[0]
    pools = parts[1].split("-") if len(parts) > 1 and parts[1] else []
    reranker = parts[2] if len(parts) > 2 and parts[2] else "passthrough"

    return strategy, pools, reranker


def format_tid(strategy: str, pools: list[str], reranker: str = "passthrough") -> str:
    """Build a tid string from strategy, pools, and reranker."""
    tid = f"{strategy}__{'-'.join(pools)}"
    if reranker != "passthrough":
        tid += f"__{reranker}"
    return tid
