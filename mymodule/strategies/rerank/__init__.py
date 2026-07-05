"""Reranker auto-discovery registry (lazy).

Any .py file or package under strategies/rerank/ containing a BaseReranker
subclass is auto-registered; the file/directory name becomes the reranker name.

Discovery happens lazily on the first ``get_reranker()`` call so that heavy
dependencies (e.g. lightgbm) are not imported at module-import time, avoiding
OpenMP conflicts with other native libraries.
"""

import inspect
import pkgutil
from importlib import import_module
from pathlib import Path

from mymodule.strategies.rerank.base import BaseReranker

RERANK_REGISTRY: dict[str, type[BaseReranker]] = {}
_DISCOVERED = False


def _discover_rerankers() -> None:
    """Register rerankers found under ``strategies/rerank/`` into ``RERANK_REGISTRY``.

    ``pkgutil.iter_modules`` picks up both .py files and package directories.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name == "base":
            continue
        module = import_module(f"mymodule.strategies.rerank.{info.name}")
        for _name, attr in inspect.getmembers(module, inspect.isclass):
            if issubclass(attr, BaseReranker) and attr is not BaseReranker:
                RERANK_REGISTRY[info.name] = attr
                break
    _DISCOVERED = True


def get_reranker(name: str, **kwargs) -> BaseReranker:
    """Instantiate a reranker by name."""
    _discover_rerankers()
    if name not in RERANK_REGISTRY:
        available = ", ".join(sorted(RERANK_REGISTRY.keys()))
        raise KeyError(f"No reranker '{name}'. Available: [{available}]")
    return RERANK_REGISTRY[name](**kwargs)
