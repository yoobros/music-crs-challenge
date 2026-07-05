"""Strategy auto-discovery registry.

Any .py file in strategies/ containing a BaseStrategy subclass is registered
automatically; the file name becomes the strategy name.

TID format: {strategy}__{pool1-pool2-...}__{reranker}
The response generator is not part of the TID; it is selected via
env `MYMODULE_RESPONSE_GEN` / CLI `--response-gen` (noop | pas | auto) and
env `MYMODULE_CHAT_PROVIDER` / CLI `--response-provider` (ollama | openai).
"""

import inspect
import os
import pkgutil
from importlib import import_module
from pathlib import Path

from loguru import logger

from mymodule.strategies.base import BaseStrategy
from mymodule.utils.tid import parse_tid

# Auto-scan: find BaseStrategy subclasses in every .py under strategies/.
STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {}
_DISCOVERED = False


def _discover_strategies() -> None:
    """Register strategy classes on demand.

    Importing every strategy at package import time can initialize native
    runtimes before GBM has a chance to load LightGBM, which is fragile on
    macOS. Discovery stays lazy but keeps the public registry/get_strategy
    behavior unchanged after the first lookup.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        # base is the ABC; underscore-prefixed modules are shared helpers, not strategies.
        if info.name == "base" or info.name.startswith("_"):
            continue
        module = import_module(f"mymodule.strategies.{info.name}")
        for attr_name, attr in inspect.getmembers(module, inspect.isclass):
            if issubclass(attr, BaseStrategy) and attr is not BaseStrategy:
                STRATEGY_REGISTRY[info.name] = (f"mymodule.strategies.{info.name}", attr_name)
                break
    _DISCOVERED = True


def get_strategy(tid: str, **kwargs) -> BaseStrategy:
    """Parse the tid and return a strategy instance.

    Args:
        tid: "{strategy}__{pools}__{reranker}".
        **kwargs: fallback, ensemble, responder_name, responder_provider, etc.
            responder_name defaults to env `MYMODULE_RESPONSE_GEN`, then "noop";
            responder_provider defaults to env `MYMODULE_CHAT_PROVIDER`, then "ollama".
    """
    strategy_name, pool_names, reranker_name = parse_tid(tid)
    _discover_strategies()

    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"No strategy '{strategy_name}'. Available: [{available}]")

    # Env-derived defaults (explicit kwargs win).
    if "responder_name" not in kwargs:
        kwargs["responder_name"] = os.environ.get("MYMODULE_RESPONSE_GEN", "noop")
    if "responder_provider" not in kwargs:
        kwargs["responder_provider"] = os.environ.get("MYMODULE_CHAT_PROVIDER", "ollama")

    module_path, class_name = STRATEGY_REGISTRY[strategy_name]
    module = import_module(module_path)
    cls = getattr(module, class_name)
    logger.info(
        f"Strategy: {class_name} | pools: {pool_names} | reranker: {reranker_name} | "
        f"responder: {kwargs['responder_name']}(provider={kwargs['responder_provider']})"
    )
    return cls(tid=tid, pool_names=pool_names, reranker_name=reranker_name, **kwargs)


def show_available() -> str:
    """Return the list of available components."""
    _discover_strategies()
    from mymodule.strategies._aliases import DEFAULT_FALLBACK, available_pool_names
    from mymodule.strategies.rerank import RERANK_REGISTRY
    from mymodule.strategies.response import RESPONSE_REGISTRY

    lines = [
        "TID format: {strategy}__{pool1-pool2-...}__{reranker}",
        "  reranker omitted → passthrough",
        "  pools combined with '-' (RRF fusion)",
        "",
        f"Strategies:     {', '.join(sorted(STRATEGY_REGISTRY.keys()))}",
        f"Ensemble pools: {', '.join(available_pool_names())}",
        f"Rerankers:      {', '.join(sorted(RERANK_REGISTRY.keys()))}",
        f"Responders:     {', '.join(sorted(RESPONSE_REGISTRY.keys()))}  "
        f"(logic via --response-gen / MYMODULE_RESPONSE_GEN; provider via "
        f"--response-provider / MYMODULE_CHAT_PROVIDER; default: noop / ollama)",
        "",
        "Examples:",
        "  ensemble__bm25-qemb_metadata                # 2-way RRF",
        "  ensemble__bm25_qmr-qemb_twotower            # nested + bag RRF",
        "",
        f"kwargs: --fallback '{DEFAULT_FALLBACK}' (default; '-'-separated, or 'popularity'),",
        "        --ensemble rrf (default, or rbc), --response-gen noop (default), --response-provider ollama (default)",
    ]
    return "\n".join(lines)
