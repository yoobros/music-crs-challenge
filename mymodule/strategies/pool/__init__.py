"""Pool auto-discovery registry.

Any .py file in strategies/pool/ containing a BasePool subclass is registered
automatically; the file name becomes the pool name.
"""

import inspect
import pkgutil
from importlib import import_module
from pathlib import Path

from mymodule.strategies.pool.base import BasePool

POOL_REGISTRY: dict[str, type[BasePool]] = {}

_pkg_dir = Path(__file__).parent
for _info in pkgutil.iter_modules([str(_pkg_dir)]):
    if _info.name == "base":
        continue
    _module = import_module(f"mymodule.strategies.pool.{_info.name}")
    for _attr_name, _attr in inspect.getmembers(_module, inspect.isclass):
        if issubclass(_attr, BasePool) and _attr is not BasePool:
            POOL_REGISTRY[_info.name] = _attr
            break


def get_pool(name: str, **kwargs) -> BasePool:
    """Create a pool instance by name."""
    if name not in POOL_REGISTRY:
        available = ", ".join(sorted(POOL_REGISTRY.keys()))
        raise KeyError(f"No pool '{name}'. Available: [{available}]")
    return POOL_REGISTRY[name](**kwargs)
