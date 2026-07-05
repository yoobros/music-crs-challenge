"""Centralized loguru configuration for the `mymodule` package.

One-shot wrapper around `loguru.logger.configure` called from
`mymodule/__init__.py`. Every downstream module uses the same `logger`
instance — import with ``from mymodule.utils.log import logger`` (or the
shorter ``from loguru import logger`` which yields the same singleton).

Design rules
------------
- **Single configuration point**: this module is the only place that calls
  `logger.remove()` / `logger.add()`. Do NOT configure loguru elsewhere.
- **Opinionated format**: coloured, single-line, with level + caller. Keeps
  CLI output scannable without a verbose timestamp on every line.
- **Env-controlled level**: ``MYMODULE_LOG_LEVEL`` overrides the default
  ``INFO``. Accepts any loguru level name (``DEBUG``, ``WARNING`` ...).
- **Idempotent**: repeated calls of `ensure_logging_configured` are no-ops,
  so tests / subprocesses don't stack handlers.
"""

from __future__ import annotations

import os
import sys
import threading

from loguru import logger

_LOCK = threading.Lock()
_CONFIGURED = False

_DEFAULT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{name}:{line}</cyan> <level>{message}</level>"
)


def ensure_logging_configured() -> bool:
    """Idempotently install the project's loguru handler.

    Returns True on first call, False on subsequent calls.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return False
    with _LOCK:
        if _CONFIGURED:
            return False
        level = os.getenv("MYMODULE_LOG_LEVEL", "INFO").upper()
        logger.remove()
        logger.add(
            sys.stderr,
            level=level,
            format=_DEFAULT_FORMAT,
            colorize=True,
            backtrace=False,
            diagnose=False,
        )
        _CONFIGURED = True
        return True


__all__ = ["logger", "ensure_logging_configured"]
