"""Centralized `.env` loader for the mymodule package.

One-shot, thread-safe wrapper around `python-dotenv.load_dotenv()`. Called once
from `mymodule/__init__.py` at import time so that every downstream
`os.getenv(...)` call in the codebase transparently sees variables declared in
the repo-root `.env` file.

Design rules
------------
- **Single source of truth**: this module is the only place that calls
  `load_dotenv()`. Do NOT call it from other modules — duplicate loads are
  wasteful and confuse precedence.
- **Real shell env wins**: `override=False` so values set in the actual
  process environment are never clobbered by `.env`. Makes CI / shell overrides
  Just Work (`MYMODULE_LLM_OPENAI_API_KEY=... uv run ...`).
- **Missing `.env` is fine**: `dotenv_path` defaults to searching upward; if no
  `.env` exists we silently no-op. The repo also ships `.env.example` as a
  template.
- **Missing `python-dotenv` is fine**: fail-soft import so evaluator-only
  subprocesses (which have their own venvs) don't crash if the dep is absent.
"""

from __future__ import annotations

import threading
from pathlib import Path

_LOCK = threading.Lock()
_LOADED = False


def ensure_env_loaded() -> bool:
    """Idempotently load `.env` from the repo root.

    Returns True on first successful load, False on subsequent calls or when
    `python-dotenv` / `.env` are absent.
    """
    global _LOADED
    if _LOADED:
        return False
    with _LOCK:
        if _LOADED:
            return False
        try:
            from dotenv import load_dotenv
        except ImportError:
            _LOADED = True  # don't retry every import
            return False

        repo_root = Path(__file__).resolve().parents[2]
        env_path = repo_root / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
        _LOADED = True
        return True
