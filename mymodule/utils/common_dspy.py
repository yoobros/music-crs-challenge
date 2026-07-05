"""Shared DSPy / LM plumbing used by response generators (`pas/`, `auto/`).

Used by `mymodule/strategies/response/base_dspy.py`, `pas/generator.py`,
`auto/module.py` etc. The qemb retrieval side no longer uses DSPy
(it uses raw Qwen3-Embedding only), so all instructed-query helpers were removed.

Provider selection
------------------
`provider` is one of `"ollama" | "openai"`. The matching environment block
(`MYMODULE_LLM_OLLAMA_*` or `MYMODULE_LLM_OPENAI_*`) drives `dspy.LM(...)`
configuration. A per-provider flag caches configured state so re-instantiating
the same generator does not rebuild the LM.

`dspy.configure(lm=...)` sets a process-global LM, so only the *most recent*
provider is active. In practice each inference run picks one provider.

Environment variables
---------------------
    MYMODULE_LLM_OLLAMA_URL       Ollama base URL; falls back to
                                   MYMODULE_EMB_OLLAMA_URL, then localhost:11434.
    MYMODULE_LLM_MODEL             Ollama model tag (default qwen3.5:2b).
    MYMODULE_LLM_OPENAI_URL        OpenAI-compat endpoint
                                   (default https://models.github.ai/inference).
    MYMODULE_LLM_OPENAI_MODEL      Upstream model name (prefixed with `openai/`).
    MYMODULE_LLM_OPENAI_API_KEY    API key for the OpenAI-compatible endpoint.
    MYMODULE_LLM_MAX_TOKENS        Max generated tokens (default 512).
    MYMODULE_LLM_TIMEOUT           Per-call seconds (default 120).
    MYMODULE_LLM_THINK             "1" enables thinking tokens, "0" disables (default "0").
    MYMODULE_LLM_JSON_MODE         "1" asks OpenAI-compatible backends for a JSON object response.
    MYMODULE_LLM_MIN_INTERVAL      Min seconds between requests to the active
                                   provider (default 0 = off).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Literal

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


Provider = Literal["ollama", "openai"]


# ---------------------------------------------------------------------------
# LM configuration — provider-scoped caches.
# ---------------------------------------------------------------------------
_CONFIG_LOCK = threading.Lock()
_CONFIGURED: dict[str, bool] = {"ollama": False, "openai": False}


def _resolve_ollama_url() -> str:
    return (
        os.getenv("MYMODULE_LLM_OLLAMA_URL") or os.getenv("MYMODULE_EMB_OLLAMA_URL") or "http://localhost:11434"
    ).rstrip("/")


def _resolve_openai_api_key() -> str:
    return os.getenv("MYMODULE_LLM_OPENAI_API_KEY") or "dummy"


def _resolve_lm_seed(seed: int | None) -> int | None:
    if seed is not None:
        return seed
    raw = os.getenv("MYMODULE_LLM_SEED")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve_lm_temperature(temperature: float | None) -> float:
    if temperature is not None:
        return temperature
    try:
        return float(os.getenv("MYMODULE_LLM_TEMPERATURE", "0.7"))
    except ValueError:
        return 0.7


def _json_mode_enabled() -> bool:
    raw = os.getenv("MYMODULE_LLM_JSON_MODE", "").strip().lower()
    return raw in ("1", "true", "yes", "on", "json")


def _is_google_openai_endpoint(url: str) -> bool:
    return "generativelanguage.googleapis.com" in url.lower()


def _build_ollama_lm(*, seed: int | None = None, temperature: float | None = None) -> Any:
    url = _resolve_ollama_url()
    model = os.getenv("MYMODULE_LLM_MODEL", "qwen3.5:2b")
    max_tokens = int(os.getenv("MYMODULE_LLM_MAX_TOKENS", "512"))
    timeout = int(os.getenv("MYMODULE_LLM_TIMEOUT", "120"))
    think = os.getenv("MYMODULE_LLM_THINK", "0") == "1"
    kwargs: dict[str, Any] = dict(
        model=f"ollama_chat/{model}",
        api_base=url,
        api_key="ollama",
        max_tokens=max_tokens,
        temperature=_resolve_lm_temperature(temperature),
        timeout=timeout,
        think=think,
    )
    seed_v = _resolve_lm_seed(seed)
    if seed_v is not None:
        kwargs["seed"] = seed_v
    return dspy.LM(**kwargs)


def _build_openai_lm(*, seed: int | None = None, temperature: float | None = None) -> Any:
    url = os.getenv("MYMODULE_LLM_OPENAI_URL", "https://models.github.ai/inference").rstrip("/")
    model = os.getenv("MYMODULE_LLM_OPENAI_MODEL", "gpt-4.1-nano")
    api_key = _resolve_openai_api_key()
    max_tokens = int(os.getenv("MYMODULE_LLM_MAX_TOKENS", "512"))
    timeout = int(os.getenv("MYMODULE_LLM_TIMEOUT", "120"))
    think = os.getenv("MYMODULE_LLM_THINK", "0") == "1"
    if not model.startswith("openai/"):
        model = f"openai/{model}"
    kwargs: dict[str, Any] = dict(
        model=model,
        api_base=url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=_resolve_lm_temperature(temperature),
        timeout=timeout,
    )
    if not _is_google_openai_endpoint(url):
        # vLLM/SGLang-backed OpenAI-compat endpoints honor enable_thinking.
        # Gemini's OpenAI-compatible endpoint rejects unknown extra_body fields.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think}}
    if _json_mode_enabled():
        kwargs["response_format"] = {"type": "json_object"}
    seed_v = _resolve_lm_seed(seed)
    if seed_v is not None and not _is_google_openai_endpoint(url):
        kwargs["seed"] = seed_v
    return dspy.LM(**kwargs)


def ensure_lm_configured(provider: Provider, *, seed: int | None = None, temperature: float | None = None) -> None:
    """Configure a provider-scoped `dspy.LM` once. Thread-safe.

    `seed` and `temperature` (or `MYMODULE_LLM_SEED` / `MYMODULE_LLM_TEMPERATURE`
    env) get forwarded to LiteLLM/OpenAI for reproducibility — providers that
    honor `seed` (vLLM, SGLang, OpenAI) will produce deterministic outputs given
    fixed (seed, temperature, prompt). The first caller wins; later calls with
    different values are ignored (cached config).
    """
    if _CONFIGURED.get(provider):
        return
    with _CONFIG_LOCK:
        if _CONFIGURED.get(provider):
            return
        if dspy is None:
            raise RuntimeError("dspy not installed. `uv add dspy`.")
        if provider == "ollama":
            lm = _build_ollama_lm(seed=seed, temperature=temperature)
        elif provider == "openai":
            lm = _build_openai_lm(seed=seed, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider {provider!r}. Expected 'ollama' or 'openai'.")
        dspy.configure(lm=lm)
        _CONFIGURED[provider] = True


# ---------------------------------------------------------------------------
# Rate limiter — minimum interval between provider calls.
# ---------------------------------------------------------------------------
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_LAST: dict[str, float] = {"ollama": 0.0, "openai": 0.0}


def min_interval_sec() -> float:
    try:
        v = float(os.getenv("MYMODULE_LLM_MIN_INTERVAL", "0") or "0")
    except ValueError:
        return 0.0
    return max(v, 0.0)


def apply_rate_limit(provider: Provider) -> None:
    """Sleep so at least `MYMODULE_LLM_MIN_INTERVAL` elapses since the last call."""
    interval = min_interval_sec()
    if interval <= 0:
        return
    with _RATE_LIMIT_LOCK:
        last = _RATE_LIMIT_LAST.get(provider, 0.0)
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _RATE_LIMIT_LAST[provider] = time.monotonic()


class RateLimitedPredictor:
    """Transparent wrapper that enforces `apply_rate_limit` on every call.

    Forwards attribute access to the inner `dspy.Predict` instance so ckpt
    loading and inspection keep working. Only useful when
    `MYMODULE_LLM_MIN_INTERVAL > 0` — callers should skip the wrap otherwise.
    """

    __slots__ = ("_inner", "_provider")

    def __init__(self, inner: Any, provider: Provider) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_provider", provider)

    def __call__(self, **kwargs: Any) -> Any:
        apply_rate_limit(self._provider)
        return self._inner(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Conversation-state formatters — line-based form, used by response generators.
# (The qemb instruction-generator pipeline was removed; only the response side
# still needs these.)
# ---------------------------------------------------------------------------

_PROFILE_FIELDS = [
    "age_group",
    "country_name",
    "preferred_language",
    "preferred_musical_culture",
    "gender",
]

# HH/HL/LH/LL: first char = goal clarity, second char = track-level specificity.
_SPECIFICITY_MAP = {
    "HH": "specific goal, specific track",
    "HL": "specific goal, general track",
    "LH": "general goal, specific track",
    "LL": "general goal, general track",
}


def _first(values: Any, fallback: str = "unknown") -> str:
    """Track-metadata fields are lists; take the first non-empty element."""
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str) and values:
        return values
    return fallback


_CHAT_EXPAND_MODES = ("off", "minimal", "selective")


def _resolve_chat_expand_mode() -> str:
    """Read `MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT`; default `selective` (PAS path).

    Invalid / unknown values silently fall back to `off` so a typo never
    silently changes prompt shape.
    """
    raw = os.getenv("MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT", "selective").strip().lower()
    return raw if raw in _CHAT_EXPAND_MODES else "off"


def _format_music_content(content: str, kv: Any, mode: str = "off") -> str:
    """Resolve a `music` role's track_id (UUID) to text per `mode`.

    Modes (controlled by `MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT`):

    - ``off``       — return raw content (UUID hex). `kv` ignored.
                       Legacy fallback; safest when conditioning is unclear.
    - ``minimal``   — `'"Title" by Artist (Album, Year)'`  ← all music turns
    - ``selective`` — handled by `fmt_chat_history` directly (renders relevant
                       priors as `minimal` and DROPS irrelevant priors). This
                       function is invoked with `mode="minimal"` for relevant
                       turns under selective mode.

    Falls back to raw content when `kv` is None, the lookup misses, or the
    metadata can't be assembled into a non-empty rendering.
    """
    if mode == "off" or kv is None:
        return content
    try:
        meta = kv.get_track_meta(content)
    except Exception:
        return content
    if not meta:
        return content
    title = _first(meta.get("track_name"), fallback="")
    artist = _first(meta.get("artist_name"), fallback="")
    parts: list[str] = []
    if title:
        parts.append(f'"{title}"')
    if artist:
        parts.append(f"by {artist}")

    # minimal: include album + year (no tags)
    album = _first(meta.get("album_name"), fallback="")
    release_date = _first(meta.get("release_date"), fallback="")
    year = release_date[:4] if release_date and len(release_date) >= 4 and release_date[:4].isdigit() else ""
    paren_bits: list[str] = []
    if album and album != "unknown":
        paren_bits.append(album)
    if year:
        paren_bits.append(year)
    if paren_bits:
        parts.append(f"({', '.join(paren_bits)})")
    rendered = " ".join(parts).strip()
    return rendered or content


def fmt_chat_history(
    chat_history: list[dict] | None,
    max_turns: int = 10,
    *,
    music_only: bool = False,
    kv: Any = None,
    selective_relevant_ids: set[str] | None = None,
) -> str:
    """Render the tail of `chat_history` as `role: content` lines.

    `music_only=True` drops `assistant` natural-language messages, keeping
    only `user` queries and `music` track lists. For retrieval-leaning tasks,
    prior nat-lang from the assistant is noise relative to the structured
    music + user-query signal. Off by default to keep `pas` / baseline `auto`
    prompt unchanged.

    `kv` is an optional `KVStore` handle. When supplied AND env
    `MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT` is set to a non-`off` mode,
    `music` role contents (track_ids = UUIDs) are resolved to metadata
    text. **Default mode `off`** preserves the UUID-only legacy behavior;
    expansion is opt-in per env. See `_format_music_content` docstring for
    mode semantics.

    `selective_relevant_ids` is consulted ONLY when the active expand mode
    is `selective`: prior music tracks whose `content` is in this set are
    expanded as in `minimal` mode (`"Title" by Artist (Album, Year)`); the
    rest are **DROPPED entirely** from the rendered output so the LLM sees a
    clean, shorter prompt with only relevant priors (no `[omitted]` noise).
    The signature docstring tells the LLM upfront that the chat history is
    pre-filtered and may not be complete. When the relevance set is empty /
    None in selective mode, behavior degrades to `off` (all music turns shown
    as raw UUIDs unchanged) so the prompt never silently drops everything
    when conditioning is misconfigured.
    """
    if not chat_history:
        return ""
    if music_only:
        chat_history = [m for m in chat_history if m.get("role") in ("user", "music")]
    tail = chat_history[-max_turns:]
    expand_mode = _resolve_chat_expand_mode() if kv is not None else "off"

    # selective mode: show relevant priors as `minimal`, drop others entirely.
    selective = expand_mode == "selective"
    selective_active = selective and bool(selective_relevant_ids)

    out: list[str] = []
    for m in tail:
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        # `Unknown message` is a dataset-side training placeholder (~5.65% of
        # GT assistant turns). Exposing it verbatim teaches the LLM that
        # `Unknown message` is an acceptable prior-turn response — replace
        # with a neutral marker so the LLM treats the slot as "no response
        # logged" instead.
        if role == "assistant" and content.strip() == "Unknown message":
            content = "[no response logged]"
        if role == "music":
            if selective_active:
                if content in selective_relevant_ids:
                    content = _format_music_content(content, kv, mode="minimal")
                else:
                    # Drop irrelevant music turn entirely — LLM is told upfront
                    # via signature docstring that chat_history is pre-filtered.
                    continue
            elif selective:
                # selective mode but no relevance set provided → degrade to off
                # (raw UUID) so we don't accidentally drop everything.
                content = _format_music_content(content, kv, mode="off")
            else:
                content = _format_music_content(content, kv, mode=expand_mode)
        out.append(f"{role}: {content}")
    return "\n".join(out)


def fmt_conversation_goal(conversation_goal: dict | None) -> str:
    """Render listener_goal + specificity (code → human text). `category` is
    available as a separate field — consumers that need it pass it explicitly."""
    if not conversation_goal:
        return "unknown"
    goal = conversation_goal.get("listener_goal", "unknown")
    spec_code = conversation_goal.get("specificity", "")
    spec = _SPECIFICITY_MAP.get(spec_code, spec_code)
    return f"goal: {goal}\nspecificity: {spec}"


def fmt_user_profile(user_profile: dict | None) -> str:
    if not user_profile:
        return "unknown"
    lines = []
    age = user_profile.get("age")
    if age is not None:
        lines.append(f"age: {age}")
    else:
        age_group = user_profile.get("age_group")
        if age_group not in (None, ""):
            lines.append(f"age_group: {age_group}")
    for f in _PROFILE_FIELDS[1:]:
        v = user_profile.get(f)
        lines.append(f"{f}: {v if v not in (None, '', []) else 'unknown'}")
    return "\n".join(lines)


__all__ = [
    "Provider",
    "ensure_lm_configured",
    "apply_rate_limit",
    "min_interval_sec",
    "RateLimitedPredictor",
    "fmt_chat_history",
    "fmt_conversation_goal",
    "fmt_user_profile",
    "_first",
]
