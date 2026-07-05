"""Embedding clients + qemb retrieval pipeline.

`OllamaEmbedder` — thin wrapper around Ollama `/api/embeddings`.
`OpenAIEmbedder` — thin wrapper around OpenAI `/v1/embeddings`.
Both satisfy the `Embedder` protocol and are selected via `MYMODULE_EMB_PROVIDER`.

`InstructedQueryEmbedder` — composes a doc-distribution-aligned query string from prior music turns
(`title: ..., artist: ..., album: ...` lowercase, matching talkpl's
`entity_str()` format) plus the lowercase current user_query, then
passes raw text to an `Embedder` (no instructed-query prefix — talkpl's
doc-side `metadata-qwen3` was built without one).

The instructed-prefix / DSPy path that previously lived here has been
removed; raw text + entity_str + rich doc performed best in ablations.

Environment variables
---------------------
    MYMODULE_EMB_PROVIDER       embedding provider: "ollama" (default) or "openai"
    MYMODULE_EMB_OLLAMA_URL     Ollama server base URL (default http://localhost:11434)
    MYMODULE_EMB_MODEL          embedding model name (default qwen3-embedding:0.6b for ollama;
                                legacy fallback for openai)
    MYMODULE_EMB_OPENAI_URL     OpenAI-compatible embedding base URL (openai provider)
    MYMODULE_EMB_OPENAI_MODEL   OpenAI-compatible embedding model (preferred for openai)
    MYMODULE_EMB_OPENAI_API_KEY API key for embedding endpoint
    MYMODULE_EMB_TIMEOUT        request timeout seconds (default 30)
    MYMODULE_LLM_OPENAI_URL     fallback OpenAI-compatible base URL (openai provider)
    MYMODULE_LLM_OPENAI_API_KEY fallback API key for OpenAI-compatible endpoint
"""

import os
import re
import threading
import time
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import requests
from loguru import logger

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_TIMEOUT = 30
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5


def normalize_model_id(model: str) -> str:
    """`qwen3-embedding:0.6b` → `qwen3-embedding-0.6b` (prefix-/key-safe)."""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "-", model.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s.lower()


class OllamaEmbedder:
    """Ollama `/api/embeddings` wrapper.

    `embed(text, instruction=None)` — encode raw text. The optional
    `instruction` arg prepends the Qwen3-Embedding instructed-query
    format `Instruct: {inst}\\nQuery: {text}` for callers that need it.
    The production pipeline does **not** use `instruction` because
    talkpl's doc-side embeddings were built from raw text — putting the
    query in instructed-prefix form misaligns the distribution.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.url = (url or os.getenv("MYMODULE_EMB_OLLAMA_URL") or DEFAULT_URL).rstrip("/")
        self.model = model or os.getenv("MYMODULE_EMB_MODEL") or DEFAULT_MODEL
        self.timeout = float(timeout if timeout is not None else os.getenv("MYMODULE_EMB_TIMEOUT", DEFAULT_TIMEOUT))
        self.model_id = normalize_model_id(self.model)

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=min(self.timeout, 5))
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str, instruction: str | None = None) -> np.ndarray:
        """Text → float32 embedding. Retries up to 3x on transport errors."""
        prompt = f"Instruct: {instruction}\nQuery: {text}" if instruction else text
        body = {"model": self.model, "prompt": prompt}
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = requests.post(f"{self.url}/api/embeddings", json=body, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                vec = data.get("embedding")
                if not vec:
                    raise ValueError(f"empty embedding in response: {data}")
                return np.asarray(vec, dtype=np.float32)
            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE**attempt)
                continue
        raise RuntimeError(f"Ollama embed failed after {_MAX_RETRIES} retries: {last_err}")

    def __repr__(self) -> str:
        return f"OllamaEmbedder(url={self.url}, model={self.model}, model_id={self.model_id})"


# ---------------------------------------------------------------------------
# Embedder protocol + OpenAI provider + factory
# ---------------------------------------------------------------------------

EmbProvider = Literal["ollama", "openai"]


@runtime_checkable
class Embedder(Protocol):
    """Minimal protocol that both OllamaEmbedder and OpenAIEmbedder satisfy."""

    url: str
    model: str
    model_id: str

    def health(self) -> bool: ...
    def embed(self, text: str, instruction: str | None = None) -> np.ndarray: ...


class OpenAIEmbedder:
    """OpenAI-compatible `/v1/embeddings` wrapper.

    Works with any OpenAI-compatible server (vLLM, GitHub Models,
    OpenAI cloud, ...). Uses `requests` directly — no SDK dependency.

    Unlike `OllamaEmbedder`, an explicit model is **required** because the
    correct model name depends on the endpoint. Prefer
    `MYMODULE_EMB_OPENAI_MODEL`; `MYMODULE_EMB_MODEL` remains as a legacy
    fallback for existing environments.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.url = (
            url
            or os.getenv("MYMODULE_EMB_OPENAI_URL")
            or os.getenv("MYMODULE_LLM_OPENAI_URL")
            or "https://models.github.ai/inference"
        ).rstrip("/")
        self.model = model or os.getenv("MYMODULE_EMB_OPENAI_MODEL") or os.getenv("MYMODULE_EMB_MODEL")
        if not self.model:
            raise ValueError(
                "MYMODULE_EMB_OPENAI_MODEL or MYMODULE_EMB_MODEL is required for openai embedding provider. "
                "Set it in .env or pass a model explicitly."
            )
        self.api_key = (
            api_key
            or os.getenv("MYMODULE_EMB_OPENAI_API_KEY")
            or os.getenv("MYMODULE_LLM_OPENAI_API_KEY")
            or "dummy"
        )
        self.timeout = float(timeout if timeout is not None else os.getenv("MYMODULE_EMB_TIMEOUT", DEFAULT_TIMEOUT))
        self.model_id = normalize_model_id(self.model)

    def health(self) -> bool:
        try:
            r = requests.get(
                f"{self.url}/v1/models",
                headers=self._headers(),
                timeout=min(self.timeout, 5),
            )
            return r.status_code == 200
        except Exception:
            return False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, text: str, instruction: str | None = None) -> np.ndarray:
        prompt = f"Instruct: {instruction}\nQuery: {text}" if instruction else text
        body = {"model": self.model, "input": prompt}
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = requests.post(
                    f"{self.url}/v1/embeddings",
                    json=body,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                vecs = data.get("data")
                if not vecs or not vecs[0].get("embedding"):
                    raise ValueError(f"empty embedding in response: {data}")
                return np.asarray(vecs[0]["embedding"], dtype=np.float32)
            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE**attempt)
                continue
        raise RuntimeError(f"OpenAI embed failed after {_MAX_RETRIES} retries: {last_err}")

    def embed_batch(
        self,
        texts: list[str],
        instruction: str | None = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Batch variant — sends `input: list[str]` per request.

        Most OpenAI-compatible servers (vLLM, OpenAI cloud) accept a list and
        return embeddings in the same order. Falls back to single-text retries
        on per-batch errors.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prompts = [f"Instruct: {instruction}\nQuery: {t}" if instruction else t for t in texts]
        out: list[np.ndarray] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            body = {"model": self.model, "input": chunk}
            last_err: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    r = requests.post(
                        f"{self.url}/v1/embeddings",
                        json=body,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
                    r.raise_for_status()
                    data = r.json()
                    vecs = data.get("data") or []
                    if len(vecs) != len(chunk):
                        raise ValueError(f"embedding count mismatch: requested {len(chunk)} got {len(vecs)}")
                    out.extend(np.asarray(v["embedding"], dtype=np.float32) for v in vecs)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_BACKOFF_BASE**attempt)
                    continue
            if last_err is not None:
                raise RuntimeError(
                    f"OpenAI embed_batch failed at chunk {i}-{i + len(chunk)} after {_MAX_RETRIES} retries: {last_err}"
                )
        return np.stack(out)

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(url={self.url}, model={self.model}, model_id={self.model_id})"


def resolve_emb_provider() -> EmbProvider:
    """Read `MYMODULE_EMB_PROVIDER`; default 'ollama'."""
    raw = os.getenv("MYMODULE_EMB_PROVIDER", "ollama").strip().lower()
    if raw not in ("ollama", "openai"):
        logger.warning(f"Unknown embedding provider '{raw}'; expected 'ollama' or 'openai'. Defaulting to 'ollama'.")
        return "ollama"
    return raw


def get_embedder(provider: EmbProvider | None = None, **kwargs: Any) -> Embedder:
    """Factory: return an OllamaEmbedder or OpenAIEmbedder based on provider.

    `provider` defaults to `resolve_emb_provider()` (env-driven). Remaining
    kwargs are forwarded to the concrete constructor.
    """
    if provider is None:
        provider = resolve_emb_provider()
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    return OllamaEmbedder(**kwargs)


# ---------------------------------------------------------------------------
# Query embedding — talkpl entity_str format (winner)
# ---------------------------------------------------------------------------


def _compose_talkpl_metadata_query(
    user_query: str,
    chat_history: list[dict] | None,
    kv_store: Any | None,
) -> str:
    """Build a doc-distribution-aligned query string (legacy compose).

    Mirrors talkpl's `utils.entity_str(meta_info)` (used to build the
    `metadata-qwen3` doc embeddings) for every prior music turn:

        title: {track_name[0].lower()}, artist: {artist_name lowercase joined}, album: {album_name lowercase joined}

    Appends the current user_query (lowercase) as a final line. Falls
    back to raw user_query when kv_store is unavailable or no music
    turns exist (e.g. T1 cold-start) — empty entity_str block + raw
    query is the same behavior the old Stage A path had.
    """
    parts: list[str] = []
    if chat_history and kv_store is not None:
        for msg in chat_history:
            if msg.get("role") != "music":
                continue
            try:
                meta = kv_store.get_track_meta(msg.get("content", ""))
            except Exception:
                meta = None
            if not meta:
                continue
            tn = meta.get("track_name") or []
            an = meta.get("artist_name") or []
            al = meta.get("album_name") or []
            title = (tn[0] if isinstance(tn, list) and tn else str(tn or "")).strip().lower()
            artist = ", ".join(str(x) for x in (an if isinstance(an, list) else [an]) if x).lower()
            album = ", ".join(str(x) for x in (al if isinstance(al, list) else [al]) if x).lower()
            if title or artist or album:
                parts.append(f"title: {title}, artist: {artist}, album: {album}")
    uq = (user_query or "").strip().lower()
    if uq:
        parts.append(uq)
    return "\n".join(parts) if parts else (user_query or "")


def _compose_talkpl_metadata_rich_query(
    user_query: str,
    chat_history: list[dict] | None,
    kv_store: Any | None,
) -> str:
    """Rich-aligned query composition — Q-T distribution match.

    Mirrors `mymodule.feature.kvdb._compose_track_metadata_rich_text` (used to
    build `metadata_rich-qwen3` track embeddings) for every prior music turn:

        title: x, artist: y, album: z, year: YYYY, popularity: N, tags: t1, t2, ...

    The current `user_query` is appended as the final line (lowercase).
    Falls back to raw user_query when kv_store is unavailable or no music
    turns exist (cold-start).

    OPT-IN — enable by setting env `MYMODULE_QEMB_QUERY_COMPOSITION=rich`.
    The default remains the legacy compose.
    """
    parts: list[str] = []
    if chat_history and kv_store is not None:
        for msg in chat_history:
            if msg.get("role") != "music":
                continue
            try:
                meta = kv_store.get_track_meta(msg.get("content", ""))
            except Exception:
                meta = None
            if not meta:
                continue
            tn = meta.get("track_name") or []
            an = meta.get("artist_name") or []
            al = meta.get("album_name") or []
            title = (tn[0] if isinstance(tn, list) and tn else str(tn or "")).strip().lower()
            artist = ", ".join(str(x) for x in (an if isinstance(an, list) else [an]) if x).lower()
            album = ", ".join(str(x) for x in (al if isinstance(al, list) else [al]) if x).lower()

            release_date = meta.get("release_date") or []
            release_first = release_date[0] if isinstance(release_date, list) and release_date else release_date
            year = ""
            if isinstance(release_first, str) and len(release_first) >= 4 and release_first[:4].isdigit():
                year = release_first[:4]

            popularity = meta.get("popularity")
            pop = str(popularity) if popularity not in (None, "", []) else ""

            tag_list = meta.get("tag_list") or []
            if not isinstance(tag_list, list):
                tag_list = [tag_list]
            tags = ", ".join(str(t).strip().lower() for t in tag_list[:5] if t).strip()

            line = f"title: {title}, artist: {artist}, album: {album}"
            if year:
                line += f", year: {year}"
            if pop:
                line += f", popularity: {pop}"
            if tags:
                line += f", tags: {tags}"
            parts.append(line)
    uq = (user_query or "").strip().lower()
    if uq:
        parts.append(uq)
    return "\n".join(parts) if parts else (user_query or "")


def _resolve_query_compose() -> tuple[str, Any]:
    """Read `MYMODULE_QEMB_QUERY_COMPOSITION` env. Returns (vector_type_prefix, fn).

    - default `legacy` → (`talkpl_metadata`, `_compose_talkpl_metadata_query`)
    - opt-in `rich` → (`talkpl_metadata_rich`, `_compose_talkpl_metadata_rich_query`)
    Unknown values silently degrade to legacy so typos don't change retrieval shape.
    """
    raw = os.getenv("MYMODULE_QEMB_QUERY_COMPOSITION", "legacy").strip().lower()
    if raw == "rich":
        return "talkpl_metadata_rich", _compose_talkpl_metadata_rich_query
    return "talkpl_metadata", _compose_talkpl_metadata_query


class InstructedQueryEmbedder:
    """Conversation → entity_str-format query → embedding vector.

    For each prior music turn, looks up track metadata via
    KVStore and emits a single `title: ..., artist: ..., album: ...`
    lowercase line — matching the doc-side `metadata-qwen3` distribution.
    Concatenates all prior music turn lines + the lowercase current
    user_query, then passes raw text to an `Embedder` (no instructed
    prefix — talkpl's doc was built without one).

    Caches the resulting embedding under
    `query:embedding:{talkpl_metadata-{model_id}}:{sha256-of-composed-text}`
    in the shared KVStore. Cache lookup uses the read-only KV instance;
    write requires a writable handle (single-process inference path) and is
    best-effort — if the writable open fails (e.g. another process holds
    the lock), the embedding is still returned, just not cached.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        provider: EmbProvider | None = None,
    ) -> None:
        self.raw = embedder or get_embedder(provider)
        self._kv: Any = None  # None = not tried, False = failed, KVStore = ready
        self._kv_writable: Any = None  # same tri-state for write-mode handle

    def _get_kv(self) -> Any:
        if self._kv is None:
            try:
                from mymodule.feature.kvdb import KVStore

                self._kv = KVStore.open(read_only=True)
            except Exception as e:
                logger.warning(
                    f"[query-emb] KVStore open failed ({type(e).__name__}: {e}); "
                    "track-meta lookup disabled — falling back to raw user_query."
                )
                self._kv = False
        return self._kv if self._kv else None

    def _get_kv_writable(self) -> Any:
        """Lazy writable KV handle for cache writes. Returns None if unavailable.

        First failure (e.g. RocksDB lock held by another process) is sticky
        within this embedder instance — we don't retry on every call.
        """
        if self._kv_writable is None:
            try:
                from mymodule.feature.kvdb import KVStore

                self._kv_writable = KVStore.open(read_only=False)
            except Exception as e:
                logger.warning(
                    f"[query-emb] writable KV open failed ({type(e).__name__}: {e}); "
                    "cache writes disabled — embeddings will be recomputed each run."
                )
                self._kv_writable = False
        return self._kv_writable if self._kv_writable else None

    def embed_query(
        self,
        user_query: str,
        *,
        chat_history: list[dict] | None = None,
        # The four fields below are kept for caller-API compatibility
        # (qemb pool passes them) but are unused in the winner pipeline.
        user_profile: dict | None = None,
        conversation_goal: dict | None = None,
        goal_progress_assessments: Any = None,
        thought: str | None = None,
    ) -> np.ndarray:
        kv = self._get_kv()
        composition_name, compose_fn = _resolve_query_compose()
        text = compose_fn(user_query, chat_history, kv)
        model_id = self.raw.model_id

        # Cache hit (read-only KV instance is fine).
        if kv is not None:
            cached = kv.get_query_embedding(model_id, text, composition=composition_name)
            if cached is not None:
                return cached

        vec = self.raw.embed(text, instruction=None)

        # Best-effort cache write — only attempted if a writable KV is obtainable.
        kv_w = self._get_kv_writable()
        if kv_w is not None:
            try:
                kv_w.put_query_embedding(model_id, text, vec, composition=composition_name)
            except Exception as e:
                logger.warning(f"[query-emb] cache write skipped: {type(e).__name__}: {e}")
        return vec

    def __repr__(self) -> str:
        return f"InstructedQueryEmbedder(raw={self.raw})"


# ---------------------------------------------------------------------------
# Process-wide singleton — kept for response helpers that share the embedder
# ---------------------------------------------------------------------------

_SHARED_INSTRUCTED_LOCK = threading.Lock()
_SHARED_INSTRUCTED: InstructedQueryEmbedder | None = None


def get_shared_instructed_embedder(provider: EmbProvider | None = None) -> InstructedQueryEmbedder:
    """Lazy process-wide `InstructedQueryEmbedder` singleton.

    `provider` is routed through `get_embedder(provider)`. Defaults to
    `resolve_emb_provider()` (env-driven).
    """
    global _SHARED_INSTRUCTED
    if _SHARED_INSTRUCTED is not None:
        return _SHARED_INSTRUCTED
    with _SHARED_INSTRUCTED_LOCK:
        if _SHARED_INSTRUCTED is None:
            _SHARED_INSTRUCTED = InstructedQueryEmbedder(provider=provider)
    return _SHARED_INSTRUCTED
