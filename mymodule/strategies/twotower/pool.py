"""TwoTower pool: LoRA-fine-tuned Qwen3-Embedding query → cached doc vecs search.

Inference path:
  1. Compose query body via `text_compose.compose_query` (KV + tag_freq).
  2. Encode with `QwenEmbedderTorch` carrying a PEFT adapter.
  3. Cosine top-K over pre-built doc cache — two search backends:
     a. GPU/CPU exact matmul (default): ``q_t @ doc_t.T`` on torch device
     b. LanceDB IVF_HNSW_SQ ANN (opt-in): ``MYMODULE_TWOTOWER_ANN=1``
  4. Drop session-seen tracks, return top `n_candidates`.

Default artifact layout (matches sasrec convention — colocated with the module):

    mymodule/strategies/twotower/ckpt/twotower/       (PEFT adapter dir)
    mymodule/strategies/twotower/data/doc_cache.npz   (adapter-encoded tracks)
    mymodule/strategies/twotower/data/doc_cache.ann.lancedb/  (ANN index, built on demand)

Custom variants supply `adapter_path` / `doc_cache_path` kwargs; the
`twotower_<variant>` alias in `_aliases.py` maps to per-variant paths.

Env vars:
    MYMODULE_TWOTOWER_DEVICE    cpu|cuda  (default: autodetect)
    MYMODULE_TWOTOWER_ANN       1=LanceDB IVF_HNSW_SQ ANN, 0=exact matmul (default)
    MYMODULE_TWOTOWER_ANN_NPROBES  ANN nprobes (default 200, 47K tracks recall≈99.7%)

To free VRAM for other GPU models, set:
    MYMODULE_TWOTOWER_DEVICE=cpu MYMODULE_TWOTOWER_ANN=1
    → encoder + search fully on CPU, VRAM ≈ 0
"""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from mymodule.strategies.pool.base import BasePool
from mymodule.strategies.twotower.encoder import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_LEN,
    QwenEmbedderTorch,
    autodetect_device,
    resolve_max_body_tokens,
)
from mymodule.strategies.twotower.text_compose import build_tag_freq, compose_query, load_tag_freq
from mymodule.utils.seen import extract_session_seen_tracks, take_unseen_pairs

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER = _MODULE_DIR / "ckpt" / "twotower"
DEFAULT_DOC_CACHE = _MODULE_DIR / "data" / "doc_cache.npz"
DEFAULT_SUMMARY_MAP = _MODULE_DIR / "data" / "query_summaries.jsonl"
_DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def build_query_body(
    user_query: str,
    chat_history: list[dict],
    conversation_goal: dict | None,
    *,
    kv,
    tag_freq: dict,
    tokenizer,
    max_body_tokens: int,
    summary_map: dict[str, str] | None,
    ablate=None,
) -> str:
    """Shared query-body composition for serving (TwoTowerPool) AND in-training
    val/devset eval, so both render byte-identical query bodies — same summary
    injection (keyed by `summary_key`), same `max_body_tokens` budget, same
    overflow guarantee inside `compose_query`. Returns the body WITHOUT the
    `Instruct:`/`Query:` prefix; callers add it via `format_query` at encode time.

    This is the single source of truth referenced by `.claude/skills/train/twotower.md`:
    train-mid devset eval and production inference MUST go through here so they can
    never drift on budget / summary / truncation again.
    """
    session_summary = None
    if summary_map:
        from mymodule.strategies.twotower.query_summary import summary_key

        session_summary = summary_map.get(summary_key(chat_history, user_query))
    return compose_query(
        user_query,
        chat_history,
        conversation_goal,
        kv,
        tag_freq,
        session_summary=session_summary,
        tokenizer=tokenizer,
        max_body_tokens=max_body_tokens,
        ablate=ablate,
    )


def _resolve_dtype(dtype: str) -> str:
    dtype_name = os.getenv("MYMODULE_TWOTOWER_DTYPE", dtype).strip().lower()
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(f"MYMODULE_TWOTOWER_DTYPE must be one of {sorted(_DTYPE_MAP)}, got {dtype_name!r}")
    return dtype_name


def _load_doc_cache(path: Path, adapter: Path) -> tuple[list[str], np.ndarray]:
    """Load (track_ids, doc_vecs) from a `.npz` cache and sanity-check the adapter binding."""
    if not path.exists():
        raise FileNotFoundError(
            f"Doc cache not found: {path}. Build it with "
            f"`python -m mymodule.strategies.twotower.doc_cache --adapter {adapter} --out {path}`."
        )
    z = np.load(path, allow_pickle=True)
    tids = z["track_ids"].astype(str).tolist()
    vecs = z["doc_vecs"].astype(np.float32)
    try:
        meta = json.loads(str(z["meta"].item())) if "meta" in z.files else {}
    except Exception:
        meta = {}
    cached_adapter = meta.get("adapter")
    if cached_adapter is None:
        logger.warning(
            f"[twotower-pool] doc cache {path} was built without an adapter (meta.adapter is null). "
            "Adapter retrieval against a base-encoded cache mixes vector spaces — rebuild the cache."
        )
    elif Path(cached_adapter).resolve() != adapter.resolve() and Path(cached_adapter).name != adapter.name:
        logger.warning(
            f"[twotower-pool] doc cache adapter ({cached_adapter}) ≠ runtime adapter ({adapter}). "
            "Retrieval will run in a mismatched vector space — rebuild the cache to fix."
        )
    logger.info(f"[twotower-pool] loaded doc cache {path} ({vecs.shape[0]} tracks × {vecs.shape[1]} dim)")
    return tids, vecs


class TwoTowerPool(BasePool):
    """LoRA Qwen3-Embedding two-tower pool — query encoded online, docs precomputed."""

    def __init__(
        self,
        n_candidates: int = 100,
        adapter_path: str | Path | None = None,
        doc_cache_path: str | Path | None = None,
        summary_map_path: str | Path | None = None,
        base_model: str = DEFAULT_BASE_MODEL,
        max_len: int = DEFAULT_MAX_LEN,
        device: str | None = None,
        encode_batch: int = 16,
        dtype: str = "float32",
        load_docs: bool = True,
        query_cache_path: str | Path | list | None = None,
        **kwargs,
    ) -> None:
        self.n_candidates = n_candidates
        self.encode_batch = encode_batch
        self.device = device or autodetect_device()

        self.adapter_path = Path(adapter_path) if adapter_path else DEFAULT_ADAPTER
        self.doc_cache_path = Path(doc_cache_path) if doc_cache_path else DEFAULT_DOC_CACHE
        self.summary_map_path = Path(summary_map_path) if summary_map_path else DEFAULT_SUMMARY_MAP
        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"TwoTower adapter not found: {self.adapter_path}. "
                "Train with: uv run python -m mymodule.strategies.twotower.train"
            )

        dtype = _resolve_dtype(dtype)
        # Lazy-embedder params: the (heavy, GPU) encoder is built on first actual
        # encode. With query_cache_path set + all body hashes cached, it is NEVER
        # built → a 5-fold 8B bag serves on one GPU without OOM (5×30GB would not fit).
        self._embedder_params = dict(
            base_model=base_model,
            adapter_path=self.adapter_path,
            device=self.device,
            dtype=_DTYPE_MAP[dtype],
            max_len=max_len,
        )
        # body-hash → query vec map (built from query_cache.py npz). Skips the encoder.
        self._query_cache = self._load_query_caches(query_cache_path) if query_cache_path else None

        if self._query_cache is None:
            # Production path UNCHANGED: eager encoder, tokenizer from it.
            self.embedder = QwenEmbedderTorch(**self._embedder_params)
            self._compose_tokenizer = self.embedder.tokenizer
        else:
            # Cache-read serving: tokenizer-only (cheap, CPU) for byte-identical body
            # composition; encoder deferred (lazy) so a cache hit never loads the 8B.
            from transformers import AutoTokenizer

            self.embedder = None
            self._compose_tokenizer = AutoTokenizer.from_pretrained(
                base_model, padding_side="left", truncation_side="left", trust_remote_code=True
            )
        self.max_body_tokens = resolve_max_body_tokens(self._compose_tokenizer, max_len)

        # ANN config (MYMODULE_TWOTOWER_ANN=1 → LanceDB IVF_HNSW_SQ)
        use_ann = os.getenv("MYMODULE_TWOTOWER_ANN", "0") == "1"
        self._ann_nprobes = int(os.getenv("MYMODULE_TWOTOWER_ANN_NPROBES", "200"))
        self._ann_table = None

        # `load_docs=False` → query-only mode: skip the 47K-track doc cache so a
        # caller (e.g. query_cache.py building the query-side OOF cache) can
        # compose/encode queries with a fold adapter whose doc cache is not built
        # yet. `generate*` then raise — only compose_query_body / encode_query_bodies
        # are usable. The doc-side path is otherwise unchanged.
        if load_docs:
            self._track_ids, self._doc_vecs = _load_doc_cache(self.doc_cache_path, self.adapter_path)
            if use_ann:
                from mymodule.strategies.twotower.ann_index import (
                    ann_index_path,
                    build_or_load_ann_index,
                )

                idx_path = ann_index_path(self.doc_cache_path)
                self._ann_table = build_or_load_ann_index(self._track_ids, self._doc_vecs, idx_path)
                # ANN search needs no GPU doc tensor → saves VRAM
                self._doc_t = None
            else:
                self._doc_t = torch.from_numpy(self._doc_vecs).to(self.device)
            logger.info(
                f"TwoTowerPool: adapter={self.adapter_path.name} cache={self.doc_cache_path.name} "
                f"n_tracks={len(self._track_ids)} dim={self._doc_vecs.shape[1]} "
                f"max_body_tokens={self.max_body_tokens} device={self.device} "
                f"search={'ann(nprobes=' + str(self._ann_nprobes) + ')' if use_ann else 'exact'}"
            )
        else:
            self._track_ids = []
            self._doc_vecs = np.empty((0, 0), dtype=np.float32)
            self._doc_t = None
            logger.info(
                f"TwoTowerPool (query-only): adapter={self.adapter_path.name} "
                f"max_body_tokens={self.max_body_tokens} device={self.device}"
            )

    @staticmethod
    def _load_query_caches(query_cache_path) -> dict[str, np.ndarray]:
        """Build body_hash → query_vec from one or more query_cache.py npz files.

        Accepts a path, a glob string, or a list. Merges all matches (body-hash keys
        are dataset-agnostic, so devset + blindset caches coexist). A cache without
        `query_body_hashes` (pre-hash build) is skipped with a warning.
        """
        import glob as _glob

        paths: list[Path] = []
        items = query_cache_path if isinstance(query_cache_path, (list, tuple)) else [query_cache_path]
        for it in items:
            s = str(it)
            paths.extend(Path(p) for p in (_glob.glob(s) if any(c in s for c in "*?[") else [s]))
        cache: dict[str, np.ndarray] = {}
        for p in paths:
            if not Path(p).exists():
                continue
            z = np.load(p, allow_pickle=True)
            if "query_body_hashes" not in z.files:
                logger.warning(f"[twotower-pool] {p} has no query_body_hashes (old build) — skip; rebuild to use.")
                continue
            hashes = z["query_body_hashes"].astype(str).tolist()
            vecs = z["query_vecs"].astype(np.float32)
            for h, v in zip(hashes, vecs):
                cache[h] = v
        if not cache:
            raise FileNotFoundError(
                f"query_cache_path={query_cache_path} resolved to no usable (body-hashed) caches. "
                "Rebuild with the body-hash version of query_cache.py."
            )
        logger.info(f"[twotower-pool] query cache: {len(cache)} body-hashed vecs (encoder skipped on hits)")
        return cache

    def _get_embedder(self):
        """Lazy encoder. Built only when actually needed (cache miss / no query cache)."""
        if self.embedder is None:
            logger.info(f"[twotower-pool] lazy-loading encoder ({self.adapter_path.name})")
            self.embedder = QwenEmbedderTorch(**self._embedder_params)
        return self.embedder

    @cached_property
    def _kv(self):
        try:
            from mymodule.feature.kvdb import KVStore

            return KVStore.open(read_only=True)
        except Exception as e:
            logger.warning(f"[twotower-pool] KV unavailable ({type(e).__name__}: {e}); using raw user_query.")
            return None

    @cached_property
    def _tag_freq(self) -> dict[str, int]:
        try:
            freq = load_tag_freq()
            logger.info(f"[twotower-pool] loaded tag_freq cache ({len(freq)} tags)")
            return freq
        except (FileNotFoundError, RuntimeError):
            logger.info("[twotower-pool] tag_freq cache missing — building from track metadata")
            from datasets import load_dataset

            ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
            return build_tag_freq(ds)

    @cached_property
    def _summary_map(self) -> dict[str, str]:
        """Load pre-baked `(chat_history, user_msg) → session_summary` JSONL.

        Missing or empty → empty dict (Session-so-far line auto-skipped).
        Build with: `python -m mymodule.strategies.twotower.query_summary`.
        """
        from mymodule.strategies.twotower.query_summary import load_query_summaries

        if not self.summary_map_path.exists():
            logger.info(
                f"[twotower-pool] query_summary cache absent at {self.summary_map_path} — "
                "'Session so far:' will be skipped. Pre-bake with `python -m "
                "mymodule.strategies.twotower.query_summary`."
            )
            return {}
        m = load_query_summaries(self.summary_map_path)
        logger.info(f"[twotower-pool] loaded {len(m)} session summaries from {self.summary_map_path}")
        return m

    def _lookup_summary(self, chat_history: list[dict], user_query: str) -> str | None:
        if not self._summary_map:
            return None
        from mymodule.strategies.twotower.query_summary import summary_key

        return self._summary_map.get(summary_key(chat_history, user_query))

    def compose_query_body(
        self,
        user_query: str,
        chat_history: list[dict],
        conversation_goal: dict | None,
    ) -> str:
        """Compose the query text body exactly as the retrieval path does.

        Exposed so the query-side cache builder (`query_cache.py`) renders byte-for-byte
        the same body the inference pool would, guaranteeing the cached query vectors
        match online retrieval.
        """
        return build_query_body(
            user_query,
            chat_history,
            conversation_goal,
            kv=self._kv,
            tag_freq=self._tag_freq,
            tokenizer=self._compose_tokenizer,
            max_body_tokens=self.max_body_tokens,
            summary_map=self._summary_map,
        )

    def encode_query_bodies(
        self, bodies: list[str], batch_size: int | None = None, normalize: bool = True
    ) -> np.ndarray:
        """Batch-encode composed query bodies with the loaded adapter → (N, D) L2-normalized."""
        return self._get_embedder().encode_queries(
            bodies, batch_size=batch_size or self.encode_batch, normalize=normalize
        )

    def _retrieve_with_scores(
        self,
        user_query: str,
        chat_history: list[dict],
        conversation_goal: dict | None,
    ) -> list[tuple[str, float]]:
        if self._ann_table is None and self._doc_t is None:
            raise RuntimeError(
                "TwoTowerPool was constructed with load_docs=False (query-only mode); "
                "retrieval/generate is unavailable. Build the doc cache and reconstruct with load_docs=True."
            )
        q_body = self.compose_query_body(user_query, chat_history, conversation_goal)
        if self._query_cache is not None:
            import hashlib

            h = hashlib.sha256(q_body.encode("utf-8")).hexdigest()
            cached = self._query_cache.get(h)
            if cached is not None:
                q_vec = cached.reshape(1, -1)
            else:
                logger.warning("[twotower-pool] query cache MISS -> lazy-encoding (loads 8B); rebuild cache.")
                q_vec = self.encode_query_bodies([q_body], batch_size=1, normalize=True)
        else:
            q_vec = self.encode_query_bodies([q_body], batch_size=1, normalize=True)

        seen = extract_session_seen_tracks(chat_history)
        k = self.n_candidates + len(seen)

        if self._ann_table is not None:
            from mymodule.strategies.twotower.ann_index import search_ann

            scored = search_ann(self._ann_table, q_vec[0], k=k, nprobes=self._ann_nprobes)
        else:
            q_t = torch.from_numpy(q_vec).to(self.device)
            with torch.no_grad():
                sims = (q_t @ self._doc_t.T)[0]
                top = sims.topk(min(k, sims.shape[0]))
            scored = [
                (self._track_ids[int(idx)], float(score)) for idx, score in zip(top.indices.cpu(), top.values.cpu())
            ]

        return take_unseen_pairs(scored, seen, self.n_candidates)

    def generate(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[str]:
        return [tid for tid, _ in self._retrieve_with_scores(user_query, chat_history, conversation_goal)]

    def generate_with_scores(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[tuple[str, float]]:
        return self._retrieve_with_scores(user_query, chat_history, conversation_goal)
