"""RocksDB-based key-value store for track/user embeddings and metadata.

Point-lookup only. Use EmbeddingStore (LanceDB) for ANN search.

Prefix schema (all keys/values are bytes):
    track:embedding:{vector_type}:{track_id}              → float32 bytes
    track:meta:{track_id}                                  → json bytes
    track:crawl:{track_id}                                 → json bytes
    user:embedding:{emb_type}:{model_id}:{id_or_key}      → float32 bytes
    user:meta:{user_id}                                    → json bytes
    query:embedding:{vector_type}:{text_hash}              → float32 bytes
    _meta:dim:{vector_type}                                → ascii int bytes
    _meta:user_emb:{dim|fields|prompt_tpl}:{emb_type}:{model_id}
    _meta:query_emb:{dim|count|last_build}:{vector_type}
    _meta:crawl:{count|ok_count|built_at|source_jsonl}    → utf-8 bytes
    _meta:vector_types                                     → json list bytes
    _meta:built_at                                         → iso timestamp bytes

Usage:
    # One-time base build (track + user)
    uv run python -m mymodule.feature.kvdb --build

    # User meta embeddings (requires Ollama)
    uv run python -m mymodule.feature.kvdb --build-user-meta-emb

    # Track metadata-rich embeddings (requires Ollama, ~30 min)
    uv run python -m mymodule.feature.kvdb --build-metadata-rich

    # Query embedding cache prebuild (requires Ollama)
    uv run python -m mymodule.feature.kvdb --build-query-emb --dataset all

    # Crawl JSONL → KV ingest (no external HTTP, disk IO only)
    uv run python -m mymodule.feature.kvdb --build-crawl

    # In code
    kv = KVStore.open_or_build()
    vec = kv.get_track_embedding(track_id, "cf-bpr")
    meta = kv.get_track_meta(track_id)
    crawl = kv.get_track_crawl(track_id)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from loguru import logger
from rocksdict import AccessType, Options, Rdict, WriteBatch
from tqdm import tqdm

DB_PATH = Path(__file__).parent / ".rocksdb"

VECTOR_TYPES: list[str] = [
    "cf-bpr",
    "audio-laion_clap",
    "image-siglip2",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
    "metadata_rich-qwen3_embedding_0.6b",
]

# Subset of VECTOR_TYPES that the HuggingFace `Track-Embeddings` dataset carries
# directly — `--build` ingests these. The remaining types (currently just
# `metadata_rich-qwen3`) are built locally via `--build-metadata-rich`;
# bringing them into the HF loop would fail with a missing-column error.
_HF_VECTOR_TYPES: list[str] = [vt for vt in VECTOR_TYPES if vt != "metadata_rich-qwen3_embedding_0.6b"]

USER_VECTOR_TYPES: list[str] = ["cf-bpr"]

# --- Prefix schema (v3.1 consolidated) ---
#
# Unified user embedding prefix:
#   user:embedding:{emb_type}:{model_id}:{id_or_key}
#     emb_type = "cfbpr"  → id_or_key = user_id, model_id = "talkpl-bpr"
#     emb_type = "meta"   → id_or_key = meta_key (shared), model_id = Ollama slug e.g. "qwen3-embedding-0-6b"
#
_P_TRACK_EMB = "track:embedding:"
_P_TRACK_META = "track:meta:"
_P_TRACK_CRAWL = "track:crawl:"  # track:crawl:{track_id} → crawl JSONL record (json bytes)
_P_USER_EMB = "user:embedding:"  # user:embedding:{emb_type}:{model_id}:{id_or_key}
_P_USER_META = "user:meta:"
_P_QUERY_EMB = "query:embedding:"  # query:embedding:{vector_type}:{text_hash}
_META_DIM = "_meta:dim:"  # per track vector_type
_META_USER_EMB_DIM = "_meta:user_emb:dim:"  # per {emb_type}:{model_id}
_META_USER_EMB_FIELDS = "_meta:user_emb:fields:"  # per {emb_type}:{model_id}, json list
_META_USER_EMB_PROMPT = "_meta:user_emb:prompt_tpl:"  # per {emb_type}:{model_id}, utf-8
_META_QUERY_EMB_DIM = "_meta:query_emb:dim:"  # per query vector_type
_META_QUERY_EMB_COUNT = "_meta:query_emb:count:"  # per query vector_type, total entries written
_META_QUERY_EMB_LAST_BUILD = "_meta:query_emb:last_build:"  # per query vector_type, ISO timestamp
_META_CRAWL_COUNT = "_meta:crawl:count"  # ascii int — total records in last build
_META_CRAWL_OK_COUNT = "_meta:crawl:ok_count"  # ascii int — ok=True records in last build
_META_CRAWL_BUILT_AT = "_meta:crawl:built_at"  # ISO timestamp utf-8
_META_CRAWL_SOURCE_JSONL = "_meta:crawl:source_jsonl"  # source JSONL path utf-8 (provenance)
_META_VECTOR_TYPES = "_meta:vector_types"
_META_BUILT_AT = "_meta:built_at"

# model_id value for user:embedding (cfbpr family)
CFBPR_MODEL_ID = "talkpl-bpr"

# Query embedding cache namespace.
#
# `vector_type` is `{QUERY_COMPOSITION}-{model_id}` — mirrors track's
# `metadata_rich-qwen3_embedding_0.6b` style. `QUERY_COMPOSITION` names the
# composition recipe in `mymodule.feature.ollama_embed::_compose_talkpl_metadata_query`.
# Rename (e.g. `talkpl_metadata` → `talkpl_metadata_v2`) when its semantics
# change in a way that doesn't show up in the composed text.
QUERY_COMPOSITION = "talkpl_metadata"

# Fields (in order) used to build the meta_key. Users sharing the same tuple of
# values share one embedding → cold users can reuse the same key. `preferred_*`
# fields live in devset session.user_profile (not in User-Metadata HF), so build
# enriches from devset scan.
USER_META_KEY_FIELDS = [
    "age_group",
    "country_code",
    "gender",
    "preferred_language",
    "preferred_musical_culture",
]

USER_META_EMB_PROMPT_TPL = (
    "User profile: age group is {age_group}; country {country_name} ({country_code}); "
    "gender {gender}; preferred language {preferred_language}; prefers {preferred_musical_culture} music."
)

_BATCH_SIZE = 5000

# rocksdict instance cache. Keyed by (path, read_only) — read-only and write
# instances must not be aliased: read-only opens bypass the LOCK file (so they
# can coexist with an active writer in another process), while write opens
# require the file lock.
_INSTANCE_CACHE: "dict[tuple[str, bool], KVStore]" = {}


def _k_track_emb(vt: str, tid: str) -> bytes:
    return f"{_P_TRACK_EMB}{vt}:{tid}".encode()


def _k_track_meta(tid: str) -> bytes:
    return f"{_P_TRACK_META}{tid}".encode()


def _k_track_crawl(tid: str) -> bytes:
    return f"{_P_TRACK_CRAWL}{tid}".encode()


def _k_user_emb(emb_type: str, model_id: str, id_or_key: str) -> bytes:
    """Unified user embedding key: `user:embedding:{emb_type}:{model_id}:{id_or_key}`."""
    return f"{_P_USER_EMB}{emb_type}:{model_id}:{id_or_key}".encode()


def _k_user_meta(uid: str) -> bytes:
    return f"{_P_USER_META}{uid}".encode()


def query_vector_type(model_id: str, composition: str | None = None) -> str:
    """`{composition}-{model_id}` — track-style single string namespace.

    `composition` overrides the default `QUERY_COMPOSITION` constant (used by
    legacy callers). Pass `talkpl_metadata_rich` for the rich opt-in. Unknown
    values still go through unchanged — the caller is responsible.
    """
    return f"{composition or QUERY_COMPOSITION}-{model_id}"


def _hash_query_text(text: str) -> str:
    """Stable short fingerprint of composed query text (sha256 hex, 16 chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _k_query_emb(vector_type: str, text_hash: str) -> bytes:
    return f"{_P_QUERY_EMB}{vector_type}:{text_hash}".encode()


def build_meta_key(meta: dict) -> str:
    """Join USER_META_KEY_FIELDS values with `|`; missing fields become 'unknown'."""
    parts = []
    for f in USER_META_KEY_FIELDS:
        v = meta.get(f)
        parts.append(str(v) if v not in (None, "", []) else "unknown")
    return "|".join(parts)


def build_meta_prompt(meta: dict) -> str:
    """Build the LLM input prompt; missing fields become 'unknown'."""
    safe = {}
    for f in ["age_group", "country_code", "country_name", "gender", "preferred_language", "preferred_musical_culture"]:
        v = meta.get(f)
        safe[f] = str(v) if v not in (None, "", []) else "unknown"
    return USER_META_EMB_PROMPT_TPL.format(**safe)


_RICH_TAG_FIRST_N = 5


def _first_rich(values, fallback: str = "") -> str:
    """First non-empty value (for list-typed metadata fields)."""
    if isinstance(values, list) and values:
        for v in values:
            if v not in (None, ""):
                return str(v)
        return fallback
    if values in (None, ""):
        return fallback
    return str(values)


def _compose_track_metadata_rich_text(row: dict) -> str:
    """Compose rich entity_str for a track. Same lowercase doc-style format used
    by `EmbeddingStore.build_track_metadata_rich` so the resulting KV embeddings
    live in the same vector space as the LanceDB table."""
    title = _first_rich(row.get("track_name")).strip().lower()
    artist_raw = row.get("artist_name") or []
    if isinstance(artist_raw, list):
        artist = ", ".join(str(x) for x in artist_raw if x).lower()
    else:
        artist = str(artist_raw).lower()
    album_raw = row.get("album_name") or []
    if isinstance(album_raw, list):
        album = ", ".join(str(x) for x in album_raw if x).lower()
    else:
        album = str(album_raw).lower()
    line = f"title: {title}, artist: {artist}, album: {album}"
    extras: list[str] = []
    rd = row.get("release_date")
    if rd:
        extras.append(f"year: {str(rd)[:4]}")
    pop = row.get("popularity")
    if pop is not None:
        extras.append(f"popularity: {pop}")
    tags = row.get("tag_list") or []
    if isinstance(tags, list) and tags:
        tag_str = ", ".join(str(t).lower() for t in tags[:_RICH_TAG_FIRST_N] if t)
        if tag_str:
            extras.append(f"tags: {tag_str}")
    if extras:
        line += ", " + ", ".join(extras)
    return line


class KVStore:
    """RocksDB K/V store for embeddings + metadata.

    Raw mode: all keys/values are bytes — embeddings as float32 raw bytes, metadata as JSON utf-8 bytes.
    """

    def __init__(self, db: Rdict) -> None:
        self._db = db
        self._dim_cache: dict[str, int] = {}
        self._user_dim_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def _open_db(cls, path: Path, read_only: bool = True) -> Rdict:
        opts = Options(raw_mode=True)
        if read_only:
            # Bypasses the LOCK file → multiple read-only handles + an active
            # writer (e.g. running inference) can coexist across processes.
            return Rdict(str(path), options=opts, access_type=AccessType.read_only())
        return Rdict(str(path), options=opts)

    @classmethod
    def open(cls, db_path: str | Path = DB_PATH, read_only: bool = True) -> KVStore:
        """Open an existing KVStore. Default is read-only (multi-reader safe).

        Pass `read_only=False` only when the caller needs to mutate the store
        (currently: query-instruction / query-embedding cache writes from
        `InstructedQueryEmbedder` and `feature.build_emb`). Concurrent write
        opens across processes will fail on the LOCK file by design.
        """
        db_path = Path(db_path)
        if not db_path.exists() or not any(db_path.iterdir()):
            raise FileNotFoundError(
                f"KVStore not built at {db_path}. Run: uv run python -m mymodule.feature.kvdb --build"
            )
        key = (str(db_path.resolve()), read_only)
        cached = _INSTANCE_CACHE.get(key)
        if cached is not None:
            return cached
        inst = cls(cls._open_db(db_path, read_only=read_only))
        _INSTANCE_CACHE[key] = inst
        return inst

    @classmethod
    def open_or_build(cls, db_path: str | Path = DB_PATH, read_only: bool = True) -> KVStore:
        db_path = Path(db_path)
        if not db_path.exists() or not any(db_path.iterdir()):
            cls.build(db_path)
        return cls.open(db_path, read_only=read_only)

    @classmethod
    def build(cls, db_path: str | Path = DB_PATH, force: bool = False) -> KVStore:
        """Build the store from HF datasets. Always opens RocksDB in write mode.

        Returns the write-mode instance (callers expect to be able to keep
        writing during the same process, e.g. `--build-user-meta-emb`). For a
        read-only follow-up, call `KVStore.open(read_only=True)` after this.
        """
        db_path = Path(db_path)
        if force and db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        db = cls._open_db(db_path, read_only=False)

        # --- Track embeddings ---
        logger.info("Loading Track-Embeddings ...")
        track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        logger.info(f"  {len(track_ds)} tracks, writing {len(_HF_VECTOR_TYPES)} HF vector types")
        for vt in _HF_VECTOR_TYPES:
            dim = 0
            written = 0
            wb = WriteBatch(raw_mode=True)
            for row in tqdm(track_ds, desc=f"  {vt}"):
                vec = row[vt]
                if not vec:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                dim = int(arr.shape[0])
                wb.put(_k_track_emb(vt, row["track_id"]), arr.tobytes())
                written += 1
                if written % _BATCH_SIZE == 0:
                    db.write(wb)
                    wb = WriteBatch(raw_mode=True)
            db.write(wb)
            db[f"{_META_DIM}{vt}".encode()] = str(dim).encode()
            logger.info(f"    {vt}: {written} embeddings, dim={dim}")

        db[_META_VECTOR_TYPES.encode()] = json.dumps(VECTOR_TYPES).encode()

        # --- Track metadata ---
        logger.info("Loading Track-Metadata ...")
        meta_written = _write_hf_metadata(
            db=db,
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            splits=["all_tracks", "train", "test"],
            id_field="track_id",
            key_fn=_k_track_meta,
            desc="  track meta",
        )
        logger.info(f"  track meta: {meta_written}")

        # --- User embeddings (cfbpr, from HF User-Embeddings) ---
        # stored under unified prefix: user:embedding:cfbpr:talkpl-bpr:{user_id}
        logger.info("Loading User-Embeddings ...")
        user_written = 0
        user_dim = 0
        for split in ["train", "test_warm", "test_cold"]:
            try:
                user_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings", split=split)
            except Exception as e:
                logger.info(f"  skip split={split}: {type(e).__name__}: {e}")
                continue
            wb = WriteBatch(raw_mode=True)
            for row in tqdm(user_ds, desc=f"  user cfbpr [{split}]"):
                vec = row.get("cf-bpr")
                if not vec:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                user_dim = int(arr.shape[0])
                wb.put(_k_user_emb("cfbpr", CFBPR_MODEL_ID, row["user_id"]), arr.tobytes())
                user_written += 1
            db.write(wb)
        if user_dim:
            db[f"{_META_USER_EMB_DIM}cfbpr:{CFBPR_MODEL_ID}".encode()] = str(user_dim).encode()
        logger.info(f"  user cfbpr: {user_written}, dim={user_dim}")

        # --- User metadata ---
        logger.info("Loading User-Metadata ...")
        user_meta_written = _write_hf_metadata(
            db=db,
            dataset_name="talkpl-ai/TalkPlayData-Challenge-User-Metadata",
            splits=["train", "test_warm", "test_cold", "test", "all_users"],
            id_field="user_id",
            key_fn=_k_user_meta,
            desc="  user meta",
        )
        logger.info(f"  user meta: {user_meta_written}")

        db[_META_BUILT_AT.encode()] = datetime.now(timezone.utc).isoformat().encode()
        logger.info(f"RocksDB ready at {db_path}")
        return cls(db)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def embedding_dim(self, vector_type: str) -> int:
        if vector_type not in self._dim_cache:
            raw = self._db.get(f"{_META_DIM}{vector_type}".encode())
            if raw is None:
                raise KeyError(f"Unknown vector_type '{vector_type}' (not in kvdb)")
            self._dim_cache[vector_type] = int(raw.decode())
        return self._dim_cache[vector_type]

    def user_embedding_dim(self, emb_type: str = "cfbpr", model_id: str | None = None) -> int:
        """unified user emb dim lookup. model_id default: cfbpr → talkpl-bpr."""
        if model_id is None:
            model_id = CFBPR_MODEL_ID if emb_type == "cfbpr" else ""
        cache_key = f"{emb_type}:{model_id}"
        if cache_key not in self._user_dim_cache:
            raw = self._db.get(f"{_META_USER_EMB_DIM}{emb_type}:{model_id}".encode())
            if raw is None:
                raise KeyError(
                    f"Unknown user emb ({emb_type}, {model_id}) — not in kvdb. Run build / build-user-meta-emb."
                )
            self._user_dim_cache[cache_key] = int(raw.decode())
        return self._user_dim_cache[cache_key]

    def get_track_embedding(self, track_id: str, vector_type: str) -> np.ndarray | None:
        raw = self._db.get(_k_track_emb(vector_type, track_id))
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def get_track_meta(self, track_id: str) -> dict | None:
        raw = self._db.get(_k_track_meta(track_id))
        if raw is None:
            return None
        return json.loads(raw)

    def get_track_crawl(self, track_id: str, fields: list[str] | None = None) -> dict | None:
        """Return crawl record (from `mymodule.feature.crawl` JSONL) for `track_id`.

        If `fields` is given, return only those keys (absent keys omitted, no
        None padding). Returns None when the record itself is missing.
        """
        raw = self._db.get(_k_track_crawl(track_id))
        if raw is None:
            return None
        rec = json.loads(raw)
        if fields is None:
            return rec
        return {k: rec[k] for k in fields if k in rec}

    def get_user_embedding(
        self,
        user_id: str,
        emb_type: str = "cfbpr",
        model_id: str | None = None,
        enriched_meta: dict | None = None,
    ) -> np.ndarray | None:
        """Unified user embedding lookup.

        - `emb_type="cfbpr"`: direct user_id lookup (model_id default talkpl-bpr).
        - `emb_type="meta"`: user_id → meta → meta_key → embedding. If `enriched_meta`
          is given, it is used instead of the kvdb user:meta record.
        """
        if emb_type == "cfbpr":
            mid = model_id or CFBPR_MODEL_ID
            raw = self._db.get(_k_user_emb("cfbpr", mid, user_id))
        elif emb_type == "meta":
            if model_id is None:
                raise ValueError("emb_type='meta' requires model_id (Ollama slug).")
            meta = enriched_meta if enriched_meta is not None else self.get_user_meta(user_id)
            if not meta:
                return None
            meta_key = build_meta_key(meta)
            raw = self._db.get(_k_user_emb("meta", model_id, meta_key))
        else:
            raise ValueError(f"Unknown emb_type '{emb_type}' (expected 'cfbpr' or 'meta')")
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def get_user_meta(self, user_id: str) -> dict | None:
        raw = self._db.get(_k_user_meta(user_id))
        if raw is None:
            return None
        return json.loads(raw)

    # Legacy alias (deprecated): prefer get_user_embedding(..., emb_type="meta", model_id=...)
    def get_user_meta_embedding(
        self,
        user_id: str,
        model_id: str,
        enriched_meta: dict | None = None,
    ) -> np.ndarray | None:
        return self.get_user_embedding(user_id, emb_type="meta", model_id=model_id, enriched_meta=enriched_meta)

    def user_meta_emb_dim(self, model_id: str) -> int:
        raw = self._db.get(f"{_META_USER_EMB_DIM}meta:{model_id}".encode())
        if raw is None:
            raise KeyError(f"user_meta_emb not built for model_id='{model_id}'")
        return int(raw.decode())

    # ------------------------------------------------------------------
    # Query embedding cache (talkpl_metadata-{model_id} vector_type)
    # ------------------------------------------------------------------

    def get_query_embedding(self, model_id: str, text: str, composition: str | None = None) -> np.ndarray | None:
        """Look up cached embedding for composed query text. None on miss.

        `composition` selects the namespace (default `talkpl_metadata`).
        """
        vt = query_vector_type(model_id, composition)
        raw = self._db.get(_k_query_emb(vt, _hash_query_text(text)))
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def put_query_embedding(self, model_id: str, text: str, vec: np.ndarray, composition: str | None = None) -> None:
        """Store embedding for composed query text. Caller must hold write-mode KV."""
        vt = query_vector_type(model_id, composition)
        arr = np.asarray(vec, dtype=np.float32)
        self._db[_k_query_emb(vt, _hash_query_text(text))] = arr.tobytes()

    def query_emb_dim(self, model_id: str, composition: str | None = None) -> int | None:
        """Cached query embedding dim (from prebuild meta), or None if never built."""
        vt = query_vector_type(model_id, composition)
        raw = self._db.get(f"{_META_QUERY_EMB_DIM}{vt}".encode())
        return int(raw.decode()) if raw else None

    def get_track_embeddings_batch(self, track_ids: list[str], vector_type: str) -> list[np.ndarray | None]:
        return [self.get_track_embedding(tid, vector_type) for tid in track_ids]

    def iter_user_metas(self) -> Iterator[tuple[str, dict]]:
        """(user_id, meta_dict) prefix scan."""
        prefix = _P_USER_META.encode()
        it = self._db.iter()
        it.seek(prefix)
        while it.valid():
            k = it.key()
            if not k.startswith(prefix):
                break
            uid = k[len(prefix) :].decode()
            yield uid, json.loads(it.value())
            it.next()

    def iter_track_embeddings(self, vector_type: str) -> Iterator[tuple[str, np.ndarray]]:
        """Prefix scan over (track_id, vector) pairs."""
        prefix = f"{_P_TRACK_EMB}{vector_type}:".encode()
        it = self._db.iter()
        it.seek(prefix)
        while it.valid():
            k = it.key()
            if not k.startswith(prefix):
                break
            tid = k[len(prefix) :].decode()
            yield tid, np.frombuffer(it.value(), dtype=np.float32).copy()
            it.next()

    def iter_track_crawl(self) -> Iterator[tuple[str, dict]]:
        """Prefix scan over (track_id, crawl_record) pairs, for bulk consumers like BM25Pool."""
        prefix = _P_TRACK_CRAWL.encode()
        it = self._db.iter()
        it.seek(prefix)
        while it.valid():
            k = it.key()
            if not k.startswith(prefix):
                break
            tid = k[len(prefix) :].decode()
            yield tid, json.loads(it.value())
            it.next()

    def built_at(self) -> str | None:
        raw = self._db.get(_META_BUILT_AT.encode())
        return raw.decode() if raw else None

    # ------------------------------------------------------------------
    # Crawl meta accessors
    # ------------------------------------------------------------------

    def crawl_count(self) -> int | None:
        raw = self._db.get(_META_CRAWL_COUNT.encode())
        return int(raw.decode()) if raw else None

    def crawl_ok_count(self) -> int | None:
        raw = self._db.get(_META_CRAWL_OK_COUNT.encode())
        return int(raw.decode()) if raw else None

    def crawl_built_at(self) -> str | None:
        raw = self._db.get(_META_CRAWL_BUILT_AT.encode())
        return raw.decode() if raw else None

    def crawl_source_jsonl(self) -> str | None:
        raw = self._db.get(_META_CRAWL_SOURCE_JSONL.encode())
        return raw.decode() if raw else None

    # ------------------------------------------------------------------
    # User meta embedding build
    # ------------------------------------------------------------------

    @classmethod
    def build_user_meta_embeddings(
        cls,
        db_path: str | Path = DB_PATH,
        embedder=None,
        force: bool = False,
        emb_provider: str | None = None,
    ) -> None:
        """Extract unique meta_keys from user metas, embed them, and store in kvdb.

        Idempotent: already-stored keys are skipped unless `force=True`.
        Without `embedder`, uses `get_embedder(emb_provider)`.
        """
        from mymodule.feature.ollama_embed import get_embedder

        embedder = embedder or get_embedder(emb_provider)
        if not embedder.health():
            raise RuntimeError(
                f"Embedding server unreachable at {embedder.url}. "
                "Check MYMODULE_EMB_PROVIDER / MYMODULE_EMB_OLLAMA_URL / MYMODULE_EMB_OPENAI_URL"
            )

        db_path = Path(db_path)
        if not db_path.exists() or not any(db_path.iterdir()):
            raise FileNotFoundError(
                f"KVStore not built at {db_path}. Run base build first: uv run python -m mymodule.feature.kvdb --build"
            )
        inst = cls.open(db_path, read_only=False)
        model_id = embedder.model_id
        logger.info(f"Building user meta embeddings via {embedder} → model_id={model_id}")

        # 1. collect all user metas
        metas: dict[str, dict] = dict(inst.iter_user_metas())
        logger.info(f"  {len(metas)} users in kvdb user:meta")

        # 2. enrich preferred_* fields from train-session user_profile only —
        #    using test (devset) profiles could leak test-user preferences into
        #    embeddings shared by train users with the same meta combo.
        logger.info("  enriching from user_profile (train split only) ...")
        enrichment: dict[str, dict] = {}
        for split in ["train"]:
            try:
                ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=split)
            except Exception as e:
                logger.info(f"    skip devset[{split}]: {type(e).__name__}")
                continue
            for s in ds:
                up = s.get("user_profile") or {}
                uid = up.get("user_id") or s.get("user_id")
                if not uid:
                    continue
                prev = enrichment.setdefault(uid, {})
                for f in ["preferred_language", "preferred_musical_culture", "country_name", "age_group"]:
                    if up.get(f) and not prev.get(f):
                        prev[f] = up[f]

        enriched = 0
        updated_users = 0
        for uid, meta in metas.items():
            add = enrichment.get(uid)
            if not add:
                continue
            changed = False
            for k, v in add.items():
                if not meta.get(k):
                    meta[k] = v
                    enriched += 1
                    changed = True
            if changed:
                # persist enriched meta back to kvdb so later lookup by user_id produces
                # the same meta_key used at embedding generation time.
                inst._db[_k_user_meta(uid)] = json.dumps(meta, ensure_ascii=False, default=str).encode()
                updated_users += 1
        logger.info(f"  enriched {enriched} fields across {updated_users} users (written back)")

        # 3. unique meta_key set
        meta_key_to_prompt: dict[str, str] = {}
        for meta in metas.values():
            key = build_meta_key(meta)
            if key not in meta_key_to_prompt:
                meta_key_to_prompt[key] = build_meta_prompt(meta)
        total_keys = len(meta_key_to_prompt)
        logger.info(f"  unique meta_key: {total_keys}")

        # 4. check already-stored keys (idempotent)
        existing = 0
        if not force:
            for key in meta_key_to_prompt:
                if inst._db.get(_k_user_emb("meta", model_id, key)) is not None:
                    existing += 1
            logger.info(f"  already-cached keys: {existing} (will skip)")

        # 5. embed + store (unified prefix: user:embedding:meta:{model_id}:{meta_key})
        to_generate = [
            (k, p)
            for k, p in meta_key_to_prompt.items()
            if force or inst._db.get(_k_user_emb("meta", model_id, k)) is None
        ]
        logger.info(f"  generating {len(to_generate)} embeddings ...")

        dim = 0
        ok = 0
        failed: list[str] = []
        for key, prompt in tqdm(to_generate, desc="  ollama embed"):
            try:
                vec = embedder.embed(prompt)
            except Exception as e:
                failed.append(key)
                logger.info(f"    FAIL meta_key='{key}': {e}")
                continue
            dim = int(vec.shape[0])
            inst._db[_k_user_emb("meta", model_id, key)] = vec.tobytes()
            ok += 1

        # 6. record meta keys (unified _meta:user_emb:* prefix)
        if dim:
            inst._db[f"{_META_USER_EMB_DIM}meta:{model_id}".encode()] = str(dim).encode()
            inst._db[f"{_META_USER_EMB_FIELDS}meta:{model_id}".encode()] = json.dumps(USER_META_KEY_FIELDS).encode()
            inst._db[f"{_META_USER_EMB_PROMPT}meta:{model_id}".encode()] = USER_META_EMB_PROMPT_TPL.encode()

        logger.info(f"  done. ok={ok} failed={len(failed)} dim={dim} (total keys: {total_keys})")
        if failed:
            logger.info(f"  first 5 failed keys: {failed[:5]}")

    # ------------------------------------------------------------------
    # Crawl JSONL → KV ingestion
    # ------------------------------------------------------------------
    @classmethod
    def build_track_crawl(
        cls,
        jsonl_path: str | Path,
        db_path: str | Path = DB_PATH,
    ) -> None:
        """Ingest crawled JSONL (from `mymodule.feature.crawl`) into KV.

        Idempotent overwrite per track_id; existing `track:crawl:` entries not in
        the JSONL are left untouched (additive). Meta keys (`_meta:crawl:*`) always
        reflect the last call's stats.
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Crawl JSONL not found: {jsonl_path}")
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        db = cls._open_db(db_path, read_only=False)
        try:
            count = 0
            ok_count = 0
            wb = WriteBatch(raw_mode=True)
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tid = rec.get("track_id")
                    if not isinstance(tid, str) or not tid:
                        continue
                    wb.put(_k_track_crawl(tid), json.dumps(rec, ensure_ascii=False).encode("utf-8"))
                    count += 1
                    if rec.get("ok") is True:
                        ok_count += 1
                    if count % _BATCH_SIZE == 0:
                        db.write(wb)
                        wb = WriteBatch(raw_mode=True)
            db.write(wb)
            db[_META_CRAWL_COUNT.encode()] = str(count).encode()
            db[_META_CRAWL_OK_COUNT.encode()] = str(ok_count).encode()
            db[_META_CRAWL_BUILT_AT.encode()] = datetime.now(timezone.utc).isoformat().encode()
            db[_META_CRAWL_SOURCE_JSONL.encode()] = str(jsonl_path).encode()
            logger.success(f"crawl ingest: {count} records ({ok_count} ok) from {jsonl_path}")
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Track metadata_rich embedding build (Ollama batch, write to KV)
    # ------------------------------------------------------------------
    @classmethod
    def build_track_metadata_rich_embeddings(
        cls,
        db_path: str | Path = DB_PATH,
        embedder=None,
        workers: int = 8,
        limit: int | None = None,
        force: bool = False,
        emb_provider: str | None = None,
    ) -> None:
        """Build `metadata_rich-qwen3_embedding_0.6b` track embeddings into KV.

        Mirrors `mymodule.feature.store --build-metadata-rich` (which targets LanceDB) but
        writes to KV for point-lookup callers (`get_track_embedding`,
        `get_track_embeddings_batch`) — used by PAS helpers and selective gate.

        Doc input form (per track):
            title: <track>, artist: <artist>, album: <album>, year: <YYYY>,
            popularity: <pop>, tags: <tag1, tag2, ...>

        Uses parallel Ollama HTTP calls (default 8 workers). Idempotent: existing
        entries skipped unless `force=True`. ~30 min for 47K tracks @ 8 workers.
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from datasets import load_dataset
        from rocksdict import WriteBatch

        from mymodule.feature.ollama_embed import get_embedder

        embedder = embedder or get_embedder(emb_provider)
        if not embedder.health():
            raise RuntimeError(
                f"Embedding server unreachable at {embedder.url}. "
                "Check MYMODULE_EMB_PROVIDER / MYMODULE_EMB_OLLAMA_URL / MYMODULE_EMB_OPENAI_URL"
            )

        db_path = Path(db_path)
        if not db_path.exists() or not any(db_path.iterdir()):
            raise FileNotFoundError(
                f"KVStore not built at {db_path}. Run base build first: uv run python -m mymodule.feature.kvdb --build"
            )

        vt = "metadata_rich-qwen3_embedding_0.6b"
        inst = cls.open(db_path, read_only=False)
        logger.info(f"Building track metadata_rich embeddings via {embedder} → vt={vt}")

        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
        n = len(ds) if limit is None else min(limit, len(ds))
        logger.info(f"  {n}/{len(ds)} tracks (limit={limit})")

        # Skip already-present entries unless force.
        skipped_existing = 0
        texts: list[tuple[str, str]] = []
        for i in range(n):
            row = ds[i]
            tid = row["track_id"]
            if not force:
                existing = inst._db.get(_k_track_emb(vt, tid))
                if existing is not None:
                    skipped_existing += 1
                    continue
            texts.append((tid, _compose_track_metadata_rich_text(row)))
        logger.info(f"  to encode: {len(texts)} (skipped existing: {skipped_existing}, use --force to re-embed)")

        if not texts:
            logger.success("Nothing to encode — all entries already present.")
            return

        # Per-thread embedder to avoid HTTP connection sharing issues.
        tlocal = threading.local()

        def _emb(text: str) -> np.ndarray:
            if not hasattr(tlocal, "raw"):
                from mymodule.feature.ollama_embed import get_embedder

                tlocal.raw = get_embedder(emb_provider)
            return tlocal.raw.embed(text, instruction=None)

        wb = WriteBatch(raw_mode=True)
        written = 0
        errors = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_emb, text): tid for tid, text in texts}
            from tqdm import tqdm

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  encode {vt}"):
                tid = futures[fut]
                try:
                    vec = fut.result()
                    wb.put(_k_track_emb(vt, tid), np.asarray(vec, dtype=np.float32).tobytes())
                    written += 1
                    if written % _BATCH_SIZE == 0:
                        inst._db.write(wb)
                        wb = WriteBatch(raw_mode=True)
                except Exception as e:
                    errors += 1
                    logger.warning(f"track {tid[:8]}… failed: {type(e).__name__}: {e}")
        inst._db.write(wb)

        elapsed = time.time() - t0
        logger.success(
            f"Wrote {written} embeddings into KV under '{vt}' "
            f"(errors={errors}, skipped_existing={skipped_existing}) in {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Sync metadata_rich vectors from LanceDB → KV (no Ollama)
    # ------------------------------------------------------------------
    @classmethod
    def sync_metadata_rich_from_lancedb(
        cls,
        db_path: str | Path = DB_PATH,
        lancedb_path: str | Path | None = None,
        vector_type: str = "metadata_rich-qwen3_embedding_0.6b",
    ) -> int:
        """Copy LanceDB metadata_rich vectors into KV. Used by the unified
        `mymodule.feature.store --build-metadata-rich` flow (default behavior)
        so a single Ollama pass populates both stores. Skips encoding entirely
        — just reads from the LanceDB table and writes to KV.

        Idempotent: writes overwrite existing KV keys with the same content.
        Returns the number of vectors written.
        """
        import lancedb  # noqa: PLC0415

        from mymodule.feature.store import _TABLE_NAME
        from mymodule.feature.store import DB_PATH as LANCE_DB_PATH

        lancedb_path = Path(lancedb_path) if lancedb_path else LANCE_DB_PATH
        ldb = lancedb.connect(str(lancedb_path))
        table_name = _TABLE_NAME[vector_type]
        if table_name not in (ldb.table_names() if hasattr(ldb, "table_names") else ldb.list_tables()):
            raise RuntimeError(
                f"LanceDB table '{table_name}' not found at {lancedb_path}. "
                f"Run `mymodule.feature.store --build-metadata-rich` first."
            )

        table = ldb.open_table(table_name)
        n_rows = table.count_rows()
        logger.info(f"  LanceDB table '{table_name}': {n_rows} rows → KV (vector_type='{vector_type}')")

        df = table.to_arrow()
        track_ids = df["track_id"].to_pylist()
        vectors = df["vector"].to_pylist()
        if len(track_ids) != len(vectors):
            raise RuntimeError(f"track_id/vector length mismatch: {len(track_ids)} vs {len(vectors)}")

        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        kv = cls._open_db(db_path, read_only=False)
        wb = WriteBatch(raw_mode=True)
        written = 0
        for tid, vec in tqdm(zip(track_ids, vectors), total=len(track_ids), desc=f"  → kv:{vector_type}"):
            if vec is None or tid is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            wb.put(_k_track_emb(vector_type, tid), arr.tobytes())
            written += 1
            if written % _BATCH_SIZE == 0:
                kv.write(wb)
                wb = WriteBatch(raw_mode=True)
        kv.write(wb)
        # Persist dim metadata so KVStore.track_embedding_dim() works.
        if vectors:
            sample = np.asarray(vectors[0], dtype=np.float32)
            kv[f"{_META_DIM}{vector_type}".encode()] = str(int(sample.shape[0])).encode()
        logger.success(f"  KV: wrote {written} embeddings under '{vector_type}'")
        return written

    # ------------------------------------------------------------------
    # Query embedding cache prebuild
    # ------------------------------------------------------------------

    @classmethod
    def build_query_embeddings(
        cls,
        dataset: str = "all",
        db_path: str | Path = DB_PATH,
        force: bool = False,
        emb_provider: str | None = None,
        workers: int = 8,
    ) -> None:
        """Eagerly populate `query:embedding:{vector_type}:{text_hash}` for every
        (user_query, chat_history) tuple in the requested dataset(s).

        `dataset`: one of "devset", "blindset_A", "blindset_B", "all".

        Idempotent: existing vector_type+text_hash entries are skipped unless `force=True`.
        """
        from mymodule.feature.ollama_embed import (
            InstructedQueryEmbedder,
            _resolve_query_compose,
        )

        embedder = InstructedQueryEmbedder(provider=emb_provider)
        if not embedder.raw.health():
            raise RuntimeError(
                f"Embedding server unreachable at {embedder.raw.url}. "
                "Check MYMODULE_EMB_PROVIDER / MYMODULE_EMB_OLLAMA_URL / MYMODULE_EMB_OPENAI_URL"
            )

        db_path = Path(db_path)
        if not db_path.exists() or not any(db_path.iterdir()):
            raise FileNotFoundError(
                f"KVStore not built at {db_path}. Run base build first: uv run python -m mymodule.feature.kvdb --build"
            )
        inst = cls.open(db_path, read_only=False)
        model_id = embedder.raw.model_id
        vector_type_prefix, compose_fn = _resolve_query_compose()
        vt = f"{vector_type_prefix}-{model_id}"
        logger.info(
            f"Building query embeddings via {embedder.raw} → vector_type={vt}, "
            f"dataset={dataset} (composition={vector_type_prefix})"
        )

        # 1. enumerate (user_query, chat_history) tuples
        pairs = _enumerate_query_pairs(dataset)
        logger.info(f"  enumerated {len(pairs)} (session, turn) pairs from {dataset}")

        # 2. compose + dedupe by text_hash
        seen_hashes: set[str] = set()
        unique_texts: list[str] = []
        for uq, hist in pairs:
            text = compose_fn(uq, hist, inst)
            h = _hash_query_text(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            unique_texts.append(text)
        logger.info(f"  unique composed texts: {len(unique_texts)} (after dedupe)")

        # 3. skip already-cached
        to_generate: list[str] = []
        for text in unique_texts:
            h = _hash_query_text(text)
            if not force and inst._db.get(_k_query_emb(vt, h)) is not None:
                continue
            to_generate.append(text)
        logger.info(f"  already-cached: {len(unique_texts) - len(to_generate)}, to generate: {len(to_generate)}")

        # 4. embed loop — parallel via ThreadPoolExecutor.
        # Ollama HTTP server queues concurrent requests internally; client-side
        # parallelism cuts HTTP/connection overhead. RocksDB writes from threads
        # are safe within a single process.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dim = 0
        ok = 0
        failed_hashes: list[str] = []

        def _embed_one(text: str) -> tuple[str, "np.ndarray | None", "Exception | None"]:
            h = _hash_query_text(text)
            try:
                vec = embedder.raw.embed(text, instruction=None)
                return h, vec, None
            except Exception as e:
                return h, None, e

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(_embed_one, t) for t in to_generate]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  ollama embed (query, x{workers})"):
                h, vec, err = fut.result()
                if err is not None or vec is None:
                    failed_hashes.append(h)
                    if err is not None:
                        logger.info(f"    FAIL hash={h}: {err}")
                    continue
                dim = int(vec.shape[0])
                inst._db[_k_query_emb(vt, h)] = vec.tobytes()
                ok += 1

        # 5. meta keys
        if dim:
            inst._db[f"{_META_QUERY_EMB_DIM}{vt}".encode()] = str(dim).encode()
        # Total existing entries for this vector_type after the run
        prefix = f"{_P_QUERY_EMB}{vt}:".encode()
        total_existing = 0
        it = inst._db.iter()
        it.seek(prefix)
        while it.valid():
            k = it.key()
            if not k.startswith(prefix):
                break
            total_existing += 1
            it.next()
        inst._db[f"{_META_QUERY_EMB_COUNT}{vt}".encode()] = str(total_existing).encode()
        inst._db[f"{_META_QUERY_EMB_LAST_BUILD}{vt}".encode()] = datetime.now(timezone.utc).isoformat().encode()

        logger.info(f"  done. ok={ok} failed={len(failed_hashes)} dim={dim} total_in_kv={total_existing}")
        if failed_hashes:
            logger.info(f"  first 5 failed hashes: {failed_hashes[:5]}")


def _enumerate_query_pairs(dataset: str) -> list[tuple[str, list[dict]]]:
    """Enumerate (user_query, chat_history) tuples from train/devset/blindset_A/B/all.

    Lazy imports to avoid circular dependencies.

    ``train`` prefills query embeddings needed for OOF feature builds of
    trainable rerankers (e.g. GBM). ~15K sessions × ~24 turns ≈ 360K queries,
    so it is slow (dominated by embedding-API latency).
    """
    valid = {"train", "devset", "blindset_A", "blindset_B", "all"}
    if dataset not in valid:
        raise ValueError(f"dataset={dataset!r} not in {valid}")

    sources: list[str] = ["train", "devset", "blindset_A", "blindset_B"] if dataset == "all" else [dataset]
    pairs: list[tuple[str, list[dict]]] = []
    for src in sources:
        if src == "train":
            pairs.extend(_pairs_from_trainset())
        elif src == "devset":
            pairs.extend(_pairs_from_devset())
        elif src == "blindset_A":
            pairs.extend(_pairs_from_blindset("blindset_A"))
        elif src == "blindset_B":
            pairs.extend(_pairs_from_blindset("blindset_B"))
    return pairs


def _pairs_from_trainset() -> list[tuple[str, list[dict]]]:
    """(user_query, chat_history) pairs for every (session, turn) in the HF train split.

    Train sessions have variable turn counts (max ~24); uses the same
    ``parse_devset_turn`` logic as the devset path.
    """
    from mymodule.run_inference_devset import parse_devset_turn

    try:
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    except Exception as e:
        logger.warning(f"  trainset load failed: {type(e).__name__}: {e}")
        return []
    out: list[tuple[str, list[dict]]] = []
    for item in tqdm(list(ds), desc="  enumerate trainset"):
        # Sessions have variable length; enumerate up to the max turn_number.
        max_turn = max(c["turn_number"] for c in item["conversations"])
        for target_turn in range(1, max_turn + 1):
            try:
                parsed = parse_devset_turn(
                    conversations=item["conversations"],
                    target_turn_number=target_turn,
                    conversation_goal=item.get("conversation_goal"),
                    goal_progress_assessments=item.get("goal_progress_assessments"),
                    user_profile=item.get("user_profile"),
                )
            except (IndexError, KeyError):
                # skip malformed turns
                continue
            out.append((parsed["user_query"], parsed["chat_history"]))
    return out


def _pairs_from_devset() -> list[tuple[str, list[dict]]]:
    from mymodule.run_inference_devset import parse_devset_turn

    try:
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    except Exception as e:
        logger.warning(f"  devset load failed: {type(e).__name__}: {e}")
        return []
    out: list[tuple[str, list[dict]]] = []
    for item in tqdm(list(ds), desc="  enumerate devset"):
        for target_turn in range(1, 9):
            parsed = parse_devset_turn(
                conversations=item["conversations"],
                target_turn_number=target_turn,
                conversation_goal=item.get("conversation_goal"),
                goal_progress_assessments=item.get("goal_progress_assessments"),
                user_profile=item.get("user_profile"),
            )
            out.append((parsed["user_query"], parsed["chat_history"]))
    return out


def _pairs_from_blindset(name: str) -> list[tuple[str, list[dict]]]:
    from mymodule.run_inference_blindset import BLINDSET_DATASETS, parse_blindset_item

    hf_path = BLINDSET_DATASETS.get(name)
    if hf_path is None:
        logger.info(f"  {name}: HF path not registered yet — skip")
        return []
    try:
        ds = load_dataset(hf_path, split="test")
    except Exception as e:
        logger.warning(f"  {name} load failed: {type(e).__name__}: {e}")
        return []
    out: list[tuple[str, list[dict]]] = []
    for item in tqdm(list(ds), desc=f"  enumerate {name}"):
        parsed = parse_blindset_item(item)
        out.append((parsed["user_query"], parsed["chat_history"]))
    return out


def _write_hf_metadata(
    db: Rdict,
    dataset_name: str,
    splits: list[str],
    id_field: str,
    key_fn,
    desc: str,
) -> int:
    """Bulk-write an HF metadata dataset into kvdb, using every loadable split."""
    seen: set[str] = set()
    written = 0
    for split in splits:
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.info(f"  skip {dataset_name} split={split}: {type(e).__name__}")
            continue
        wb = WriteBatch(raw_mode=True)
        for row in tqdm(ds, desc=f"{desc} [{split}]"):
            _id = row.get(id_field)
            if not _id or _id in seen:
                continue
            seen.add(_id)
            meta = {k: v for k, v in row.items() if k != id_field}
            wb.put(key_fn(_id), json.dumps(meta, ensure_ascii=False, default=str).encode())
            written += 1
            if written % _BATCH_SIZE == 0:
                db.write(wb)
                wb = WriteBatch(raw_mode=True)
        db.write(wb)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RocksDB KV store management")
    parser.add_argument("--build", action="store_true", help="Build KV store from HuggingFace datasets")
    parser.add_argument(
        "--build-user-meta-emb",
        action="store_true",
        help="Generate user meta embeddings (env: MYMODULE_EMB_PROVIDER, MYMODULE_EMB_OLLAMA_URL, MYMODULE_EMB_MODEL)",
    )
    parser.add_argument(
        "--build-metadata-rich",
        action="store_true",
        help="Build metadata_rich track embeddings into KV "
        "(env: MYMODULE_EMB_PROVIDER, MYMODULE_EMB_OLLAMA_URL, MYMODULE_EMB_MODEL). "
        "~30 min for 47K tracks @ 8 workers. "
        "Mirror of `store --build-metadata-rich` for LanceDB; same compose function.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel Ollama HTTP workers for --build-metadata-rich / --build-query-emb",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap track count for --build-metadata-rich (debug)",
    )
    parser.add_argument(
        "--build-query-emb",
        action="store_true",
        help="Prebuild query embedding cache for train/devset/blindset "
        "(env: MYMODULE_EMB_PROVIDER, MYMODULE_EMB_OLLAMA_URL, MYMODULE_EMB_MODEL)",
    )
    parser.add_argument(
        "--build-crawl",
        action="store_true",
        help="Ingest crawled metadata JSONL (from mymodule.feature.crawl) into KV "
        "under `track:crawl:` prefix. Idempotent overwrite per track_id; meta keys "
        "(_meta:crawl:*) record last build stats. No external HTTP — disk IO only.",
    )
    parser.add_argument(
        "--crawl-jsonl",
        type=Path,
        default=None,
        help="Source JSONL for --build-crawl (default: mymodule/feature/.crawl/lyrics.jsonl)",
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "devset", "blindset_A", "blindset_B", "all"],
        default="all",
        help="Which dataset(s) to enumerate for --build-query-emb (default: all). "
        "`train` is for GBM reranker OOF builds (~360K queries, embedding-API latency bound).",
    )
    parser.add_argument(
        "--query-composition",
        choices=["legacy", "rich"],
        default=None,
        help="Override query composition for --build-query-emb. `legacy` (default) "
        "= title/artist/album from prior music turns; `rich` = also year/popularity/tags "
        "(experimental). When unset, falls back to env "
        "`MYMODULE_QEMB_QUERY_COMPOSITION` (default `legacy`).",
    )
    parser.add_argument("--force", action="store_true", help="Remove existing DB / regenerate")
    parser.add_argument("--db-path", default=None, help="Custom DB path")
    parser.add_argument(
        "--emb-provider",
        choices=["ollama", "openai"],
        default=None,
        help="Embedding provider override (default: MYMODULE_EMB_PROVIDER env, or ollama)",
    )
    args = parser.parse_args()

    if args.query_composition is not None:
        os.environ["MYMODULE_QEMB_QUERY_COMPOSITION"] = args.query_composition

    path = Path(args.db_path) if args.db_path else DB_PATH

    if args.build:
        KVStore.build(path, force=args.force)
    elif args.build_user_meta_emb:
        KVStore.build_user_meta_embeddings(path, force=args.force, emb_provider=args.emb_provider)
    elif args.build_metadata_rich:
        KVStore.build_track_metadata_rich_embeddings(
            path, workers=args.workers, limit=args.limit, force=args.force, emb_provider=args.emb_provider
        )
    elif args.build_query_emb:
        KVStore.build_query_embeddings(
            dataset=args.dataset,
            db_path=path,
            force=args.force,
            emb_provider=args.emb_provider,
            workers=args.workers,
        )
    elif args.build_crawl:
        jsonl_path = args.crawl_jsonl or (Path(__file__).parent / ".crawl" / "lyrics.jsonl")
        KVStore.build_track_crawl(jsonl_path, db_path=path)
    else:
        parser.print_help()
