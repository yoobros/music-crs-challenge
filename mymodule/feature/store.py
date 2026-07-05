"""LanceDB-based embedding store for **ANN search only**.

Separation of concerns:
- Single track/user vector point lookups: `mymodule.feature.kvdb.KVStore` (rocksdict).
- This module: LanceDB ANN search (`search`) + popularity ranking (`get_popular_tracks`).

Usage:
    # One-time build (LanceDB ANN index)
    uv run python -m mymodule.feature.store --build

    # metadata_rich-qwen3 LanceDB table build (requires Ollama, ~30 min);
    # creates the 'tracks_metadata_rich_qwen3_embedding_0_6b' table.
    uv run python -m mymodule.feature.store --build-metadata-rich

    # In code
    store = EmbeddingStore.open_or_build()
    results = store.search("cf-bpr", query_vec, top_k=100)
    top = store.get_popular_tracks(20)
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import lancedb
import numpy as np
from datasets import load_dataset
from loguru import logger

DB_PATH = Path(__file__).parent / ".lancedb"

VECTOR_TYPES: list[str] = [
    "cf-bpr",
    "audio-laion_clap",
    "image-siglip2",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
    # Built locally by mymodule.feature.store --build-metadata-rich — same Qwen3
    # encoder but with year, popularity, and tag_list inline alongside the
    # talkpl entity_str. Lives in the same .lancedb/ as the others.
    # Pairs with the `qemb_metadata_rich` pool alias.
    "metadata_rich-qwen3_embedding_0.6b",
]

# Subset of VECTOR_TYPES that the HuggingFace `Track-Embeddings` dataset carries
# directly — `--build` ingests these. The remaining types (currently just the
# `metadata_rich-qwen3` variant) are built locally via the dedicated CLI flag
# (`--build-metadata-rich`); calling `--build` for them would fail with a
# missing-column error.
_HF_VECTOR_TYPES: list[str] = [vt for vt in VECTOR_TYPES if vt != "metadata_rich-qwen3_embedding_0.6b"]

_TABLE_NAME = {vt: f"tracks_{vt.replace('-', '_')}" for vt in VECTOR_TYPES}
# Locally-built rich variant used a slightly different naming convention
# (`0_6b` instead of `0.6b`) — special-case here so it matches the table
# that mymodule.feature.store --build-metadata-rich created. LanceDB OSS doesn't
# support rename, so adapt the lookup instead of re-encoding 47K tracks.
_TABLE_NAME["metadata_rich-qwen3_embedding_0.6b"] = "tracks_metadata_rich_qwen3_embedding_0_6b"
_POPULARITY_TABLE = "popularity"

# IVF_HNSW_SQ index default — for ~47K tracks, num_partitions=256 (~sqrt(N))
# gives top-100 recall ~99.7% (nprobes=200 at search time, see search_with_scores).
# Build cost: ~6s per table.
_ANN_NUM_PARTITIONS = 256


def _ensure_ann_index(table: lancedb.table.Table, metric: str = "cosine") -> None:
    """Create an IVF_HNSW_SQ index on the table if missing; no-op otherwise.

    Also supports ``dot`` metric tables such as cf-bpr (handled by LanceDB).
    """
    try:
        existing = table.list_indices()
    except Exception as e:
        logger.warning(f"  list_indices failed on {table.name}: {e}; skip ANN index")
        return
    if existing:
        logger.info(f"  ANN index already on {table.name}: {[idx.name for idx in existing]}")
        return
    n_rows = table.count_rows()
    if n_rows < 1000:
        logger.info(f"  skip ANN index on {table.name} (only {n_rows} rows; brute-force is fine)")
        return
    logger.info(f"  building IVF_HNSW_SQ index on {table.name} (rows={n_rows}, metric={metric})…")
    t0 = time.time()
    table.create_index(
        metric=metric,
        index_type="IVF_HNSW_SQ",
        num_partitions=_ANN_NUM_PARTITIONS,
        m=20,
        ef_construction=300,
    )
    logger.success(f"  ANN index built on {table.name} in {time.time() - t0:.1f}s")


class EmbeddingStore:
    """LanceDB ANN search + popularity."""

    def __init__(self, db: lancedb.DBConnection) -> None:
        self._db = db
        self._track_tables: dict[str, lancedb.table.Table] = {}
        self._popularity: list[str] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, db_path: str | Path = DB_PATH) -> EmbeddingStore:
        db = lancedb.connect(str(db_path))
        return cls(db)

    @classmethod
    def open_or_build(cls, db_path: str | Path = DB_PATH) -> EmbeddingStore:
        db_path = Path(db_path)
        if not db_path.exists() or not any(db_path.iterdir()):
            return cls.build(db_path)
        return cls.open(db_path)

    @classmethod
    def build(cls, db_path: str | Path = DB_PATH) -> EmbeddingStore:
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_path))

        existing = set(db.table_names() if hasattr(db, "table_names") else db.list_tables())

        # --- Track embeddings ---
        track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        track_ids = track_ds["track_id"]
        logger.info(f"Building LanceDB: {len(track_ids)} tracks")

        for vt in _HF_VECTOR_TYPES:
            table_name = _TABLE_NAME[vt]
            if table_name in existing:
                logger.info(f"  skip {table_name} (exists)")
                continue
            vectors = track_ds[vt]
            data = []
            dim = 0
            for tid, vec in zip(track_ids, vectors):
                if len(vec) == 0:
                    continue
                dim = len(vec)
                data.append({"track_id": tid, "vector": np.array(vec, dtype=np.float32)})
            db.create_table(table_name, data)
            logger.info(f"  created {table_name} ({dim}d, {len(data)} rows, skipped {len(track_ids) - len(data)})")
            _ensure_ann_index(db.open_table(table_name), metric=("dot" if vt == "cf-bpr" else "cosine"))

        # --- Popularity ---
        if _POPULARITY_TABLE not in existing:
            train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
            counter: Counter[str] = Counter()
            for item in train_ds:
                for msg in item["conversations"]:
                    if msg["role"] == "music":
                        counter[msg["content"]] += 1
            ranked = [tid for tid, _ in counter.most_common()]
            data = [{"track_id": tid, "rank": i} for i, tid in enumerate(ranked)]
            db.create_table(_POPULARITY_TABLE, data)
            logger.info(f"  created {_POPULARITY_TABLE} ({len(ranked)} tracks)")
        else:
            logger.info(f"  skip {_POPULARITY_TABLE} (exists)")

        logger.info(f"LanceDB ready at {db_path}")
        return cls(db)

    # ------------------------------------------------------------------
    # Metadata-rich track embeddings (local Ollama-based build)
    # ------------------------------------------------------------------
    @classmethod
    def build_track_metadata_rich(
        cls,
        db_path: str | Path = DB_PATH,
        embedder=None,
        workers: int = 8,
        limit: int | None = None,
        force: bool = False,
        emb_provider: str | None = None,
    ) -> EmbeddingStore:
        """Build the `metadata_rich-qwen3_embedding_0.6b` LanceDB table.

        HuggingFace `Track-Embeddings` does NOT carry this column — we compute
        locally via Ollama using a richer doc text (title/artist/album/year/
        popularity/tags) than the `metadata-qwen3` standard build.

        Same vector space as the qemb retrieval pool (`qemb_metadata_rich`).
        Idempotent: existing table is skipped unless `force=True` (drops + re-builds).

        Mirrors the build pattern of `KVStore.build_track_metadata_rich_embeddings`
        — same compose function (via `_compose_track_metadata_rich_text`) so KV
        and LanceDB stay vector-space aligned.
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        from mymodule.feature.kvdb import _compose_track_metadata_rich_text
        from mymodule.feature.ollama_embed import get_embedder

        embedder = embedder or get_embedder(emb_provider)
        if not embedder.health():
            raise RuntimeError(
                f"Embedding server unreachable at {embedder.url}. "
                "Check MYMODULE_EMB_PROVIDER / MYMODULE_EMB_OLLAMA_URL / MYMODULE_EMB_OPENAI_URL"
            )

        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_path))

        vt = "metadata_rich-qwen3_embedding_0.6b"
        table_name = _TABLE_NAME[vt]
        existing = set(db.table_names() if hasattr(db, "table_names") else db.list_tables())
        if table_name in existing and not force:
            logger.info(f"Table '{table_name}' already exists — use --force to rebuild. Open and return.")
            return cls(db)
        if table_name in existing and force:
            logger.warning(f"Dropping existing table '{table_name}' (--force)")
            db.drop_table(table_name)

        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
        n = len(ds) if limit is None else min(limit, len(ds))
        logger.info(f"Encoding {n}/{len(ds)} tracks via Ollama ({embedder}) → '{table_name}'")

        # Compose all text up front — cheap, ~1s for 47K.
        texts: list[tuple[str, str]] = []
        for i in range(n):
            row = ds[i]
            tid = row["track_id"]
            texts.append((tid, _compose_track_metadata_rich_text(row)))
        logger.info(f"  prepared {len(texts)} input texts (sample: {texts[0][1][:120]}…)")

        # Per-thread embedder to avoid HTTP connection sharing issues.
        tlocal = threading.local()

        def _emb(text: str) -> np.ndarray:
            if not hasattr(tlocal, "raw"):
                from mymodule.feature.ollama_embed import get_embedder

                tlocal.raw = get_embedder(emb_provider)
            return tlocal.raw.embed(text, instruction=None)

        results: list[dict] = []
        errors = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_emb, text): tid for tid, text in texts}
            with tqdm(total=len(futures), desc=f"  encode {table_name}", dynamic_ncols=True) as pbar:
                for fut in as_completed(futures):
                    tid = futures[fut]
                    try:
                        vec = fut.result()
                        results.append({"track_id": tid, "vector": vec.astype(np.float32)})
                    except Exception as e:
                        errors += 1
                        logger.warning(f"track {tid[:8]}… failed: {type(e).__name__}: {e}")
                    pbar.update(1)
                    if pbar.n % 1000 == 0:
                        pbar.set_postfix(rate=f"{pbar.n / (time.time() - t0):.1f}/s", err=errors)

        if not results:
            raise RuntimeError("No vectors produced — aborting LanceDB write.")
        db.create_table(table_name, results)
        logger.success(
            f"Created LanceDB table '{table_name}' with {len(results)} rows (errors={errors}, {time.time() - t0:.1f}s)"
        )
        _ensure_ann_index(db.open_table(table_name), metric="cosine")
        return cls(db)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _get_track_table(self, vector_type: str) -> lancedb.table.Table:
        if vector_type not in self._track_tables:
            self._track_tables[vector_type] = self._db.open_table(_TABLE_NAME[vector_type])
        return self._track_tables[vector_type]

    def search(self, vector_type: str, query_vector: np.ndarray, top_k: int = 100) -> list[str]:
        """ANN search over a single vector type → list of track_ids."""
        return [tid for tid, _ in self.search_with_scores(vector_type, query_vector, top_k)]

    def search_with_scores(
        self,
        vector_type: str,
        query_vector: np.ndarray,
        top_k: int = 100,
        nprobes: int = 200,
    ) -> list[tuple[str, float]]:
        """ANN search → list of (track_id, similarity_score); higher score = better match.

        - cosine metric: ``1 - _distance`` = cosine similarity
        - dot metric (cf-bpr): ``-_distance`` = inner product (LanceDB negates
          dot product into a distance)

        ``nprobes`` controls IVF_HNSW_SQ partition search width; default 200 gives
        top-100 recall ~99.7% at ~22ms on ~47K tracks. LanceDB falls back to
        brute force automatically when the index is missing.
        """
        table = self._get_track_table(vector_type)
        metric = "dot" if vector_type == "cf-bpr" else "cosine"
        results = table.search(query_vector.tolist()).metric(metric).nprobes(nprobes).limit(top_k).to_list()
        out: list[tuple[str, float]] = []
        for r in results:
            dist = float(r.get("_distance", 0.0))
            score = (1.0 - dist) if metric == "cosine" else -dist
            out.append((r["track_id"], score))
        return out

    def get_popular_tracks(self, top_k: int = 20) -> list[str]:
        """Track_ids ranked by train-set popularity."""
        if self._popularity is None:
            table = self._db.open_table(_POPULARITY_TABLE)
            rows = table.search().limit(top_k * 10).to_list()
            rows.sort(key=lambda r: r["rank"])
            self._popularity = [r["track_id"] for r in rows]
        return self._popularity[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB embedding store management")
    parser.add_argument("--build", action="store_true", help="Build LanceDB from HuggingFace datasets")
    parser.add_argument(
        "--build-metadata-rich",
        action="store_true",
        help="Build the metadata_rich LanceDB table via embedding provider (~25 min). "
        "By default also syncs the result into KV (rocksdict) so a single "
        "pass populates BOTH stores. Pass --no-kv-sync to skip.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel embedding workers (--build-metadata-rich)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks (debug)")
    parser.add_argument("--force", action="store_true", help="Drop existing table before rebuild")
    parser.add_argument(
        "--no-kv-sync",
        action="store_true",
        help="With --build-metadata-rich: skip the auto-sync to KV (LanceDB only).",
    )
    parser.add_argument(
        "--emb-provider",
        choices=["ollama", "openai"],
        default=None,
        help="Embedding provider override (default: MYMODULE_EMB_PROVIDER env, or ollama)",
    )
    args = parser.parse_args()

    if args.build:
        EmbeddingStore.build()
    elif args.build_metadata_rich:
        EmbeddingStore.build_track_metadata_rich(
            workers=args.workers, limit=args.limit, force=args.force, emb_provider=args.emb_provider
        )
        if not args.no_kv_sync:
            from mymodule.feature.kvdb import KVStore

            logger.info("Auto-syncing metadata_rich vectors LanceDB → KV (~9 sec, no Ollama) …")
            KVStore.sync_metadata_rich_from_lancedb()
            logger.success("Both stores ready.")
    else:
        parser.print_help()
