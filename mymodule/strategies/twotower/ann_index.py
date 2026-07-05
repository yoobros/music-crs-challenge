"""LanceDB IVF_HNSW_SQ ANN index for TwoTower doc vectors.

Built once from doc_cache(.npz) and reused; follows the IVF_HNSW_SQ pattern
of feature/store.py.

Index path convention (same directory as doc_cache):
    data/doc_cache_<adapter>.npz
    data/doc_cache_<adapter>.ann.lancedb/   ← this file

Env vars:
    MYMODULE_TWOTOWER_ANN=1             : enable ANN (default 0 = exact matmul)
    MYMODULE_TWOTOWER_ANN_NPROBES=200   : partition search width
                                          47K / 256 partitions → nprobes=200 ≈ 99.7% recall@100
                                          nprobes=64 → ~99% recall@100 (3× faster)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

_ANN_NUM_PARTITIONS = 256  # ≈ sqrt(47K), same as feature/store.py
_DEFAULT_NPROBES = 200


def ann_index_path(doc_cache_path: Path) -> Path:
    """Derive the ANN index path from a doc_cache .npz path."""
    return doc_cache_path.with_suffix(".ann.lancedb")


def build_or_load_ann_index(
    tids: list[str],
    vecs: np.ndarray,
    index_path: Path,
) -> object:  # lancedb.table.Table
    """Load the ANN index, or build a new one.

    When index_path exists, only open_table runs (fast); otherwise an
    IVF_HNSW_SQ index is built from vecs.
    """
    import lancedb

    index_path = Path(index_path)

    if index_path.exists():
        try:
            db = lancedb.connect(str(index_path))
            table = db.open_table("tracks")
            logger.info(f"[twotower-ann] loaded ANN index ({table.count_rows()} tracks) ← {index_path}")
            return table
        except Exception as e:
            logger.warning(f"[twotower-ann] index corrupt, rebuilding: {e}")

    logger.info(f"[twotower-ann] building IVF_HNSW_SQ index ({len(tids)} tracks × {vecs.shape[1]} dim) → {index_path}")
    import pyarrow as pa

    db = lancedb.connect(str(index_path))
    data = pa.table(
        {
            "track_id": pa.array(tids, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vecs.flatten().tolist(), type=pa.float32()),
                vecs.shape[1],
            ),
        }
    )
    table = db.create_table("tracks", data=data, mode="overwrite")

    num_partitions = min(_ANN_NUM_PARTITIONS, max(1, len(tids) // 4))
    table.create_index(
        metric="cosine",
        index_type="IVF_HNSW_SQ",
        num_partitions=num_partitions,
    )
    logger.success(f"[twotower-ann] index built (partitions={num_partitions}, tracks={len(tids)})")
    return table


def search_ann(
    table: object,  # lancedb.table.Table
    query_vec: np.ndarray,
    k: int,
    nprobes: int = _DEFAULT_NPROBES,
) -> list[tuple[str, float]]:
    """ANN search; returns (track_id, cosine_similarity) descending."""
    results = table.search(query_vec.tolist(), query_type="vector").metric("cosine").nprobes(nprobes).limit(k).to_list()
    # LanceDB cosine distance = 1 − cosine_similarity (assumes L2-normed vectors)
    return [(r["track_id"], 1.0 - r["_distance"]) for r in results]
