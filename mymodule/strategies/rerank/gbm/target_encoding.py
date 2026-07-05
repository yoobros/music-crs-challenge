"""Leakage-aware target encoding helpers for GBM reranking.

The encoder intentionally uses only coarse, inference-available fields:
rank buckets, category, specificity, and turn number. It does not encode
track_id/user_id/session_id, because those would be high-cardinality shortcuts
with poor blindset generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mymodule.cv import CVConfig, make_folds

RANK_BUCKETS = ("r1", "r2_3", "r4_10", "r11_20", "r21_50", "r51_100", "r101p", "missing")


@dataclass(frozen=True)
class TargetEncodingSource:
    out_col: str
    source_col: str
    kind: str


def infer_sources(feature_columns: list[str], mode: str) -> list[TargetEncodingSource]:
    """Choose target-encoding sources from existing GBM feature columns."""
    if mode == "none":
        return []
    if mode != "rank":
        raise ValueError(f"unknown target encoding mode: {mode}")

    sources: list[TargetEncodingSource] = []
    for col in feature_columns:
        if col.startswith("rank_"):
            sources.append(TargetEncodingSource(out_col=f"te_{col}", source_col=col, kind="rank_bucket"))
    for col in ("category", "specificity", "turn_number"):
        if col in feature_columns:
            sources.append(TargetEncodingSource(out_col=f"te_{col}", source_col=col, kind="category"))
    return sources


def _key_series(df: pd.DataFrame, source: TargetEncodingSource) -> pd.Series:
    if source.source_col not in df.columns:
        return pd.Series(["missing"] * len(df), index=df.index, dtype="object")

    s = df[source.source_col]
    if source.kind == "rank_bucket":
        values = pd.to_numeric(s, errors="coerce")
        bucketed = pd.cut(
            values,
            bins=[0, 1, 3, 10, 20, 50, 100, float("inf")],
            labels=list(RANK_BUCKETS[:-1]),
            right=True,
        )
        out = bucketed.astype("object")
        return out.where(values.notna(), "missing").astype(str)

    if source.kind == "category":
        return s.where(s.notna(), "missing").astype(str)

    raise ValueError(f"unknown target encoding source kind: {source.kind}")


def _fit_mapping(
    df: pd.DataFrame,
    source: TargetEncodingSource,
    *,
    label_col: str,
    alpha: float,
    global_mean: float,
) -> dict[str, float]:
    keys = _key_series(df, source)
    stats = df[label_col].astype(float).groupby(keys, sort=True).agg(["sum", "count"])
    values = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
    return {str(k): float(v) for k, v in values.items()}


def _transform_with_mappings(
    df: pd.DataFrame,
    sources: list[TargetEncodingSource],
    mappings: dict[str, dict[str, float]],
    *,
    global_mean: float,
) -> pd.DataFrame:
    out = df.copy()
    for source in sources:
        keys = _key_series(out, source)
        mapping = mappings.get(source.out_col, {})
        out[source.out_col] = keys.map(mapping).fillna(global_mean).astype("float32")
    return out


def _fit_state(
    df: pd.DataFrame,
    sources: list[TargetEncodingSource],
    *,
    label_col: str,
    alpha: float,
    global_mean: float,
) -> dict[str, Any]:
    mappings = {
        source.out_col: _fit_mapping(
            df,
            source,
            label_col=label_col,
            alpha=alpha,
            global_mean=global_mean,
        )
        for source in sources
    }
    return {
        "version": 1,
        "alpha": float(alpha),
        "global_mean": float(global_mean),
        "feature_columns": [source.out_col for source in sources],
        "sources": [
            {
                "out_col": source.out_col,
                "source_col": source.source_col,
                "kind": source.kind,
                "mapping": mappings[source.out_col],
            }
            for source in sources
        ],
    }


def fit_transform_target_encoding(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    feature_columns: list[str],
    *,
    mode: str = "rank",
    label_col: str = "label",
    session_col: str = "session_id",
    alpha: float = 50.0,
    folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cross-fit train target encodings and fit train-only state for inference."""
    sources = infer_sources(feature_columns, mode)
    if not sources:
        return df_train.copy(), df_valid.copy(), {}

    global_mean = float(df_train[label_col].astype(float).mean())
    full_state = _fit_state(
        df_train,
        sources,
        label_col=label_col,
        alpha=alpha,
        global_mean=global_mean,
    )
    full_sources = [
        TargetEncodingSource(
            out_col=src["out_col"],
            source_col=src["source_col"],
            kind=src["kind"],
        )
        for src in full_state["sources"]
    ]
    full_mappings = {src["out_col"]: src["mapping"] for src in full_state["sources"]}
    df_valid_te = _transform_with_mappings(df_valid, full_sources, full_mappings, global_mean=global_mean)

    df_train_te = df_train.copy()
    for col in full_state["feature_columns"]:
        df_train_te[col] = global_mean

    session_ids = df_train[session_col].drop_duplicates().tolist()
    if folds <= 1 or len(session_ids) < 2:
        df_train_te = _transform_with_mappings(df_train, full_sources, full_mappings, global_mean=global_mean)
        return df_train_te, df_valid_te, full_state

    cfg = CVConfig(num_folds=min(folds, len(session_ids)), fold_idx=None, split_seed=seed)
    cv_folds = make_folds(session_ids, group_fn=lambda s: s, config=cfg)
    for train_idx, holdout_idx in cv_folds:
        train_sids = {session_ids[i] for i in train_idx}
        holdout_sids = {session_ids[i] for i in holdout_idx}
        fit_part = df_train[df_train[session_col].isin(train_sids)]
        holdout_mask = df_train[session_col].isin(holdout_sids)
        fold_mean = float(fit_part[label_col].astype(float).mean()) if len(fit_part) else global_mean
        fold_state = _fit_state(
            fit_part,
            sources,
            label_col=label_col,
            alpha=alpha,
            global_mean=fold_mean,
        )
        fold_sources = [
            TargetEncodingSource(
                out_col=src["out_col"],
                source_col=src["source_col"],
                kind=src["kind"],
            )
            for src in fold_state["sources"]
        ]
        fold_mappings = {src["out_col"]: src["mapping"] for src in fold_state["sources"]}
        transformed = _transform_with_mappings(
            df_train.loc[holdout_mask],
            fold_sources,
            fold_mappings,
            global_mean=fold_mean,
        )
        for col in full_state["feature_columns"]:
            df_train_te.loc[holdout_mask, col] = transformed[col].to_numpy()

    return df_train_te, df_valid_te, full_state


def transform_target_encoding(df: pd.DataFrame, state: dict[str, Any] | None) -> pd.DataFrame:
    """Apply a saved target-encoding state to inference rows."""
    if not state or not state.get("sources"):
        return df

    sources = [
        TargetEncodingSource(
            out_col=src["out_col"],
            source_col=src["source_col"],
            kind=src["kind"],
        )
        for src in state["sources"]
    ]
    mappings = {src["out_col"]: src.get("mapping", {}) for src in state["sources"]}
    return _transform_with_mappings(
        df,
        sources,
        mappings,
        global_mean=float(state.get("global_mean", 0.0)),
    )
