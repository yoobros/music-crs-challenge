"""Shared GBM feature transforms for train and inference."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

BASE_CAT_COLS = ["category", "specificity"]
DERIVED_CAT_COLS = ["best_pool"]


def categorical_columns(feat_cols: list[str]) -> list[str]:
    return [c for c in [*BASE_CAT_COLS, *DERIVED_CAT_COLS] if c in feat_cols]


def _without_rrf(feat_cols: list[str]) -> list[str]:
    return [c for c in feat_cols if c not in {"rank_rrf", "score_rrf"}]


def _rank_cols(pool_names: list[str], columns: list[str]) -> list[str]:
    available = set(columns)
    return [f"rank_{name}" for name in pool_names if f"rank_{name}" in available]


def _add_expert_rank_features(df: pd.DataFrame, pool_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    rank_cols = _rank_cols(pool_names, list(out.columns))
    if not rank_cols:
        return out, []

    ranks = out[rank_cols].apply(pd.to_numeric, errors="coerce")
    derived_cols = [
        "pool_hit_count",
        "pool_min_rank",
        "pool_mean_rank",
        "pool_rank_std",
        "pool_rank_span",
        "best_pool",
    ]
    out["pool_hit_count"] = ranks.notna().sum(axis=1).astype("int16")
    out["pool_min_rank"] = ranks.min(axis=1).astype("float32")
    out["pool_mean_rank"] = ranks.mean(axis=1).astype("float32")
    out["pool_rank_std"] = ranks.std(axis=1).fillna(0.0).astype("float32")
    out["pool_rank_span"] = (ranks.max(axis=1) - ranks.min(axis=1)).fillna(0.0).astype("float32")
    out["best_pool"] = ranks.idxmin(axis=1).fillna("rank_<none>").str.removeprefix("rank_").astype(str)

    for left, right in combinations(rank_cols, 2):
        left_pool = left.removeprefix("rank_")
        right_pool = right.removeprefix("rank_")
        col = f"rank_gap_{left_pool}__{right_pool}"
        out[col] = (ranks[left] - ranks[right]).astype("float32")
        derived_cols.append(col)

    return out, derived_cols


def apply_feature_mode(
    df: pd.DataFrame,
    feat_cols: list[str],
    pool_names: list[str],
    feature_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply optional GBM feature-mode transforms.

    Modes:
    - ``default``: existing feature set.
    - ``no_rrf``: remove rank_rrf/score_rrf shortcut features.
    - ``expert``: remove RRF shortcut features and add candidate-source
      disagreement features from per-pool ranks.
    """

    if feature_mode == "default":
        return df.copy(), list(feat_cols)
    if feature_mode == "no_rrf":
        return df.copy(), _without_rrf(feat_cols)
    if feature_mode == "expert":
        out, derived_cols = _add_expert_rank_features(df, pool_names)
        return out, [*_without_rrf(feat_cols), *derived_cols]
    if feature_mode == "crossenc":
        # ce_score + per-pool signals, RRF excluded
        base = _without_rrf(feat_cols)
        ce_cols = [c for c in ["ce_score", "ce_rank", "ce_mean", "ce_std"] if c in df.columns]
        return df.copy(), [*ce_cols, *base]
    raise ValueError(f"unknown GBM feature mode: {feature_mode}")
