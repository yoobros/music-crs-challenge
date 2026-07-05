"""CLI: load the OOF parquet, train LightGBM lambdarank, save the ckpt.

Usage:
    uv run python -m mymodule.strategies.rerank.gbm.train --tid <tid>

Outputs:
- ``mymodule/strategies/rerank/gbm/ckpt/{pool_signature}/model.txt``
- ``mymodule/strategies/rerank/gbm/ckpt/{pool_signature}/feature_meta.json``

GBMReranker auto-loads from the same directory at inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# On macOS, if lightgbm's OpenMP loads *after* mymodule deps (datasets/pyarrow),
# lgb.train can segfault. So (1) set the KMP env at process start and (2) import
# lightgbm first to pin the OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from mymodule.cv import CVConfig, split_items  # noqa: E402
from mymodule.strategies.rerank.gbm._specs import get_pool_specs, pool_signature  # noqa: E402
from mymodule.strategies.rerank.gbm.feature_transforms import (  # noqa: E402
    BASE_CAT_COLS,
    apply_feature_mode,
    categorical_columns,
)
from mymodule.utils.tid import parse_tid  # noqa: E402

CKPT_ROOT = Path(__file__).parent / "ckpt"
OOF_ROOT = Path(__file__).parent / "oof"


def _resolve_feat_path(arg_path: "str | None", sig: str) -> "Path | None":
    """Resolve the OOF feature location.

    Layout precedence:
    1. explicit ``--features`` argument (file or dir)
    2. partitioned dir: ``{OOF_ROOT}/{sig}/``
    3. legacy single file: ``{OOF_ROOT}/{sig}.parquet``
    """
    if arg_path:
        p = Path(arg_path)
        return p if p.exists() else None
    p_dir = OOF_ROOT / sig
    if p_dir.is_dir() and any(p_dir.iterdir()):
        return p_dir
    p_file = OOF_ROOT / f"{sig}.parquet"
    if p_file.is_file():
        return p_file
    return None


def _read_oof(feat_path: Path) -> pd.DataFrame:
    """Read either a partitioned dir or a single .parquet into a DataFrame."""
    if feat_path.is_dir():
        # pyarrow dataset reader merges all partitions (turn_number restored)
        import pyarrow.parquet as pq

        return pq.read_table(feat_path).to_pandas()
    return pd.read_parquet(feat_path)


def _feature_columns(df: pd.DataFrame, pool_names: list[str]) -> list[str]:
    """Determine the feature-column order for training.

    The OOF parquet holds ``rank_<pool>`` / ``score_<pool>`` columns per logical
    pool, plus RRF and categorical columns.
    """
    specs = get_pool_specs(pool_names)
    cols: list[str] = []
    for spec in specs:
        for prefix in ("rank_", "score_"):
            c = f"{prefix}{spec.name}"
            if c in df.columns:
                cols.append(c)
            else:
                logger.warning(f"feature column '{c}' missing in OOF parquet — skipping")
    # RRF + categorical
    cols += ["rank_rrf", "score_rrf", "turn_number"]
    cols += [c for c in BASE_CAT_COLS if c in df.columns]
    # CE score columns are added by crossenc feature_mode in apply_feature_mode()
    return cols


def _make_dataset(df: pd.DataFrame, feat_cols: list[str], cat_orders: dict[str, list[str]]):
    """Build an ``lgb.Dataset`` with (session_id, turn_number) group sizes.

    The DataFrame must be pre-sorted by (session_id, turn_number).
    """

    df = df.sort_values(["session_id", "turn_number"]).reset_index(drop=True)
    group = df.groupby(["session_id", "turn_number"], sort=False).size().tolist()
    X = df[feat_cols].copy()
    if "turn_number" in X.columns:
        X["turn_number"] = X["turn_number"].astype("int16")
    cats_in_use = categorical_columns(feat_cols)
    for c in cats_in_use:
        X[c] = pd.Categorical(X[c], categories=cat_orders[c])
    return lgb.Dataset(
        X,
        label=df["label"].astype(int),
        group=group,
        categorical_feature=cats_in_use,
        free_raw_data=False,
    )


def _monotone_constraints(feat_cols: list[str]) -> list[int]:
    """LightGBM monotone constraints matching rank/score feature semantics."""

    constraints: list[int] = []
    for col in feat_cols:
        if col.startswith("rank_") or col == "ce_rank":
            constraints.append(-1)  # lower rank = better
        elif col.startswith("score_") or col in ("ce_score", "ce_mean"):
            constraints.append(1)  # higher score = better
        else:
            constraints.append(0)
    return constraints


def _build_lgb_params(args: argparse.Namespace, feat_cols: list[str]) -> dict:
    params = dict(
        objective=args.objective,
        metric="ndcg",
        ndcg_eval_at=[20],
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        min_gain_to_split=args.min_gain_to_split,
        cat_l2=args.cat_l2,
        cat_smooth=args.cat_smooth,
        min_data_per_group=args.min_data_per_group,
        feature_fraction_seed=args.seed,
        bagging_seed=args.seed,
        deterministic=True,
        # Large lambdarank datasets can segfault in LightGBM's OpenMP row-wise
        # path on macOS. Single-threaded col-wise training is slower but stable.
        num_threads=1,
        force_col_wise=True,
        verbose=-1,
    )
    if args.lambdarank_truncation_level is not None:
        params["lambdarank_truncation_level"] = args.lambdarank_truncation_level
    if args.monotone_rank_score:
        params["monotone_constraints"] = _monotone_constraints(feat_cols)
        params["monotone_constraints_method"] = "intermediate"
    return params


def main() -> None:
    p = argparse.ArgumentParser(description="GBM reranker — LightGBM lambdarank trainer")
    p.add_argument("--tid", required=True, help="TID whose pool part defines the feature set")
    p.add_argument("--features", type=str, default=None, help="override the OOF parquet path")
    p.add_argument("--out-dir", type=str, default=None, help="override the ckpt output dir")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # Conservative defaults — the dominant score_rrf feature plus the train↔eval
    # distribution gap makes overfitting likely; keep capacity low and suppress
    # variance via depth/leaf-data limits, L1/L2, and bagging, with a small
    # learning rate, many boost rounds, and long early stopping.
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--max-depth", type=int, default=5, help="tree depth cap (strong overfit control)")
    p.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=500,
        help="min samples per leaf (conservative default for this dataset size)",
    )
    p.add_argument("--feature-fraction", type=float, default=0.7, help="column subsample per iteration")
    p.add_argument("--bagging-fraction", type=float, default=0.8, help="row subsample per iteration")
    p.add_argument("--bagging-freq", type=int, default=5, help="bagging period (every k iters)")
    p.add_argument("--lambda-l1", type=float, default=0.1, help="L1 regularization")
    p.add_argument("--lambda-l2", type=float, default=1.0, help="L2 regularization")
    p.add_argument("--min-gain-to-split", type=float, default=0.0, help="min gain to split")
    p.add_argument(
        "--feature-mode",
        choices=["default", "no_rrf", "expert", "crossenc"],
        default="default",
        help=(
            "default: current features; no_rrf: drop rank_rrf/score_rrf; "
            "expert: no_rrf + source disagreement features; "
            "crossenc: ce_score + pool (no RRF)"
        ),
    )
    p.add_argument("--objective", choices=["lambdarank", "rank_xendcg"], default="lambdarank")
    p.add_argument(
        "--lambdarank-truncation-level",
        type=int,
        default=None,
        help="LambdaRank top cutoff focus",
    )
    p.add_argument("--cat-l2", type=float, default=10.0, help="categorical split L2 regularization")
    p.add_argument("--cat-smooth", type=float, default=10.0, help="categorical noise smoothing")
    p.add_argument("--min-data-per-group", type=int, default=100, help="minimum rows per categorical group")
    p.add_argument(
        "--monotone-rank-score",
        action="store_true",
        help="Constrain rank_* decreasing and score_* increasing for robustness experiments",
    )
    p.add_argument(
        "--target-encoding",
        choices=["none", "rank"],
        default="none",
        help="Add smoothed target-encoding features from coarse rank/category buckets.",
    )
    p.add_argument("--te-alpha", type=float, default=50.0, help="Target encoding smoothing strength")
    p.add_argument("--te-folds", type=int, default=5, help="Cross-fit folds for train target encoding rows")
    p.add_argument("--num-boost-round", type=int, default=5000)
    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument(
        "--ce-oof",
        type=str,
        default=None,
        help="CrossEncoder OOF scores parquet (ce_oof_scores.parquet). JOIN on session_id/turn_number/track_id.",
    )
    args = p.parse_args()

    _strategy, pool_names, reranker = parse_tid(args.tid)
    if reranker != "gbm":
        logger.warning(f"TID reranker part is '{reranker}', not 'gbm'; using the pool part only.")
    sig = pool_signature(pool_names)

    # Layouts: ``{sig}/`` (partitioned) and legacy ``{sig}.parquet`` (single file).
    # Precedence: explicit argument > partition dir > legacy single file.
    feat_path = _resolve_feat_path(args.features, sig)
    if feat_path is None:
        raise FileNotFoundError(
            f"OOF feature parquet not found at {OOF_ROOT}/{sig}/ or {OOF_ROOT}/{sig}.parquet. "
            f"Run build_features first:"
            f"\n  uv run python -m mymodule.strategies.rerank.gbm.build_features --tid {args.tid}"
        )

    logger.info(f"Loading OOF features: {feat_path}")
    df = _read_oof(feat_path)
    logger.info(f"Rows={len(df)}, cols={len(df.columns)}, positives={df['label'].sum()}")

    # CE OOF scores JOIN (optional)
    if args.ce_oof:
        ce_path = Path(args.ce_oof)
        if not ce_path.exists():
            raise FileNotFoundError(f"CE OOF parquet not found: {ce_path}")
        ce_df = pd.read_parquet(ce_path)
        logger.info(f"CE OOF scores loaded: {len(ce_df)} rows, columns={list(ce_df.columns)}")
        # ce_all_scores.parquet may carry ce_score_fold0..4 → derive ce_mean + ce_std
        fold_score_cols = [c for c in ce_df.columns if c.startswith("ce_score_fold")]
        if fold_score_cols:
            logger.info(f"computing ce_mean/ce_std from {fold_score_cols}")
            ce_df["ce_mean"] = ce_df[fold_score_cols].mean(axis=1).astype("float32")
            ce_df["ce_std"] = ce_df[fold_score_cols].std(axis=1).fillna(0.0).astype("float32")
            ce_df["ce_score"] = ce_df["ce_mean"]
        join_cols = ["session_id", "turn_number", "track_id"]
        n_before = len(df)
        df = df.merge(ce_df, on=join_cols, how="left")
        n_matched = df["ce_score"].notna().sum()
        logger.info(f"CE OOF JOIN: {n_before} rows → {n_matched} matched ({n_matched / n_before:.1%})")
        # ce_rank: within-group rank by ce_score (lower = better)
        df["ce_rank"] = df.groupby(["session_id", "turn_number"])["ce_score"].rank(
            ascending=False, method="min", na_option="bottom"
        )

    base_feat_cols = _feature_columns(df, pool_names)
    df, feat_cols = apply_feature_mode(df, base_feat_cols, pool_names, args.feature_mode)
    logger.info(f"Feature columns ({len(feat_cols)}): {feat_cols}")

    # Pre-extract categorical orders so training and inference use identical ones.
    cat_orders: dict[str, list[str]] = {}
    for c in categorical_columns(feat_cols):
        if c in df.columns:
            cat_orders[c] = sorted(df[c].dropna().astype(str).unique().tolist())

    # Group-aware train/valid split (by session) via `mymodule.cv.split_items`.
    session_ids = df["session_id"].drop_duplicates().tolist()
    train_idx, val_idx = split_items(
        session_ids,
        group_fn=lambda s: s,
        config=CVConfig(),  # is_active=False → val_ratio split
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )
    train_sids = {session_ids[i] for i in train_idx}
    val_sids = {session_ids[i] for i in val_idx}
    df_tr = df[df["session_id"].isin(train_sids)]
    df_va = df[df["session_id"].isin(val_sids)]
    logger.info(
        f"Train: {len(df_tr)} rows ({len(train_sids)} sessions) / Valid: {len(df_va)} rows ({len(val_sids)} sessions)"
    )

    target_encoding_meta = None
    if args.target_encoding != "none":
        from mymodule.strategies.rerank.gbm.target_encoding import fit_transform_target_encoding

        df_tr, df_va, target_encoding_meta = fit_transform_target_encoding(
            df_tr,
            df_va,
            feat_cols,
            mode=args.target_encoding,
            alpha=args.te_alpha,
            folds=args.te_folds,
            seed=args.seed,
        )
        feat_cols += target_encoding_meta.get("feature_columns", [])
        logger.info(
            f"Target encoding enabled: mode={args.target_encoding}, "
            f"features={target_encoding_meta.get('feature_columns', [])}"
        )

    ds_tr = _make_dataset(df_tr, feat_cols, cat_orders)
    ds_va = _make_dataset(df_va, feat_cols, cat_orders)

    params = _build_lgb_params(args, feat_cols)
    booster = lgb.train(
        params,
        ds_tr,
        num_boost_round=args.num_boost_round,
        valid_sets=[ds_tr, ds_va],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)],
    )

    out_dir = Path(args.out_dir) if args.out_dir else CKPT_ROOT / sig
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    # Max rows per (session, turn) group = the RRF top-K used in training;
    # record it in meta so inference truncates to the same top-K.
    top_k_rerank = int(df.groupby(["session_id", "turn_number"], sort=False).size().max())
    meta = {
        "feature_columns": feat_cols,
        "base_feature_columns": base_feat_cols,
        "feature_mode": args.feature_mode,
        "categorical_orders": cat_orders,
        "pool_names": [s.name for s in get_pool_specs(pool_names)],
        "top_k_rerank": top_k_rerank,
        "params": params,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "best_iteration": booster.best_iteration,
        "best_score": dict(booster.best_score) if booster.best_score else None,
        "target_encoding": target_encoding_meta,
    }
    (out_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.success(f"Saved GBM ckpt → {out_dir}")


if __name__ == "__main__":
    main()
