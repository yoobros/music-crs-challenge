"""GBM (LightGBM lambdarank) reranker — self-contained package.

OOF feature build, LightGBM training, and inference (``GBMReranker``) all live
in this folder. Workflow:

    # 1. Build OOF parquet (fold-aware routing is automatic)
    uv run python -m mymodule.strategies.rerank.gbm.build_features --tid <tid>

    # 2. Train LightGBM → save ckpt
    uv run python -m mymodule.strategies.rerank.gbm.train --tid <tid>

``GBMReranker`` in this ``__init__.py`` is auto-registered under the name
``"gbm"`` via the ``pkgutil.iter_modules`` discovery in
``mymodule.strategies.rerank.__init__``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Import LightGBM before this package pulls in any other potentially native
# runtime. On macOS, loading the Booster after package-local imports can
# segfault in libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm as lgb  # noqa: E402
from loguru import logger

from mymodule.strategies.rerank.base import BaseReranker
from mymodule.strategies.rerank.gbm._specs import pool_signature
from mymodule.strategies.rerank.gbm.feature_transforms import apply_feature_mode

CKPT_ROOT = Path(__file__).parent / "ckpt"
DEFAULT_FALLBACK_FEATURE = "<unk>"

# Candidate window the reranker operates on (rerank on top of post-RRF results).
# The actual value comes from the trained ckpt meta ``top_k_rerank`` (keeps
# train/inference distributions aligned); this default applies only when absent.
TOP_K_RERANK_FALLBACK = 100


def _weighted_rank_fuse(
    original_order: list[str],
    gbm_order: list[str],
    original_weight: float,
    gbm_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    """Fuse original post-RRF order with GBM order using weighted reciprocal rank."""

    if original_weight <= 0:
        return list(gbm_order)

    all_tids = list(dict.fromkeys([*original_order, *gbm_order]))
    fallback_rank = len(all_tids) + 1
    original_rank = {tid: i + 1 for i, tid in enumerate(original_order)}
    gbm_rank = {tid: i + 1 for i, tid in enumerate(gbm_order)}

    def score(tid: str) -> float:
        return original_weight / (rrf_k + original_rank.get(tid, fallback_rank)) + gbm_weight / (
            rrf_k + gbm_rank.get(tid, fallback_rank)
        )

    return sorted(
        all_tids,
        key=lambda tid: (-score(tid), original_rank.get(tid, fallback_rank), gbm_rank.get(tid, fallback_rank)),
    )


def _original_rrf_weight_from_env() -> float:
    raw = os.getenv("MYMODULE_GBM_ORIGINAL_RRF_WEIGHT", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(f"Invalid MYMODULE_GBM_ORIGINAL_RRF_WEIGHT={raw!r}; using 0.0")
        return 0.0


def _logical_pool_groups(pool_names: list[str]) -> dict[str, list[str]]:
    """``["qemb_twotower_fold0", ..., "bm25"]`` → ``{"qemb_twotower": [...folds...], "bm25": ["bm25"]}``.

    BaseStrategy.predict passes expanded pool names, but OOF training features
    only have logical-name columns; normalize fold expansion here and rebuild a
    single logical score by averaging native scores.
    """
    import re

    _FOLD_SUFFIX = re.compile(r"^(.+)_fold\d+$")
    groups: dict[str, list[str]] = {}
    for n in pool_names:
        m = _FOLD_SUFFIX.match(n)
        if m:
            logical = m.group(1)
            groups.setdefault(logical, []).append(n)
        elif n.endswith("_all"):
            # strip the _all suffix to get the logical name
            logical = n[: -len("_all")]
            groups.setdefault(logical, []).append(n)
        else:
            groups.setdefault(n, []).append(n)
    return groups


def _logical_pool_rankings(
    pool_rankings: dict[str, list[tuple[str, float]]],
    pool_names: list[str],
) -> dict[str, list[tuple[str, float]]]:
    """Normalize expanded pool rankings from BaseStrategy to logical names.

    Multi-fold results are merged by **averaging native scores**. Merging via
    RRF would put scores on a completely different scale than the OOF training
    features (single-fold native scores); the native average keeps the
    train/inference distributions aligned.
    """
    from collections import defaultdict

    groups = _logical_pool_groups(pool_names)
    out: dict[str, list[tuple[str, float]]] = {}
    for logical_name, members in groups.items():
        if len(members) == 1 and members[0] == logical_name:
            out[logical_name] = pool_rankings.get(members[0], [])
            continue
        # Multi-member logical group: average native scores per track across
        # members. Tracks appearing in only some folds average over those folds;
        # missing entries are excluded from the mean, keeping the scale stable.
        accum: dict[str, list[float]] = defaultdict(list)
        for m in members:
            for tid, score in pool_rankings.get(m, []):
                accum[tid].append(float(score))
        merged = {tid: sum(scores) / len(scores) for tid, scores in accum.items()}
        out[logical_name] = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return out


class GBMReranker(BaseReranker):
    """LightGBM lambdarank reranker.

    Trained models live in ``mymodule/strategies/rerank/gbm/ckpt/{pool_signature}/``;
    the ``pool_names`` (or ``tid``) kwarg routes to the matching ckpt.

    At inference, features are built on the fly from ``pool_rankings`` /
    ``fused_scores`` / ``pool_names`` passed by ``BaseStrategy.predict`` and
    candidates are sorted by ``booster.predict``.
    """

    def __init__(
        self,
        pool_names: list[str] | None = None,
        tid: str | None = None,
        ckpt_root: str | Path | None = None,
        **kwargs,
    ) -> None:
        env_ckpt_root = os.getenv("MYMODULE_GBM_CKPT_ROOT")
        self._ckpt_root = Path(ckpt_root or env_ckpt_root) if (ckpt_root or env_ckpt_root) else CKPT_ROOT
        self._loaded_signature: str | None = None
        self.booster: Any = None  # lightgbm.Booster, lazy import
        self.meta: dict[str, Any] | None = None

        # Lazy auto-load: attempt ckpt loading immediately when pool_names is given.
        if pool_names is None and tid is not None:
            from mymodule.utils.tid import parse_tid

            _strategy, parsed_pool_names, _reranker = parse_tid(tid)
            pool_names = parsed_pool_names
        if pool_names:
            try:
                self._load_for(pool_names)
            except FileNotFoundError as e:
                logger.warning(f"GBMReranker: ckpt not found for {pool_names} ({e}); will no-op until trained.")

    # ------------------------------------------------------------------
    # ckpt loading
    # ------------------------------------------------------------------

    def _ckpt_dir_for(self, sig: str) -> Path:
        return self._ckpt_root / sig

    def _load_for(self, pool_names: list[str]) -> None:
        sig = pool_signature(_to_logical_pool_names(pool_names))
        ckpt_dir = self._ckpt_dir_for(sig)
        model_path = ckpt_dir / "model.txt"
        meta_path = ckpt_dir / "feature_meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"GBM ckpt missing in {ckpt_dir}")
        self.booster = lgb.Booster(model_file=str(model_path))
        self.meta = json.loads(meta_path.read_text())
        self._loaded_signature = sig
        logger.info(f"GBMReranker: loaded ckpt for signature '{sig}' from {ckpt_dir}")

    # ------------------------------------------------------------------
    # rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list[str],
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
        pool_rankings: dict[str, list[tuple[str, float]]] | None = None,
        fused_scores: dict[str, float] | None = None,
        pool_names: list[str] | None = None,
    ) -> list[str]:
        # Graceful no-op paths: missing signals / missing ckpt / pool combo mismatch.
        if pool_rankings is None or fused_scores is None or pool_names is None:
            return candidates
        try:
            # exclude the "ce" pseudo pool from the signature
            sig_pool_names = [n for n in pool_names if n != "ce"]
            target_sig = pool_signature(_to_logical_pool_names(sig_pool_names))
        except KeyError:
            return candidates  # unregistered pool combination (e.g. turn-1 fallback)

        if target_sig != self._loaded_signature:
            try:
                self._load_for(pool_names)
            except FileNotFoundError:
                return candidates

        assert self.booster is not None and self.meta is not None  # for type checker
        import numpy as np
        import pandas as pd

        # Rerank on top of post-RRF results: truncate candidates to the same
        # top-K used in training to keep train/inference distributions aligned.
        # ``candidates`` is the fused (post-RRF) result from
        # ``BaseStrategy.predict``, so taking the first N is sufficient.
        top_k_rerank = int(self.meta.get("top_k_rerank", TOP_K_RERANK_FALLBACK))
        candidates_top = candidates[:top_k_rerank]

        logical_rankings = _logical_pool_rankings(pool_rankings, pool_names)
        # RRF ranks are assigned within top-K only, matching the 1..top_k_rerank
        # range of ``rank_rrf`` in the training OOF parquet.
        fs_top = {tid: float(fused_scores.get(tid, 0.0)) for tid in candidates_top}
        rrf_sorted = sorted(fs_top.items(), key=lambda x: x[1], reverse=True)
        rrf_rank = {tid: r for r, (tid, _) in enumerate(rrf_sorted)}
        # per-logical-pool dict
        idx_by_pool: dict[str, dict[str, tuple[int, float]]] = {
            name: {tid: (r, s) for r, (tid, s) in enumerate(scored)} for name, scored in logical_rankings.items()
        }
        # Build feature matrix in the column order the model was trained with.
        feat_cols: list[str] = self.meta["feature_columns"]
        base_feat_cols: list[str] = self.meta.get("base_feature_columns", feat_cols)
        feature_mode = self.meta.get("feature_mode", "default")
        cat_orders: dict[str, list[str]] = self.meta.get("categorical_orders", {})
        booster_cats = getattr(self.booster, "pandas_categorical", None)
        known_cat_orders = [cat_orders[c] for c in ("category", "specificity") if c in cat_orders]
        if (
            "turn_number" in feat_cols
            and "turn_number" not in cat_orders
            and booster_cats
            and len(booster_cats) == len(known_cat_orders) + 1
            and booster_cats[1:] == known_cat_orders
        ):
            cat_orders = {"turn_number": booster_cats[0], **cat_orders}
        turn_number = max((m.get("turn_number", 0) for m in (chat_history or [])), default=0) + 1
        cg = conversation_goal or {}
        category = str(cg.get("category", DEFAULT_FALLBACK_FEATURE))
        specificity = str(cg.get("specificity", DEFAULT_FALLBACK_FEATURE))
        rows: list[dict[str, Any]] = []
        for tid in candidates_top:
            row: dict[str, Any] = {
                "rank_rrf": rrf_rank.get(tid, len(candidates_top)) + 1,
                "score_rrf": fs_top[tid],
                "turn_number": int(turn_number),
                "category": category,
                "specificity": specificity,
            }
            for name, idx in idx_by_pool.items():
                if tid in idx:
                    r, s = idx[tid]
                    row[f"rank_{name}"] = int(r) + 1
                    row[f"score_{name}"] = float(s)
                else:
                    row[f"rank_{name}"] = np.nan
                    row[f"score_{name}"] = np.nan
            rows.append(row)
        X = pd.DataFrame(rows)
        # CE pseudo pool "ce" → rename BEFORE apply_feature_mode
        # (that mode looks for ce_score/ce_rank, not rank_ce/score_ce)
        if "rank_ce" in X.columns:
            X = X.rename(columns={"rank_ce": "ce_rank", "score_ce": "ce_score"})
        X, _derived_feat_cols = apply_feature_mode(
            X,
            base_feat_cols,
            self.meta.get("pool_names", _to_logical_pool_names(pool_names)),
            feature_mode,
        )
        target_encoding_meta = self.meta.get("target_encoding")
        if target_encoding_meta:
            from mymodule.strategies.rerank.gbm.target_encoding import transform_target_encoding

            X = transform_target_encoding(X, target_encoding_meta)
        # fill missing training columns with NaN
        for col in feat_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feat_cols]
        # Assign categorical dtypes using the category orders from training.
        for cat_col, order in cat_orders.items():
            if cat_col in X.columns:
                X[cat_col] = pd.Categorical(X[cat_col], categories=order)

        scores = self.booster.predict(X)
        # Stable sort: mergesort on negated scores preserves original order on ties.
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        # Candidates beyond top-K are appended in their original RRF order; the
        # caller (BaseStrategy) truncates to the final top_k anyway.
        reordered = [candidates_top[int(i)] for i in order]
        original_rrf_weight = _original_rrf_weight_from_env()
        if original_rrf_weight > 0:
            reordered = _weighted_rank_fuse(candidates_top, reordered, original_weight=original_rrf_weight)
        rest = candidates[top_k_rerank:]
        return reordered + rest


def _to_logical_pool_names(pool_names: list[str]) -> list[str]:
    """Normalize expanded pool names (e.g. qemb_twotower_fold0..4) to the
    logical names recognized by ``_specs.KNOWN_POOL_SPECS``."""
    import re

    _FOLD_SUFFIX = re.compile(r"^(.+)_fold\d+$")
    out: list[str] = []
    for n in pool_names:
        m = _FOLD_SUFFIX.match(n)
        if m:
            out.append(m.group(1))
        else:
            out.append(n)
    return out
