"""Evaluation script with per-turn / user-split / specificity / category filters.

Computes nDCG + Recall + diversity itself (no evaluator submodule changes),
with turn / warm-cold / HH-HL-LH-LL / A-K filtering.

Usage:
    # all turns, all users
    uv run python scripts/evaluate_by_turn.py --tid <tid> [<tid2> ...]

    # turn 1 only
    uv run python scripts/evaluate_by_turn.py --tid <tid> --turn 1

    # warm-cold breakdown (overall + warm + cold)
    uv run python scripts/evaluate_by_turn.py --tid <tid> --split-breakdown

    # warm only
    uv run python scripts/evaluate_by_turn.py --tid <tid> --user-split warm

    # specificity breakdown (overall + HH + HL + LH + LL)
    uv run python scripts/evaluate_by_turn.py --tid <tid> --specificity-breakdown

    # single specificity filter
    uv run python scripts/evaluate_by_turn.py --tid <tid> --specificity HH

    # category breakdown (overall + A-K)
    uv run python scripts/evaluate_by_turn.py --tid <tid> --category-breakdown

    # single category filter
    uv run python scripts/evaluate_by_turn.py --tid <tid> --category H

    # turn breakdown (overall + T1..T8)
    uv run python scripts/evaluate_by_turn.py --tid <tid> --turn-breakdown

    # combined filters (AND-chain): warm + HH + category H
    uv run python scripts/evaluate_by_turn.py --tid <tid> --user-split warm --specificity HH --category H

    # all breakdowns at once (additive)
    uv run python scripts/evaluate_by_turn.py --tid <tid> --split-breakdown \\
        --specificity-breakdown --category-breakdown --turn-breakdown

Rows with fewer than SMALL_N_THRESHOLD matched turns get a `[small N=<count>]` marker.
"""

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = PROJECT_ROOT / "music-crs-evaluator" / "exp" / "ground_truth" / "devset.json"
EVALUATOR_INFERENCE_ROOT = PROJECT_ROOT / "music-crs-evaluator" / "exp" / "inference"
MYMODULE_INFERENCE_ROOT = PROJECT_ROOT / "mymodule" / "exp" / "inference"
TOTAL_CATALOG_SIZE = 47071


def ndcg_at_k(predicted: list[str], ground_truth_id: str, k: int) -> float:
    for i, tid in enumerate(predicted[:k]):
        if tid == ground_truth_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def recall_at_k(predicted: list[str], ground_truth_id: str, k: int) -> float:
    """One ground-truth track per turn, so Recall@K = Hit Rate @K (binary per turn)."""
    return 1.0 if ground_truth_id in predicted[:k] else 0.0


def catalog_diversity(all_track_ids: list[list[str]]) -> float:
    unique = set()
    for tracks in all_track_ids:
        unique.update(tracks)
    return len(unique) / TOTAL_CATALOG_SIZE if TOTAL_CATALOG_SIZE else 0.0


def lexical_diversity(all_responses: list[str]) -> float:
    total_bigrams = 0
    unique_bigrams: set[tuple[str, str]] = set()
    for resp in all_responses:
        tokens = resp.lower().split()
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            unique_bigrams.add(bigram)
            total_bigrams += 1
    return len(unique_bigrams) / total_bigrams if total_bigrams else 0.0


NDCG_KS = [1, 10, 20]
RECALL_KS = [20, 100]
SMALL_N_THRESHOLD = 50  # rows with fewer matched turns get a noise-warning marker


def _normalize_split(raw: str | None) -> str:
    """devset user_profile.user_split(`test_warm`, `test_cold`) → `warm` / `cold` / `unknown`."""
    if not raw:
        return "unknown"
    s = str(raw).lower().replace("test_", "").strip()
    return s if s in {"warm", "cold", "train"} else "unknown"


@lru_cache(maxsize=1)
def _session_split_map() -> dict[str, str]:
    """session_id → normalized user_split from HF devset(test split)."""
    from datasets import load_dataset

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    return {s["session_id"]: _normalize_split((s.get("user_profile") or {}).get("user_split")) for s in ds}


SPECIFICITY_VALUES = ("HH", "HL", "LH", "LL")
CATEGORY_VALUES = tuple(chr(ord("A") + i) for i in range(11))  # A~K


def _normalize_specificity(raw: str | None) -> str:
    """conversation_goal.specificity (HH/HL/LH/LL) → upper/strip; else 'unknown'."""
    if not raw:
        return "unknown"
    s = str(raw).strip().upper()
    return s if s in SPECIFICITY_VALUES else "unknown"


def _normalize_category(raw: str | None) -> str:
    """conversation_goal.category (A~K) → upper/strip; else 'unknown'."""
    if not raw:
        return "unknown"
    s = str(raw).strip().upper()
    return s if s in CATEGORY_VALUES else "unknown"


@lru_cache(maxsize=1)
def _session_specificity_map() -> dict[str, str]:
    """session_id → conversation_goal.specificity from HF devset(test split)."""
    from datasets import load_dataset

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    return {s["session_id"]: _normalize_specificity((s.get("conversation_goal") or {}).get("specificity")) for s in ds}


@lru_cache(maxsize=1)
def _session_category_map() -> dict[str, str]:
    """session_id → conversation_goal.category from HF devset(test split)."""
    from datasets import load_dataset

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    return {s["session_id"]: _normalize_category((s.get("conversation_goal") or {}).get("category")) for s in ds}


def _find_inference_path(tid: str, dataset: str) -> Path | None:
    path = EVALUATOR_INFERENCE_ROOT / dataset / f"{tid}.json"
    if path.exists():
        return path
    alt = MYMODULE_INFERENCE_ROOT / dataset / f"{tid}.json"
    if alt.exists():
        return alt
    return None


def _compute(
    preds: list[dict],
    gt: dict[tuple, str],
) -> dict:
    ndcg_scores = {k: [] for k in NDCG_KS}
    recall_scores = {k: [] for k in RECALL_KS}
    all_tracks: list[list[str]] = []
    all_responses: list[str] = []

    for pred in preds:
        key = (pred["session_id"], pred["turn_number"])
        gt_id = gt.get(key)
        if gt_id is None:
            continue
        for k in NDCG_KS:
            ndcg_scores[k].append(ndcg_at_k(pred["predicted_track_ids"], gt_id, k))
        for k in RECALL_KS:
            recall_scores[k].append(recall_at_k(pred["predicted_track_ids"], gt_id, k))
        all_tracks.append(pred["predicted_track_ids"])
        resp = pred.get("predicted_response", "")
        if resp:
            all_responses.append(resp)

    result = {}
    for k in NDCG_KS:
        result[f"nDCG@{k}"] = sum(ndcg_scores[k]) / len(ndcg_scores[k]) if ndcg_scores[k] else 0.0
    for k in RECALL_KS:
        result[f"R@{k}"] = sum(recall_scores[k]) / len(recall_scores[k]) if recall_scores[k] else 0.0
    result["catalog_div"] = catalog_diversity(all_tracks)
    result["lexical_div"] = lexical_diversity(all_responses)
    result["matched"] = len(ndcg_scores[1])
    result["max_pred_len"] = max((len(p["predicted_track_ids"]) for p in preds), default=0)
    return result


def _emit_stratum(
    tid: str,
    label: str,
    subset: list[dict],
    gt: dict[tuple, str],
    turn_breakdown: bool,
) -> list[dict]:
    """Emit one row for `subset` (labeled `label`), plus one row per turn when
    `turn_breakdown` is True. Per-turn label is `T{n}` for the overall stratum,
    else `{label}/T{n}` (e.g., `warm/T1`, `HH/T1`, `warm+HH+H/T1`)."""
    out = [{"tid": tid, "split": label, **_compute(subset, gt)}]
    if turn_breakdown:
        for t in sorted({r["turn_number"] for r in subset}):
            tsub = [r for r in subset if r["turn_number"] == t]
            tlabel = f"T{t}" if label == "overall" else f"{label}/T{t}"
            out.append({"tid": tid, "split": tlabel, **_compute(tsub, gt)})
    return out


def evaluate(
    tid: str,
    gt: dict[tuple, str],
    turns: list[int] | None,
    dataset: str,
    split_filter: str | None,
    split_breakdown: bool,
    spec_filter: str | None = None,
    spec_breakdown: bool = False,
    cat_filter: str | None = None,
    cat_breakdown: bool = False,
    turn_breakdown: bool = False,
) -> list[dict]:
    """Return list of rows, one per stratum. Each row has `tid`, `split` (label
    column shared by user-split, specificity, category, and turn), and metrics.

    Filters AND-chain: any combination of `split_filter`/`spec_filter`/`cat_filter`
    intersects (e.g., warm+HH+H). When any filter is active, *categorical*
    breakdowns are ignored (only the filtered subset row is emitted), but
    `turn_breakdown` still applies and expands that subset by turn.
    Categorical breakdowns are additive: when all are set, the output contains
    overall + 2 user-split rows + 4 specificity rows + 11 category rows.
    `turn_breakdown` further expands each of those rows into per-turn rows.
    """
    path = _find_inference_path(tid, dataset)
    if path is None:
        return [{"tid": tid, "split": "—", "error": "not found"}]

    preds = json.load(open(path))
    if turns:
        preds = [r for r in preds if r["turn_number"] in turns]

    rows: list[dict] = []
    need_split = bool(split_filter or split_breakdown)
    need_spec = bool(spec_filter or spec_breakdown)
    need_cat = bool(cat_filter or cat_breakdown)
    ssplit = _session_split_map() if need_split else {}
    sspec = _session_specificity_map() if need_spec else {}
    scat = _session_category_map() if need_cat else {}

    # Combined-filter mode: AND-chain across any subset of {split,spec,cat}.
    if split_filter or spec_filter or cat_filter:
        subset = preds
        labels: list[str] = []
        if split_filter:
            tgt = split_filter.lower()
            subset = [r for r in subset if ssplit.get(r["session_id"]) == tgt]
            labels.append(tgt)
        if spec_filter:
            tgt = spec_filter.upper()
            subset = [r for r in subset if sspec.get(r["session_id"]) == tgt]
            labels.append(tgt)
        if cat_filter:
            tgt = cat_filter.upper()
            subset = [r for r in subset if scat.get(r["session_id"]) == tgt]
            labels.append(tgt)
        rows.extend(_emit_stratum(tid, "+".join(labels), subset, gt, turn_breakdown))
        return rows

    # Breakdown mode (or default overall).
    rows.extend(_emit_stratum(tid, "overall", preds, gt, turn_breakdown))

    if split_breakdown:
        for label in ("warm", "cold"):
            subset = [r for r in preds if ssplit.get(r["session_id"]) == label]
            if subset:
                rows.extend(_emit_stratum(tid, label, subset, gt, turn_breakdown))

    if spec_breakdown:
        for label in SPECIFICITY_VALUES:
            subset = [r for r in preds if sspec.get(r["session_id"]) == label]
            if subset:
                rows.extend(_emit_stratum(tid, label, subset, gt, turn_breakdown))

    if cat_breakdown:
        for label in CATEGORY_VALUES:
            subset = [r for r in preds if scat.get(r["session_id"]) == label]
            if subset:
                rows.extend(_emit_stratum(tid, label, subset, gt, turn_breakdown))

    return rows


def main():
    parser = argparse.ArgumentParser(description="Per-turn / user-split filterable evaluation")
    parser.add_argument("--tid", nargs="+", required=True, help="tid(s) to evaluate")
    parser.add_argument("--turn", nargs="*", type=int, default=None, help="turn numbers to evaluate (default: all)")
    parser.add_argument(
        "--dataset",
        default="devset",
        help="Inference dataset subdir (e.g. devset, devset_topk100). Recall@100 needs top-k>=100.",
    )
    parser.add_argument(
        "--user-split",
        choices=["warm", "cold"],
        default=None,
        help="Filter to one user_split (warm/cold); suppresses overall/breakdown rows.",
    )
    parser.add_argument(
        "--split-breakdown",
        action="store_true",
        help="Also print per warm/cold metrics in addition to overall.",
    )
    parser.add_argument(
        "--specificity",
        choices=list(SPECIFICITY_VALUES),
        default=None,
        help="Filter to one conversation_goal.specificity (HH/HL/LH/LL); prints only that row.",
    )
    parser.add_argument(
        "--specificity-breakdown",
        action="store_true",
        help="Also print per HH/HL/LH/LL metrics (additive with --split-breakdown).",
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_VALUES),
        default=None,
        help="Filter to one conversation_goal.category (A-K); prints only that row.",
    )
    parser.add_argument(
        "--category-breakdown",
        action="store_true",
        help="Also print per A-K metrics (additive with other breakdowns).",
    )
    parser.add_argument(
        "--turn-breakdown",
        action="store_true",
        help="Additionally expand each stratum per turn (T1, T2, ...); additive with other breakdowns/filters.",
    )
    args = parser.parse_args()

    gt_all = json.load(open(GROUND_TRUTH_PATH))
    gt = {(r["session_id"], r["turn_number"]): r["ground_truth_track_id"] for r in gt_all}
    if args.turn:
        gt = {k: v for k, v in gt.items() if k[1] in args.turn}

    turn_label = f"turn {args.turn}" if args.turn else "all turns"
    active_filters = [
        ("split", args.user_split),
        ("specificity", args.specificity),
        ("category", args.category),
    ]
    active_filters = [(k, v) for k, v in active_filters if v]
    if active_filters:
        strata = "filter=" + "+".join(f"{k}:{v}" for k, v in active_filters)
        if args.turn_breakdown:
            strata += " +turn-breakdown"
    else:
        parts = []
        if args.split_breakdown:
            parts.append("warm/cold")
        if args.specificity_breakdown:
            parts.append("HH/HL/LH/LL")
        if args.category_breakdown:
            parts.append("A~K")
        if args.turn_breakdown:
            parts.append("Tn")
        strata = "split=overall" if not parts else f"breakdown=overall+{'+'.join(parts)}"
    print(f"Evaluating: {turn_label} ({len(gt)} gt), dataset={args.dataset}, {strata}\n")

    split_w = 14  # combined filter labels (e.g., "warm+HH+H") may exceed 8 chars
    header = (
        f"{'tid':<40} {'split':>{split_w}} "
        f"{'nDCG@1':>7} {'nDCG@10':>8} {'nDCG@20':>8} "
        f"{'R@20':>7} {'R@100':>7} {'cat_div':>8} {'lex_div':>8} {'len':>4}"
    )
    print(header)
    print("-" * len(header))

    for tid in args.tid:
        rows = evaluate(
            tid=tid,
            gt=gt,
            turns=args.turn,
            dataset=args.dataset,
            split_filter=args.user_split,
            split_breakdown=args.split_breakdown,
            spec_filter=args.specificity,
            spec_breakdown=args.specificity_breakdown,
            cat_filter=args.category,
            cat_breakdown=args.category_breakdown,
            turn_breakdown=args.turn_breakdown,
        )
        for row in rows:
            if "error" in row:
                print(f"{row['tid']:<40} {row['split']:>{split_w}} {row['error']}")
            else:
                small = f"  [small N={row['matched']}]" if row["matched"] < SMALL_N_THRESHOLD else ""
                print(
                    f"{row['tid']:<40} {row['split']:>{split_w}} "
                    f"{row['nDCG@1']:>7.4f} {row['nDCG@10']:>8.4f} "
                    f"{row['nDCG@20']:>8.4f} {row['R@20']:>7.4f} {row['R@100']:>7.4f} "
                    f"{row['catalog_div']:>8.4f} {row['lexical_div']:>8.4f} {row['max_pred_len']:>4}"
                    f"{small}"
                )


if __name__ == "__main__":
    main()
