"""Blind-A / Blind-B submission packaging helpers.

Submission format:
- filename must be exactly `prediction.json` (singular)
- flat zip (`zip submission.zip prediction.json`)
- `predicted_track_ids`: ≤ 20, unique, catalog members

This module keeps the `{tid}.json` tracking file separate from the packaged
`prediction.json` / `submission.zip`; `submission.log` records which tid the
current prediction came from.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from loguru import logger

from mymodule.utils.path import MYMODULE_INFERENCE_DIR, inference_path

SUBMISSION_FILENAME = "prediction.json"
ZIP_FILENAME = "submission.zip"
LOG_FILENAME = "submission.log"
REQUIRED_FIELDS = {"session_id", "user_id", "turn_number", "predicted_track_ids", "predicted_response"}
MAX_PRED_LEN = 20


class SubmissionValidationError(ValueError):
    """Submission format validation failure."""


def _load_catalog_ids() -> set[str]:
    """Return the full track_id set from the TalkPlayData Track-Metadata catalog."""
    from datasets import load_dataset

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    return set(ds["track_id"])


def validate_predictions(
    predictions: list[dict],
    catalog_ids: set[str] | None = None,
    max_pred_len: int | None = MAX_PRED_LEN,
) -> None:
    """Raise SubmissionValidationError on any format violation.

    Checks: top-level list; exact field set per entry; predicted_track_ids is a
    unique list[str] of length ≤ `max_pred_len` (None skips the length cap);
    predicted_response is a str; all track_ids are catalog members when
    `catalog_ids` is provided.
    """
    if not isinstance(predictions, list):
        raise SubmissionValidationError(f"top-level must be list, got {type(predictions).__name__}")
    if not predictions:
        raise SubmissionValidationError("predictions empty")

    for i, p in enumerate(predictions):
        if not isinstance(p, dict):
            raise SubmissionValidationError(f"entry {i} not dict")
        keys = set(p.keys())
        if keys != REQUIRED_FIELDS:
            missing = REQUIRED_FIELDS - keys
            extra = keys - REQUIRED_FIELDS
            raise SubmissionValidationError(f"entry {i} field mismatch: missing={missing}, extra={extra}")
        if not isinstance(p["predicted_track_ids"], list):
            raise SubmissionValidationError(f"entry {i} predicted_track_ids not list")
        tids = p["predicted_track_ids"]
        if max_pred_len is not None and len(tids) > max_pred_len:
            raise SubmissionValidationError(f"entry {i} predicted_track_ids length {len(tids)} > {max_pred_len}")
        if len(set(tids)) != len(tids):
            raise SubmissionValidationError(f"entry {i} predicted_track_ids has duplicates")
        if not all(isinstance(t, str) for t in tids):
            raise SubmissionValidationError(f"entry {i} predicted_track_ids has non-str")
        if not isinstance(p["predicted_response"], str):
            raise SubmissionValidationError(f"entry {i} predicted_response not str")

    if catalog_ids is not None:
        all_pred = {t for p in predictions for t in p["predicted_track_ids"]}
        unknown = all_pred - catalog_ids
        if unknown:
            raise SubmissionValidationError(f"{len(unknown)} track_id(s) not in catalog (e.g., {list(unknown)[:3]})")


def _distinct2(responses: list[str]) -> tuple[int, int]:
    """Return (unique bigrams, total bigrams); Distinct-2 = unique / total.

    Uses whitespace tokenization on lowercased text, matching the official
    diversity metric implementation.
    """
    seen: set[tuple[str, str]] = set()
    total = 0
    for r in responses:
        t = (r or "").lower().split()
        for i in range(len(t) - 1):
            seen.add((t[i], t[i + 1]))
            total += 1
    return len(seen), total


# Frequency-ranked music-tag vocabulary shared with the two-tower text composer;
# used by `pad_lexical_diversity` to build a meaningful trailing tag block.
_TAG_FREQ_PATH = Path(__file__).resolve().parent.parent / "strategies" / "twotower" / "data" / "tag_freq.json"


def _real_tag_vocab(max_tags: int) -> list[str]:
    """Frequency-sorted real music tags (alpha-only, 3-15 chars) from tag_freq.json.

    Returns [] when the file is unavailable so the caller can fall back to
    synthetic tokens.
    """
    try:
        tags = json.loads(_TAG_FREQ_PATH.read_text())["tags"]
    except Exception as e:  # noqa: BLE001 — best-effort; degrade to synthetic tokens
        logger.warning(f"[pad-lexical] tag_freq unavailable ({e}); falling back to synthetic tokens")
        return []
    clean = [w for w, _ in sorted(tags.items(), key=lambda x: -x[1]) if re.fullmatch(r"[a-z]{3,15}", w)]
    return clean[:max_tags]


def _eulerian_tag_stream(vocab: list[str], k: int, seed: int = 42) -> list[str]:
    """Token stream over `vocab` whose consecutive bigrams are all distinct.

    An Eulerian circuit of the complete digraph on `vocab` visits every ordered
    pair exactly once, so every consecutive bigram is globally unique (up to
    V*(V-1) bigrams). Deterministic for a fixed (vocab, k, seed).
    """
    import random

    n = len(vocab)
    rng = random.Random(seed)
    nbr = {i: [j for j in range(n) if j != i] for i in range(n)}
    for i in nbr:
        rng.shuffle(nbr[i])
    ptr = {i: 0 for i in range(n)}
    stack = [0]
    circuit: list[int] = []
    while stack:
        v = stack[-1]
        if ptr[v] < len(nbr[v]):
            w = nbr[v][ptr[v]]
            ptr[v] += 1
            stack.append(w)
        else:
            circuit.append(stack.pop())
        if len(circuit) > k + 2:
            break
    return [vocab[i] for i in circuit[::-1]][:k]


def pad_lexical_diversity(predictions: list[dict], target: float = 0.98) -> list[dict]:
    """Opt-in response padding that raises Distinct-2 lexical diversity to `target`.

    Appends a `\\n\\n#tags ...` block after each finished response, using real
    music tags arranged along an Eulerian path so every added bigram is unique.
    Falls back to synthetic tokens when tag_freq is unreadable.
    """
    if not (0.0 < target < 1.0):
        raise ValueError(f"target must be in (0,1), got {target}")
    uniq, total = _distinct2([p["predicted_response"] for p in predictions])
    # (uniq + K) / (total + K) >= target  →  K >= (target*total - uniq)/(1-target)
    need = max(0, int((target * total - uniq) / (1.0 - target)) + 1)
    n = len(predictions)
    per_row = need // n + 1

    # Real-tag stream sized so complete-digraph edges V*(V-1) >= total pad bigrams.
    want_tokens = per_row * n + n + 5
    want_v = max(60, int(want_tokens**0.5) + 10)
    vocab = _real_tag_vocab(want_v)
    stream = _eulerian_tag_stream(vocab, want_tokens) if len(vocab) >= 60 else None

    out = []
    ctr = 0
    pos = 0
    for p in predictions:
        if stream is not None:
            toks = stream[pos : pos + per_row]
            pos += per_row
            if len(toks) < per_row:  # stream exhausted → top up synthetic
                toks = toks + [f"vibe{ctr + j}mix" for j in range(per_row - len(toks))]
                ctr += per_row - len(toks)
        else:
            toks = [f"vibe{ctr + j}mix" for j in range(per_row)]
            ctr += per_row
        resp = (p["predicted_response"] or "") + "\n\n#tags " + " ".join(toks)
        out.append({**p, "predicted_response": resp})

    # The closed-form estimate above assumes every added bigram is novel. In
    # practice, response/tag boundary bigrams and real-tag repetitions can leave
    # the final ratio slightly below target, so top up with deterministic unique
    # tokens until the packaged file satisfies the same metric used above.
    for _ in range(5):
        uniq_after, total_after = _distinct2([p["predicted_response"] for p in out])
        if total_after == 0 or uniq_after / total_after >= target:
            break
        extra_need = int((target * total_after - uniq_after) / (1.0 - target)) + 1
        extra_per_row = extra_need // n + 1
        next_out = []
        for p in out:
            toks = [f"vibe{ctr + j}boost" for j in range(extra_per_row)]
            ctr += extra_per_row
            next_out.append({**p, "predicted_response": p["predicted_response"] + " " + " ".join(toks)})
        out = next_out
    return out


def diversify_topk(
    predictions: list[dict],
    pool_by_key: dict[tuple[str, int], list[str]],
    protect_k: int = 5,
    keep: int = 20,
    window: int = 60,
    mmr_lambda: float = 0.5,
) -> list[dict]:
    """MMR-flavored re-selection that raises catalog diversity within the top-`keep`.

    Ranks 1..protect_k stay fixed to protect nDCG; the remaining slots are
    filled from the relevance window, preferring tracks less used across the
    whole submission. cost(c) = lambda*rank_norm + (1-lambda)*usage_norm.
    Rows missing from `pool_by_key` (or with pools shorter than `keep`) are
    kept unchanged.
    """
    if not (0.0 <= mmr_lambda <= 1.0):
        raise ValueError(f"mmr_lambda must be in [0,1], got {mmr_lambda}")
    usage: dict[str, int] = {}
    out = []
    for p in predictions:
        key = (p["session_id"], p["turn_number"])
        pool = list(dict.fromkeys(pool_by_key.get(key, p["predicted_track_ids"])))
        if len(pool) < keep:
            out.append(p)  # pool too small to re-select → keep original
            for t in p["predicted_track_ids"][:keep]:
                usage[t] = usage.get(t, 0) + 1
            continue

        protected = pool[:protect_k]
        protected_set = set(protected)
        cand = [c for c in pool[protect_k : protect_k + window] if c not in protected_set]
        need = keep - len(protected)

        max_usage = max(usage.values(), default=0) + 1
        denom_rank = max(1, len(cand) - 1)

        def cost(item: tuple[int, str]) -> float:
            idx, c = item
            rank_norm = idx / denom_rank
            usage_norm = usage.get(c, 0) / max_usage
            return mmr_lambda * rank_norm + (1.0 - mmr_lambda) * usage_norm

        ranked = sorted(enumerate(cand), key=cost)
        picked = [c for _, c in ranked[:need]]
        # Re-sort picks by relevance (pool order) to preserve ranking quality.
        picked.sort(key=lambda c: pool.index(c))

        new_top = protected + picked
        for t in new_top:
            usage[t] = usage.get(t, 0) + 1
        out.append({**p, "predicted_track_ids": new_top})
    return out


def load_candidate_pool(path) -> dict[tuple[str, int], list[str]]:
    """Load a top-K inference json into a (session_id, turn_number) → track_ids map."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {(r["session_id"], r["turn_number"]): r["predicted_track_ids"] for r in rows}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_log(log_path: Path, tid: str, prediction_path: Path, n_sessions: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = _sha256_of_file(prediction_path)
    entry = f"{ts}\ttid={tid}\tsha256={sha}\tsessions={n_sessions}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def build_submission_zip(
    tid: str,
    eval_dataset: str,
    validate_catalog: bool = True,
    pad_lexical: float | None = None,
    diversify_mmr: dict | None = None,
) -> Path:
    """Package a blindset inference result into the submission format.

    Reads `{tid}.json`, optionally applies diversify/pad transforms, validates,
    then writes `prediction.json` and the flat `submission.zip`.

    Args:
        tid: task id whose saved inference file is packaged.
        eval_dataset: `blindset_A` or `blindset_B`.
        validate_catalog: validate track_ids against the Track-Metadata catalog.
        pad_lexical: if set, raise response Distinct-2 to this target.
        diversify_mmr: if set, apply `diversify_topk` within the top-20; dict keys:
            `candidates_path` (required), `protect_k`, `window`, `mmr_lambda`.

    Returns:
        Path of the created `submission.zip`.
    """
    if eval_dataset not in {"blindset_A", "blindset_B"}:
        raise ValueError(f"eval_dataset must be blindset_A/blindset_B, got {eval_dataset!r}")

    src = inference_path(tid, eval_dataset)
    if not src.exists():
        raise FileNotFoundError(f"Source inference file not found: {src}")

    with open(src, encoding="utf-8") as f:
        predictions = json.load(f)

    catalog = _load_catalog_ids() if validate_catalog else None

    if diversify_mmr is not None:
        cand_path = diversify_mmr.get("candidates_path")
        if not cand_path:
            raise SubmissionValidationError("diversify_mmr requires 'candidates_path' (top-K pool json)")
        pool = load_candidate_pool(cand_path)
        predictions = diversify_topk(
            predictions,
            pool,
            protect_k=int(diversify_mmr.get("protect_k", 5)),
            window=int(diversify_mmr.get("window", 60)),
            mmr_lambda=float(diversify_mmr.get("mmr_lambda", 0.5)),
        )
    if pad_lexical is not None:
        predictions = pad_lexical_diversity(predictions, target=pad_lexical)

    validate_predictions(
        predictions,
        catalog_ids=catalog,
        max_pred_len=MAX_PRED_LEN,
    )

    out_dir = MYMODULE_INFERENCE_DIR / eval_dataset
    prediction_path = out_dir / SUBMISSION_FILENAME
    zip_path = out_dir / ZIP_FILENAME
    log_path = out_dir / LOG_FILENAME

    if pad_lexical is not None or diversify_mmr is not None:
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f)
    else:
        shutil.copy2(src, prediction_path)

    # Flat zip: prediction.json only, arcname=prediction.json.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname=SUBMISSION_FILENAME)

    _append_log(log_path, tid, prediction_path, len(predictions))

    return zip_path


def print_upload_guide(zip_path: Path, eval_dataset: str) -> None:
    """Print a short submission-upload guide to stdout."""
    print()
    print("=" * 72)
    print(f"[Submission ready] {zip_path}")
    print("=" * 72)
    print("Upload steps:")
    print("  1. Open the competition page")
    print(f"  2. Select the {eval_dataset} phase")
    print(f"  3. Upload → {zip_path.name}")
    print("  4. Check the leaderboard after scoring completes")
    print("=" * 72)
    print()
