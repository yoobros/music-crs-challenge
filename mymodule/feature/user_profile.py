"""User profile = `user:meta:*` (demographics) + `user:history:*` (prior interactions).

Bundles a user's demographic metadata and prior train-split session history
into a single `UserProfile` object so different callers (PAS responder,
retrieval pool, reranker, ...) share the same personalization signal.

Storage schema (RocksDB, key prefix `user:history:`):

    user:meta:{user_id}     → JSON (populated during kvdb build)
    user:history:{user_id}  → JSON
        {
            "sessions": [
                {
                    "session_id":   str,
                    "session_date": str (ISO date, e.g. "2011-12-26"),
                    "turn_pairs":   [[user_content, music_track_id], ...]
                },
                ...                  ← ascending session_date
            ],
            "n_sessions": int,
            "n_turns":    int
        }

`build_user_profiles()` scans the HF `train` split once, grouping each user's
conversation-music pairs by session (one-shot, ~3-5 min). At inference,
`get_user_profile()` is an O(1) point-lookup. Cold users (no train history or
missing user_id) yield `None` or an empty history.

Selection helpers:
- `select_prior_pairs(profile, filter_fn, top_k)` — pair-level filter
  (substring / cosine / etc.), caller-defined.
- `filter_before_date(profile, target_date, ...)` / `filter_excluding_session(...)`
  — session-level timing filters that avoid self-prediction leakage in
  train-side use (compile / fewshot / in-train evaluation). Sessions sharing
  the same session_date have no ordering, so `exclude_same_day=True` (default)
  conservatively drops them. At inference time (test/devset/blindset) all of
  train is in the past, so no timing filter is needed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from tqdm import tqdm

from mymodule.feature.kvdb import KVStore

_P_USER_HISTORY = "user:history:"

# Module-level once-only warning state for v1 schema detection. On rollout we
# silently degrade legacy `{"turn_pairs": [...]}` records to empty history (so
# inference doesn't crash), but the operator should rebuild — emit the
# instruction once per process.
_V1_LEGACY_WARNED = False


def _k_user_history(user_id: str) -> bytes:
    return f"{_P_USER_HISTORY}{user_id}".encode()


def _warn_v1_legacy_once() -> None:
    global _V1_LEGACY_WARNED
    if _V1_LEGACY_WARNED:
        return
    _V1_LEGACY_WARNED = True
    logger.warning(
        "[user_profile] user:history record uses legacy v1 schema (turn_pairs key). "
        "Rebuild with `uv run python -m mymodule.feature.user_profile --build` to "
        "upgrade to the v2 schema that carries session_id / session_date for timing "
        "filters. Until rebuilt, prior history is silently empty."
    )


# --------------------------------------------------------------------------
# Dataclass
# --------------------------------------------------------------------------


@dataclass
class PriorSession:
    """One prior conversation session within a user's history.

    `session_date` is the dataset's ISO date string (`YYYY-MM-DD`, e.g.
    `"2011-12-26"`) — kept as-is rather than parsed to allow lexicographic
    comparison without timezone ambiguity. `turn_pairs` is the same
    `(user_content, music_track_id)` shape as before, scoped to this session.
    """

    session_id: str
    session_date: str
    turn_pairs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class UserProfile:
    """Unified per-user view: demographic metadata + prior interaction history.

    `prior_sessions` is the source-of-truth (date-sorted ascending);
    `prior_pairs` / `prior_user_contents` / `prior_track_ids` are flattened
    convenience views over the entire history. `metadata` is whatever
    `KVStore.get_user_meta(user_id)` returned (age, country_code,
    preferred_language, etc.) — empty dict when KV has no entry.
    """

    user_id: str
    metadata: dict = field(default_factory=dict)
    prior_sessions: list[PriorSession] = field(default_factory=list)
    n_sessions: int = 0
    n_turns: int = 0

    @property
    def prior_pairs(self) -> list[tuple[str, str]]:
        """Flat view: all turn pairs across sessions, in session_date order."""
        return [pair for s in self.prior_sessions for pair in s.turn_pairs]

    @property
    def prior_user_contents(self) -> list[str]:
        return [uc for uc, _ in self.prior_pairs]

    @property
    def prior_track_ids(self) -> list[str]:
        return [tid for _, tid in self.prior_pairs]

    @property
    def has_metadata(self) -> bool:
        return bool(self.metadata)

    @property
    def has_history(self) -> bool:
        return any(s.turn_pairs for s in self.prior_sessions)

    @property
    def is_empty(self) -> bool:
        """True when neither metadata nor history is available."""
        return not self.has_metadata and not self.has_history


# --------------------------------------------------------------------------
# Build (train split → user:history:* KV)
# --------------------------------------------------------------------------


def _extract_pairs_from_session(session: dict[str, Any]) -> list[tuple[str, str]]:
    """Return paired ``(user_content, music_track_id)`` per turn within a session.

    TalkPlay sessions follow `user → music → assistant` order so the latest
    preceding user utterance maps to the next music GT. Music turns without
    any preceding user utterance are skipped (rare). Empty contents on either
    side are also skipped.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in session.get("conversations", []):
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "music":
            if pending_user is None:
                continue
            pairs.append((pending_user, content))
            pending_user = None
    return pairs


def build_user_profiles(kv: KVStore, hf_split: str = "train") -> dict[str, int]:
    """One-shot build: aggregate `hf_split` sessions per user_id → `user:history:*` KV.

    Each user's record stores sessions individually (not a flat pair list) so
    callers can apply session-level filters such as `filter_before_date`.
    Sessions are written in `session_date` ascending order.

    Idempotent (overwrites existing entries). user_metadata (`user:meta:*`)
    must be built separately via `kvdb --build` — this builder only touches
    the history namespace. Cold-start users (no train history) won't appear
    in KV; `get_user_profile()` handles them gracefully.

    Returns summary stats `{"users": N, "music_turns": M}` for logging.
    """
    from datasets import load_dataset  # heavy import — keep lazy

    logger.info(f"loading split {hf_split!r} for user-profile history aggregation …")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=hf_split)
    logger.info(f"sessions: {len(ds)}")

    # Group raw session dicts by user_id so we can sort each user's sessions
    # by session_date deterministically before write.
    per_user_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in tqdm(ds, desc=f"group by user_id [{hf_split}]"):
        uid = session.get("user_id") or ""
        if not uid:
            continue
        pairs = _extract_pairs_from_session(session)
        if not pairs:
            continue
        per_user_sessions[uid].append(
            {
                "session_id": session.get("session_id") or "",
                "session_date": session.get("session_date") or "",
                "turn_pairs": pairs,
            }
        )

    n_users = len(per_user_sessions)
    n_total_turns = sum(sum(len(s["turn_pairs"]) for s in sess) for sess in per_user_sessions.values())
    logger.info(f"unique users with history: {n_users}, total music turns: {n_total_turns}")

    logger.info("writing user:history:* to KV …")
    for uid, sessions in tqdm(per_user_sessions.items(), desc="kv write"):
        # Stable date-ascending order (empty date strings sort first).
        sessions_sorted = sorted(sessions, key=lambda s: (s["session_date"], s["session_id"]))
        record = {
            "sessions": sessions_sorted,
            "n_sessions": len(sessions_sorted),
            "n_turns": sum(len(s["turn_pairs"]) for s in sessions_sorted),
        }
        kv._db[_k_user_history(uid)] = json.dumps(record, ensure_ascii=False).encode()  # noqa: SLF001
    logger.success(f"build_user_profiles done — {n_users} users in KV")
    return {"users": n_users, "music_turns": n_total_turns}


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


def _get_history_raw(kv: KVStore, user_id: str) -> tuple[list[PriorSession], int, int]:
    """Internal: fetch history JSON → (sorted PriorSession list, n_sessions, n_turns).

    Returns `([], 0, 0)` when the user has no `user:history:` entry. The
    on-disk format is the v2 schema (`sessions: [...]`); legacy v1
    (`turn_pairs: [...]`) records are silently treated as empty so callers
    don't crash mid-rollout (rebuild via `--build` once to upgrade).
    """
    raw = kv._db.get(_k_user_history(user_id))  # noqa: SLF001
    if raw is None:
        return [], 0, 0
    rec = json.loads(raw)
    # Legacy v1 schema (flat turn_pairs without per-session metadata) cannot
    # carry timing info. Warn once and degrade to empty so the operator
    # rebuilds; safer than silently dropping the date dimension on read.
    if "sessions" not in rec and "turn_pairs" in rec:
        _warn_v1_legacy_once()
        return [], 0, 0
    sessions_raw = rec.get("sessions") or []
    sessions: list[PriorSession] = []
    for s in sessions_raw:
        pairs_raw = s.get("turn_pairs") or []
        pairs: list[tuple[str, str]] = [(uc, tid) for uc, tid in pairs_raw if uc and tid]
        if not pairs:
            continue
        sessions.append(
            PriorSession(
                session_id=s.get("session_id") or "",
                session_date=s.get("session_date") or "",
                turn_pairs=pairs,
            )
        )
    n_sessions = int(rec.get("n_sessions", len(sessions)))
    n_turns = int(rec.get("n_turns", sum(len(s.turn_pairs) for s in sessions)))
    return sessions, n_sessions, n_turns


def get_user_profile(kv: KVStore, user_id: str) -> UserProfile | None:
    """Return combined `UserProfile` for `user_id`, or None when truly unknown.

    "Truly unknown" = neither `user:meta:{uid}` nor `user:history:{uid}` exists.
    When only one half is present, the other is returned as empty.
    """
    if not user_id:
        return None
    try:
        meta = kv.get_user_meta(user_id) or {}
    except Exception:
        meta = {}
    sessions, n_sessions, n_turns = _get_history_raw(kv, user_id)
    if not meta and not sessions:
        return None
    return UserProfile(
        user_id=user_id,
        metadata=meta,
        prior_sessions=sessions,
        n_sessions=n_sessions,
        n_turns=n_turns,
    )


# --------------------------------------------------------------------------
# Session-level timing filters (leakage-safe variants for train-side use)
# --------------------------------------------------------------------------


def filter_before_date(
    profile: UserProfile,
    target_date: str,
    *,
    exclude_same_day: bool = True,
) -> UserProfile:
    """Return a NEW `UserProfile` with only sessions strictly before `target_date`.

    `target_date` is an ISO date string (`YYYY-MM-DD`). Comparison is
    lexicographic — equivalent to date ordering thanks to the ISO format.

    `exclude_same_day=True` (default, safest): keep only sessions whose
    `session_date < target_date`. Same-day siblings carry no time-of-day
    info in the dataset, so including them risks future-leakage.

    `exclude_same_day=False`: keep sessions with `session_date <= target_date`.
    Use only when same-day overlap is explicitly acceptable.

    Empty / missing `target_date` returns the profile unchanged.
    """
    if not target_date:
        return profile
    keep: list[PriorSession] = []
    for s in profile.prior_sessions:
        sd = s.session_date or ""
        if not sd:
            # No date → cannot order safely; drop in strict mode.
            if not exclude_same_day:
                keep.append(s)
            continue
        if exclude_same_day:
            if sd < target_date:
                keep.append(s)
        else:
            if sd <= target_date:
                keep.append(s)
    return UserProfile(
        user_id=profile.user_id,
        metadata=profile.metadata,
        prior_sessions=keep,
        n_sessions=len(keep),
        n_turns=sum(len(s.turn_pairs) for s in keep),
    )


def filter_excluding_session(profile: UserProfile, target_session_id: str) -> UserProfile:
    """Return a NEW `UserProfile` with `target_session_id` removed.

    Direct self-inclusion guard for train-side use (when the current session
    happens to be in the user's own history). No-op when `target_session_id`
    is empty or not present.
    """
    if not target_session_id:
        return profile
    keep = [s for s in profile.prior_sessions if s.session_id != target_session_id]
    if len(keep) == len(profile.prior_sessions):
        return profile  # no change → cheap return
    return UserProfile(
        user_id=profile.user_id,
        metadata=profile.metadata,
        prior_sessions=keep,
        n_sessions=len(keep),
        n_turns=sum(len(s.turn_pairs) for s in keep),
    )


# --------------------------------------------------------------------------
# Selection — pair-level filter-able accessor
# --------------------------------------------------------------------------


def select_prior_pairs(
    profile: UserProfile,
    *,
    filter_fn: Callable[[str, str], bool] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, str]]:
    """Filter + (optionally) head-truncate `profile.prior_pairs`.

    - `filter_fn(user_content, music_track_id) → bool` — caller-supplied
      predicate. Returns prior_pairs as-is when None (no filtering).
    - `top_k` — keep only the first K pairs that pass `filter_fn`. None →
      keep all that pass.

    Iteration order matches `profile.prior_pairs` (conversation order across
    all the user's train sessions). Caller chooses any selection strategy:
      • recency-only:    `select_prior_pairs(profile, top_k=5)`
      • substring match: `lambda uc, _: 'rock' in uc.lower()`
      • cosine-thresh:   closure capturing an embedder, see
                          `make_cosine_threshold_filter` for a default impl.
    """
    if not profile.prior_pairs:
        return []
    out: list[tuple[str, str]] = []
    for uc, tid in profile.prior_pairs:
        if filter_fn is not None and not filter_fn(uc, tid):
            continue
        out.append((uc, tid))
        if top_k is not None and len(out) >= top_k:
            break
    return out


# --------------------------------------------------------------------------
# Convenience filter factories
# --------------------------------------------------------------------------


def make_substring_filter(keyword: str, *, case_insensitive: bool = True) -> Callable[[str, str], bool]:
    """Filter: user_content contains `keyword` as a substring."""
    needle = keyword.lower() if case_insensitive else keyword

    def _f(uc: str, _tid: str) -> bool:
        haystack = uc.lower() if case_insensitive else uc
        return needle in haystack

    return _f


def make_cosine_threshold_filter(
    query: str,
    embedder: Any,
    *,
    threshold: float = 0.5,
) -> Callable[[str, str], bool]:
    """Filter: cosine(embed(query), embed(prior_user_content)) ≥ threshold.

    `embedder` is anything with a callable returning a 1D vector — typically
    `mymodule.feature.ollama_embed.OllamaEmbedder` (`.embed(text)`) or the
    instructed-query variant (`.embed_query(text)`). Both interfaces are
    detected at call time.

    Vectors are normalized so the dot product equals cosine similarity. On
    embedder failure (None / dim mismatch / exception), the filter returns
    False so the caller gracefully gets fewer pairs rather than crashing.
    """
    import numpy as np

    def _embed(text: str) -> Any:
        if hasattr(embedder, "embed_query"):
            return embedder.embed_query(text)
        if hasattr(embedder, "embed"):
            return embedder.embed(text)
        if callable(embedder):
            return embedder(text)
        raise AttributeError(f"embedder {type(embedder).__name__} has no embed/embed_query method")

    try:
        qv = _embed(query)
        if qv is None:
            return lambda *_args: False
        qarr = np.asarray(qv, dtype=np.float32)
        qnorm = qarr / (np.linalg.norm(qarr) + 1e-9)
    except Exception as e:
        logger.warning(f"[user_profile] cosine filter init failed ({type(e).__name__}: {e}); filter rejects all.")
        return lambda *_args: False

    def _f(uc: str, _tid: str) -> bool:
        try:
            pv = _embed(uc)
            if pv is None:
                return False
            parr = np.asarray(pv, dtype=np.float32)
            if parr.shape != qarr.shape:
                return False
            pnorm = parr / (np.linalg.norm(parr) + 1e-9)
            return float(np.dot(qnorm, pnorm)) >= threshold
        except Exception:
            return False

    return _f


def score_prior_pairs_by_query(
    profile: UserProfile,
    query: str,
    embedder: Any,
    *,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> list[tuple[str, str, float]]:
    """Return prior pairs scored against `query`, sorted strongest-first.

    Unlike `select_prior_pairs(filter_fn=make_cosine_threshold_filter(...))`
    — which yields only `(uc, tid)` in conversation order — this returns the
    cosine similarity alongside each pair so the caller can:
      • sort by relevance (strongest match to the current query first), and
      • attach a per-pair label (e.g., strong / moderate / weak).

    Behaviour:
      • Pairs below `threshold` are dropped (same gate semantics as the
        existing filter helper).
      • Result is sorted by `sim` descending. Ties keep conversation order.
      • `top_k` truncates AFTER sorting so the most relevant `k` win.
      • On embedder failure (None vector / dim mismatch / exception) the
        affected pair is skipped; full failure returns `[]` so the caller
        can fall back to recency.
    """
    import numpy as np

    if not profile.prior_pairs:
        return []

    def _embed(text: str) -> Any:
        if hasattr(embedder, "embed_query"):
            return embedder.embed_query(text)
        if hasattr(embedder, "embed"):
            return embedder.embed(text)
        if callable(embedder):
            return embedder(text)
        raise AttributeError(f"embedder {type(embedder).__name__} has no embed/embed_query method")

    try:
        qv = _embed(query)
        if qv is None:
            return []
        qarr = np.asarray(qv, dtype=np.float32)
        qnorm = qarr / (np.linalg.norm(qarr) + 1e-9)
    except Exception as e:
        logger.warning(
            f"[user_profile] score_prior_pairs_by_query init failed ({type(e).__name__}: {e}); returning empty list."
        )
        return []

    scored: list[tuple[str, str, float]] = []
    for uc, tid in profile.prior_pairs:
        try:
            pv = _embed(uc)
            if pv is None:
                continue
            parr = np.asarray(pv, dtype=np.float32)
            if parr.shape != qarr.shape:
                continue
            pnorm = parr / (np.linalg.norm(parr) + 1e-9)
            sim = float(np.dot(qnorm, pnorm))
            if sim >= threshold:
                scored.append((uc, tid, sim))
        except Exception:
            continue

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k] if top_k is not None else scored


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    p.add_argument(
        "--build",
        action="store_true",
        help="Aggregate train split into user:history:* (one-shot, ~3-5 min).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="HF dataset split used for --build (default: train).",
    )
    p.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="user_id to inspect after build (or stand-alone).",
    )
    args = p.parse_args()

    if args.build:
        kv = KVStore.open(read_only=False)
        stats = build_user_profiles(kv, hf_split=args.split)
        logger.success(f"build stats: {stats}")
        del kv

    if args.inspect:
        kv = KVStore.open_or_build()
        profile = get_user_profile(kv, args.inspect)
        if profile is None:
            print(f"user {args.inspect!r}: unknown (no metadata and no history)")
        else:
            print(f"user_id={profile.user_id}")
            print(f"  has_metadata={profile.has_metadata}, has_history={profile.has_history}")
            print(f"  metadata keys: {sorted(profile.metadata)}")
            print(f"  n_sessions={profile.n_sessions}, n_turns={profile.n_turns}")
            for i, s in enumerate(profile.prior_sessions[:3]):
                preview = s.turn_pairs[0][0] if s.turn_pairs else "(empty)"
                if len(preview) > 60:
                    preview = preview[:57] + "…"
                print(
                    f"  session {i}: id={s.session_id[:8]}… date={s.session_date} "
                    f"n_pairs={len(s.turn_pairs)} | first uc: {preview!r}"
                )
            if len(profile.prior_sessions) > 3:
                print(f"  … +{len(profile.prior_sessions) - 3} more sessions")


if __name__ == "__main__":
    main()
