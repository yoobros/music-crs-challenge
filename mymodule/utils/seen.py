"""Session-level seen-track filtering.

Each pool drops any candidate whose track_id already appeared in the
current session's chat_history as a `role=music` turn, guaranteeing the
pool returns exactly `n_candidates` unseen track_ids (search with
`n_candidates + len(seen)` headroom, then filter+truncate via `take_unseen`).
Ensemble fusion therefore operates on all-unseen candidates with no
downstream filter required.

Devset data exploration (see
https://github.com/yoobros/recsys-challenge-2026/issues/29#issuecomment-4305203211)
shows:

- **GT repeat rate within a session: 0.0000%** (0/7000 eligible interactions).
  Filtering session-seen tracks is provably lossless.
- ~12% of bm25 top-20 slots were wasted on previously-played tracks before
  this filter.
"""

from __future__ import annotations

import os
from collections.abc import Iterable


def extract_session_seen_tracks(chat_history: Iterable[dict] | None) -> set[str]:
    """Return the set of track_ids already played in this session.

    A track is "seen" when it appears as the `content` of a message whose
    `role` is `music`. Empty / malformed entries are skipped.
    """
    if not chat_history:
        return set()
    out: set[str] = set()
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "music":
            continue
        tid = msg.get("content")
        if isinstance(tid, str) and tid:
            out.add(tid)
    return out


def seen_filter_enabled() -> bool:
    """Honor `MYMODULE_SEEN_FILTER=0` as a kill-switch for A/B testing.

    Default on. Accept `"0"`, `"false"`, `"no"` (case-insensitive) to disable.
    """
    raw = (os.getenv("MYMODULE_SEEN_FILTER") or "").strip().lower()
    return raw not in {"0", "false", "no"}


def apply_seen_filter(
    candidates: list[str],
    chat_history: Iterable[dict] | None,
) -> list[str]:
    """Drop candidates whose track_id appeared in the session's music turns.

    Legacy strategy-level helper — prefer the pool-level `take_unseen`
    path (pools search with headroom so the filtered result keeps the
    full `n_candidates` width). Retained for ad-hoc callers and tests.

    No-op when `MYMODULE_SEEN_FILTER=0` or when `chat_history` has no music
    turns (e.g. turn 1). Preserves the original ordering of the surviving
    candidates.
    """
    if not seen_filter_enabled():
        return candidates
    seen = extract_session_seen_tracks(chat_history)
    if not seen:
        return candidates
    return [c for c in candidates if c not in seen]


def take_unseen(candidates: list[str], seen: set[str], limit: int) -> list[str]:
    """Filter `candidates` by `seen` (preserve order) then truncate to `limit`.

    Intended for pool-side use: the caller has already searched with
    `n_candidates + len(seen)` headroom, so after removing seen tracks the
    result still satisfies the pool's advertised `n_candidates` width
    (unless the raw retriever itself returned fewer).

    Honors `MYMODULE_SEEN_FILTER=0`: returns `candidates[:limit]` without
    filtering. This lets A/B runs disable the filter uniformly via env
    without touching pool internals.
    """
    if not seen_filter_enabled() or not seen:
        return candidates[:limit]
    out: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        out.append(c)
        if len(out) >= limit:
            break
    return out


def take_unseen_pairs(candidates: list[tuple[str, float]], seen: set[str], limit: int) -> list[tuple[str, float]]:
    """Score-aware variant of `take_unseen`; ``candidates`` are ``(track_id, score)`` tuples.

    Same filtering as ``take_unseen([t for t, _ in candidates], seen, limit)``
    but preserves scores. Used by pool ``generate_with_scores`` implementations.
    """
    if not seen_filter_enabled() or not seen:
        return candidates[:limit]
    out: list[tuple[str, float]] = []
    for tid, score in candidates:
        if tid in seen:
            continue
        out.append((tid, score))
        if len(out) >= limit:
            break
    return out
