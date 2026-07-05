"""JSONL append + summary helpers — no KVStore, no HF.

Resume contract: `read_existing_ids(path)` returns the set of `track_id` already
present in `path`. The orchestrator skips them unless `--force` is set.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_existing_ids(path: Path) -> set[str]:
    """Return `track_id`s already in `path`. Missing/empty file → empty set."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines — treat them as if they weren't there so
                # the next crawl re-attempts. Don't raise.
                continue
            tid = rec.get("track_id")
            if isinstance(tid, str) and tid:
                out.add(tid)
    return out


def append_records(path: Path, records: Iterable[dict]) -> int:
    """Append records as JSONL. Returns count written. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_records(path: Path) -> Iterable[dict]:
    """Stream records back out of a JSONL file, skipping malformed lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


_TRACKED_FIELDS = (
    "lyrics",
    "caption",
    "chord",
    "tempo",
    "key",
    "mbid",
    "isrc",
    "label",
    "country",
    "mb_release_date",
    "mb_tags",
)


def write_summary(jsonl_path: Path, summary_path: Path | None = None) -> dict:
    """Compute and persist a summary JSON next to the JSONL file.

    Reports per-field coverage (count + %), per-field source breakdown,
    flag distribution, and cross-track duplicate-lyrics groups.
    """
    summary_path = summary_path or jsonl_path.with_suffix(jsonl_path.suffix + ".summary.json")
    flag_counts: Counter[str] = Counter()
    ok_count = 0
    total = 0
    field_present: Counter[str] = Counter()
    field_source: dict[str, Counter[str]] = {f: Counter() for f in _TRACKED_FIELDS}
    hash_to_tids: dict[str, list[str]] = defaultdict(list)

    for rec in iter_records(jsonl_path):
        total += 1
        if rec.get("ok") is True:
            ok_count += 1
        for f in rec.get("guardrail_flags") or []:
            flag_counts[f] += 1
        for field in _TRACKED_FIELDS:
            v = rec.get(field)
            if isinstance(v, list):
                if v:
                    field_present[field] += 1
            elif isinstance(v, str):
                if v.strip():
                    field_present[field] += 1
            elif v is not None:
                field_present[field] += 1
        sources_map = rec.get("sources") or {}
        for field in _TRACKED_FIELDS:
            src = sources_map.get(field)
            if src:
                field_source[field][src] += 1
        lyrics = rec.get("lyrics")
        if isinstance(lyrics, str) and lyrics.strip():
            h = hashlib.sha1(lyrics.strip().encode("utf-8")).hexdigest()[:16]
            hash_to_tids[h].append(rec.get("track_id", ""))

    duplicate_groups = sorted(
        ({"hash": h, "count": len(tids), "track_ids": tids[:10]} for h, tids in hash_to_tids.items() if len(tids) > 1),
        key=lambda g: -g["count"],
    )[:10]

    coverage_pct = {f: round(field_present[f] / total * 100, 1) if total else 0.0 for f in _TRACKED_FIELDS}

    summary = {
        "total": total,
        "ok_count": ok_count,
        "field_coverage_count": {f: field_present[f] for f in _TRACKED_FIELDS},
        "field_coverage_pct": coverage_pct,
        "field_source_breakdown": {f: dict(field_source[f]) for f in _TRACKED_FIELDS},
        "flag_counts": dict(flag_counts),
        "duplicate_lyrics_groups_top10": duplicate_groups,
        "unique_lyrics_count": len(hash_to_tids),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["read_existing_ids", "append_records", "iter_records", "write_summary"]
