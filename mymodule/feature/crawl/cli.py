"""Multi-source track metadata crawler — JSONL output + inline guardrails.

Source priority per field (free sources only):

    Field            Priority                                    Coverage*
    ─────────────────────────────────────────────────────────────────────
    lyrics           talkpl_cache → lrclib → genius (if token)   ~95%
    caption          talkpl_cache only                           ~29%
    chord            talkpl_cache only                           ~10-20%
    tempo            talkpl_cache only                           ~29%
    key              talkpl_cache only                           ~29%
    mbid/isrc/...    musicbrainz                                 ~70-90%
    mb_tags          musicbrainz                                 ~25-40%

    * estimates from the n=30 smoke; actual numbers in `.summary.json`
      after each run. talkpl_cache covers ~29% of the challenge dataset
      (13.7K of 47K tracks) — caption/chord/tempo/key are bottlenecked
      there since no free fallback exists.

Usage:

    uv run python -m mymodule.feature.crawl --limit 30 --no-genius
    uv run python -m mymodule.feature.crawl --output path/to/lyrics.jsonl

Public API:

    from mymodule.feature.crawl import (
        load_talkpl_lookup, fetch_lrclib, fetch_musicbrainz,
        validate_record, is_ok, write_summary,
    )

Per-field source provenance is recorded in `record["sources"]`. Guardrail
flags are namespaced (`lyrics_*`, `caption_*`, `tempo_*`, ...). Top-level
`ok` requires the critical fields (lyrics + caption) to pass without flags.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from mymodule.feature.crawl import guardrails, jsonl, sources, talkpl

DEFAULT_OUTPUT = Path("mymodule/feature/.crawl/lyrics.jsonl")
HF_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
HF_SPLIT = "all_tracks"


def _first(value) -> str:
    if isinstance(value, list):
        for x in value:
            if x:
                return str(x)
        return ""
    return str(value or "")


def _row_to_meta(r: dict) -> dict:
    return {
        "track_id": r["track_id"],
        "track_name": _first(r.get("track_name")),
        "artist_name": _first(r.get("artist_name")),
        "album_name": _first(r.get("album_name")),
        "duration": r.get("duration"),
    }


def crawl_one(
    meta: dict,
    talkpl_lookup: dict,
    use_lrclib: bool,
    use_genius: bool,
    use_musicbrainz: bool,
) -> dict:
    """Per-track multi-source fetch. Returns the assembled record.

    Exposed for tests and library callers that want to crawl a single track
    without going through the CLI orchestration.
    """
    rec: dict = {
        **meta,
        "sources": {},
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    tn = meta["track_name"]
    an = meta["artist_name"]

    # 1) talkpl_cache primary — covers lyrics/caption/chord/tempo/key
    talkpl_rec = talkpl_lookup.get((tn.strip().lower(), an.strip().lower()))
    talkpl_fields = talkpl.extract_fields(talkpl_rec)
    for k, v in talkpl_fields.items():
        rec[k] = v
        rec["sources"][k] = "talkpl_cache"

    # 2) lyrics fallback chain — only if talkpl missed.
    # External fetchers also return match metadata (LRCLIB / Genius's resolved
    # title/artist/duration). We attach it under `lyrics_match` so guardrails
    # can flag wrong-track matches (different recording, cover, fuzzy SEARCH
    # mismatch). talkpl_cache is exact (lc_track, lc_artist) key match → no
    # validation needed there.
    if "lyrics" not in rec and use_lrclib:
        text, match_meta = sources.fetch_lrclib(tn, an, meta.get("album_name", ""), meta.get("duration"))
        if text:
            rec["lyrics"] = text
            rec["sources"]["lyrics"] = "lrclib"
            if match_meta:
                rec["lyrics_match"] = match_meta
    if "lyrics" not in rec and use_genius:
        text, match_meta = sources.fetch_genius(tn, an)
        if text:
            rec["lyrics"] = text
            rec["sources"]["lyrics"] = "genius"
            if match_meta:
                rec["lyrics_match"] = match_meta

    # 3) MusicBrainz — independent of talkpl, fills mbid/isrc/label/country/etc.
    if use_musicbrainz:
        mb = sources.fetch_musicbrainz(tn, an)
        if mb:
            for k, v in mb.items():
                if v:
                    rec[k] = v
                    rec["sources"][k] = "musicbrainz"

    # 4) validate
    rec["guardrail_flags"] = guardrails.validate_record(rec)
    rec["ok"] = guardrails.is_ok(rec["guardrail_flags"])
    return rec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m mymodule.feature.crawl",
        description="Multi-field track metadata crawler — talkpl_cache + LRCLIB + MusicBrainz → JSONL.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL output path")
    p.add_argument("--limit", type=int, default=None, help="cap number of tracks (smoke)")
    p.add_argument("--workers", type=int, default=8, help="HTTP worker threads")
    p.add_argument("--no-lrclib", action="store_true", help="skip LRCLIB lyrics fallback")
    p.add_argument("--no-genius", action="store_true", help="skip Genius lyrics fallback")
    p.add_argument("--no-musicbrainz", action="store_true", help="skip MusicBrainz (mbid/isrc/...)")
    p.add_argument("--skip-test-env", action="store_true", help="skip the smaller talkpl test_env.tar.gz")
    p.add_argument("--force", action="store_true", help="re-crawl tracks already in JSONL")
    p.add_argument("--save-every", type=int, default=200, help="batch flush interval")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logger.info(f"Loading challenge tracks from HF: {HF_DATASET} [{HF_SPLIT}]")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    rows = [_row_to_meta(r) for r in ds]
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"Total tracks to consider: {len(rows)}")

    existing = set() if args.force else jsonl.read_existing_ids(args.output)
    todo = [m for m in rows if m["track_id"] not in existing]
    logger.info(f"existing-in-jsonl: {len(existing)} | todo: {len(todo)}")
    if not todo:
        logger.success(f"Nothing to crawl — {args.output} already covers all targets.")
        summary = jsonl.write_summary(args.output)
        logger.success(f"Summary refreshed: {summary}")
        return 0

    talkpl_lookup = talkpl.load_lookup(skip_test_env=args.skip_test_env)

    source_counts: dict[str, dict[str, int]] = {}
    flag_counts: dict[str, int] = {}
    pending: list[dict] = []

    def _flush():
        if not pending:
            return
        jsonl.append_records(args.output, pending)
        pending.clear()

    use_lrclib = not args.no_lrclib
    use_genius = not args.no_genius
    use_musicbrainz = not args.no_musicbrainz

    # MusicBrainz enforces 1 req/sec globally inside `sources.fetch_musicbrainz`.
    # ThreadPool still helps with talkpl_cache + LRCLIB (throttle-free).
    with ThreadPoolExecutor(max_workers=args.workers) as ex, tqdm(total=len(todo)) as pbar:
        futures = {
            ex.submit(crawl_one, m, talkpl_lookup, use_lrclib, use_genius, use_musicbrainz): m["track_id"] for m in todo
        }
        for fut in as_completed(futures):
            rec = fut.result()
            for f, src in (rec.get("sources") or {}).items():
                source_counts.setdefault(f, {}).setdefault(src, 0)
                source_counts[f][src] += 1
            for flg in rec["guardrail_flags"]:
                flag_counts[flg] = flag_counts.get(flg, 0) + 1
            pending.append(rec)
            pbar.update(1)
            pbar.set_postfix({"flags": sum(flag_counts.values())})
            if len(pending) >= args.save_every:
                _flush()
    _flush()

    logger.success(f"Crawl done — sources={source_counts}")
    logger.success(f"Crawl done — flag_counts={flag_counts}")
    summary = jsonl.write_summary(args.output)
    dup_groups = len(summary.get("duplicate_lyrics_groups_top10") or [])
    logger.success(f"Summary written: ok={summary['ok_count']}/{summary['total']}, dup_groups={dup_groups}")
    logger.info(f"  JSONL: {args.output}")
    logger.info(f"  Summary: {args.output.with_suffix(args.output.suffix + '.summary.json')}")
    return 0
