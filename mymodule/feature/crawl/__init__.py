"""Track metadata crawler — multi-source, JSONL output.

Free-source crawler for per-track metadata used by downstream BM25 corpora,
text embedding pipelines, and rerank features. Sources are layered:

- `talkpl_cache` (primary): talkpl-ai's `TalkPlayTools-Env` tar (lyrics,
  caption, chord, tempo, key). Exact `(track_name_lc, artist_name_lc)` match.
- `lrclib`: free + no-auth HTTP lyrics. Used as fallback when talkpl misses.
- `genius`: optional fallback (requires `MYMODULE_GENIUS_TOKEN`).
- `musicbrainz`: structured metadata (mbid, isrc, label, country,
  release_date, tags). 1 req/sec global throttle.

Output is JSONL — one record per track. Per-record `guardrail_flags` and
`ok` flag let downstream consumers filter without re-validating. RocksDB /
LanceDB ingestion is intentionally out of scope here; that lives in
`mymodule.feature.kvdb` / `mymodule.feature.store` and may consume this
JSONL via a follow-up importer.

Public API:
    load_talkpl_lookup, extract_talkpl_fields
    fetch_lrclib, fetch_genius, fetch_lyrics, fetch_musicbrainz
    validate_record, is_ok
    read_existing_ids, append_records, iter_records, write_summary
    crawl_one
"""

from mymodule.feature.crawl.cli import crawl_one
from mymodule.feature.crawl.guardrails import is_ok, validate_record
from mymodule.feature.crawl.jsonl import (
    append_records,
    iter_records,
    read_existing_ids,
    write_summary,
)
from mymodule.feature.crawl.sources import (
    fetch_genius,
    fetch_lrclib,
    fetch_lyrics,
    fetch_musicbrainz,
)
from mymodule.feature.crawl.talkpl import (
    extract_fields as extract_talkpl_fields,
)
from mymodule.feature.crawl.talkpl import (
    load_lookup as load_talkpl_lookup,
)

__all__ = [
    "crawl_one",
    "validate_record",
    "is_ok",
    "fetch_lrclib",
    "fetch_genius",
    "fetch_lyrics",
    "fetch_musicbrainz",
    "load_talkpl_lookup",
    "extract_talkpl_fields",
    "read_existing_ids",
    "append_records",
    "iter_records",
    "write_summary",
]
