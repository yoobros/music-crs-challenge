"""talkpl-ai/TalkPlayTools-Env cache loader (primary metadata source).

Streams `tool_env.tar.gz` (+ optional `test_env.tar.gz`) from HuggingFace,
extracts `cache/metadata/item_metadata.json`, and builds an in-memory lookup
keyed on `(track_name_lower, artist_name_lower)`. The orchestrator opens the
cache once at startup and queries per-track without further HTTP.

Lifted from `scripts/import_talkplay_cache.py` with KVStore writes removed.
The cache itself is large (~70 MB JSON for the full tool_env), so we persist
the extracted JSON to `LOCAL_CACHE_DIR` and reuse on repeat runs.
"""

from __future__ import annotations

import json
import os
import tarfile

import requests
from huggingface_hub import hf_hub_url
from loguru import logger

REPO_ID = "talkpl-ai/TalkPlayTools-Env"
ARCHIVE_INNER_PATH = "cache/metadata/item_metadata.json"
# Cache dir for the extracted JSON. Override via `MYMODULE_TALKPL_CACHE_DIR`.
# The default path matches the legacy `scripts/import_talkplay_cache.py` layout
# so a fresh checkout reuses an already-extracted JSON instead of re-downloading
# the 2.3 GB tar.
DEFAULT_LOCAL_DIR = os.getenv("MYMODULE_TALKPL_CACHE_DIR", "/tmp/talkplay_lyrics")
_FILENAME_PATTERN = "cache__metadata__item_metadata__{filename}.json"

# Field names exposed by the talkpl cache. The cache JSON sometimes stores
# these as lists or scalars — `_first` normalises to a single string.
TALKPL_FIELDS = ("lyrics", "caption", "chord", "tempo", "key")


def _first(v) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v or "")


def _stream_extract(filename: str, cache_dir: str = DEFAULT_LOCAL_DIR) -> dict:
    """Download `{filename}` from HF, extract `item_metadata.json` in memory.

    Resulting JSON is also written to disk so subsequent runs reuse it. Returns
    the parsed dict ({track_id: record}).
    """
    local_path = os.path.join(cache_dir, _FILENAME_PATTERN.format(filename=filename))
    if os.path.exists(local_path):
        logger.info(f"reusing local talkpl cache: {local_path}")
        with open(local_path) as f:
            return json.load(f)

    url = hf_hub_url(REPO_ID, filename, repo_type="dataset")
    logger.info(f"streaming {url} → extracting {ARCHIVE_INNER_PATH}")
    os.makedirs(cache_dir, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for member in tf:
                if member.name == ARCHIVE_INNER_PATH:
                    fh = tf.extractfile(member)
                    if fh is None:
                        raise RuntimeError(f"cannot extract {member.name}")
                    raw = fh.read()
                    with open(local_path, "wb") as out:
                        out.write(raw)
                    logger.success(f"extracted {member.size / 1024 / 1024:.1f} MB → {local_path}")
                    return json.loads(raw)
    raise RuntimeError(f"{ARCHIVE_INNER_PATH} not found in {filename}")


def _build_lookup(meta: dict) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for _tid, rec in meta.items():
        if not isinstance(rec, dict):
            continue
        tn = _first(rec.get("track_name")).strip().lower()
        an = _first(rec.get("artist_name")).strip().lower()
        if tn and an:
            out[(tn, an)] = rec
    return out


def load_lookup(skip_test_env: bool = False, cache_dir: str = DEFAULT_LOCAL_DIR) -> dict[tuple[str, str], dict]:
    """Return `(track_name_lc, artist_name_lc) → talkpl record` lookup.

    `tool_env.tar.gz` is the full corpus; `test_env.tar.gz` is a smaller subset
    that occasionally has unique entries — we merge with tool_env winning ties.
    """
    tool = _stream_extract("tool_env.tar.gz", cache_dir=cache_dir)
    logger.info(f"talkpl tool_env tracks: {len(tool)}")
    test: dict = {}
    if not skip_test_env:
        try:
            test = _stream_extract("test_env.tar.gz", cache_dir=cache_dir)
            logger.info(f"talkpl test_env tracks: {len(test)}")
        except Exception as e:
            logger.warning(f"talkpl test_env skipped ({e})")
    lookup = _build_lookup(test)
    lookup.update(_build_lookup(tool))  # tool_env wins on conflict
    logger.success(f"talkpl unique (track_name, artist_name) keys: {len(lookup)}")
    return lookup


def extract_fields(rec: dict | None) -> dict[str, str]:
    """Pull canonical talkpl fields from a record. Missing/empty → not in output."""
    if not isinstance(rec, dict):
        return {}
    out: dict[str, str] = {}
    for f in TALKPL_FIELDS:
        raw = rec.get(f)
        if isinstance(raw, list):
            if not raw:
                continue
            # `chord` is a list of chord symbols → space-joined string.
            if f == "chord":
                value = " ".join(str(c) for c in raw if c)
            else:
                value = " ".join(str(x) for x in raw if x)
        else:
            value = _first(raw)
        value = value.strip()
        if value:
            out[f] = value
    return out


__all__ = ["load_lookup", "extract_fields", "TALKPL_FIELDS", "DEFAULT_LOCAL_DIR"]
