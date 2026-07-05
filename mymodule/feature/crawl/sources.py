"""External per-track HTTP sources.

- `fetch_lrclib` / `fetch_genius`: lyrics text (free; Genius needs API token).
- `fetch_musicbrainz`: structured metadata (mbid, isrc, label, country,
  release_date, tags). MusicBrainz is free + no auth, but requires a 1 req/sec
  rate limit per their policy — enforced via a process-wide lock.

All fetchers are pure I/O: return text/dict or None. The orchestrator owns
caching, validation, and persistence.
"""

from __future__ import annotations

import os
import threading
import time

import requests
from loguru import logger

LRCLIB_GET = "https://lrclib.net/api/get"
LRCLIB_SEARCH = "https://lrclib.net/api/search"
MUSICBRAINZ_RECORDING_SEARCH = "https://musicbrainz.org/ws/2/recording/"
MUSICBRAINZ_RELEASE_LOOKUP = "https://musicbrainz.org/ws/2/release/"
USER_AGENT = "recsys-challenge-2026 metadata-crawler/0.3 (research)"

# Tight HTTP timeout — LRCLIB is usually fast; slow responses are usually
# overloaded edges, so failing quickly + falling back is preferable to hanging.
HTTP_TIMEOUT_SEC = 10


def fetch_lrclib(
    track_name: str,
    artist_name: str,
    album_name: str = "",
    duration: float | None = None,
) -> tuple[str | None, dict | None]:
    """LRCLIB GET (exact) → SEARCH (artist-exact) fallback.

    Returns `(text, match_meta)`:
      text       : lyrics or None
      match_meta : dict on hit (None on miss). Fields:
        - track_name   (str): LRCLIB-side trackName
        - artist_name  (str): LRCLIB-side artistName
        - duration     (float | None): LRCLIB-side duration (seconds)
        - method       (str): "lrclib_get_exact" | "lrclib_search"
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        params: dict = {"track_name": track_name, "artist_name": artist_name}
        if album_name:
            params["album_name"] = album_name
        if duration and duration > 0:
            # HF challenge metadata is milliseconds; LRCLIB expects seconds
            # and rejects matches whose duration differs by more than a few
            # seconds. Send seconds to keep GET (exact) on the happy path
            # instead of falling all the way through to fuzzy SEARCH.
            sec = float(duration) / 1000.0 if duration > 1000 else float(duration)
            params["duration"] = int(sec)
        r = requests.get(LRCLIB_GET, params=params, headers=headers, timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 200:
            data = r.json() or {}
            text = (data.get("plainLyrics") or "").strip()
            if text:
                return text, {
                    "track_name": data.get("trackName") or "",
                    "artist_name": data.get("artistName") or "",
                    "duration": data.get("duration"),
                    "method": "lrclib_get_exact",
                }
    except Exception:
        # Treat any GET failure as miss — SEARCH below is a best-effort retry.
        pass
    try:
        r = requests.get(
            LRCLIB_SEARCH,
            params={"q": f"{track_name} {artist_name}".strip()},
            headers=headers,
            timeout=HTTP_TIMEOUT_SEC,
        )
        if r.status_code == 200:
            results = r.json() or []
            artist_lc = artist_name.lower().strip()
            for d in results[:3]:
                if (d.get("artistName") or "").lower().strip() == artist_lc:
                    text = (d.get("plainLyrics") or "").strip()
                    if text:
                        return text, {
                            "track_name": d.get("trackName") or d.get("name") or "",
                            "artist_name": d.get("artistName") or "",
                            "duration": d.get("duration"),
                            "method": "lrclib_search",
                        }
    except Exception:
        pass
    return None, None


_genius_lock = threading.Lock()
_genius_client = None  # singleton, lazily created on first use


def _get_genius_client():
    """Lazy import + init. Returns None if `MYMODULE_GENIUS_TOKEN` is missing."""
    global _genius_client
    with _genius_lock:
        if _genius_client is not None:
            return _genius_client
        token = os.getenv("MYMODULE_GENIUS_TOKEN")
        if not token:
            return None
        try:
            import lyricsgenius  # type: ignore[import-not-found]

            client = lyricsgenius.Genius(
                token,
                timeout=HTTP_TIMEOUT_SEC,
                retries=1,
                remove_section_headers=True,
                skip_non_songs=True,
                verbose=False,
            )
            client.user_agent = USER_AGENT
            _genius_client = client
            return client
        except ImportError:
            logger.warning("lyricsgenius not installed — pip install lyricsgenius (Genius fallback skipped)")
            return None


def fetch_genius(track_name: str, artist_name: str) -> tuple[str | None, dict | None]:
    """Returns (text, match_meta). match_meta uses Genius's own resolved title/artist."""
    client = _get_genius_client()
    if client is None:
        return None, None
    try:
        song = client.search_song(track_name, artist_name)
        if song and song.lyrics:
            text = song.lyrics.strip()
            if not text:
                return None, None
            return text, {
                "track_name": getattr(song, "title", "") or "",
                "artist_name": getattr(song, "artist", "") or "",
                "duration": None,
                "method": "genius_search",
            }
    except Exception:
        return None, None
    return None, None


def fetch_lyrics(
    track_name: str,
    artist_name: str,
    album_name: str = "",
    duration: float | None = None,
    use_genius: bool = True,
) -> tuple[str | None, str, dict | None]:
    """Returns (lyrics, source, match_meta).

    source ∈ {`lrclib`, `genius`, `miss`}. match_meta is None on miss.
    """
    if not track_name or not artist_name:
        return None, "miss", None
    text, meta = fetch_lrclib(track_name, artist_name, album_name, duration)
    if text:
        return text, "lrclib", meta
    if use_genius:
        text, meta = fetch_genius(track_name, artist_name)
        if text:
            return text, "genius", meta
    return None, "miss", None


# ---- MusicBrainz -----------------------------------------------------------

# MusicBrainz public API enforces 1 req/sec per their policy. We honour it
# globally so all crawler threads serialise through the same gate.
_MB_LOCK = threading.Lock()
_MB_LAST_REQUEST_TS = [0.0]
_MB_MIN_INTERVAL_SEC = 1.05  # tiny buffer over the 1 req/sec policy


def _mb_throttle():
    with _MB_LOCK:
        delta = time.monotonic() - _MB_LAST_REQUEST_TS[0]
        if delta < _MB_MIN_INTERVAL_SEC:
            time.sleep(_MB_MIN_INTERVAL_SEC - delta)
        _MB_LAST_REQUEST_TS[0] = time.monotonic()


def _mb_get(url: str, params: dict) -> dict | None:
    """Throttled GET → JSON. Returns None on failure (network, 4xx, 5xx)."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {**params, "fmt": "json"}
    _mb_throttle()
    try:
        r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 503:
            # MB throttles aggressive callers — back off once and retry.
            time.sleep(2.0)
            _mb_throttle()
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        logger.debug(f"musicbrainz GET failed: {e}")
    return None


def _mb_pick_best_recording(results: list[dict], artist_name: str, track_name: str) -> dict | None:
    """Pick the best recording match: artist exact (case-insensitive), title exact-prefix.

    MB SEARCH returns Lucene-scored hits — we prefer artist-exact + title-equal,
    falling back to artist-exact only. Avoids returning the wrong song when the
    artist coincidentally matches another track in MB's index.
    """
    artist_lc = artist_name.lower().strip()
    title_lc = track_name.lower().strip()
    artist_exact_title_exact: list[dict] = []
    artist_exact_only: list[dict] = []
    for rec in results[:10]:
        rec_title = (rec.get("title") or "").lower().strip()
        rec_artists = rec.get("artist-credit") or []
        rec_artist_names = [
            (ac.get("name") or ac.get("artist", {}).get("name") or "").lower().strip() for ac in rec_artists
        ]
        if artist_lc not in rec_artist_names:
            continue
        if rec_title == title_lc:
            artist_exact_title_exact.append(rec)
        else:
            artist_exact_only.append(rec)
    if artist_exact_title_exact:
        return artist_exact_title_exact[0]
    if artist_exact_only:
        return artist_exact_only[0]
    return None


def fetch_musicbrainz(track_name: str, artist_name: str) -> dict | None:
    """Fetch structured metadata from MusicBrainz. Returns dict or None.

    Output keys (any subset, depending on MB coverage):
        mbid:              recording MBID (str)
        isrc:              first ISRC if present (str)
        label:             first release label name (str)
        country:           first release country (str, ISO 3166-1)
        mb_release_date:   first release date (str, YYYY-MM-DD or partial)
        mb_tags:           list of tag names (list[str], may be empty)
    """
    if not track_name or not artist_name:
        return None

    query = f'recording:"{track_name}" AND artist:"{artist_name}"'
    data = _mb_get(MUSICBRAINZ_RECORDING_SEARCH, {"query": query, "limit": 10, "inc": "tags+isrcs"})
    if not data:
        return None
    results = data.get("recordings") or []
    rec = _mb_pick_best_recording(results, artist_name, track_name)
    if not rec:
        return None

    out: dict = {"mbid": rec.get("id")}

    isrcs = rec.get("isrcs") or []
    if isrcs:
        out["isrc"] = isrcs[0]

    tags = [t.get("name") for t in (rec.get("tags") or []) if t.get("name")]
    if tags:
        out["mb_tags"] = tags

    releases = rec.get("releases") or []
    if releases:
        rel = releases[0]
        if rel.get("date"):
            out["mb_release_date"] = rel["date"]
        if rel.get("country"):
            out["country"] = rel["country"]
        # Label requires a deeper lookup — try inline first via release-group.
        labels = rel.get("label-info") or rel.get("label-info-list") or []
        if not labels and rel.get("id"):
            # Fetch the release detail for label info (one extra HTTP).
            rel_data = _mb_get(MUSICBRAINZ_RELEASE_LOOKUP + rel["id"], {"inc": "labels"})
            if rel_data:
                labels = rel_data.get("label-info") or []
        for li in labels:
            label = li.get("label") or {}
            name = label.get("name")
            if name:
                out["label"] = name
                break

    return out


__all__ = [
    "fetch_lyrics",
    "fetch_lrclib",
    "fetch_genius",
    "fetch_musicbrainz",
    "HTTP_TIMEOUT_SEC",
    "USER_AGENT",
]
