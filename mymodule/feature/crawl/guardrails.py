"""Per-field quality flags for crawled track records.

The record is a flat dict with optional fields (`lyrics`, `caption`, `chord`,
`tempo`, `key`, `mbid`, `isrc`, `label`, `country`, `mb_release_date`,
`mb_tags`). Each text-y field has its own validators; the resulting flag list
is namespaced (`lyrics_too_short`, `caption_empty`, `tempo_invalid`, ...).

Top-level `ok` (computed by the orchestrator) requires the **two highest-value
fields** — `lyrics` and `caption` — present without quality flags. The other
fields are bonus (`mbid_missing` is informational, not blocking).

Cross-track checks (e.g. duplicate lyrics across many tracks) live in the
orchestrator summary — they need the full corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

# ---- thresholds ------------------------------------------------------------

# Lyrics — long-form text. Empirical thresholds against the talkpl + LRCLIB
# corpus: under 100 chars almost always means truncated content or a wrong
# track-name match returning a snippet instead of full lyrics.
LYRICS_MIN_CHARS = 100
LYRICS_MAX_CHARS = 100_000
LYRICS_REPEATED_LINE_RATIO = 0.5

# Caption — short-form audio description (~50-200 chars typical).
CAPTION_MIN_CHARS = 30
CAPTION_MAX_CHARS = 5_000

PRINTABLE_RATIO_MIN = 0.5

# Tempo — BPM. Outside this range is almost certainly a parse error.
TEMPO_BPM_MIN = 30.0
TEMPO_BPM_MAX = 300.0

# Lyrics match-quality thresholds (used by `_flags_lyrics_match`).
# Title is allowed to be looser than artist because external sources often
# return suffixes like " - feat. X" or " (Remastered 2011)" which still mean
# the same recording.
LYRICS_TITLE_MATCH_MIN_RATIO = 0.80
LYRICS_ARTIST_MATCH_MIN_RATIO = 0.85
LYRICS_DURATION_MAX_DELTA_SEC = 10.0

# Fields whose presence we expect for a record to be `ok`. Other fields are
# bonus (their absence does not flip `ok` to False).
CRITICAL_FIELDS = ("lyrics", "caption")

# Flags that are reported but do not flip `ok` to False. These are diagnostic
# (e.g. "matched via fuzzy SEARCH not exact GET") rather than blocking.
INFORMATIONAL_FLAGS = frozenset({"lyrics_method_search_fuzzy"})

_APOLOGY_PATTERNS = [
    re.compile(r"instrumental track that has no lyrics", re.IGNORECASE),
    re.compile(r"this song is an instrumental", re.IGNORECASE),
    re.compile(r"no lyrics? (?:available|found)", re.IGNORECASE),
]
_INSTRUMENTAL_RE = re.compile(r"^\s*[\[\(]?\s*instrumental\s*[\]\)]?\s*$", re.IGNORECASE)

# ---- text-field shared checks ---------------------------------------------


def _is_empty(text) -> bool:
    return text is None or not str(text).strip()


def _is_mostly_non_printable(text: str) -> bool:
    if len(text) < 20:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t ")
    return (printable / len(text)) < PRINTABLE_RATIO_MIN


# ---- lyrics-specific -------------------------------------------------------


def _flags_lyrics(text) -> list[str]:
    flags: list[str] = []
    if _is_empty(text):
        return ["lyrics_empty"]
    txt = str(text)
    stripped_len = len(txt.strip())
    if 0 < stripped_len < LYRICS_MIN_CHARS:
        flags.append("lyrics_too_short")
    if len(txt) > LYRICS_MAX_CHARS:
        flags.append("lyrics_too_long")
    # instrumental marker — entire content is just `[Instrumental]` variants
    lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    if lines and all(_INSTRUMENTAL_RE.match(ln) for ln in lines):
        flags.append("lyrics_instrumental_marker")
    # repeated line dominance
    lc_lines = [ln.strip().lower() for ln in txt.splitlines() if ln.strip()]
    if len(lc_lines) >= 4:
        most_common = Counter(lc_lines).most_common(1)[0][1]
        if (most_common / len(lc_lines)) >= LYRICS_REPEATED_LINE_RATIO:
            flags.append("lyrics_repeated_line_dominant")
    if _is_mostly_non_printable(txt):
        flags.append("lyrics_mostly_non_printable")
    if any(p.search(txt) for p in _APOLOGY_PATTERNS):
        flags.append("lyrics_apology")
    return flags


# ---- caption-specific ------------------------------------------------------


def _flags_caption(text) -> list[str]:
    flags: list[str] = []
    if _is_empty(text):
        return ["caption_empty"]
    txt = str(text)
    stripped_len = len(txt.strip())
    if 0 < stripped_len < CAPTION_MIN_CHARS:
        flags.append("caption_too_short")
    if len(txt) > CAPTION_MAX_CHARS:
        flags.append("caption_too_long")
    if _is_mostly_non_printable(txt):
        flags.append("caption_mostly_non_printable")
    return flags


# ---- chord / tempo / key ---------------------------------------------------


def _flags_chord(text) -> list[str]:
    return ["chord_empty"] if _is_empty(text) else []


def _flags_key(text) -> list[str]:
    return ["key_empty"] if _is_empty(text) else []


def _flags_tempo(text) -> list[str]:
    if _is_empty(text):
        return ["tempo_empty"]
    try:
        bpm = float(str(text).strip())
    except ValueError:
        return ["tempo_invalid"]
    if not (TEMPO_BPM_MIN <= bpm <= TEMPO_BPM_MAX):
        return ["tempo_invalid"]
    return []


# ---- musicbrainz fields ----------------------------------------------------


def _flags_mbid(record: dict) -> list[str]:
    return ["mbid_missing"] if _is_empty(record.get("mbid")) else []


# ---- lyrics match-quality (record["lyrics_match"]) -------------------------


def _normalise_for_compare(s: str) -> str:
    """Lowercase + strip + collapse whitespace. No unicode normalisation —
    `SequenceMatcher.ratio()` handles minor character differences gracefully."""
    return " ".join(str(s).strip().lower().split())


_SUFFIX_SEPARATORS = " ([-/"


def _is_suffix_extension(shorter: str, longer: str) -> bool:
    """Detect "shorter is `longer` minus a tail-suffix" pattern.

    Real-world external-source returns frequently append:
        ` (feat. X)`, ` - Remastered 2011`, ` (Remix)`, ` / Live`, ` feat. ...`
    These don't change the underlying recording, so we should NOT flag a
    title/artist mismatch for them.

    A bare substring test would be too lax — `"artist"` inside
    `"some other artist"` is a *different* artist that happens to share
    a token. We require the longer string to **start with** the shorter
    one, then the next character must be a recognised separator.
    """
    if not shorter or not longer or len(longer) <= len(shorter):
        return False
    if not longer.startswith(shorter):
        return False
    return longer[len(shorter)] in _SUFFIX_SEPARATORS


def _name_similarity(a: str, b: str) -> float:
    """Asymmetric similarity for short names (track / artist).

    Returns 1.0 when one string is the other plus a recognised tail-suffix.
    Otherwise falls back to `SequenceMatcher.ratio()`. This rejects
    `"artist"` ≟ `"some other artist"` (no shared prefix) while accepting
    `"letting go"` ≟ `"letting go (feat. sarah green)"`.
    """
    if not a or not b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if _is_suffix_extension(shorter, longer):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _query_duration_seconds(record: dict) -> float | None:
    """HF challenge metadata stores `duration` in milliseconds (e.g. 263520
    for a ~4:23 track). LRCLIB returns seconds. Heuristic: anything > 1000 is
    treated as ms and converted. `0` / negative is treated as missing — HF
    occasionally has 0-duration rows for tracks without analysis."""
    raw = record.get("duration")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v / 1000.0 if v > 1000 else v


def _flags_lyrics_match(record: dict) -> list[str]:
    """Validate that fetched lyrics actually came from the requested track.

    Reads `record["lyrics_match"]` (set by the orchestrator only when lyrics
    came from an external HTTP source — talkpl_cache is exact key-match so
    no validation needed). Compares matched track_name / artist_name / duration
    to the query record's own fields.

    Flags raised:
        lyrics_title_mismatch         — title similarity below threshold
        lyrics_artist_mismatch        — artist similarity below threshold
        lyrics_duration_mismatch      — |matched - query| > 10s
        lyrics_method_search_fuzzy    — informational (matched via fuzzy SEARCH
                                        rather than exact GET; does not flip ok)
    """
    match = record.get("lyrics_match")
    if not isinstance(match, dict):
        return []
    flags: list[str] = []

    query_tn = _normalise_for_compare(record.get("track_name") or "")
    query_an = _normalise_for_compare(record.get("artist_name") or "")
    matched_tn = _normalise_for_compare(match.get("track_name") or "")
    matched_an = _normalise_for_compare(match.get("artist_name") or "")

    if query_tn and matched_tn:
        if _name_similarity(matched_tn, query_tn) < LYRICS_TITLE_MATCH_MIN_RATIO:
            flags.append("lyrics_title_mismatch")

    if query_an and matched_an:
        if _name_similarity(matched_an, query_an) < LYRICS_ARTIST_MATCH_MIN_RATIO:
            flags.append("lyrics_artist_mismatch")

    matched_dur = match.get("duration")
    query_dur_sec = _query_duration_seconds(record)
    if matched_dur is not None and query_dur_sec is not None:
        try:
            if abs(float(matched_dur) - query_dur_sec) > LYRICS_DURATION_MAX_DELTA_SEC:
                flags.append("lyrics_duration_mismatch")
        except (TypeError, ValueError):
            pass

    method = match.get("method") or ""
    if method in ("lrclib_search", "genius_search"):
        flags.append("lyrics_method_search_fuzzy")

    return flags


# ---- composition -----------------------------------------------------------


def validate_record(record: dict) -> list[str]:
    """Return a sorted list of namespaced flag names for `record`.

    Flag list is alphabetically sorted for diff-friendly summaries. Empty list
    ⇔ all checked fields pass. Non-critical absence (e.g. `mbid_missing`) is
    reported but does not by itself prevent `ok` — that decision lives in
    `is_ok()`.
    """
    flags: list[str] = []
    flags.extend(_flags_lyrics(record.get("lyrics")))
    flags.extend(_flags_lyrics_match(record))
    flags.extend(_flags_caption(record.get("caption")))
    flags.extend(_flags_chord(record.get("chord")))
    flags.extend(_flags_tempo(record.get("tempo")))
    flags.extend(_flags_key(record.get("key")))
    flags.extend(_flags_mbid(record))
    return sorted(flags)


def is_ok(flags: list[str]) -> bool:
    """A record is `ok` iff every critical field has no blocking quality flag.

    `mbid_missing`, `chord_empty`, etc. do NOT block `ok` — they're reported
    but treated as nice-to-have. Flags in `INFORMATIONAL_FLAGS` (e.g.
    `lyrics_method_search_fuzzy`) are also exempt. Critical fields are
    `lyrics` and `caption`.
    """
    flag_set = set(flags) - INFORMATIONAL_FLAGS
    for field in CRITICAL_FIELDS:
        for f in flag_set:
            if f.startswith(field + "_"):
                return False
    return True


__all__ = [
    "LYRICS_MIN_CHARS",
    "LYRICS_MAX_CHARS",
    "LYRICS_REPEATED_LINE_RATIO",
    "CAPTION_MIN_CHARS",
    "CAPTION_MAX_CHARS",
    "PRINTABLE_RATIO_MIN",
    "TEMPO_BPM_MIN",
    "TEMPO_BPM_MAX",
    "LYRICS_TITLE_MATCH_MIN_RATIO",
    "LYRICS_ARTIST_MATCH_MIN_RATIO",
    "LYRICS_DURATION_MAX_DELTA_SEC",
    "CRITICAL_FIELDS",
    "INFORMATIONAL_FLAGS",
    "validate_record",
    "is_ok",
]
