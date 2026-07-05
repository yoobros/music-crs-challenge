"""Doc + Query composition for LoRA two-tower fine-tuning.

Single source for query/doc text composition. Replaces the prior `v2`/`v3`
split — features are toggled via keyword flags rather than versioned function
names.

Public API:
    build_tag_freq(track_ds)             -> dict[str, int]   (with disk caching)
    load_tag_freq()                      -> dict[str, int]
    clean_tags(tags, freq_map, ...)      -> list[str]
    compose_doc(row, freq_map)           -> str
    compose_query(user_msg, chat_history, conversation_goal, kv, freq_map,
                  *, tokenizer=None, max_body_tokens=240)
                                         -> str

The `Instruct:`/`Query:` prefix is added downstream by
`mymodule.strategies.twotower.encoder.format_query`. compose_query returns the body only.
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

# ---------------------------------------------------------------------------
# Goal short-phrase mappings (data-explorer 2026-05-10 verified)
# ---------------------------------------------------------------------------

SPECIFICITY_PHRASE: dict[str, str] = {
    "HH": "specific goal, specific track",
    "HL": "specific goal, vague track",
    "LH": "vague goal, specific track",
    "LL": "vague goal, vague track",
}

CATEGORY_PHRASE: dict[str, str] = {
    "A": "sonic characteristics",
    "B": "lyrical themes",
    "C": "album artwork",
    "D": "activity fit",
    "E": "guided discovery",
    "F": "vague memory",
    "G": "mood and emotion",
    "H": "artist exploration",
    "I": "exact popular hit",
    "J": "popular trends",
    "K": "era sound",
}


# ---------------------------------------------------------------------------
# Tag normalization constants
# ---------------------------------------------------------------------------

_BRITISH_AMERICAN_VARIANTS: dict[str, str] = {
    "favourites": "favorites",
    "favourite": "favorite",
    "favourite song": "favorite song",
    "favourite songs": "favorite songs",
    "favourites song": "favorites song",
    "favourites songs": "favorites songs",
    "colour": "color",
    "favourite bands": "favorite bands",
    "favourite band": "favorite band",
}

# Patterns that mark known noise — applied AFTER lowercase+strip.
_BLACKLIST_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\d+ of \d+ stars?$"),  # "10 of 10 stars"
    re.compile(r"^\d+\s*stars?$"),  # "5 stars"
    re.compile(r"^[a-z]{2,5}-?fm$"),  # "wber", "cimx-fm"
    re.compile(r"^i (love|like|listen|always|just|absolutely)\b"),
    re.compile(r"^my (favo?u?rite|fav)\b"),
    re.compile(r"^favo?u?rite\s+song"),
    re.compile(r"^\d{1,2}\s*of\s*\d{1,2}$"),  # "8 of 10"
)

_HIGH_UNICODE = re.compile(r"[-￿]")
_MAX_TAG_LEN = 30

# Cache file — bumped to v2 because British normalization is now applied at build time.
_TAG_FREQ_CACHE = Path(__file__).parent / "data" / "tag_freq.json"
_TAG_FREQ_CACHE_VERSION = "v2_british_normalized"


# ---------------------------------------------------------------------------
# spaCy lazy load — shared by _summarise_queries and _smart_lower
# ---------------------------------------------------------------------------

try:
    import spacy

    _SPACY_MODEL = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_MODEL = None


_KEEP_POS = {"NOUN", "ADJ", "PROPN", "VERB"}
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "b",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "both",
        "but",
        "by",
        "c",
        "can",
        "cannot",
        "could",
        "d",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "due",
        "during",
        "e",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "f",
        "few",
        "for",
        "former",
        "formerly",
        "from",
        "further",
        "g",
        "get",
        "gets",
        "getting",
        "go",
        "goes",
        "going",
        "gone",
        "got",
        "gotten",
        "h",
        "had",
        "has",
        "have",
        "having",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "j",
        "k",
        "l",
        "last",
        "lately",
        "later",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "m",
        "made",
        "make",
        "makes",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "n",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "o",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "overall",
        "own",
        "p",
        "per",
        "perhaps",
        "please",
        "q",
        "r",
        "rather",
        "re",
        "really",
        "regarding",
        "s",
        "same",
        "say",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "seven",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "t",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "u",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "v",
        "very",
        "via",
        "w",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "x",
        "y",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "z",
    }
)


# ---------------------------------------------------------------------------
# Tag cleaning primitives
# ---------------------------------------------------------------------------


def _normalize_tag(t: Any) -> str:
    """lowercase + collapse whitespace + strip trailing punctuation."""
    if t is None:
        return ""
    s = str(t).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:!?")
    return s


def _normalise_spelling(tag: str) -> str:
    """British → American spelling normalization."""
    return _BRITISH_AMERICAN_VARIANTS.get(tag, tag)


def _is_noise(tag: str) -> bool:
    if not tag:
        return True
    if len(tag) > _MAX_TAG_LEN:
        return True
    if _HIGH_UNICODE.search(tag):
        return True
    for pat in _BLACKLIST_PATTERNS:
        if pat.search(tag):
            return True
    return False


def build_tag_freq(track_ds: Iterable[dict], cache: Path | None = _TAG_FREQ_CACHE) -> dict[str, int]:
    """Scan track_ds once, return {normalized_tag: track_count}.

    Each tag counts AT MOST ONCE per track. Result cached to `cache` json with
    a version stamp so subsequent calls skip the scan. British→American
    normalization is applied at build time so freq lookups in `clean_tags`
    line up with the normalized tag.
    """
    if cache is not None and cache.exists():
        try:
            with cache.open() as f:
                obj = json.load(f)
            if isinstance(obj, dict) and obj.get("_version") == _TAG_FREQ_CACHE_VERSION:
                tags = obj.get("tags", {})
                logger.info(f"[text-compose] loaded tag_freq cache {cache} ({len(tags)} tags)")
                return {k: int(v) for k, v in tags.items()}
            logger.warning(
                f"[text-compose] tag_freq cache version mismatch ({obj.get('_version')!r} "
                f"≠ {_TAG_FREQ_CACHE_VERSION!r}); rebuilding"
            )
        except Exception as e:
            logger.warning(f"[text-compose] tag_freq cache load failed ({e}); rebuilding")
    counter: Counter[str] = Counter()
    n_rows = 0
    for row in track_ds:
        n_rows += 1
        tags = row.get("tag_list") or []
        if not isinstance(tags, list):
            tags = [tags]
        seen: set[str] = set()
        for raw in tags:
            t = _normalize_tag(raw)
            if not t:
                continue
            t = _normalise_spelling(t)
            if t in seen:
                continue
            seen.add(t)
            counter[t] += 1
    out = dict(counter)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"_version": _TAG_FREQ_CACHE_VERSION, "tags": out}, ensure_ascii=False))
    logger.info(f"[text-compose] built tag_freq from {n_rows} tracks ({len(out)} unique tags)")
    return out


def load_tag_freq(cache: Path = _TAG_FREQ_CACHE) -> dict[str, int]:
    """Load existing tag_freq cache. Raises if not built yet or version mismatch."""
    if not cache.exists():
        raise FileNotFoundError(f"tag_freq cache missing at {cache} — run `build_tag_freq(track_ds)` first.")
    with cache.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or obj.get("_version") != _TAG_FREQ_CACHE_VERSION:
        raise RuntimeError(
            f"tag_freq cache version mismatch at {cache} "
            f"(got {obj.get('_version')!r}, expected {_TAG_FREQ_CACHE_VERSION!r}); rebuild required."
        )
    tags = obj.get("tags", {})
    return {k: int(v) for k, v in tags.items()}


def clean_tags(
    tags: Any,
    freq_map: dict[str, int],
    *,
    min_freq: int = 5,
    top_n: int = 20,
    normalize_british: bool = True,
    substring_dedup: bool = True,
    preserve_input_order: bool = True,
) -> list[str]:
    """Unified tag cleaner shared between doc and query composition.

    Pipeline:
        1. normalize (lowercase, whitespace, trailing punctuation)
        2. British → American spelling (if enabled)
        3. dedup (preserves first occurrence)
        4. noise filter (blacklist regex + length / Unicode cap)
        5. freq filter (`freq_map[t] >= min_freq`)
        6. substring dedup (if enabled) — for `a ⊂ b`, keep the higher-freq
           one; on tie keep the longer; preserve relative order
        7. ordering — input order (preserve_input_order=True) or
           `(-freq, alphabetical)`
        8. top_n cap
    """
    if not tags:
        return []
    if not isinstance(tags, list):
        tags = [tags]

    # Stages 1-3: normalize → spelling → dedup
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tags:
        t = _normalize_tag(raw)
        if not t:
            continue
        if normalize_british:
            t = _normalise_spelling(t)
        if t in seen:
            continue
        seen.add(t)
        normalized.append(t)

    # Stage 4: noise filter
    # Stage 5: freq filter
    filtered: list[str] = [t for t in normalized if not _is_noise(t) and freq_map.get(t, 0) >= min_freq]

    # Stage 6: substring dedup
    if substring_dedup and len(filtered) > 1:
        kept: list[str] = []
        for t in filtered:
            drop = False
            to_remove: list[str] = []
            for k in kept:
                if t == k:
                    drop = True
                    break
                if k in t or t in k:
                    tf, kf = freq_map.get(t, 0), freq_map.get(k, 0)
                    # Higher freq wins; on tie, longer wins
                    t_wins = (tf > kf) or (tf == kf and len(t) > len(k))
                    if t_wins:
                        to_remove.append(k)
                    else:
                        drop = True
                        break
            if drop:
                continue
            for r in to_remove:
                kept.remove(r)
            kept.append(t)
        filtered = kept

    # Stage 7: ordering
    if not preserve_input_order:
        filtered.sort(key=lambda x: (-freq_map.get(x, 0), x))

    # Stage 8: top_n cap
    return filtered[:top_n]


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------


def _first_str(values: Any, fallback: str = "") -> str:
    if isinstance(values, list) and values:
        return str(values[0]).strip()
    if isinstance(values, str):
        return values.strip()
    return fallback


def _join_strs(values: Any, sep: str = ", ", limit: int | None = None) -> str:
    if values is None:
        return ""
    if not isinstance(values, list):
        values = [values]
    items = [str(x).strip() for x in values if x not in (None, "")]
    if limit is not None:
        items = items[:limit]
    return sep.join(items)


def _year(release_date: Any) -> str:
    rd = _first_str(release_date, "")
    if rd and len(rd) >= 4 and rd[:4].isdigit():
        return rd[:4]
    return ""


def _popularity(pop: Any) -> str:
    if pop in (None, "", []):
        return ""
    try:
        return str(int(round(float(pop))))
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Bucket helpers (used by query-side global aggregates)
# ---------------------------------------------------------------------------


def _bucket_year_str(y: str) -> str:
    """Decade bucket — "1995" → "1990s". Standard 10-year boundary."""
    if not y:
        return ""
    try:
        yi = int(y)
        return f"{yi // 10 * 10}s"
    except (TypeError, ValueError):
        return ""


def _bucket_pop_str(p: str) -> str:
    """10-point popularity bucket — "67" → "p60-69"."""
    if not p:
        return ""
    try:
        pi = int(p)
        lo = pi // 10 * 10
        return f"p{lo}-{lo + 9}"
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Casing — preserve proper nouns
# ---------------------------------------------------------------------------

_SMART_LOWER_DEFAULT = object()


def _smart_lower(text: str, nlp: Any = _SMART_LOWER_DEFAULT) -> str:
    """Lowercase except spaCy-detected PROPN tokens; preserves whitespace.

    `nlp` not passed (default) → use module-level `_SPACY_MODEL` if available.
    `nlp=None` → force plain `str.lower()` fallback (used by tests and by
    callers that don't want the spaCy cost).
    """
    if not text:
        return ""
    if nlp is _SMART_LOWER_DEFAULT:
        nlp = _SPACY_MODEL
    if nlp is None:
        return text.lower()
    try:
        doc = nlp(text)
    except Exception:
        return text.lower()
    out: list[str] = []
    for tok in doc:
        out.append(tok.text if tok.pos_ == "PROPN" else tok.text.lower())
        out.append(tok.whitespace_)
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Natural-language phrase helpers (Gemini-aligned doc composition)
# ---------------------------------------------------------------------------


def _decade_bucket_phrase(year_str: str) -> str:
    """4-digit year → 'early/mid/late <decade>s'. '1993' → 'early 1990s'."""
    if not year_str or len(year_str) < 4 or not year_str[:4].isdigit():
        return ""
    y = int(year_str[:4])
    decade = (y // 10) * 10
    last = y % 10
    bucket = "early" if last <= 3 else ("mid" if last <= 6 else "late")
    return f"{bucket} {decade}s"


def _popularity_phrase(pop_str: str) -> str:
    """0-100 popularity → coarse adjective phrase."""
    if not pop_str or not pop_str.lstrip("-").isdigit():
        return ""
    p = int(pop_str)
    if p <= 20:
        return "obscure"
    if p <= 40:
        return "niche"
    if p <= 60:
        return "moderate popularity"
    if p <= 80:
        return "highly popular"
    return "mainstream hit"


def _era_genre_phrase(year_str: str, primary_genre: str) -> str:
    """Combine decade bucket + primary genre into a single retrieval-strong phrase.

    Examples::

        _era_genre_phrase("1987", "synth-pop") -> "late 1980s synth-pop"
        _era_genre_phrase("2003", "indie rock") -> "early 2000s indie rock"
        _era_genre_phrase("",     "techno")    -> "techno"
        _era_genre_phrase("1995", "")          -> "mid 1990s"

    Used by the doc-side `era-genre` component to give the era-sound (K) cell
    stronger lexical co-occurrence between era and genre. Falls back
    gracefully when either field is missing.
    """
    decade = _decade_bucket_phrase(year_str)
    primary_genre = (primary_genre or "").strip()
    if not decade and not primary_genre:
        return ""
    if decade and primary_genre:
        return f"{decade} {primary_genre}"
    return decade or primary_genre


def _canonical_alias(text: str) -> str:
    """Lowercase + collapse whitespace + strip surrounding punctuation."""
    if not text:
        return ""
    s = re.sub(r"\s+", " ", str(text).strip().lower())
    return s.strip(".,;:!?\"'`-")


def _alias_line(*fields: str) -> str:
    """Build a ``a | b | c`` canonical alias line from arbitrary identity fields.

    Dedups case-insensitively. Empty fields are skipped. Returns "" if no
    survivable field. Used by the doc-side `alias` component to give the
    exact-popular-hit (I) cell lexical robustness for slightly-misspelled or
    differently-cased references in the user query.
    """
    seen: set[str] = set()
    aliases: list[str] = []
    for raw in fields:
        c = _canonical_alias(raw)
        if c and c not in seen:
            seen.add(c)
            aliases.append(c)
    return " | ".join(aliases)


def _tempo_phrase(bpm_str: Any) -> str:
    """BPM string → coarse tempo phrase. '120' → 'mid-tempo'."""
    if bpm_str in (None, "", []):
        return ""
    try:
        bpm = float(bpm_str)
    except (TypeError, ValueError):
        return ""
    if bpm < 70:
        return "slow"
    if bpm < 110:
        return "mid-tempo"
    if bpm < 140:
        return "upbeat"
    return "fast"


def _extract_propn_priority_tags(tags: list[str], nlp: Any = _SMART_LOWER_DEFAULT) -> list[str]:
    """Reorder tags so PROPN-bearing tags come first (preserves casing).

    spaCy tokenizes each tag; if any token is PROPN, the tag is hoisted to the
    front of the result list. Within each group, original input order is kept.
    Returns the same set of tags, just reordered. spaCy unavailable → input
    order unchanged.
    """
    if nlp is _SMART_LOWER_DEFAULT:
        nlp = _SPACY_MODEL
    if nlp is None or not tags:
        return list(tags)
    propn: list[str] = []
    rest: list[str] = []
    for tag in tags:
        try:
            doc = nlp(tag)
            has_propn = any(tok.pos_ == "PROPN" for tok in doc)
        except Exception:
            has_propn = False
        (propn if has_propn else rest).append(tag)
    return propn + rest


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Cap text at max_chars at the last word boundary, strip trailing punct."""
    if not text or len(text) <= max_chars:
        return (text or "").strip()
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut.rstrip(",.;:! ").strip()


def _split_csv(s: Any) -> list[str]:
    """Split 'a, b, c' or list → cleaned list of strings."""
    if not s:
        return []
    if isinstance(s, list):
        items = [str(x).strip() for x in s if x not in (None, "")]
    else:
        items = [x.strip() for x in str(s).split(",")]
    return [it for it in items if it]


# ---------------------------------------------------------------------------
# Ablation toggles (used by eval.py / extract_pairs.py to isolate component
# contributions). Keep the literal strings stable — they're referenced from
# CLI flags and external scripts.
# ---------------------------------------------------------------------------

# Four compose components that can be independently disabled for ablation /
# evaluation. Each constant is the CLI/external token used by eval.py,
# extract_pairs.py, doc_cache.py.
#
#   cat-tail (query EOS):
#       Append the lexical category phrase to the very end of `Looking for:`
#       so EOS pooling sees the cat axis next to the user message.
#   goal-cue (query Context):
#       Regex-mine `listener_goal` text for era / popularity / time-recency
#       words and surface them as `era cue: …`, `popularity cue: …`,
#       `time cue: …` lines in Context.
#   era-genre (doc L2 paren):
#       Combine decade phrase with the primary genre token (`late 1980s
#       synth-pop`) instead of emitting the bare decade.
#   alias (doc L4.5):
#       Emit a lowercase canonical alias line — `Also known as: title |
#       artist | album` — to give the doc lexical robustness.
ABLATE_NO_CAT_TAIL = "no-cat-tail"  # disable cat-tail (EOS category anchor)
ABLATE_NO_GOAL_CUE = "no-goal-cue"  # disable goal-cue (era/pop/time regex on listener_goal)
ABLATE_NO_ERA_GENRE = "no-era-genre"  # disable era-genre (decade × primary_genre combo)
ABLATE_NO_ALIAS = "no-alias"  # disable alias (canonical 'Also known as:' line)

ALL_ABLATIONS: tuple[str, ...] = (
    ABLATE_NO_CAT_TAIL,
    ABLATE_NO_GOAL_CUE,
    ABLATE_NO_ERA_GENRE,
    ABLATE_NO_ALIAS,
)


def _resolve_ablate(ablate: Any) -> set[str]:
    """Accept None / str / iterable[str] → return a normalised set."""
    if not ablate:
        return set()
    if isinstance(ablate, str):
        items = [ablate]
    else:
        items = list(ablate)
    out: set[str] = set()
    for it in items:
        s = str(it).strip().lower()
        if s and s != "none":
            out.add(s)
    return out


# ---------------------------------------------------------------------------
# Doc composition — Gemini-aligned: natural prose + `Suitable for:` EOS anchor
# ---------------------------------------------------------------------------


def compose_doc(
    row: dict,
    freq_map: dict[str, int],
    *,
    crawl: dict | None = None,
    synth: dict | None = None,
    top_n_tags: int = 15,
    min_freq: int = 2,
    substring_dedup: bool = False,
    tag_propn_priority: bool = True,
    ablate: Any = None,
) -> str:
    """Compose a retrieval-aligned doc string for a track.

    The output is intentionally **natural-language prose** rather than
    `key: value` lines. Last line is ``Suitable for: u1; u2; ...`` so that
    last-token pooling aligns with the query's ``Looking for: ...`` anchor.

    Sources
    -------
    - ``row``    — HF Track-Metadata row (track_name, artist_name, album_name,
                  release_date, popularity, tag_list).
    - ``crawl``  — Optional dict from ``KVStore.get_track_crawl(tid)`` with
                  ``mb_tags``, ``caption``, ``lyrics``, ``label``, ``country``,
                  ``tempo``, ``key``.
    - ``synth``  — Optional dict ``{"mood": str, "themes": str, "use_cases":
                  list[str]}`` from
                  ``mymodule.strategies.twotower.synth_doc`` (DSPy-generated,
                  JSONL-cached). Drives the Mood / Themes / Suitable for lines.

    Layout (lines that have no signal are skipped — Suitable for is also
    dropped when both ``synth`` and crawl-derived mood/themes are empty,
    rather than emitting a meaningless EOS anchor):

        L1: "Title" by Artist (Year, primary_genre).
        L2: From Album, released on Label (Country, decade phrase).
        L3: Mood: m1, m2, m3.
        L4: Themes: t1, t2, t3.
        L5: Notable tags: tag1, tag2, ...
        L6: Tempo: 120 BPM (mid-tempo). Key: A minor.
        L7: Popularity: highly popular (73/100).
        L8: Suitable for: u1; u2; u3; u4; u5.   ← EOS anchor (★)
    """
    ablate_set = _resolve_ablate(ablate)
    title = _first_str(row.get("track_name"), "")
    artist = _join_strs(row.get("artist_name"))
    album = _join_strs(row.get("album_name"))
    year = _year(row.get("release_date"))
    decade_phrase = _decade_bucket_phrase(year)
    pop = _popularity(row.get("popularity"))
    pop_phrase = _popularity_phrase(pop)
    # `alias` component — canonical alias line; off → empty string → line skipped.
    alias_combo = "" if ABLATE_NO_ALIAS in ablate_set else _alias_line(title, artist, album)

    raw_tag_list = row.get("tag_list") or []
    cleaned_tags = clean_tags(
        raw_tag_list,
        freq_map,
        min_freq=min_freq,
        top_n=top_n_tags,
        normalize_british=True,
        substring_dedup=substring_dedup,
        preserve_input_order=True,
    )
    if tag_propn_priority:
        cleaned_tags = _extract_propn_priority_tags(cleaned_tags)

    crawl = crawl or {}
    mb_tags = _split_csv(crawl.get("mb_tags"))
    label = (crawl.get("label") or "").strip()
    country = (crawl.get("country") or "").strip()
    tempo_raw = crawl.get("tempo")
    tempo_p = _tempo_phrase(tempo_raw)
    key = (crawl.get("key") or "").strip()

    synth = synth or {}
    mood = (synth.get("mood") or "").strip()
    themes = (synth.get("themes") or "").strip()
    use_cases = synth.get("use_cases") or []
    if isinstance(use_cases, str):
        use_cases = [u.strip() for u in use_cases.split(";") if u.strip()]
    use_cases = [u.strip().rstrip(".") for u in use_cases if u and u.strip()][:5]

    # Primary genre: mb_tags first (curated), then first HF tag.
    primary_genre = ""
    if mb_tags:
        primary_genre = mb_tags[0]
    elif raw_tag_list:
        primary_genre = str(_first_str(raw_tag_list, "") or (raw_tag_list[0] if isinstance(raw_tag_list, list) else ""))

    lines: list[str] = []

    # L1 — identity
    head_parts: list[str] = []
    if title:
        head_parts.append(f'"{title}"')
    if artist:
        head_parts.append(f"by {artist}")
    paren_bits: list[str] = []
    if year:
        paren_bits.append(year)
    if primary_genre:
        paren_bits.append(primary_genre)
    head = " ".join(head_parts).strip()
    if paren_bits:
        head = f"{head} ({', '.join(paren_bits)})" if head else f"({', '.join(paren_bits)})"
    if head:
        lines.append(f"{head}.")

    # L2 — release context. The `era-genre` component replaces the standalone
    # decade phrase with an era × primary-genre combo (e.g., 'late 1980s
    # synth-pop') so the era-sound (K) cell query gains stronger lexical
    # co-occurrence. Ablation `no-era-genre` reverts to the standalone decade.
    era_genre = "" if ABLATE_NO_ERA_GENRE in ablate_set else _era_genre_phrase(year, primary_genre)
    rel_parts: list[str] = []
    if album:
        rel_parts.append(f"From {album}")
    label_country: list[str] = []
    if country:
        label_country.append(country)
    if era_genre:
        label_country.append(era_genre)
    elif decade_phrase:
        label_country.append(decade_phrase)
    if label:
        prefix = "released on" if rel_parts else "Released on"
        rel_bit = f"{prefix} {label}"
        if label_country:
            rel_bit += f" ({', '.join(label_country)})"
        rel_parts.append(rel_bit)
    elif label_country and rel_parts:
        rel_parts[-1] = rel_parts[-1] + f" ({', '.join(label_country)})"
    elif label_country:
        rel_parts.append(f"Released {', '.join(label_country)}")
    if rel_parts:
        lines.append(", ".join(rel_parts) + ".")

    # L3 — Mood (from synth only — heuristic mood from caption is too noisy)
    if mood:
        lines.append(f"Mood: {mood}.")

    # L4 — Themes
    if themes:
        lines.append(f"Themes: {themes}.")

    # L4.5 — Also known as (canonical alias). The `alias` component feeds the
    # exact-popular-hit (I) cell with lexical-robust forms of identity. Placed
    # between Themes and Notable tags so it sits in the EOS-adjacent block.
    if alias_combo:
        lines.append(f"Also known as: {alias_combo}.")

    # L5 — Notable tags (PROPN-priority sorted)
    if cleaned_tags:
        lines.append(f"Notable tags: {', '.join(cleaned_tags)}.")

    # L6 — Tempo / Key
    tempo_key_bits: list[str] = []
    if tempo_raw not in (None, "", []):
        try:
            bpm_int = int(round(float(tempo_raw)))
            if tempo_p:
                tempo_key_bits.append(f"Tempo: {bpm_int} BPM ({tempo_p})")
            else:
                tempo_key_bits.append(f"Tempo: {bpm_int} BPM")
        except (TypeError, ValueError):
            pass
    if key:
        tempo_key_bits.append(f"Key: {key}")
    if tempo_key_bits:
        lines.append(". ".join(tempo_key_bits) + ".")

    # L7 — Popularity
    if pop and pop_phrase:
        lines.append(f"Popularity: {pop_phrase} ({pop}/100).")
    elif pop:
        lines.append(f"Popularity: {pop}/100.")

    # L8 — Suitable for (EOS anchor) — only when LLM use_cases available.
    # Skipping here is intentional: a low-quality derived anchor would pollute
    # last-token pooling. Run `python -m mymodule.strategies.twotower.synth_doc`
    # to populate use_cases for all tracks.
    if use_cases:
        lines.append(f"Suitable for: {'; '.join(use_cases)}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query-side helpers
# ---------------------------------------------------------------------------


# goal-cue helpers — regex / dictionary mining for era / popularity / time cues
# from listener_goal text. These augment the PROPN/NER pull (vague-memory F
# anchor) with lexical signals that target the exact-popular-hit (I),
# popular-trends (J), and era-sound (K) cells which lack PROPN density.

_ERA_NUMERIC_PATTERN = re.compile(r"\b(?:(?:19|20)\d{2}s?|'?\d{2}s)\b", re.IGNORECASE)
_DECADE_WORD_TO_PHRASE: dict[str, str] = {
    "thirties": "1930s",
    "forties": "1940s",
    "fifties": "1950s",
    "sixties": "1960s",
    "seventies": "1970s",
    "eighties": "1980s",
    "nineties": "1990s",
    "aughts": "2000s",
    "noughties": "2000s",
    "tens": "2010s",
    "twenties": "2020s",
}
_DECADE_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _DECADE_WORD_TO_PHRASE) + r")\b",
    re.IGNORECASE,
)

_POP_CUE_PHRASES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "top-tier",
        re.compile(
            r"\b(?:top|biggest|#1|number\s+one|chart[- ]?topping|smash\s+hit|mega[- ]?hit|huge\s+hit|massive\s+hit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "popular",
        re.compile(
            r"\b(?:popular|mainstream|famous|well[- ]?known)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "obscure",
        re.compile(
            r"\b(?:obscure|hidden\s+gem|underrated|underground|lesser[- ]?known"
            r"|under\s+the\s+radar|deep\s+cut|niche)\b",
            re.IGNORECASE,
        ),
    ),
)

_TIME_CUE_PHRASES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "present",
        re.compile(
            r"\b(?:right\s+now|nowadays|these\s+days|this\s+(?:week|month|year)"
            r"|recent(?:ly)?|latest|brand[- ]?new|trending|currently)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "classic",
        re.compile(
            r"\b(?:classic|oldies|vintage|retro|throwback|nostalgic|nostalgia|old[- ]?school|yesteryear)\b",
            re.IGNORECASE,
        ),
    ),
)


def _normalize_era_token(tok: str) -> str:
    """Normalize raw era surface to canonical 'YYYYs' form. '80s'→'1980s', '1990'→'1990s'."""
    t = tok.strip().lower().lstrip("'")
    if not t:
        return ""
    if t.endswith("s"):
        head = t[:-1]
    else:
        head = t
    if not head.isdigit():
        return ""
    if len(head) == 2:
        # '80' → '1980' if >= 30 else '2080' fallback (no listener_goal references 2080s)
        century = "19" if int(head) >= 30 else "20"
        head = century + head
    elif len(head) == 4:
        # Exact year. Bucket to decade.
        head = head[:3] + "0"
    else:
        return ""
    return f"{head}s"


def _extract_goal_cues(conversation_goal: Any) -> dict[str, list[str]]:
    """Mine era / popularity / time-recency cues from ``listener_goal``.

    Returns a dict::

        {
            "era":  ["1980s", "1990s"],   # canonical decade buckets, dedup, in surface order
            "pop":  ["top-tier"],         # one of {top-tier, popular, obscure} — first match wins
            "time": ["present"],          # one of {present, classic} — first match wins
        }

    Pop / time use mutually-exclusive bucketing (first regex match wins) because
    contradictory cues in the same goal text are rare and the bucket label is
    what we want EOS to see.
    """
    if not conversation_goal or not isinstance(conversation_goal, dict):
        return {"era": [], "pop": [], "time": []}
    text = conversation_goal.get("listener_goal") or ""
    if not isinstance(text, str) or not text.strip():
        return {"era": [], "pop": [], "time": []}

    eras: list[str] = []
    seen_eras: set[str] = set()
    for m in _ERA_NUMERIC_PATTERN.finditer(text):
        canon = _normalize_era_token(m.group(0))
        if canon and canon not in seen_eras:
            seen_eras.add(canon)
            eras.append(canon)
    for m in _DECADE_WORD_PATTERN.finditer(text):
        canon = _DECADE_WORD_TO_PHRASE[m.group(0).lower()]
        if canon not in seen_eras:
            seen_eras.add(canon)
            eras.append(canon)

    pop_label: list[str] = []
    for label, pat in _POP_CUE_PHRASES:
        if pat.search(text):
            pop_label.append(label)
            break

    time_label: list[str] = []
    for label, pat in _TIME_CUE_PHRASES:
        if pat.search(text):
            time_label.append(label)
            break

    return {"era": eras, "pop": pop_label, "time": time_label}


def _natural_goal_phrase(conversation_goal: Any) -> str:
    """Render conversation_goal (category + specificity) as a natural fragment.

    No "Session Goal:" label — the phrase is meant to be embedded inside the
    `Context: ...` sentence so the user-intent reads naturally. We deliberately
    do NOT inject `listener_goal` raw text (it can contain annotator-leaked GT
    hints — see data-explorer 2026-05-17 analysis).

    Examples:
        HH + J (popular trends)  → "wants a specific popular-trends track"
        HL + G (mood and emotion)→ "wants tracks matching a specific mood-and-emotion angle"
        LH + F (vague memory)    → "trying to recall a particular track from a vague vague-memory sense"
        LL + A (sonic chars)     → "exploring various sonic-characteristics tracks"
    """
    if not conversation_goal or not isinstance(conversation_goal, dict):
        return ""
    cat = (conversation_goal.get("category") or "").strip().upper()
    spec = (conversation_goal.get("specificity") or "").strip().upper()
    cat_phrase = CATEGORY_PHRASE.get(cat, "")
    if not cat_phrase:
        return ""
    if spec == "HH":
        return f"wants a specific {cat_phrase} track"
    if spec == "HL":
        return f"wants tracks matching a specific {cat_phrase} angle"
    if spec == "LH":
        return f"trying to recall a particular track from a vague {cat_phrase} sense"
    if spec == "LL":
        return f"exploring various {cat_phrase} tracks"
    return f"interested in {cat_phrase} tracks"


def _extract_query_keywords(
    chat_history: list[dict] | None,
    *,
    nlp: Any = _SMART_LOWER_DEFAULT,
    top_n: int = 8,
) -> list[str]:
    """Extract NOUN/PROPN/ADJ tokens from prior user messages (NOT current turn).

    PROPN preserved with original casing AND prioritized first (matches the
    doc-side `_extract_propn_priority_tags` convention). Returns a deduped
    ordered list. spaCy unavailable / no prior user messages → empty list.

    Why primary (not fallback): even when an LLM `session_summary` is
    available, the raw lexical anchors from prior queries are independent
    high-precision signal (genre names, artist names, mood adjectives) that
    the summary may compress away. We surface them in the Context line.
    """
    if nlp is _SMART_LOWER_DEFAULT:
        nlp = _SPACY_MODEL
    if not chat_history or nlp is None:
        return []
    user_texts = [(msg.get("content") or "").strip() for msg in chat_history if msg.get("role") == "user"]
    user_texts = [t for t in user_texts if t]
    if not user_texts:
        return []
    propn: list[str] = []
    other: list[str] = []
    seen_lower: set[str] = set()
    for txt in user_texts:
        try:
            doc = nlp(txt)
        except Exception:
            continue
        for tok in doc:
            if tok.pos_ not in {"NOUN", "PROPN", "ADJ"}:
                continue
            if tok.is_stop:
                continue
            term = tok.text if tok.pos_ == "PROPN" else tok.lemma_.lower()
            if len(term) < 3:
                continue
            key = term.lower()
            if key in seen_lower or key in _STOPWORDS:
                continue
            seen_lower.add(key)
            (propn if tok.pos_ == "PROPN" else other).append(term)
    return (propn + other)[:top_n]


def _extract_goal_entities(
    conversation_goal: Any,
    *,
    nlp: Any = _SMART_LOWER_DEFAULT,
    top_n: int = 6,
) -> list[str]:
    """Extract retrieval-strong entities from ``conversation_goal.listener_goal``.

    Keeps only PROPN tokens + selected NER spans (PERSON / ORG / WORK_OF_ART /
    GPE / NORP / LANGUAGE / EVENT). Generic NOUN / ADJ are excluded because the
    spec×category phrase already encodes the goal axis — passing those again
    just adds noise. Examples of what survives:

        "find a specific Clubroot song"                      → ["Clubroot"]
        "identify Jim Guthrie's Sword & Sworcery Lp"         → ["Jim Guthrie", "Sword & Sworcery"]
        "find one specific K-Pop song from the early 2010s"  → ["K-Pop"]
        "uplifting songs by Indonesian artists"              → ["Indonesian"]
        "vague memory of a melancholic mood"                 → []

    The annotator's listener_goal is a dataset-provided input under challenge
    rules, so using the surfaced entities is not a GT leak — we just avoid
    pulling the entire free-text into the query.
    """
    if nlp is _SMART_LOWER_DEFAULT:
        nlp = _SPACY_MODEL
    if not conversation_goal or nlp is None:
        return []
    if not isinstance(conversation_goal, dict):
        return []
    goal_text = conversation_goal.get("listener_goal") or ""
    if not isinstance(goal_text, str) or not goal_text.strip():
        return []
    try:
        doc = nlp(goal_text)
    except Exception:
        return []

    keep_labels = {"PERSON", "ORG", "WORK_OF_ART", "GPE", "NORP", "LANGUAGE", "EVENT"}
    out: list[str] = []
    seen_lower: set[str] = set()

    # Pass 1 — NER spans (multi-token entities preserved as-is).
    for ent in doc.ents:
        if ent.label_ not in keep_labels:
            continue
        txt = ent.text.strip().strip("'\"`")
        key = txt.lower()
        if len(txt) < 3 or key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(txt)

    # Pass 2 — standalone PROPN tokens not already covered by an NER span.
    for tok in doc:
        if tok.pos_ != "PROPN" or tok.is_stop:
            continue
        if tok.ent_iob_ in ("B", "I"):
            continue
        txt = tok.text.strip().strip("'\"`")
        key = txt.lower()
        if len(txt) < 3 or key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(txt)

    return out[:top_n]


def _iter_music_metas(chat_history: list[dict] | None, kv_store: Any | None):
    """Yield ``meta`` dicts for every ``music`` role in chat_history (in order)."""
    if not chat_history or kv_store is None:
        return
    for msg in chat_history:
        if msg.get("role") != "music":
            continue
        tid = msg.get("content")
        if not tid:
            continue
        try:
            meta = kv_store.get_track_meta(tid)
        except Exception:
            continue
        if meta:
            yield meta


def _format_recent_listens(
    chat_history: list[dict] | None,
    kv_store: Any | None,
    *,
    max_items: int = 5,
) -> list[str]:
    """Return formatted track lines in **recent-first** order.

    Each entry: ``"Title" by Artist (Year, genre)``. Empty fields are skipped
    within the parens. We reverse the chronological order because earlier
    experiments showed recent-first improved retrieval (the most recent listen
    is the strongest signal for the next pick).
    """
    metas = list(_iter_music_metas(chat_history, kv_store))
    if not metas:
        return []
    metas.reverse()  # recent-first
    lines: list[str] = []
    for meta in metas[:max_items]:
        title = _first_str(meta.get("track_name"), "")
        artist = _first_str(meta.get("artist_name"), "")
        release = _first_str(meta.get("release_date"), "")
        year = release[:4] if release and len(release) >= 4 and release[:4].isdigit() else ""
        tags = meta.get("tag_list") or []
        genre = ""
        if isinstance(tags, list) and tags:
            genre = str(tags[0]).strip()
        head_bits: list[str] = []
        if title:
            head_bits.append(f'"{title}"')
        if artist:
            head_bits.append(f"by {artist}")
        paren: list[str] = []
        if year:
            paren.append(year)
        if genre:
            paren.append(genre)
        line = " ".join(head_bits)
        if paren:
            line = f"{line} ({', '.join(paren)})" if line else f"({', '.join(paren)})"
        if line:
            lines.append(line)
    return lines


def _aggregate_tags_list(
    chat_history: list[dict] | None,
    kv_store: Any | None,
    freq_map: dict[str, int],
    *,
    top_n: int = 10,
) -> list[str]:
    """Distinct first-occurrence-ordered tags across chat_history music tracks."""
    order: "OrderedDict[str, int]" = OrderedDict()
    for meta in _iter_music_metas(chat_history, kv_store):
        for t in clean_tags(
            meta.get("tag_list") or [],
            freq_map,
            normalize_british=True,
            substring_dedup=True,
            preserve_input_order=True,
        ):
            if t not in order:
                order[t] = len(order)
            if len(order) >= top_n:
                break
        if len(order) >= top_n:
            break
    return list(order.keys())


def _aggregate_era_phrase(chat_history: list[dict] | None, kv_store: Any | None) -> str:
    """Compact era span phrase from chat_history. '1990s' / '1990s-2000s' / ''."""
    decades: set[str] = set()
    for meta in _iter_music_metas(chat_history, kv_store):
        y = _year(meta.get("release_date"))
        if y:
            b = _bucket_year_str(y)
            if b:
                decades.add(b)
    if not decades:
        return ""
    s = sorted(decades)
    return s[0] if len(s) == 1 else f"{s[0]}-{s[-1]}"


def _aggregate_pop_phrase(chat_history: list[dict] | None, kv_store: Any | None) -> str:
    """Median popularity → natural phrase (matches `_popularity_phrase` buckets)."""
    pops: list[int] = []
    for meta in _iter_music_metas(chat_history, kv_store):
        p = _popularity(meta.get("popularity"))
        if p:
            try:
                pops.append(int(p))
            except ValueError:
                pass
    if not pops:
        return ""
    median = sorted(pops)[len(pops) // 2]
    return _popularity_phrase(str(median))


# ---------------------------------------------------------------------------
# Token-budget aware assembly
# ---------------------------------------------------------------------------


def _measure_tokens(text: str, tokenizer: Any) -> int:
    """Token count, falling back to char/3.5 heuristic when tokenizer is None."""
    if tokenizer is None:
        return int(len(text) / 3.5) + 1
    return len(tokenizer.encode(text, add_special_tokens=False))


# Drop order (lowest priority first). ``looking_for`` (with optional cat tail)
# is the EOS anchor and must NEVER drop; the most recent listen is also
# preserved unless the budget is so tight that nothing else remains.
_QUERY_DROP_ORDER: tuple[str, ...] = (
    "session_summary",  # LLM-generated 1-sentence summary
    "context_aggregates",  # era / pop / tags inside Context
    "context_keywords_full",  # cap keywords list at 4 instead of 8
    "context_keywords",  # drop keywords from Context entirely
    "recent_middle",  # drop middle entries of Recent listens
    "context_goal_cues",  # drop era/pop/time cues mined from goal text
    "context_goal_phrase",  # drop the goal/category phrase from Context
    "recent_first",  # drop all but the most recent listen
    # `looking_for` (+ tail_cat) and `recent_last` are never dropped
)


def _assemble_query(components: dict[str, Any], enabled: dict[str, bool]) -> str:
    """Reassemble the natural-prose query body from selected components.

    Output shape (lines with no signal are skipped)::

        Recent listens: <recent>; <older>; ...
        Session so far: <LLM summary sentence>.
        Context: <goal phrase>. <era>; <pop>; tags: ...; lexical cues — <PROPN/NER + prior keywords>.
        Looking for: <current turn user message>            ← EOS anchor (★)
    """
    sections: list[str] = []

    # L1 — Recent listens (recent-first; drops governed by recent_first/middle).
    recent_lines: list[str] = list(components.get("recent_lines") or [])
    if recent_lines:
        keep = list(recent_lines)
        if not enabled.get("recent_first", True) and len(keep) > 1:
            keep = [keep[0]]
        elif not enabled.get("recent_middle", True) and len(keep) > 2:
            keep = [keep[0], keep[-1]]
        sections.append("Recent listens: " + "; ".join(keep) + ".")

    # L2 — Session so far (LLM summary).
    if enabled.get("session_summary") and components.get("session_summary"):
        text = components["session_summary"].rstrip(". ").strip()
        if text:
            sections.append(f"Session so far: {text}.")

    # L3 — Context: goal phrase + aggregates + keywords + goal-mined cues
    # (single sentence).
    ctx_bits: list[str] = []
    if enabled.get("context_goal_phrase") and components.get("goal_phrase"):
        ctx_bits.append(components["goal_phrase"])
    if enabled.get("context_aggregates"):
        agg: list[str] = []
        era = components.get("era") or ""
        pop = components.get("pop") or ""
        tags = components.get("tags") or []
        if era:
            agg.append(f"era {era}")
        if pop:
            agg.append(pop)
        if tags:
            agg.append("tags: " + ", ".join(tags[:8]))
        if agg:
            ctx_bits.append("; ".join(agg))
    # `goal-cue` component — goal-mined cues (era / popularity / time).
    # Independent of history aggregates so deeper turns still surface the
    # listener_goal axis.
    if enabled.get("context_goal_cues"):
        cue_bits: list[str] = []
        g_era = components.get("goal_era") or []
        g_pop = components.get("goal_pop") or []
        g_time = components.get("goal_time") or []
        if g_era:
            cue_bits.append("era cue: " + ", ".join(g_era))
        if g_pop:
            cue_bits.append("popularity cue: " + ", ".join(g_pop))
        if g_time:
            cue_bits.append("time cue: " + ", ".join(g_time))
        if cue_bits:
            ctx_bits.append("; ".join(cue_bits))
    if enabled.get("context_keywords") and components.get("keywords"):
        n = 8 if enabled.get("context_keywords_full", True) else 4
        kw = components["keywords"][:n]
        if kw:
            ctx_bits.append("lexical cues — " + ", ".join(kw))
    if ctx_bits:
        sections.append("Context: " + ". ".join(ctx_bits) + ".")

    # L4 — Looking for (EOS anchor, never dropped). The `cat-tail` component
    # appends the lexical category label at the very tail so EOS pooling sees
    # the category axis right next to the user message.
    tail = ""
    if enabled.get("tail_cat") and components.get("cat_lexical"):
        tail = f" — {components['cat_lexical']}"
    sections.append(f"Looking for: {components['looking_for']}{tail}")
    return "\n".join(sections)


def _build_query_components(
    user_msg: str,
    chat_history: list[dict] | None,
    conversation_goal: dict | None,
    kv_store: Any | None,
    freq_map: dict[str, int],
    *,
    session_summary: str | None = None,
    ablate: set[str] | None = None,
) -> dict[str, Any]:
    """Pre-compute every component as an independent value.

    The natural-prose assembly happens in ``_assemble_query``. Components are
    raw values (lists / strings) so the assembly can decide formatting and
    drop logic uniformly.
    """
    looking_for_raw = (user_msg or "").strip()
    looking_for = _smart_lower(looking_for_raw) or looking_for_raw

    # Lexical cues = prior-turn NOUN/PROPN/ADJ + listener_goal PROPN/NER.
    # Goal entities first so PROPN-priority survives the merge dedup.
    goal_entities = _extract_goal_entities(conversation_goal)
    prior_keywords = _extract_query_keywords(chat_history)
    merged_keywords: list[str] = []
    seen: set[str] = set()
    for term in (*goal_entities, *prior_keywords):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        merged_keywords.append(term)

    ablate_set = ablate or set()

    # `cat-tail` component — lexical category label for the EOS-adjacent tail
    # of `Looking for:`. `no-cat-tail` ablation: blank it out → tail dropped.
    cat_lexical = ""
    if ABLATE_NO_CAT_TAIL not in ablate_set and conversation_goal and isinstance(conversation_goal, dict):
        cat = (conversation_goal.get("category") or "").strip().upper()
        cat_lexical = CATEGORY_PHRASE.get(cat, "")

    # `goal-cue` component — era / popularity / time-recency cues mined from
    # listener_goal text. `no-goal-cue` ablation: return empty buckets → cue
    # line dropped.
    if ABLATE_NO_GOAL_CUE in ablate_set:
        goal_cues = {"era": [], "pop": [], "time": []}
    else:
        goal_cues = _extract_goal_cues(conversation_goal)

    return {
        "recent_lines": _format_recent_listens(chat_history, kv_store),
        "session_summary": (session_summary or "").strip(),
        "goal_phrase": _natural_goal_phrase(conversation_goal),
        "keywords": merged_keywords,
        "tags": _aggregate_tags_list(chat_history, kv_store, freq_map),
        "era": _aggregate_era_phrase(chat_history, kv_store),
        "pop": _aggregate_pop_phrase(chat_history, kv_store),
        "cat_lexical": cat_lexical,
        "goal_era": goal_cues["era"],
        "goal_pop": goal_cues["pop"],
        "goal_time": goal_cues["time"],
        "looking_for": looking_for or "(no current message)",
    }


# ---------------------------------------------------------------------------
# Public: compose_query
# ---------------------------------------------------------------------------


def compose_query(
    user_msg: str,
    chat_history: list[dict] | None,
    conversation_goal: dict | None,
    kv_store: Any | None,
    freq_map: dict[str, int],
    *,
    session_summary: str | None = None,
    tokenizer: Any = None,
    max_body_tokens: int = 240,
    ablate: Any = None,
) -> str:
    """Compose the retrieval-aligned natural-language query body.

    Output structure (Gemini-aligned, see ``_assemble_query`` for shape):

      Recent listens: <recent-first track list>
      Session so far: <optional LLM-summary sentence>
      Context: <goal phrase>. <era>; <pop>; tags: …; lexical cues — …
      Looking for: <current user message>     ← EOS anchor (matches doc-side Suitable for:)

    The ``Instruct:``/``Query:`` prefix is added downstream by
    ``mymodule.strategies.twotower.encoder.format_query``; ``max_body_tokens``
    should therefore equal ``model_max_len - prefix_tokens``.

    ``session_summary`` is an optional pre-baked LLM summary
    (``mymodule.strategies.twotower.query_summary``) keyed by
    ``(session_id, turn_number)``. When absent, the Session-so-far line is
    skipped and the lexical spaCy keywords still drive the Context line.

    Drop order (when over budget) is ``_QUERY_DROP_ORDER`` — lowest-priority
    pieces fall off first. ``Looking for:`` is never dropped.
    """
    ablate_set = _resolve_ablate(ablate)
    components = _build_query_components(
        user_msg,
        chat_history,
        conversation_goal,
        kv_store,
        freq_map,
        session_summary=session_summary,
        ablate=ablate_set,
    )

    enabled: dict[str, bool] = {
        "recent_listens": True,
        "recent_first": True,
        "recent_middle": True,
        "session_summary": True,
        "context_goal_phrase": True,
        "context_aggregates": True,
        "context_goal_cues": True,
        "context_keywords": True,
        "context_keywords_full": True,
        "tail_cat": True,
    }
    body = _assemble_query(components, enabled)

    drop_iter = iter(_QUERY_DROP_ORDER)
    while _measure_tokens(body, tokenizer) > max_body_tokens:
        try:
            target = next(drop_iter)
        except StopIteration:
            break  # only Looking for + most-recent listen remain — bail
        if enabled.get(target):
            enabled[target] = False
            body = _assemble_query(components, enabled)

    # Hard guarantee: even when the un-droppable `Looking for:` (current user
    # message) alone exceeds the budget, the body MUST fit so that
    # `format_query`'s prefix doesn't push the total past model_max_len — else
    # the encoder's left-truncation (truncation_side='left' for last-token
    # pooling) eats into the `Instruct:` prefix. Token-truncate from the LEFT so
    # the EOS anchor (`Looking for:` tail = the pooling position) survives.
    if _measure_tokens(body, tokenizer) > max_body_tokens:
        if tokenizer is not None:
            # decode↔re-encode drifts +1-2 tokens at the truncation boundary, and
            # format_query's prefix/body join can add one more. Leave 2-token
            # headroom and RE-CHECK so the final body never re-encodes above budget
            # (else the encoder's left-truncation clips the Instruct prefix).
            target = max(8, max_body_tokens - 2)
            ids = tokenizer.encode(body, add_special_tokens=False)
            body = tokenizer.decode(ids[-target:])
            while _measure_tokens(body, tokenizer) > target:
                ids = tokenizer.encode(body, add_special_tokens=False)
                if len(ids) <= 1:
                    break
                body = tokenizer.decode(ids[1:])
        else:
            body = body[-int(max_body_tokens * 3.5) :]

    return body


__all__ = [
    "SPECIFICITY_PHRASE",
    "CATEGORY_PHRASE",
    "build_tag_freq",
    "load_tag_freq",
    "clean_tags",
    "compose_doc",
    "compose_query",
]
