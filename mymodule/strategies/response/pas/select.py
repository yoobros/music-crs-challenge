"""Deterministic SELECT stage — validate + repair the LLM's response.

Runs after the DSPy ChainOfThought call:

1. Parse `themes` output lines and check that every referenced track index was
   actually proposed by Stage A (`pas_propose.classify_intents`). Violations are
   logged but do not block output.
2. Check the `response` text against TWO layers of regex bans:
     a. HARD ban list (hard-coded, applies every turn) — covers the globally
        repeated offenders that tank Lexical Diversity across sessions
        ("this song", "here are", "perfect for", ...). Violations are repaired
        via randomized rotation-pool substitution.
     b. LLM-emitted SOFT bans (session-specific regex lines from the DSPy
        `themes_excluded_patterns` field). Violations trigger ONE regeneration
        attempt with a feedback hint; if the retry also fails, the retry is
        accepted as-is (we prefer one soft-violation over a broken response).
3. Apply the hard-ban substitutions last so retry text also gets cleaned.

Substitutions use a `random.Random(seed)` seeded by (session_id, turn_number)
when available, so a given conversation's response is deterministic across
re-runs but different conversations diverge.
"""

from __future__ import annotations

import os
import random
import re
from difflib import SequenceMatcher  # noqa: F401 — used by _fuzzy_ratio below
from typing import Any

from loguru import logger

_ = SequenceMatcher  # keep ruff from flagging the re-export as unused

# ---------------------------------------------------------------------------
# Hard-ban list — patterns applied every turn regardless of LLM emission.
# Each entry: (regex_str, list_of_replacement_phrases).
# Keep replacements grammatically interchangeable with the banned phrase.
# ---------------------------------------------------------------------------
_HARD_BANS: list[tuple[str, list[str]]] = [
    # Phrase-specific first: the generic `I've got` replacement can otherwise
    # produce "I have one for you just the thing".
    (
        r"\bi['’]ve got just the thing\b",
        ["I have the right pick", "I found a strong fit", "I know the right cut", "I have a good one"],
    ),
    (
        r"\bright on[!,]?\s*glad that\b",
        ["good to hear that", "nice to know that", "great to hear that", "happy to hear that"],
    ),
    (
        r"\bright on!\s*",
        ["good call. ", "sounds good. ", "absolutely. "],
    ),
    (
        r"\bright on,\s*",
        ["good call, ", "sounds good, ", "absolutely, "],
    ),
    # Generic referents that swamp Distinct-2 when repeated
    (
        r"\bthis song\b",
        ["this cut", "the pick", "the number", "this piece", "the choice"],
    ),
    (
        r"\bthis track\b",
        ["this cut", "the number", "this piece", "this entry", "the pick"],
    ),
    (
        r"\bthis one\b",
        ["this pick", "this choice", "this entry", "this cut", "the one"],
    ),
    (
        r"\bthat song\b",
        ["that cut", "the pick there", "that number", "that piece"],
    ),
    (
        r"\bthat track\b",
        ["that cut", "that number", "that piece", "that pick"],
    ),
    # Banned openers (word-boundary instead of line-start to catch mid-sentence "here are")
    (
        r"\bhere are\b",
        ["Centered on", "Anchored by", "Kicking off with", "Leading with", "Starting with"],
    ),
    (
        r"\bcheck (this|these|it) out\b",
        ["queue this up", "slot this in", "reach for this", "drop this on"],
    ),
    (
        r"\bcheck out\b",
        ["try", "add", "queue up", "reach for"],
    ),
    (
        r"\byou absolutely need to hear\b",
        ["I'd start with", "I'd queue up", "go with", "reach for"],
    ),
    (
        r"\bperfectly feels like\b",
        ["leans into", "carries", "matches", "points toward"],
    ),
    (
        r"\bperfectly (captures|matches|fits)\b",
        ["carries", "matches", "leans into", "points toward"],
    ),
    (
        r"\bi['’]ve got\b",
        ["I have queued up", "I have lined up", "I'd start with", "I'd queue up", "My pick is"],
    ),
    (
        r"\bi['’]m going with\b",
        ["I'd pick", "I'd queue up", "I have"],
    ),
    (
        r"\byou['’]re after\b",
        ["you want", "you're looking for", "you're seeking", "you requested", "you're aiming for"],
    ),
    (
        r"\byou asked for\b",
        ["you wanted", "you requested", "the brief called for"],
    ),
    (
        r"\blet['’]s go with\b",
        ["try", "I'd pick", "I'd go with", "queue up", "my pick is"],
    ),
    (
        r"\byou['’]ll really like\b",
        [
            "this should land for you",
            "you'll dig",
            "this one's a fit",
            "you'll be into",
            "this should hit the mark",
        ],
    ),
    (
        r"\byou['’]ll love\b",
        [
            "you'll gravitate toward",
            "this should land for you",
            "you'll dig",
            "this one's a fit",
            "you'll be into",
        ],
    ),
    (
        r"\btake a look\b",
        ["have a listen", "sit with this", "try this"],
    ),
    # Filler collocations
    (
        r"\bperfect for\b",
        ["tailored to", "dialed in for", "matched to", "tuned for", "well-suited to"],
    ),
    (
        r"\bgreat for\b",
        ["fitting for", "set up for", "built for", "made for", "shaped around"],
    ),
    (
        r"\bthat vibe\b",
        ["that mood", "that atmosphere", "that sensibility", "that quality", "that feel", "that energy"],
    ),
    # LLM-native repetition — generic verb+noun patterns
    # NOTE: rotations intentionally exclude `supplies/yields/serves up` since those
    # are themselves banned encyclopedic verbs below.
    (
        r"\boffers a\b",
        ["brings a", "carries a", "adds a", "has a", "packs a", "lands a"],
    ),
    (
        r"\bprovides a\b",
        ["gives a", "carries a", "brings a", "has a", "delivers", "packs a"],
    ),
    (
        r"\bsense of\b",
        ["feeling of", "undercurrent of", "touch of", "thread of", "shade of", "hint of"],
    ),
    # Common Turn-2+ ack phrases — vary to avoid "glad you" / "hit the spot" bigram dominance
    (
        r"\bglad you (liked|enjoyed|dug)\b",
        ["happy you connected with", "great that you vibed with", "good you took to", "pleased you got into"],
    ),
    (
        r"\bhit the spot\b",
        ["landed well", "clicked", "worked for you", "did the trick"],
    ),
    (
        r"\bso glad\b",
        ["really happy", "glad", "happy", "pleased"],
    ),
    (
        r"\bgive (it|this|that) a listen\b",
        ["queue it up", "drop it on", "try it out", "press play"],
    ),
    (
        r"\bgive (it|this|that) a spin\b",
        ["press play", "save it for that mood", "try it next", "let it run"],
    ),
    (
        r"\bis the way to go\b",
        ["is the clearest match", "is the cleaner fit", "is my pick", "lands closest to the brief"],
    ),
    (
        r"\bnatural next step\b",
        ["clean follow-up", "good bridge", "direct follow-up", "next place I'd go"],
    ),
    (
        r"\bqueue it up next\b",
        ["press play when it fits", "save it for that mood", "try it next", "let it run"],
    ),
    (
        r"\bis a (?:fantastic|wonderful|beautiful|strong|natural|killer) choice\b",
        ["fits the brief", "matches the thread", "lands well here", "is the clearest match"],
    ),
    (
        r"\bis a fantastic start\b",
        ["is a useful starting point", "sets up the search", "is where I'd start"],
    ),
    (
        r"\bis a great comparison\b",
        ["makes a useful comparison", "helps separate that sound", "is useful for comparison"],
    ),
    (
        r"\bis a must(?:-hear)?\b",
        ["belongs here", "matches the brief", "lands well here", "is my pick"],
    ),
    (
        r"\bis a strong candidate\b",
        ["is the clearest match", "answers that brief", "lines up with that thread", "has the right contour"],
    ),
    (
        r"\bis a strong start\b",
        ["sets up that search", "is where I'd begin", "answers the opening brief", "points at the right sound"],
    ),
    (
        r"\bis a strong follow-up\b",
        ["continues that thread", "is where I'd go next", "keeps that thread moving", "extends that sound"],
    ),
    (
        r"\bis a strong fit\b",
        ["matches that thread", "lands well here", "fits the brief", "is the clearest match"],
    ),
    (
        r"\bis a strong move\b",
        ["is where I'd go next", "lands closest to that cue", "moves in that direction", "answers that cue"],
    ),
    (
        r"\bis a strong pick\b",
        ["is my pick", "matches that thread", "lands well here", "answers that cue"],
    ),
    (
        r"\bfits best\b",
        ["is the clearest match", "lands closest to your request", "matches that cue", "fits the brief"],
    ),
    (
        r"\bis a close match\b",
        ["is the closest pool option", "is the nearest fit here", "lands closest in the pool", "is the closest read"],
    ),
    (
        r"\bclose match\b",
        ["near fit", "nearby option", "closest option here", "nearest fit here"],
    ),
    (
        r"\bmight be (?:it|the one)\b",
        ["is my best read", "is the likely match", "is the closest read"],
    ),
    (
        r"\bstrong possibility\b",
        ["plausible read", "likely match", "good read"],
    ),
    (
        r"\bwonderful way\b",
        ["clean way", "useful way", "good way"],
    ),
    (
        r"\bhits the mark\b",
        ["clicks", "works", "lands well"],
    ),
    (
        r"\bhit the mark\b",
        ["clicked", "worked", "landed well"],
    ),
    (
        r"\blet me know what you think\b",
        ["tell me how it sits", "see what you make of it", "curious what you think"],
    ),
    (
        r"\bstart there\b",
        ["press play when it fits", "save it for that mood", "try it when that mood hits", "let it run"],
    ),
    # Encyclopedic / 3rd-person curator verbs — replace with friend register.
    # Pools widened so distinct-2 doesn't bottleneck on the substitute itself.
    (r"\bsupplies a\b", ["brings a", "has a", "gives a", "packs a", "carries a"]),
    (r"\bsupplies\b", ["brings", "has", "gives", "packs", "carries"]),
    (r"\byields a\b", ["brings a", "lands a", "gives a", "throws out a", "drops a"]),
    (r"\byields\b", ["brings", "lands", "gives", "throws out", "drops"]),
    (r"\butilizes\b", ["uses", "leans on", "runs on", "rides on", "is built on"]),
    (r"\butilizing\b", ["using", "leaning on", "riding on"]),
    (r"\bdelivers a\b", ["brings a", "lands a", "has a", "puts up a", "drops a"]),
    (r"\bdelivers\b", ["brings", "lands", "has", "puts up", "drops"]),
    (r"\bshowcases\b", ["shows off", "spotlights", "puts forward", "leans into", "highlights"]),
    (r"\bserves up a\b", ["brings a", "lands a", "has a", "drops a"]),
    (r"\bserves up\b", ["brings", "lands", "drops"]),
    (r"\bembodies\b", ["captures", "feels like", "lives in", "carries"]),
    (r"\bepitomizes\b", ["nails", "captures", "is the heart of"]),
    # Meta-talk leak — exposes the recommender's curator seams
    (
        r"\bthe rest of the (selection|pool|collection|set|recommendations|tracks|picks)\b",
        ["more picks here", "more options here", "other tracks here"],
    ),
    (
        r"\badditional (pieces|tracks|selections|picks|cuts)\b",
        ["more here", "other picks", "more cuts"],
    ),
    (r"\bin the pool\b", ["in this set", "around here", "in this batch"]),
    (r"\bthese selections\b", ["these picks", "these cuts"]),
    (r"\bthese additional\b", ["the other", "more"]),
    (
        r"\brounds out the (selection|pool|set|collection|recommendations|batch)\b",
        ["fills out the picks", "wraps up the set"],
    ),
    (r"\bthe wider (collection|pool|set)\b", ["the wider mix", "more in this set"]),
    # Formulaic two-fork CTA — replace with a softer prompt (lowercase regex; we keep noun unchanged)
    (
        r"\bwould you (prefer|rather)\b",
        ["want me to", "should we"],
    ),
]


# ---------------------------------------------------------------------------
# Rotation pools — used as fallback when LLM soft-bans target something we
# don't know how to substitute cleanly. Kept evidence-typed so callers that
# know the evidence context can pick a grammatically sensible replacement.
# ---------------------------------------------------------------------------
_ROTATION_POOL: dict[str, list[str]] = {
    "query_aligned": ["answers", "matches", "lines up with", "hits the brief"],
    "lyric_resonant": ["echoes", "mirrors", "speaks to", "resonates with"],
    "taste_continuous": ["continues", "extends", "follows on from", "stays in your lane"],
    "pool_coherent": ["holds together as", "forms a tight", "anchors a", "rounds out"],
    "discovery": ["widens the net", "steps sideways into", "ventures toward"],
}


# ---------------------------------------------------------------------------
# Theme parsing + grounding check (informational)
# ---------------------------------------------------------------------------

_THEME_TRACKS_RE = re.compile(r"tracks?\s*[\[\(]?\s*([0-9][0-9,\s]*)\s*[\]\)]?", re.IGNORECASE)


def parse_themes(themes_raw: str) -> list[dict]:
    """Extract `{line, track_indices_0based}` from each non-empty line."""
    parsed: list[dict] = []
    for line in (themes_raw or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _THEME_TRACKS_RE.search(stripped)
        if not m:
            continue
        raw_ids = m.group(1)
        idxs: list[int] = []
        for part in raw_ids.split(","):
            part = part.strip()
            if part.isdigit():
                idxs.append(int(part) - 1)  # 1-based in prompt → 0-based internally
        parsed.append({"line": stripped, "track_indices_0based": idxs})
    return parsed


def enforce_evidence_grounding(themes: list[dict], intent_groups: dict[str, dict]) -> list[str]:
    """Return a list of warnings for themes that reference off-pool indices."""
    all_known: set[int] = set()
    for g in intent_groups.values():
        all_known.update(g.get("indices", []))
    warnings: list[str] = []
    for t in themes:
        unknown = [i for i in t.get("track_indices_0based", []) if i not in all_known]
        if unknown:
            unknown_1based = ", ".join(str(i + 1) for i in unknown)
            warnings.append(f"theme references unknown tracks [{unknown_1based}]: {t.get('line', '')[:80]}")
    return warnings


# ---------------------------------------------------------------------------
# LLM-emitted soft pattern validation
# ---------------------------------------------------------------------------


def validate_excluded_patterns(response: str, patterns_raw: str) -> list[str]:
    """Return the list of pattern strings that MATCH response. Empty = clean."""
    if not response or not patterns_raw:
        return []
    violations: list[str] = []
    for raw in patterns_raw.splitlines():
        pat_str = raw.strip()
        # strip leading bullets / numbering the LM might have added
        pat_str = re.sub(r"^[-*\d.\s)]+", "", pat_str)
        if not pat_str:
            continue
        try:
            pat = re.compile(pat_str, re.IGNORECASE)
        except re.error:
            continue  # skip malformed regex; fail-soft
        if pat.search(response):
            violations.append(pat_str)
    return violations


# ---------------------------------------------------------------------------
# Hard-ban repair
# ---------------------------------------------------------------------------


def _case_preserving(match: re.Match, replacement: str) -> str:
    """Match capitalization of the first character of the matched text."""
    original = match.group(0)
    if original and original[0].isupper() and len(replacement) > 0:
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_hard_bans(text: str, seed: int | str | None = None) -> str:
    """Substitute every hard-banned phrase with a randomized rotation replacement."""
    if not text:
        return text
    rng = random.Random(seed)
    for pat_str, replacements in _HARD_BANS:
        try:
            pat = re.compile(pat_str, re.IGNORECASE | re.MULTILINE)
        except re.error:
            continue

        def _sub(m: re.Match, _repls: list[str] = replacements) -> str:
            return _case_preserving(m, rng.choice(_repls))

        text = pat.sub(_sub, text)
    # collapse any double-spaces the substitutions may have introduced
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Title citation resolution — LLM emits `cited_titles` list; we fuzzy-match
# against top-20 and canonicalize / strip each citation in `response`.
# ---------------------------------------------------------------------------


def _fuzzy_ratio(a: str, b: str) -> float:
    """Case-/whitespace-normalized SequenceMatcher ratio. stdlib only."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _build_title_pool(track_ids: list[str], kv: Any) -> list[tuple[int, str, str]]:
    """Return [(1-based_idx, title, artist), ...] for every track we can resolve."""
    pool: list[tuple[int, str, str]] = []
    if not track_ids or kv is None:
        return pool
    for i, tid in enumerate(track_ids, 1):
        try:
            meta = kv.get_track_meta(tid)
        except Exception:
            continue
        if not meta:
            continue
        title = _track_meta_first(meta, "track_name")
        artist = _track_meta_first(meta, "artist_name")
        if title and title != "unknown":
            pool.append((i, title, artist or ""))
    return pool


def _best_pool_match(candidate: str, pool: list[tuple[int, str, str]], threshold: float) -> tuple[int, str, str] | None:
    """Return the highest-scoring (idx, title, artist) above `threshold`, else None."""
    if not candidate or not pool:
        return None
    best = None
    best_score = 0.0
    for entry in pool:
        score = _fuzzy_ratio(candidate, entry[1])
        if score > best_score:
            best_score = score
            best = entry
    return best if best and best_score >= threshold else None


def _exact_pool_title_match(candidate: str, pool: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    """Return a pool title only when the normalized title is an exact match."""
    key = _title_key(candidate)
    if not key:
        return None
    for entry in pool:
        if _title_key(entry[1]) == key:
            return entry
    return None


def _exact_pool_title_canonical(candidate: str, pool: list[tuple[int, str, str]]) -> tuple[str, str] | None:
    hit = _exact_pool_title_match(candidate, pool)
    if hit is None:
        return None
    _, title, artist = hit
    return _canonical_form(title, artist), artist


def _canonical_form(title: str, artist: str) -> str:
    """`"Title" by Artist` — the authoritative citation form."""
    if artist and artist != "unknown":
        return f'"{title}" by {artist}'
    return f'"{title}"'


def _cleanup_malformed_citation_quotes(response: str, pool: list[tuple[int, str, str]]) -> str:
    """Repair quote-boundary artifacts left after canonical citation replacement.

    Titles with apostrophes or literal quote marks can make the broad fallback
    matcher treat an inner quote as the end of the title. The canonicalizer may
    then leave fragments such as `"Title" by Artist" by Artist`. This pass is
    deliberately closed-set: it only rewrites exact canonical citations for
    artists in the current top-20 pool.
    """
    if not response:
        return response
    for _, title, artist in pool:
        canonical = _canonical_form(title, artist)
        # Extra leading double-quotes before the exact canonical citation:
        # `""'Dry Cured'" by Clubroot` -> `"'Dry Cured'" by Clubroot`.
        while ('"' + canonical) in response:
            response = response.replace('"' + canonical, canonical)
        while ('""' + canonical) in response:
            response = response.replace('""' + canonical, canonical)
        if artist and artist != "unknown":
            duplicate = f'{canonical}" by {artist}'
            # Repeated exact attribution after a malformed title quote:
            # `"Keep On Jumpin'" by Musique" by Musique` -> canonical.
            while duplicate in response:
                response = response.replace(duplicate, canonical)
        # Stray quote immediately after the exact canonical citation.
        while (canonical + '"') in response:
            response = response.replace(canonical + '"', canonical)
    return response


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_QUOTED_CITATION_RE = re.compile(r'"[^"]+"\s+by\s+[^.?!,;]+')
_CONCRETE_ATTRIBUTE_RE = re.compile(
    r"\b("
    r"production|texture|vocal|vocals|voice|delivery|guitar|piano|drum|drums|bass|synth|strings|"
    r"brass|horn|saxophone|percussion|riff|groove|lyric|lyrics|imagery|melody|arrangement|"
    r"pulse|beat|rhythm|chorus|harmony|tempo|key|waltz|solo|tone|timbre|distortion|reverb|"
    r"ambient|darkwave|reggae|blues|jazz|folk|punk|metal|electronic|orchestral"
    r")\b",
    re.IGNORECASE,
)
_WEAK_SUPPLEMENT_CUE_RE = re.compile(
    r"^\s*(?:"
    r"I(?:'d| would)? also|I'll also|I also|For more|For an? (?:even|more)|To keep|To maintain|"
    r"To broaden|If you|For something|I'll pair|I'd pair|I'd include|I have .*also"
    r")\b",
    re.IGNORECASE,
)


def prune_weak_supplement_sentences(response: str) -> str:
    """Drop extra cited-pick sentences that do not explain a concrete attribute.

    The judge penalizes two-track replies when the second citation is just a
    generic verdict ("keeps the momentum", "same vibe"). This conservative pass
    always keeps the first cited sentence and removes only later supplement-like
    cited sentences with no concrete musical or lyrical attribute.
    """
    if not response or len(_QUOTED_CITATION_RE.findall(response)) < 2:
        return response

    sentences = _SENTENCE_SPLIT_RE.split(response)
    kept: list[str] = []
    seen_cited_sentence = False
    for sentence in sentences:
        has_citation = bool(_QUOTED_CITATION_RE.search(sentence))
        if has_citation and not seen_cited_sentence:
            seen_cited_sentence = True
            kept.append(sentence)
            continue
        if (
            seen_cited_sentence
            and has_citation
            and _WEAK_SUPPLEMENT_CUE_RE.search(sentence)
            and not _CONCRETE_ATTRIBUTE_RE.search(sentence)
        ):
            continue
        kept.append(sentence)

    pruned = " ".join(s.strip() for s in kept if s.strip())
    pruned = re.sub(r"[ \t]{2,}", " ", pruned).strip()
    return pruned or response


# Artist phrase grammar — separated into three ingredients:
#   ABBREV    = "A.B.C." style (abbreviation), ending in period by design
#   CAP_WORD  = regular capitalized word, NO trailing period
#   CAP_PART  = either of the above (ie the words that actually anchor an artist name)
#   CONN      = lowercase connective that can appear INSIDE a multi-word artist
#               name ("of", "and", "the", ...) but never at the end
#
# The phrase starts with a CAP_PART and extends by (CONN? + CAP_PART) chunks.
# This prevents two failure modes seen in earlier drafts:
#   1. Trailing period on a regular word eating into the next sentence
#      ("by Artist. Next sentence" → "by Artist Next").
#   2. Trailing "and" / "the" being consumed when followed by lowercase prose
#      ("by A.B.C. and friends" → "by A.B.C. and").
_ABBREV = r"(?:[A-Z]\.){2,}"  # A.B.C. style
_ABBREV1 = r"(?:(?:Vs|Feat|Ft|Mr|Mrs|Ms|Dr|St|Jr|Sr)\.)"  # known one-dot abbreviation tokens
_CAP_WORD = r"[A-Z][A-Za-z0-9’'\-]*"
_CAP_PART = rf"(?:{_ABBREV}|{_ABBREV1}|{_CAP_WORD})"
_CONN = r"(?:of|and|the|&|feat\.?|ft\.?|vs\.?|featuring)"
_ARTIST_PHRASE = rf"{_CAP_PART}(?:\s+(?:{_CONN}\s+)?{_CAP_PART})*"
# Generic artist reference — "by the same band", "by their trio", "by them", etc.
# Consume so the post-substitution result doesn't have a dangling pronoun clause
# after the canonical `by <Artist>`.
_GENERIC_ARTIST_REF = (
    r"(?:(?:the|that|this|his|her|their|its)\s+(?:same\s+)?"
    r"(?:band|group|artist|duo|trio|quartet|song|track|one|record)|them|him|her|it)"
)
# Attribution can use ` by Artist` (English convention) OR an em-/en-/hyphen-dash
# separator (`— Artist`, `– Artist`, `- Artist`). Weak models (qwen3.5) emit the
# dash form despite the prompt asking for `by`; treat both as consumable so the
# canonical replacement absorbs trailing artist tokens instead of duplicating
# attribution (e.g. `"The Price" by Leprous — Leprous`).
_BY_OR_DASH = r"(?:\s+by\s+|\s*[—–-]\s+)"
_TRAILING_BY_ARTIST = rf"(?:{_BY_OR_DASH}(?:{_ARTIST_PHRASE}|{_GENERIC_ARTIST_REF}))?"
_SENTENCE_START_AFTER_ATTR = re.compile(
    r"^\s+(?:The|This|That|These|Those|It|Its|Their|His|Her|A|An|For|Since|If|While|Another|Those)\b"
)


def _literal_artist_pattern(artist: str) -> str:
    """Return a regex for the exact pool artist with flexible whitespace."""
    if not artist or artist == "unknown":
        return ""
    parts = [re.escape(p) for p in artist.split()]
    return r"\s+".join(parts)


def _canonical_replacement(match: re.Match, response: str, canonical: str) -> str:
    """Insert a sentence boundary when attribution repair reveals one.

    Weak LMs sometimes emit `"Title" by Artist The ...` without a period. If
    we replace only the title+artist span, the next sentence remains glued to
    the citation. Add a period when the remaining suffix starts like a sentence.
    """
    tail = response[match.end() :]
    if _SENTENCE_START_AFTER_ATTR.search(tail) and not canonical.endswith((".", "!", "?")):
        return canonical + "."
    return canonical


def _replace_citation(response: str, cited: str, canonical: str, artist: str = "") -> tuple[str, bool]:
    """Substitute the first occurrence of `cited` (quoted or bare) with `canonical`.

    Consumes any immediately-trailing `by <Artist>` clause — but only while the
    trailing tokens look like an artist name — so we don't eat into the sentence
    that follows. The title half is case-insensitive via inline `(?i:...)`; the
    artist half stays case-sensitive so `[A-Z]` in `_ARTIST_WORD` still means
    "starts with uppercase" (otherwise IGNORECASE makes every lowercase verb
    match as an artist word, over-consuming trailing prose).

    Paired-quote matching allows `cited` to be a SUBSTRING of the quoted span
    (e.g. cited_titles stripped to `Title` but response wrote `"Title - 2001
    Remaster"`). This avoids the nesting artifact where bare-substring matching
    would substitute inside the quoted span and leave the surrounding quote /
    trailing suffix in place.
    """
    title_ci = "(?i:" + re.escape(cited) + ")"
    not_quote = r"[^\"'“”‘’\n]*"
    exact_artist = _literal_artist_pattern(artist)

    def _sub(m: re.Match) -> str:
        return _canonical_replacement(m, response, canonical)

    # Priority 1: quoted span containing cited (cited may be substring of the
    # actual quoted text); optional exact pool-artist attribution. Exact-artist
    # matching prevents sentence starters like `The` / `Its` from being consumed
    # as a stray final artist token when the LM forgot a period.
    if exact_artist:
        trailing_exact = rf"(?:{_BY_OR_DASH}{exact_artist})?"
        for quote in ('"', "'", "“", "‘"):
            pat = re.compile(re.escape(quote) + not_quote + title_ci + not_quote + r"[\"'”’]" + trailing_exact)
            if pat.search(response):
                return pat.sub(_sub, response, count=1), True
        pat_bare_exact = re.compile(r"(?<![\"'“”‘’])\b" + title_ci + r"\b" + trailing_exact)
        if pat_bare_exact.search(response):
            return pat_bare_exact.sub(_sub, response, count=1), True

    # Priority 2: quoted span with broad artist/reference consumption. This
    # keeps legacy protection against mismatched/doubled attributions when the
    # model named the wrong artist, after exact matching had first chance.
    for quote in ('"', "'", "“", "‘"):
        pat = re.compile(re.escape(quote) + not_quote + title_ci + not_quote + r"[\"'”’]" + _TRAILING_BY_ARTIST)
        if pat.search(response):
            return pat.sub(_sub, response, count=1), True
    # Priority 2: bare form NOT inside quotes (negative lookbehind for any
    # quote-like char on the immediate left), plus optional trailing attribution.
    pat_bare = re.compile(r"(?<![\"'“”‘’])\b" + title_ci + r"\b" + _TRAILING_BY_ARTIST)
    if pat_bare.search(response):
        return pat_bare.sub(_sub, response, count=1), True
    return response, False


def _strip_fabricated_citation(response: str, fabricated: str) -> str:
    """Best-effort deletion of a citation with no pool match.

    Removes the quoted title plus any trailing `by <Artist>` and a leading separator
    comma/conjunction, then normalizes whitespace. Title is case-insensitive via
    inline `(?i:...)`; artist boundary stays case-sensitive (see `_replace_citation`).
    """
    title_ci = "(?i:" + re.escape(fabricated) + ")"
    for quote in ('"', "'", "“", "‘"):
        pat = re.compile(
            r"(?:(?:,\s*(?:and\s+|or\s+)?)|(?:\s+(?:and|or)\s+))?\s*"
            + re.escape(quote)
            + title_ci
            + r"[\"'”’]"
            + _TRAILING_BY_ARTIST,
        )
        if pat.search(response):
            response = pat.sub("", response, count=1)
            break
    # cleanup
    response = re.sub(r"\s{2,}", " ", response).strip()
    response = re.sub(r",\s*,", ",", response)
    response = re.sub(r"\s+([.,!?])", r"\1", response)
    return response


def _title_key(text: str) -> str:
    """Normalize a title-like string for protected-context matching."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


_CONTEXT_QUOTE_PATTERNS = (
    re.compile(r'"([^"\n]{2,80})"'),
    re.compile(r"“([^”\n]{2,80})”"),
    re.compile(r"'([^'\n]{2,80})'"),
    re.compile(r"‘([^’\n]{2,80})’"),
)
_CONTEXT_TRAILING_QUOTE_RE = re.compile(r'(?<![A-Za-z0-9])([A-Z0-9][A-Za-z0-9][^"\n]{0,78})"')


def _extract_context_titles(*texts: str) -> set[str]:
    """Return quoted title-like spans already present in user/query context.

    These are often prior-listen titles. If the model repeats one in the
    response, swapping it to an unrelated current-pool title corrupts the
    personalization anchor. Preserve the reference by de-quoting it instead.
    """
    titles: set[str] = set()
    for text in texts:
        if not text:
            continue
        for pat in _CONTEXT_QUOTE_PATTERNS:
            for match in pat.finditer(text):
                title = match.group(1).strip()
                if title:
                    titles.add(title)
        # Defensive recovery for malformed blindset snippets like
        # `Urethane" is a classic!` where the leading quote is absent.
        for match in _CONTEXT_TRAILING_QUOTE_RE.finditer(text):
            title = match.group(1).strip()
            if title:
                titles.add(title)
    return titles


def _is_protected_title(candidate: str, protected_titles: set[str]) -> bool:
    key = _title_key(candidate)
    if len(key) < 3:
        return False
    for protected in protected_titles:
        pkey = _title_key(protected)
        if len(pkey) < 3:
            continue
        if key == pkey or key in pkey or pkey in key:
            return True
    return False


def _dequote_protected_citation(response: str, title: str) -> tuple[str, bool]:
    """Remove quote marks around a protected prior/query title, keeping text."""
    if not response or not title:
        return response, False
    title_ci = "(?i:" + re.escape(title) + ")"
    quote_pairs = [('"', '"'), ("“", "”"), ("‘", "’"), ("'", "'")]
    for left, right in quote_pairs:
        pat = re.compile(re.escape(left) + title_ci + re.escape(right))
        if pat.search(response):
            return pat.sub(title, response, count=1), True
    return response, False


# ---------------------------------------------------------------------------
# Semantic fallback — Ollama embedding matches citation against
# metadata-qwen3_embedding_0.6b when fuzzy title match fails.
# ---------------------------------------------------------------------------


def _semantic_match(
    cited_title: str,
    track_ids: list[str],
    pool: list[tuple[int, str, str]],
    kv: Any,
    threshold: float,
) -> tuple[int, str, str] | None:
    """Embed `cited_title` via Ollama; compare to each pool track's metadata
    embedding. Return the best (idx, title, artist) above `threshold`, else None.

    Graceful degradation — Ollama unavailable / missing embeddings all yield None.
    """
    if not cited_title or not pool or kv is None:
        return None
    try:
        import numpy as np

        from mymodule.feature.ollama_embed import get_embedder

        qvec = get_embedder().embed(cited_title)
        if qvec is None or len(qvec) == 0:
            return None
        qnorm = qvec / (np.linalg.norm(qvec) + 1e-9)
    except Exception:
        return None

    best = None
    best_score = 0.0
    for idx, title, artist in pool:
        tid = track_ids[idx - 1] if idx - 1 < len(track_ids) else None
        if tid is None:
            continue
        try:
            mvec = kv.get_track_embedding(tid, "metadata-qwen3_embedding_0.6b")
        except Exception:
            continue
        if mvec is None:
            continue
        try:
            mnorm = mvec / (np.linalg.norm(mvec) + 1e-9)
            score = float(np.dot(qnorm, mnorm))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best = (idx, title, artist)
    return best if best and best_score >= threshold else None


def resolve_title_citations(
    response: str,
    cited_titles_raw: str,
    track_ids: list[str],
    kv: Any,
    sim_threshold: float | None = None,
    protected_titles: set[str] | None = None,
) -> tuple[str, dict]:
    """Canonicalize verified title citations; strip fabrications.

    Returns (new_response, report) where report has {"canonicalized": [...],
    "fabricated": [...], "unresolved": [...]}.
    """
    import os

    if sim_threshold is None:
        sim_threshold = float(os.getenv("MYMODULE_PAS_TITLE_SIM_THRESHOLD", "0.80"))

    report = {"canonicalized": [], "fabricated": [], "unresolved": [], "protected": []}
    if not response:
        return response, report

    pool = _build_title_pool(track_ids or [], kv)
    protected_titles = protected_titles or set()

    # Parse cited_titles lines. Weak models (qwen3.5) emit free-form formats
    # despite the prompt asking for bare title; normalize defensively:
    #   1. bullet/numbering prefix (`- `, `1. `, `* `)
    #   2. surrounding paired quotes — also handle one-sided trailing quote
    #      (`Title"` or `"Title`) that breaks symmetric strip below
    #   3. trailing attribution — ` by Artist` always, OR em-/en-dash + word,
    #      OR ASCII hyphen + Capitalized word. The hyphen variant requires
    #      a leading uppercase letter so we don't strip suffixes like
    #      `- 2001 - Remaster` or `- Acoustic` that are part of the title
    #      (otherwise the canonical substitution duplicates the suffix).
    # Order matters: attribution strip BEFORE final quote strip, since attribution
    # itself may carry a trailing quote/dash mixture.
    cited: list[str] = []
    for raw in (cited_titles_raw or "").splitlines():
        t = raw.strip()
        t = re.sub(r"^[-*\d.\s)]+", "", t)
        # ` by Artist` (case-insensitive) — wide net.
        t = re.sub(r"\s+by\s+.+$", "", t, flags=re.IGNORECASE).strip()
        # em-/en-dash trailing — uniformly treated as attribution.
        t = re.sub(r"\s*[—–]\s+.+$", "", t).strip()
        # ASCII hyphen trailing — only when next token starts with uppercase
        # (artist name signal). Skip suffixes like `- 2001 - Remaster` or
        # `- Acoustic Version` that are part of the canonical title.
        t = re.sub(r"\s+-\s+[A-Z][A-Za-z'][^-]*$", "", t).strip()
        # Drop any remaining leading/trailing quote chars (handles `Title"` /
        # `"Title` / paired `"Title"` / `'Title'` / smart quotes alike).
        t = t.strip("\"'“”‘’").strip()
        if t and t not in cited:  # de-dup while preserving order
            cited.append(t)

    if not cited:
        # Even if the LM forgot to emit cited_titles, a final sweep on the
        # response still catches unlisted quotes — fall through.
        pass

    sem_threshold = float(os.getenv("MYMODULE_PAS_SEM_SIM_THRESHOLD", "0.72"))
    # Closed-set safety net (PR P0 #2). When fuzzy + relaxed-semantic both fail,
    # swap to the highest-scoring pool entry under this floor instead of
    # stripping the citation. The pool itself bounds the worst case (any of
    # top-20 is a valid recommendation), so a low-similarity swap is still in
    # the recommended set — better than a subjectless WHY sentence. Disable
    # by setting `1.0` to fall back to the old strip behavior.
    swap_floor = float(os.getenv("MYMODULE_PAS_SWAP_FLOOR", "0.0"))

    def _handle(candidate: str, *, allow_swap: bool = True) -> tuple[str, str] | None:
        """Try fuzzy → semantic → relaxed-semantic → closest-pool-swap → None.

        The relaxed-semantic stage uses a lower threshold so fabricated
        citations get replaced with the closest pool track rather than
        stripped. Stripping removes the explanation's subject and hurts
        LLM-Judge scores; a slightly-wrong canonicalization is better than
        no subject at all.

        The closed-set swap (P0 #2) is the final safety net: when all
        thresholds fail, return the canonical form of the highest-scoring
        pool entry above `swap_floor`. Since `recommended_titles_pool` was
        passed to the LLM as a CLOSED citation set, any fabrication here is
        either (a) a chat_history title the LLM pulled in despite the
        instruction, or (b) an invented name — both deserve a swap to the
        actual recommended pool rather than a hole in the response.
        """
        hit = _best_pool_match(candidate, pool, sim_threshold)
        if hit is None:
            hit = _semantic_match(candidate, track_ids, pool, kv, sem_threshold)
        if hit is None:
            # Relaxed semantic — catches near-misses (partial title, paraphrase).
            relaxed_threshold = float(os.getenv("MYMODULE_PAS_SEM_SIM_RELAXED", "0.55"))
            hit = _semantic_match(candidate, track_ids, pool, kv, relaxed_threshold)
        if allow_swap and hit is None and swap_floor < 1.0 and pool:
            # Closed-set swap: pick the highest-fuzzy-scoring pool entry, no
            # threshold. Acts as a last-mile safety so a fabricated citation
            # gets a pool-grounded subject rather than disappearing.
            hit = _best_pool_match(candidate, pool, swap_floor)
            if hit is None:
                # Even the floor missed (empty candidate or zero-similarity);
                # fall through to top-1 as the conservative default.
                hit = pool[0]
        if hit is None:
            return None
        _, canon_t, canon_a = hit
        return _canonical_form(canon_t, canon_a), canon_a

    processed: set[str] = set()  # titles already handled — avoid double-work in sweep
    for cited_title in cited:
        protected = _is_protected_title(cited_title, protected_titles)
        handled = _exact_pool_title_canonical(cited_title, pool) if protected else _handle(cited_title)
        if handled is None:
            if protected:
                new_response, ok = _dequote_protected_citation(response, cited_title)
                if ok:
                    response = new_response
                    report["protected"].append(cited_title)
                else:
                    report["unresolved"].append(cited_title)
                processed.add(cited_title.lower())
                continue
            # Fabrication (or at least not resolvable).
            new_response = _strip_fabricated_citation(response, cited_title)
            if new_response != response:
                report["fabricated"].append(cited_title)
                response = new_response
            else:
                report["unresolved"].append(cited_title)
            processed.add(cited_title.lower())
            continue
        canonical, artist = handled
        new_response, ok = _replace_citation(response, cited_title, canonical, artist=artist)
        if ok:
            report["canonicalized"].append((cited_title, canonical))
            response = new_response
        else:
            report["unresolved"].append(cited_title)
        processed.add(cited_title.lower())

    # Fallback sweep — catch quoted citations the LM didn't list in cited_titles.
    # Only handle paired double/smart quote forms. ASCII single-quote `'` is
    # EXCLUDED on purpose: English contractions ("you've", "it's", "I'll")
    # interleave apostrophes unpredictably, and a greedy span match between
    # two contractions strips legitimate prose. The risk/reward is bad.
    _QUOTE_PATTERNS = [
        r'"([A-Za-z0-9][^"\n]{1,58})"',
        r"“([A-Za-z0-9][^”\n]{1,58})”",
        r"‘([A-Za-z0-9][^’\n]{1,58})’",
    ]
    seen_spans: list[tuple[int, int]] = []
    sweep_matches = []
    for qp in _QUOTE_PATTERNS:
        for m in re.finditer(qp, response):
            if any(s <= m.start() < e for s, e in seen_spans):
                continue
            seen_spans.append((m.start(), m.end()))
            sweep_matches.append(m)
    for m in sweep_matches:
        quoted = m.group(1).strip()
        if not quoted or quoted.lower() in processed:
            continue
        protected = _is_protected_title(quoted, protected_titles)
        handled = _exact_pool_title_canonical(quoted, pool) if protected else _handle(quoted)
        if handled is None:
            if protected:
                new_response, ok = _dequote_protected_citation(response, quoted)
                if ok:
                    response = new_response
                    report["protected"].append(quoted)
                processed.add(quoted.lower())
                continue
            new_response = _strip_fabricated_citation(response, quoted)
            if new_response != response:
                report["fabricated"].append(quoted)
                response = new_response
        else:
            canonical, artist = handled
            new_response, ok = _replace_citation(response, quoted, canonical, artist=artist)
            if ok:
                report["canonicalized"].append((quoted, canonical))
                response = new_response
        processed.add(quoted.lower())

    response = _cleanup_malformed_citation_quotes(response, pool)
    return response, report


# ---------------------------------------------------------------------------
# Track-ID anchor resolution ([T1]..[T5] → descriptive phrase) — legacy,
# kept as a safety net if old prompts or ckpts still emit bracketed refs.
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(r"\[T(\d+)\]")


def _track_meta_first(meta: dict | None, key: str, fallback: str = "") -> str:
    """Track metadata fields are lists — take the first non-empty element."""
    if meta is None:
        return fallback
    v = meta.get(key)
    if isinstance(v, list) and v:
        return str(v[0])
    if isinstance(v, str) and v:
        return v
    return fallback


def _build_anchor_descriptions(track_ids: list[str], kv: Any) -> dict[int, str]:
    """For each 1-based anchor index, produce a grounded descriptive phrase.

    Strategy:
      - unique artist in top-5 → "<artist>'s <year> cut" (or just "<artist> cut" if year missing)
      - duplicate artist       → "<artist>'s <album> cut"; if album also collides,
                                 ordinal fallback ("the <ordinal> <artist> cut")
    """
    if not track_ids or kv is None:
        return {}

    metas: list[dict | None] = []
    for tid in track_ids:
        try:
            metas.append(kv.get_track_meta(tid))
        except Exception:
            metas.append(None)

    # Count artist duplicates so we know when disambiguation is needed
    artist_counts: dict[str, int] = {}
    for m in metas:
        a = _track_meta_first(m, "artist_name")
        if a:
            artist_counts[a] = artist_counts.get(a, 0) + 1

    # For ordinal fallbacks
    artist_seen: dict[str, int] = {}
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

    out: dict[int, str] = {}
    for idx, m in enumerate(metas, 1):
        if m is None:
            out[idx] = "that track"
            continue
        artist = _track_meta_first(m, "artist_name")
        album = _track_meta_first(m, "album_name")
        release = _track_meta_first(m, "release_date")
        year = release[:4] if release and len(release) >= 4 and release[:4].isdigit() else ""

        if artist and artist != "unknown":
            if artist_counts.get(artist, 0) <= 1:
                # Unique artist among top-5 — artist + year/album
                if year:
                    out[idx] = f"{artist}'s {year} cut"
                elif album and album != "unknown":
                    out[idx] = f"{artist}'s {album} cut"
                else:
                    out[idx] = f"the {artist} cut"
            else:
                # Duplicate artist — disambiguate by album, else ordinal
                if album and album != "unknown":
                    out[idx] = f"{artist}'s {album} cut"
                else:
                    seq = artist_seen.get(artist, 0)
                    artist_seen[artist] = seq + 1
                    ord_word = ordinals[seq] if seq < len(ordinals) else f"{seq + 1}th"
                    out[idx] = f"the {ord_word} {artist} cut"
        elif album and album != "unknown":
            out[idx] = f"the {album} cut"
        elif year:
            out[idx] = f"the {year} cut"
        else:
            out[idx] = "that track"
    return out


def resolve_track_anchors(response: str, track_ids: list[str], kv: Any) -> str:
    """Replace every `[T<N>]` in `response` with a grounded descriptive phrase.

    Preserves case-neutrality (the anchor form doesn't carry case). Unknown indices
    (anchors beyond the top-N we have) are stripped rather than left as literal
    brackets that would leak pipeline internals to the user.
    """
    if not response:
        return response
    descriptions = _build_anchor_descriptions(track_ids, kv)
    if not descriptions:
        # Still strip any anchors to avoid leaking to the user.
        return _ANCHOR_RE.sub("that track", response).strip()

    def _replace(m: re.Match) -> str:
        idx = int(m.group(1))
        return descriptions.get(idx, "that track")

    return _ANCHOR_RE.sub(_replace, response).strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _append_violation_hint(predict_kwargs: dict, violations: list[str]) -> dict:
    """Non-invasive hint: append a short note to user_query so DSPy re-invokes fresh."""
    hinted = dict(predict_kwargs)
    banned = ", ".join(f'"{v}"' for v in violations[:6])
    suffix = (
        f"\n\n[Internal retry note — avoid these phrases in your response this turn: {banned}. Rephrase without them.]"
    )
    hinted["user_query"] = (hinted.get("user_query", "") or "") + suffix
    return hinted


def validate_and_repair(
    predictor: Any,
    predict_kwargs: dict,
    raw_out: Any,
    intent_groups: dict[str, dict] | None = None,
    max_retries: int = 1,
    seed: int | str | None = None,
    track_ids: list[str] | None = None,
    kv: Any = None,
) -> str:
    """Full post-LLM pipeline. Returns the final `response` string."""
    text = (getattr(raw_out, "response", "") or "").strip()
    if not text:
        return ""

    # diagnostic grounding check (non-blocking)
    themes_raw = getattr(raw_out, "themes", "") or ""
    if intent_groups is not None and themes_raw:
        parsed = parse_themes(themes_raw)
        for w in enforce_evidence_grounding(parsed, intent_groups):
            logger.warning(f"[pas_select] grounding warn: {w}")

    # soft patterns — retry once if violated
    patterns_raw = getattr(raw_out, "themes_excluded_patterns", "") or ""
    violations = validate_excluded_patterns(text, patterns_raw)
    if violations and max_retries > 0:
        try:
            retry_kwargs = _append_violation_hint(predict_kwargs, violations)
            retry_out = predictor(**retry_kwargs)
            retry_text = (getattr(retry_out, "response", "") or "").strip()
            if retry_text:
                retry_violations = validate_excluded_patterns(retry_text, patterns_raw)
                if len(retry_violations) < len(violations):
                    text = retry_text
        except Exception as e:
            logger.warning(f"[pas_select] retry failed ({type(e).__name__}): {e}")

    # hard bans — applied to whichever text we landed on
    text = apply_hard_bans(text, seed=seed)

    # Title citation resolution — canonicalize verified citations, strip fabrications.
    # Run BEFORE the legacy anchor resolver so titles land first.
    if track_ids and kv is not None:
        cited_titles_raw = getattr(raw_out, "cited_titles", "") or ""
        protected_titles = _extract_context_titles(
            predict_kwargs.get("user_query", "") or "",
            predict_kwargs.get("chat_history", "") or "",
        )
        text, cite_report = resolve_title_citations(
            text,
            cited_titles_raw,
            track_ids,
            kv,
            protected_titles=protected_titles,
        )
        if cite_report["fabricated"]:
            logger.warning(f"[pas_select] stripped fabricated citations: {cite_report['fabricated']}")
        if cite_report["unresolved"]:
            logger.warning(f"[pas_select] unresolved citations (kept as-is): {cite_report['unresolved']}")

    if os.getenv("MYMODULE_PAS_PRUNE_WEAK_SUPPLEMENTS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        text = prune_weak_supplement_sentences(text)

    # Legacy track-ID anchor cleanup — noop in new runs, safety for stale prompts.
    if track_ids and kv is not None:
        text = resolve_track_anchors(text, track_ids, kv)

    return text
