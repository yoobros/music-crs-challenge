"""PAS (Propose-Assign-Select) response generator.

Pipeline:
    retrieval pools → fusion → rerank → top-k track_ids → **PasResponseGenerator**

For each (session, turn):
    1. Deterministic PROPOSE (`propose.classify_intents`) — evidence-typed
       partition of the top-20 into 1-5 intent groups.
    2. DSPy `Predict(CRSResponse)` — the LLM performs ASSIGN + SELECT silently
       in one pass, emitting `themes`, `themes_excluded_patterns`,
       `cited_titles`, `response`.
    3. Deterministic post-processor (`select.validate_and_repair`) — grounds
       themes, applies soft/hard ban repair (with one retry), canonicalizes
       verified title citations, strips fabrications.

LM provider selection (`--response-provider {ollama, openai}`) is handled by
the parent `DspyResponseGenerator`. Compiled few-shot weights (if present at
`ckpt/pas_{provider}.json`) are loaded automatically. If a parallel
`ckpt/pas_{provider}.buckets.json` sidecar exists (written by `optimize.py`),
the `DemoRouter` is activated and replaces the static demo set with a per-call
selection keyed by `(specificity, turn_position)` of the current session.

Conditional few-shot retrieval (DemoRouter)
-------------------------------------------
When `pas_{provider}.buckets.json` is found next to the standard ckpt, demos
are *no longer static*: each call retrieves up to N demos matching the current
turn's `(specificity, turn_position)` (with hierarchical fallback through
specificity-only → turn-only → global top-N). Number of demos per call
controlled by `MYMODULE_PAS_ROUTER_TOP_N` (default 1 — a single most-relevant
demo; fewer demos preserve personalization/explanation quality. Raise if
under-conditioning).

Selective chat-expand (default mode for PAS)
--------------------------------------------
Prior music turns in `chat_history` are rendered as `Title by Artist (Album,
Year)` when the **current user query** is semantically close to that prior
track's metadata — specifically `cosine(query_qwen3, prior_track_metadata_rich) ≥
MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD` (default 0.5). Other prior tracks
are masked as `music: [omitted]` (signature InputField docstring tells the LLM
to ignore those). This is **query-aware** — answers "which prior listens are
relevant to what the user is asking right now". Avoids chat-tag token soup
while keeping a clean prompt. Override via env
`MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT` (e.g. `off` for legacy UUID-only).
"""

from __future__ import annotations

import json as _json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from mymodule.feature.user_profile import (
    UserProfile,
    get_user_profile,
    select_prior_pairs,
)
from mymodule.strategies.response.base_dspy import (
    DspyResponseGenerator,
    fmt_recommended_titles_pool,
    fmt_recommended_tracks,
    fmt_tracks_overview,
    strip_dspy_markers,
    try_open_kvstore,
)
from mymodule.strategies.response.pas.compile_router import (
    DemoRouter,
    turn_number_from_chat_history,
    turn_position_from_index,
)
from mymodule.strategies.response.pas.helpers import (
    fmt_lyric_similarity_hints,
    fmt_query_similarity_hints,
    fmt_track_similarity_hints,
)
from mymodule.strategies.response.pas.propose import (
    classify_intents,
    format_intent_groups_for_prompt,
)
from mymodule.strategies.response.pas.select import validate_and_repair
from mymodule.strategies.response.pas.signature import CompactCRSResponse, CRSResponse
from mymodule.utils.common_dspy import (
    Provider,
    RateLimitedPredictor,
    fmt_chat_history,
    fmt_conversation_goal,
    fmt_user_profile,
    min_interval_sec,
)

# ---------------------------------------------------------------------------
# Debug-dump hook (env-gated; production default OFF).
#
# When MYMODULE_PAS_DEBUG_DUMP=<path.jsonl> is set, each generate() call appends
# one record capturing the LLM's raw OutputFields (themes / personalization_anchors /
# track_roles / cited_titles / response_raw) BEFORE validate_and_repair plus the
# final shipped text. Used for failure-mode analysis (compare what the model
# said vs what we emitted). Joined post-hoc against the HF devset via user_query
# + chat_history length. Best-effort; never raises into the generator path.
# ---------------------------------------------------------------------------
_DEBUG_DUMP_LOCK = threading.Lock()


def _debug_dump_path() -> str:
    return os.getenv("MYMODULE_PAS_DEBUG_DUMP", "").strip()


def _compact_signature_enabled() -> bool:
    raw = os.getenv("MYMODULE_PAS_COMPACT_SIGNATURE", "").strip().lower()
    return raw in ("1", "true", "yes", "on", "compact")


def _maybe_debug_dump(
    predict_kwargs: dict,
    raw_out: Any,
    final_text: str,
    track_ids: list[str],
) -> None:
    path = _debug_dump_path()
    if not path:
        return
    try:
        chat = predict_kwargs.get("chat_history", "") or ""
        rec = {
            "user_query": predict_kwargs.get("user_query", "") or "",
            "listener_goal": predict_kwargs.get("listener_goal", "") or "",
            "chat_history_len_chars": len(chat),
            "chat_history_preview": chat[:300],
            "top3_track_ids": list(track_ids[:3]),
            "candidate_track_ids": list(track_ids[:20]),
            "raw": {
                "personalization_anchors": getattr(raw_out, "personalization_anchors", "") or "",
                "track_roles": getattr(raw_out, "track_roles", "") or "",
                "themes": getattr(raw_out, "themes", "") or "",
                "themes_excluded_patterns": getattr(raw_out, "themes_excluded_patterns", "") or "",
                "cited_titles": getattr(raw_out, "cited_titles", "") or "",
                "response_raw": getattr(raw_out, "response", "") or "",
            },
            "final_response": final_text or "",
        }
        with _DEBUG_DUMP_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[pas-debug] dump failed ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Process-wide lexical-diversity governor.
#
# `_RESPONSE_WORD_FREQ` accumulates a Counter of content-word occurrences
# across all responses generated by this process (one `run_inference_blindset`
# / `run_inference_devset` invocation). Before each new generation call, the
# top-N most frequent content words (above a min-count floor) are passed to
# the LLM via the `overused_words` InputField on `CRSResponse`; the LLM is
# instructed to avoid them. After generation, the produced response text is
# tokenized and the counter is updated under `_LOCK`. Word matching is
# case-insensitive; the suppression list excludes stopwords and
# track-grounding tokens (title / artist / album fragments).
#
# Tunables via env:
#   MYMODULE_LEXDIV_SUPPRESS_TOPN       — # over-used words shown (default 15;
#                                          set to 0 to disable the mechanism)
#   MYMODULE_LEXDIV_SUPPRESS_MIN_COUNT  — min occurrences to qualify (default 5)
#   MYMODULE_LLM_RECOMMENDED_WITH_CRAWL — toggle crawl enrichment in the
#                                          candidate detail block (default 1)
# ---------------------------------------------------------------------------
_RESPONSE_WORD_FREQ: Counter[str] = Counter()
_RESPONSE_WORD_FREQ_LOCK = threading.Lock()
_RESPONSE_BIGRAM_FREQ: Counter[tuple[str, str]] = Counter()
_RESPONSE_BIGRAM_FREQ_LOCK = threading.Lock()
_WORD_TOKEN_RE = re.compile(r"[A-Za-z]{4,}")  # ≥4 letters skips most stopwords already
_BIGRAM_TOKEN_RE = re.compile(r"[A-Za-z']+")  # broader: bigrams keep stopwords intact
_LEXDIV_STOPWORDS: frozenset[str] = frozenset(
    {
        # Function-word fillers that slip past the 4-letter floor.
        "with",
        "this",
        "that",
        "these",
        "those",
        "have",
        "your",
        "their",
        "they",
        "them",
        "from",
        "into",
        "than",
        "then",
        "when",
        "what",
        "where",
        "while",
        "which",
        "would",
        "could",
        "should",
        "about",
        "after",
        "also",
        "been",
        "being",
        "here",
        "just",
        "like",
        "more",
        "much",
        "only",
        "over",
        "some",
        "such",
        "very",
        "well",
        "will",
        # Music-grounding tokens we WANT the LLM to keep using even if frequent.
        # Suppressing these would push the LLM toward pronouns and hurt grounding.
        "song",
        "songs",
        "track",
        "tracks",
        "album",
        "albums",
        "artist",
        "artists",
        "music",
        "musical",
        "sound",
        "sounds",
        "title",
    }
)


def _update_word_freq(text: str) -> None:
    """Tokenize `text` and bump module-level word counter (thread-safe).

    No-op for empty input. Counts each unique token once per response (set
    semantics) so a single long response doesn't dominate the budget.
    """
    if not text:
        return
    tokens = {m.lower() for m in _WORD_TOKEN_RE.findall(text)}
    tokens -= _LEXDIV_STOPWORDS
    if not tokens:
        return
    with _RESPONSE_WORD_FREQ_LOCK:
        _RESPONSE_WORD_FREQ.update(tokens)


def _get_overused_words(top_n: int, min_count: int) -> list[str]:
    """Return up to `top_n` words seen ≥ `min_count` times across responses."""
    if top_n <= 0:
        return []
    with _RESPONSE_WORD_FREQ_LOCK:
        ranked = _RESPONSE_WORD_FREQ.most_common(top_n * 2)
    return [w for w, c in ranked if c >= min_count][:top_n]


def _update_bigram_freq(text: str) -> None:
    """Tokenize `text` and bump module-level bigram counter (thread-safe).

    Bigrams are content-bearing adjacent word pairs derived after dropping
    stopwords / track-grounding tokens (same exclusion set as word_freq).
    Set semantics per response so one long response doesn't dominate the
    distinct-2 budget. Direct suppression target for Distinct-2 (response-side
    metric measured at bigram level).
    """
    if not text:
        return
    raw_tokens = [m.lower() for m in _BIGRAM_TOKEN_RE.findall(text)]
    # Apply same content filter as word_freq so we don't tax LLM on
    # function-word fillers and music-grounding nouns the response NEEDS
    # to reuse (artist/title/album/song/track).
    tokens = [t for t in raw_tokens if len(t) >= 3 and t not in _LEXDIV_STOPWORDS]
    if len(tokens) < 2:
        return
    bigrams = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
    if not bigrams:
        return
    with _RESPONSE_BIGRAM_FREQ_LOCK:
        _RESPONSE_BIGRAM_FREQ.update(bigrams)


def _get_overused_bigrams(top_n: int, min_count: int) -> list[str]:
    """Return up to `top_n` bigrams seen ≥ `min_count` times across responses.

    Each entry rendered as `"word1 word2"` so the LLM can match it visually
    when scanning its draft (regex-aware downstream is overkill — content
    avoidance is enough).
    """
    if top_n <= 0:
        return []
    with _RESPONSE_BIGRAM_FREQ_LOCK:
        ranked = _RESPONSE_BIGRAM_FREQ.most_common(top_n * 2)
    return [f"{a} {b}" for (a, b), c in ranked if c >= min_count][:top_n]


_LEXDIV_SEED_BIGRAMS: tuple[str, ...] = (
    # Seeded from smoke-run evidence: these generic opener/closer
    # collocations repeat before the process-wide counter warms up under
    # parallel generation. Keep artist/title bigrams out of this list.
    "what you",
    "see what",
    "glad that",
    "keep that",
    "i'm going",
    "going with",
    "right on",
    "give it",
    "a spin",
    "a must",
    "up next",
    "natural next",
    "next step",
    "way to",
    "to go",
    "fantastic choice",
    "wonderful choice",
    "natural choice",
    "great comparison",
    "killer choice",
    "start there",
)


def _lexdiv_seed_bigrams_from_env() -> list[str]:
    raw = os.getenv("MYMODULE_LEXDIV_SEED_BIGRAMS")
    if raw is None:
        return list(_LEXDIV_SEED_BIGRAMS)
    raw = raw.strip()
    if raw.lower() in ("", "0", "false", "off"):
        return []
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _format_overused_bigrams_for_prompt(dynamic_bigrams: list[str]) -> str:
    """Combine dynamic batch bigrams with smoke-seeded generic collocations.

    Dynamic entries stay first because they reflect the current process. Seeded
    entries cover the first parallel batch, where the counter is still empty.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in [*dynamic_bigrams, *_lexdiv_seed_bigrams_from_env()]:
        p = (phrase or "").strip().lower()
        if not p or p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    return ", ".join(ordered)


def _bucket_sidecar_path(ckpt_path: Path) -> Path:
    """Sibling sidecar path: `pas_{provider}.json` → `pas_{provider}.buckets.json`."""
    return ckpt_path.with_suffix(".buckets.json")


# ---- user_profile enrichment env helpers (single source of truth) -----------
# Runtime (`PasResponseGenerator.__init__`) AND compile (`optimize._build_examples`)
# read the same three knobs so demo / inference distribution stays aligned.
def _user_history_enabled_from_env() -> bool:
    return os.getenv("MYMODULE_PAS_USER_HISTORY_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "",
    )


def _user_history_top_k_from_env() -> int:
    return int(os.getenv("MYMODULE_PAS_USER_HISTORY_TOP_K", "5"))


def _user_history_sim_threshold_from_env() -> float:
    return float(os.getenv("MYMODULE_PAS_USER_HISTORY_SIM_THRESHOLD", "0.45"))


def _render_prior_track(tid: str, kv: Any) -> str:
    """Render a prior-listen track as `"Title" by Artist [tag1, tag2, tag3]`.

    Title + artist anchor the citation; the top-3 tag list is a compact
    style-preference signal so the LLM can infer "this user has gravitated
    toward <genre / mood>" without needing to look up extra metadata. Tags
    are omitted when the KV record has none.
    """
    if not tid:
        return ""
    meta = None
    try:
        meta = kv.get_track_meta(tid) if kv is not None else None
    except Exception:
        meta = None
    if not meta:
        return f"track {tid[:8]}…"
    title_raw = meta.get("track_name")
    artist_raw = meta.get("artist_name")
    title = title_raw[0] if isinstance(title_raw, list) and title_raw else (title_raw or "")
    artist = artist_raw[0] if isinstance(artist_raw, list) and artist_raw else (artist_raw or "")
    if not title or title == "unknown":
        return f"track {tid[:8]}…"
    base = f'"{title}" by {artist}' if artist and artist != "unknown" else f'"{title}"'
    tag_list = meta.get("tag_list") or []
    if tag_list:
        tag_str = ", ".join(str(t) for t in tag_list[:3])
        return f"{base} [{tag_str}]"
    return base


def _format_user_profile_with_history(
    profile_dict: dict | None,
    user_id: str | None,
    kv: Any,
    user_query: str,
    *,
    enabled: bool,
    top_k: int,
    sim_threshold: float,
    filter_before_date: str | None = None,
    exclude_session_id: str | None = None,
) -> str:
    """`fmt_user_profile(profile_dict)` + query-relevant prior-listens block.

    Behaviour:
      • `enabled=False` / no user_id / KV unavailable / no UserProfile in KV
        → returns `fmt_user_profile(profile_dict)` unchanged
      • cosine-threshold filter via shared instructed embedder, top-K most
        recent passing pairs; falls back to recency-only on embedder failure
      • each prior pair rendered as `- "<utterance preview>" → "<Title>" by <Artist>`

    Cost: the cosine filter calls `embedder.embed(...)` once per prior_pair
    so it scales linearly with the user's training history. For warm users
    this is bounded (avg ~10-30 prior turns).

    Leakage filters (train-side use — compile / fewshot):
      • `filter_before_date` (ISO `YYYY-MM-DD`): drop sessions whose
        `session_date >= target_date` (strict `<`, `exclude_same_day=True`).
        Day-level cutoff matches the dataset's finest temporal granularity.
      • `exclude_session_id`: defense-in-depth — drop the named session
        even if its `session_date` survives the date cutoff.
      Both default to `None` so runtime callers see no behavior change.
    """
    base = fmt_user_profile(profile_dict)
    if not enabled or not user_id or kv is None:
        return base

    profile: UserProfile | None = None
    try:
        profile = get_user_profile(kv, user_id)
    except Exception as e:
        logger.warning(f"[pas] get_user_profile({user_id!r}) failed ({type(e).__name__}: {e}); skipping history.")
        profile = None
    if profile is None or not profile.has_history:
        return base

    # Leakage filters — train-side cutoff before any history is exposed to the LLM.
    if filter_before_date or exclude_session_id:
        try:
            from mymodule.feature.user_profile import (
                filter_before_date as _filter_before_date_fn,
            )
            from mymodule.feature.user_profile import (
                filter_excluding_session as _filter_excluding_session_fn,
            )

            if filter_before_date:
                profile = _filter_before_date_fn(profile, filter_before_date, exclude_same_day=True)
            if exclude_session_id:
                profile = _filter_excluding_session_fn(profile, exclude_session_id)
        except Exception as e:
            logger.warning(
                f"[pas] leakage filter failed for user={user_id!r} "
                f"date={filter_before_date!r} sid={exclude_session_id!r} "
                f"({type(e).__name__}: {e}); falling back to demographics-only."
            )
            return base
        if profile is None or not profile.has_history:
            return base

    # Pass 1: cosine-relevant slice scored + sorted strongest-first. Each
    # entry carries its sim score so the renderer can attach a 3-tier band
    # label (`strong` / `moderate` / `weak`) — matching the similarity-hint
    # vocabulary. Embedder failure / no pair above threshold → recency fallback.
    scored: list[tuple[str, str, float]] = []
    relevance_filtered = False
    if user_query:
        try:
            from mymodule.feature.ollama_embed import get_shared_instructed_embedder
            from mymodule.feature.user_profile import (
                score_prior_pairs_by_query as _score_prior_pairs_by_query,
            )

            embedder = get_shared_instructed_embedder()
            scored = _score_prior_pairs_by_query(profile, user_query, embedder, threshold=sim_threshold, top_k=top_k)
            if scored:
                relevance_filtered = True
        except Exception as e:
            logger.warning(f"[pas] prior-history cosine scoring failed ({type(e).__name__}: {e}); using recency.")
            scored = []
    if not scored:
        # Recency fallback — pack the (uc, tid) pairs into the same triple
        # shape with a sentinel `None` sim so the downstream renderer can
        # tell the two paths apart and emit a different per-line prefix.
        recency_pairs = select_prior_pairs(profile, top_k=top_k)
        scored = [(uc, tid, float("nan")) for uc, tid in recency_pairs]
    if not scored:
        return base

    # Header signals (a) what the block is for (past asks + liked tracks),
    # (b) ordering and labelling semantics — relevance-sorted with per-line
    # band labels vs recency fallback with no scoring. The signature
    # docstring also tells the LLM how to use the band: ignore `weak` rows
    # when the mood doesn't match the current query.
    if relevance_filtered:
        header = (
            "Past asks + tracks this user liked, sorted by relevance to the current query "
            "(strongest first; cosine match on track metadata). Each entry is tagged with a "
            "3-tier band — `[strong/moderate/weak relevance]` — same vocabulary as the "
            "similarity hints. Use the band to weight the signal: `strong` entries directly "
            "support the current pick's WHY; `weak` entries should be ignored when their "
            "mood/genre doesn't match the current query (do not force a fit)."
        )
    else:
        header = (
            "Past asks + tracks this user liked, by recency (no relevance scoring — embedder "
            "unavailable or no prior turn passed the relevance threshold). Treat as background "
            "style hints, not as direct query matches. If the mood/genre of these recents "
            "doesn't match the current query, explicitly note that in `personalization_anchors` "
            "rather than forcing a connection."
        )

    lines = [base, "", header]
    for uc, tid, sim in scored:
        track_str = _render_prior_track(tid, kv)
        uc_short = uc.strip()
        if len(uc_short) > 100:
            uc_short = uc_short[:97] + "…"
        if relevance_filtered:
            band = _grade_relevance_label(sim)
            prefix = f"[{band} relevance]"
        else:
            prefix = "[recency only]"
        lines.append(f'  - {prefix} asked: "{uc_short}"')
        lines.append(f"    liked: {track_str}")
    return "\n".join(lines)


def _grade_relevance_label(sim: float) -> str:
    """3-tier band for the Past-asks block. Mirrors `helpers._grade_label`
    so the LLM sees the same vocabulary on prior listens and similarity
    hints. Bands calibrated for `metadata_rich-qwen3` cosine distributions:
    strong ≥ 0.65 / moderate ≥ 0.45 / weak < 0.45 (above the helper's
    `sim_threshold` floor — anything dropping here already cleared 0.30+).
    """
    if sim >= 0.65:
        return "strong"
    if sim >= 0.45:
        return "moderate"
    return "weak"


_QUOTED_SPAN_RE = re.compile(r""""[^"\n]{2,80}"|(?<![A-Za-z0-9])'[^'\n]{2,80}'(?![A-Za-z0-9])""")
_EXACT_MATCH_CUE_RE = re.compile(
    r"\b(?:original|exact|specific|release|version|looking\s+for|find|play|by\s+[A-Z0-9])\b",
    re.IGNORECASE,
)
_BROAD_DISCOVERY_RE = re.compile(
    r"\b(?:bands|artists|recommendations?|general\s+recommendations?|"
    r"recommend(?:\s+me)?\s+(?:a\s+few|several|some|more)|a\s+few\s+more)\b",
    re.IGNORECASE,
)
_EXPLICIT_VARIETY_RE = re.compile(
    r"\b(?:"
    r"recommend(?:\s+me)?\s+(?:a\s+few|several|some|more|other)|"
    r"give\s+me\s+(?:a\s+few|several|some|more)|"
    r"show\s+me\s+(?:a\s+couple|several|some|more)|"
    r"(?:a\s+few|some)\s+more\s+(?:bands?|artists?|songs?|tracks?|recommendations?)|"
    r"more\s+(?:bands?|artists?|songs?|tracks?|recommendations?)|"
    r"other\s+(?:bands?|artists?|songs?|tracks?|recommendations?)|"
    r"(?:artists?|tracks?)\s+or\s+(?:tracks?|artists?)"
    r")\b",
    re.IGNORECASE,
)
_GOAL_VARIETY_RE = re.compile(
    r"\b(?:multiple|several|a\s+few|a\s+couple|more\s+than\s+one|variety|options?|"
    r"multiple\s+.+(?:songs?|tracks?|recommendations?))\b",
    re.IGNORECASE,
)
_CATALOG_CONTINUATION_RE = re.compile(
    r"\b(?:what\s+other|other\s+songs?|other\s+tracks?|what\s+else|another\s+(?:one|song|track)|"
    r"more\s+(?:songs?|tracks?)|keep\s+(?:['’]?em|them)\s+coming|what['’]?s\s+next)\b",
    re.IGNORECASE,
)
_CATALOG_ANCHOR_RE = re.compile(
    r"\b(?:album(?!\s+(?:art|cover|artwork))|on\s+['\"]|from\s+['\"]|"
    r"songs?\s+(?:are\s+)?on|tracks?\s+(?:are\s+)?on)\b",
    re.IGNORECASE,
)
_TERSE_CONTINUATION_RE = re.compile(
    r"^\s*(?:what\s+else|another\s+(?:one|song|track)|keep\s+(?:['’]?em|them)\s+coming|"
    r"what['’]?s\s+next)[.!?]?\s*$",
    re.IGNORECASE,
)
_DIRECT_TITLE_REQUEST_RE = re.compile(
    r"\b(?:play|put\s+on|how\s+about|next|find|looking\s+for|specific|exact)\b",
    re.IGNORECASE,
)
_VISUAL_ARTWORK_RE = re.compile(
    r"\b(?:cover\s+art|album\s+art|artwork|album\s+cover|visual(?:ly)?|"
    r"painting|painted|color\s+palette|bold\s+colou?rs?|abstract\s+(?:cover|art|visual))\b",
    re.IGNORECASE,
)
_ALBUM_THEME_EXPLANATION_RE = re.compile(
    r"\b(?:specific\s+themes?|particular\s+musical\s+elements?|what\s+themes?|"
    r"themes?\s+.+\bexplores?\b|make\s+it\s+distinct|more\s+about\s+.+\balbum)\b",
    re.IGNORECASE,
)


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _first_meta_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _candidate_title_norms(track_ids: list[str] | None, kv: Any | None) -> set[str]:
    norms: set[str] = set()
    if not track_ids or kv is None:
        return norms
    for tid in track_ids[:20]:
        try:
            meta = kv.get_track_meta(tid)
        except Exception:
            meta = None
        if not meta:
            continue
        title = _first_meta_value(meta.get("track_name"))
        norm = _norm_title(title)
        if norm:
            norms.add(norm)
    return norms


def _quoted_request_titles(query: str) -> list[str]:
    titles: list[str] = []
    for match in _QUOTED_SPAN_RE.finditer(query or ""):
        quoted = match.group(0)[1:-1].strip()
        if not quoted:
            continue
        before = query[max(0, match.start() - 64) : match.start()].lower()
        after = query[match.end() : match.end() + 64].lower()
        # Quoted albums are often context anchors ("from 'Meteora' album"),
        # not the exact track being requested.
        if "album" in after or re.search(r"\b(?:from|on)\s*$", before):
            continue
        direct_before = re.search(
            r"\b(?:play|put\s+on|how\s+about|find|looking\s+for|specific|exact)\b[^.!?]{0,56}$",
            before,
        )
        direct_after = re.match(r"\s*(?:next|by\b)", after)
        if not (direct_before or direct_after):
            continue
        titles.append(quoted)
    return titles


def _has_requested_title_in_pool(query: str, track_ids: list[str] | None, kv: Any | None) -> bool | None:
    requested = [_norm_title(t) for t in _quoted_request_titles(query)]
    requested = [t for t in requested if t]
    if not requested:
        return None
    pool = _candidate_title_norms(track_ids, kv)
    if not pool:
        return None
    return any(req == title or req in title or title in req for req in requested for title in pool)


def _conversation_goal_text(conversation_goal: dict | None) -> str:
    if not conversation_goal:
        return ""
    fields = (
        conversation_goal.get("listener_goal"),
        conversation_goal.get("goal"),
        conversation_goal.get("description"),
        conversation_goal.get("category"),
        conversation_goal.get("specificity"),
    )
    return " ".join(str(v) for v in fields if v)


_EXACT_TITLE_ABSENT_NOTE = (
    'Requested exact title appears absent from the candidate pool: do not say "You got it" '
    "or claim the exact request is playing; briefly frame the selected track as the closest available fit."
)
_VISUAL_ARTWORK_NOTE = (
    "Visual/artwork turn: connect the cover-art clue to the selected title, but include "
    "one concrete sonic or metadata attribute (instrumentation, vocals, tempo, production, "
    "lyrics, or genre texture). Do not rely only on artwork or say to see if the sound matches."
)
_EXPLICIT_VARIETY_NOTE = (
    "Explicit variety request from {source}: use two cited tracks when the pool supports it, otherwise one. "
    "Give each pick a different concrete musical attribute and tie both back to the user's "
    "requested style, artist extension, or mood. Keep it prose, not a list."
)
_ALBUM_THEME_EXPLANATION_NOTE = (
    "Album/theme explanation turn: do not treat quoted prior titles as exact-play requests. "
    "Cite one current-pool title and explain it with a concrete theme, lyric image, riff, "
    "tempo, vocal, or production detail that answers what makes the album or artist distinct."
)


def _format_response_style_notes(
    user_query: str,
    chat_history: list[dict] | None,
    conversation_goal: dict | None,
    track_ids: list[str] | None = None,
    kv: Any | None = None,
) -> str:
    """Return narrow per-turn style guidance for known PAS drift cases.

    This is an input-shaping hint, not a deterministic post-hoc rewrite. Empty
    string means the generic PAS specificity bands should apply unchanged.
    """
    query = (user_query or "").strip()
    if not query:
        return ""
    specificity = str((conversation_goal or {}).get("specificity") or "").upper()
    has_quote = bool(_QUOTED_SPAN_RE.search(query))
    exact_title_absent = (
        has_quote
        and _DIRECT_TITLE_REQUEST_RE.search(query)
        and not _BROAD_DISCOVERY_RE.search(query)
        and _has_requested_title_in_pool(query, track_ids, kv) is False
    )

    if _ALBUM_THEME_EXPLANATION_RE.search(query):
        return _ALBUM_THEME_EXPLANATION_NOTE

    if (
        specificity == "HH"
        and has_quote
        and (_EXACT_MATCH_CUE_RE.search(query) or _DIRECT_TITLE_REQUEST_RE.search(query))
        and not _BROAD_DISCOVERY_RE.search(query)
    ):
        return (
            "Compact exact-match turn: target 18-32 words, one track, 1-2 sentences. "
            "Acknowledge the requested title/artist/album briefly, cite the selected title, "
            "give at most one concrete reason from metadata, and add no extra alternatives "
            "or follow-up question unless the user explicitly asks."
            + (f" {_EXACT_TITLE_ABSENT_NOTE}" if exact_title_absent else "")
        )

    if exact_title_absent:
        return f"Exact-title mismatch turn: target 18-36 words, one track, 1-2 sentences. {_EXACT_TITLE_ABSENT_NOTE}"

    catalog_like = bool(_CATALOG_CONTINUATION_RE.search(query))
    album_catalog_anchor = bool(_CATALOG_ANCHOR_RE.search(query))
    anchored_catalog = bool(album_catalog_anchor or has_quote)
    terse_followup = bool(chat_history and _TERSE_CONTINUATION_RE.search(query))
    query_variety = bool(_EXPLICIT_VARIETY_RE.search(query))
    goal_variety = bool(_GOAL_VARIETY_RE.search(_conversation_goal_text(conversation_goal)))
    explicit_non_album_variety = query_variety and not album_catalog_anchor
    if (
        catalog_like
        and not explicit_non_album_variety
        and not _BROAD_DISCOVERY_RE.search(query)
        and (anchored_catalog or terse_followup)
    ):
        return (
            "Compact catalog-continuation turn: target 20-34 words, one track by default, "
            "acknowledge the album/thread in no more than six words, cite the selected title, "
            "and use one concrete reason. Do not add a second pick unless the user asks for "
            "several, a few, a couple, or multiple options."
        )

    notes: list[str] = []
    if (query_variety or (goal_variety and not catalog_like)) and not (catalog_like and album_catalog_anchor):
        if query_variety and goal_variety:
            source = "current query and listener goal"
        elif query_variety:
            source = "current query"
        else:
            source = "listener goal"
        notes.append(_EXPLICIT_VARIETY_NOTE.format(source=source))
    if _VISUAL_ARTWORK_RE.search(query):
        notes.append(_VISUAL_ARTWORK_NOTE)
    if notes:
        return " ".join(notes)

    return ""


def _compute_relevant_priors(
    user_query: str,
    chat_history: list[dict],
    candidate_track_ids: list[str],
    kv: Any,
    threshold: float | None = None,
    *,
    threshold_md: float = 0.55,
    threshold_bpr: float = 0.15,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    goal_progress_assessments: Any = None,
    thought: str | None = None,
    metadata_vector: str = "metadata_rich-qwen3_embedding_0.6b",
    bpr_vector: str = "cf-bpr",
) -> set[str]:
    """Multi-signal selective gate for chat_history music turns.

    A prior music track is included if **ANY** of these signals passes its
    per-vector-space threshold:

      1. `cosine(query_emb, prior_track_md_emb) >= threshold_md` (qwen3 1024d)
         "Is this prior topically relevant to what the user is asking now?"
      2. `max_i cosine(candidate_i_md_emb, prior_md_emb) >= threshold_md` (qwen3 1024d)
         "Does this prior look like recommended tracks (semantic)?"
      3. `max_i cosine(candidate_i_bpr_emb, prior_bpr_emb) >= threshold_bpr` (cf-bpr 128d)
         "Is this prior in same listening cohort as recommended (collab)?"

    Per-vector thresholds because cosine distributions differ massively across
    spaces (qwen3 typical ~0.5-0.7, cf-bpr typical ~0.0-0.3 — single threshold
    can't cover both). Empirically calibrated defaults at ~40% pass rate
    (devset sample, see `scripts/analyze_similarity_thresholds.py`).

    `threshold` (legacy single-arg) overrides BOTH `threshold_md` and
    `threshold_bpr` to the same value when provided — kept for backward compat.

    Returns empty set on missing KV / chat_history / embedder failure.
    Per-prior failures (missing embedding, dim mismatch) are silently skipped.
    """
    if threshold is not None:
        threshold_md = threshold
        threshold_bpr = threshold
    if not chat_history or kv is None:
        return set()
    prior_ids = [m.get("content", "") for m in chat_history if m.get("role") == "music"]
    prior_ids = [p for p in prior_ids if p]
    if not prior_ids:
        return set()
    try:
        import numpy as np  # noqa: PLC0415

        from mymodule.feature.ollama_embed import get_shared_instructed_embedder

        # Signal 1: query embedding (qwen3 1024d). Skip if query empty.
        query_norm = None
        if user_query:
            qv = get_shared_instructed_embedder().embed_query(
                user_query,
                chat_history=chat_history,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress_assessments=goal_progress_assessments,
                thought=thought,
            )
            if qv is not None and len(qv) > 0:
                query_norm = qv.astype(np.float32) / (np.linalg.norm(qv) + 1e-9)

        # Signals 2 & 3: candidate embedding matrices (normalized rows).
        cand_md_unit = None
        cand_bpr_unit = None
        if candidate_track_ids:
            md_raw = kv.get_track_embeddings_batch(candidate_track_ids, metadata_vector)
            md_keep = [e for e in md_raw if e is not None]
            if md_keep:
                m = np.stack(md_keep).astype(np.float32)
                cand_md_unit = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
            bpr_raw = kv.get_track_embeddings_batch(candidate_track_ids, bpr_vector)
            bpr_keep = [e for e in bpr_raw if e is not None]
            if bpr_keep:
                m = np.stack(bpr_keep).astype(np.float32)
                cand_bpr_unit = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

        relevant: set[str] = set()
        for pid in prior_ids:
            prior_md = kv.get_track_embedding(pid, metadata_vector)
            prior_bpr = kv.get_track_embedding(pid, bpr_vector)

            # Signal 1: query ↔ prior metadata (qwen3 space).
            if query_norm is not None and prior_md is not None and prior_md.shape[-1] == query_norm.shape[-1]:
                pmd = prior_md.astype(np.float32) / (np.linalg.norm(prior_md) + 1e-9)
                if float(np.dot(query_norm, pmd)) >= threshold_md:
                    relevant.add(pid)
                    continue

            # Signal 2: max(candidates_metadata ↔ prior_metadata) — qwen3 space.
            if cand_md_unit is not None and prior_md is not None and prior_md.shape[-1] == cand_md_unit.shape[-1]:
                pmd = prior_md.astype(np.float32) / (np.linalg.norm(prior_md) + 1e-9)
                if float((cand_md_unit @ pmd).max()) >= threshold_md:
                    relevant.add(pid)
                    continue

            # Signal 3: max(candidates_bpr ↔ prior_bpr) — cf-bpr space (smaller scale).
            if cand_bpr_unit is not None and prior_bpr is not None and prior_bpr.shape[-1] == cand_bpr_unit.shape[-1]:
                pbpr = prior_bpr.astype(np.float32) / (np.linalg.norm(prior_bpr) + 1e-9)
                if float((cand_bpr_unit @ pbpr).max()) >= threshold_bpr:
                    relevant.add(pid)
                    continue
        return relevant
    except Exception as e:
        logger.warning(f"[pas] selective relevance gate failed ({type(e).__name__}: {e}); empty set.")
        return set()


class PasResponseGenerator(DspyResponseGenerator):
    """PAS response generator — deterministic PROPOSE → LLM ASSIGN+SELECT → deterministic repair."""

    signature = CRSResponse
    ckpt_basename = "pas"

    def __init__(self, provider: Provider = "ollama", **kwargs: Any) -> None:
        super().__init__(provider=provider, **kwargs)
        self._kv = try_open_kvstore()
        self._top_tracks = int(os.getenv("MYMODULE_LLM_TOP_TRACKS", "5"))
        self._overview_n = int(os.getenv("MYMODULE_LLM_OVERVIEW_TRACKS", "20"))
        self._router_top_n = int(os.getenv("MYMODULE_PAS_ROUTER_TOP_N", "1"))
        self._router: DemoRouter | None = self._maybe_load_router()
        self._fewshot_min_turn = int(os.getenv("MYMODULE_PAS_FEWSHOT_MIN_TURN", "1"))
        self._predict_lock = threading.Lock()
        # Selective chat-expand is the default for PAS — gate prior music tracks
        # via 3-signal OR (query↔prior md, cand↔prior md, cand↔prior bpr).
        # Per-vector thresholds because cosine distributions differ massively
        # (qwen3 typical ~0.5-0.7, cf-bpr typical ~0.0-0.3). Empirically calibrated
        # at ~40% pass rate (devset; see scripts/analyze_similarity_thresholds.py).
        self._chat_expand_mode = os.getenv("MYMODULE_RESPONSE_CHAT_EXPAND_FORMAT", "selective").strip().lower()
        self._selective_threshold_md = float(os.getenv("MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_MD", "0.55"))
        self._selective_threshold_bpr = float(os.getenv("MYMODULE_RESPONSE_CHAT_SELECTIVE_THRESHOLD_BPR", "0.15"))
        # Crawl enrichment for the candidate detail block (caption + lyric hook + key + tempo).
        # Coverage is partial — missing fields are silently dropped per-track.
        _crawl_flag = os.getenv("MYMODULE_LLM_RECOMMENDED_WITH_CRAWL", "1").strip().lower()
        self._with_crawl = _crawl_flag not in ("0", "false", "off", "")
        # Process-wide lexical-diversity suppressor (see module-level docstring).
        self._lexdiv_topn = int(os.getenv("MYMODULE_LEXDIV_SUPPRESS_TOPN", "15"))
        self._lexdiv_min_count = int(os.getenv("MYMODULE_LEXDIV_SUPPRESS_MIN_COUNT", "5"))
        # Bigram-level suppressor. Distinct-2 metric measures bigram diversity
        # so direct bigram avoidance is the most efficient lever. Bigrams are
        # rarer than words → lower min_count floor.
        self._lexdiv_bigram_topn = int(os.getenv("MYMODULE_LEXDIV_SUPPRESS_BIGRAM_TOPN", "12"))
        self._lexdiv_bigram_min_count = int(os.getenv("MYMODULE_LEXDIV_SUPPRESS_BIGRAM_MIN_COUNT", "3"))
        # ---- user_profile enrichment with prior session history -------------
        # `user:history:{user_id}` (built by `mymodule.feature.user_profile --build`)
        # carries this user's train-split interactions. We fold a query-relevant
        # slice into the `user_profile` InputField so Personalization (Step 1)
        # can ground on "what this user has actually engaged with before".
        # Disabled via env when the KV history namespace is not built (cold start
        # or fresh machine) — the helper degrades gracefully on None.
        self._user_history_enabled = _user_history_enabled_from_env()
        self._user_history_top_k = _user_history_top_k_from_env()
        self._user_history_sim_threshold = _user_history_sim_threshold_from_env()

        # ---- KNN demo retrieval --------------------------------------------
        # Activate only when (a) env opts in, (b) router exists and (c) sidecar
        # actually carries query embeddings. Auto-falls back to bucket mode
        # when any of these is false. Read settings into `self._knn_*` once so
        # `generate()` stays cheap.
        sel_mode = os.getenv("MYMODULE_PAS_DEMO_SELECTION", "bucket").strip().lower()
        self._knn_enabled = sel_mode == "knn" and self._router is not None and self._router.has_knn_index()
        self._knn_fetch_m = int(os.getenv("MYMODULE_PAS_KNN_FETCH_M", "10"))
        self._knn_alpha = float(os.getenv("MYMODULE_PAS_KNN_ALPHA", "0.7"))
        self._knn_category_bonus = float(os.getenv("MYMODULE_PAS_KNN_CATEGORY_BONUS", "0.06"))
        self._knn_specificity_bonus = float(os.getenv("MYMODULE_PAS_KNN_SPECIFICITY_BONUS", "0.08"))
        self._knn_turn_bonus = float(os.getenv("MYMODULE_PAS_KNN_TURN_BONUS", "0.03"))
        if self._knn_enabled:
            logger.info(
                f"[pas] KNN demo selection active: dim={self._router.embedding_dim} "
                f"model={self._router.embedding_model or '?'} "
                f"top_k={self._router_top_n} fetch_m={self._knn_fetch_m} alpha={self._knn_alpha:.2f} "
                f"bonuses(cat/spec/turn)="
                f"{self._knn_category_bonus:.2f}/{self._knn_specificity_bonus:.2f}/{self._knn_turn_bonus:.2f}"
            )
        elif sel_mode == "knn" and self._router is not None:
            logger.warning(
                "[pas] MYMODULE_PAS_DEMO_SELECTION=knn requested but sidecar has "
                "no query embeddings — falling back to bucket routing. Re-compile "
                "with `--n-pool N` to enable KNN."
            )

    def _ckpt_path(self) -> Path | None:
        raw = os.getenv("MYMODULE_PAS_FORCE_ZERO_SHOT", "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            logger.info("[pas] MYMODULE_PAS_FORCE_ZERO_SHOT is set; skipping compiled PAS demos.")
            return None
        return super()._ckpt_path()

    def _signature_for_predictor(self) -> type:
        if _compact_signature_enabled():
            return CompactCRSResponse
        return CRSResponse

    def _build_predictor(self) -> Any:
        import dspy

        signature = self._signature_for_predictor()
        pred = dspy.Predict(signature)
        if signature is CompactCRSResponse:
            logger.info("[pas] MYMODULE_PAS_COMPACT_SIGNATURE is set; using compact zero-shot PAS signature.")
        else:
            ckpt = self._ckpt_path()
            if ckpt is not None:
                try:
                    pred.load(str(ckpt))
                except Exception as e:
                    logger.warning(f"[response-gen] compiled ckpt load failed ({e}), using uncompiled.")
        if min_interval_sec() > 0:
            return RateLimitedPredictor(pred, self.provider)
        return pred

    def _maybe_load_router(self) -> DemoRouter | None:
        """Look for the bucket sidecar next to the standard DSPy ckpt and load it.

        Returns None when either ckpt is missing (no sidecar to anchor on) or the
        sidecar itself is missing — both cases fall back to standard DSPy demo
        behavior (static demos from the compiled JSON, or zero-shot if uncompiled).
        """
        ckpt = self._ckpt_path()
        if ckpt is None:
            return None
        sidecar = _bucket_sidecar_path(ckpt)
        if not sidecar.exists():
            return None
        try:
            router = DemoRouter.from_sidecar(sidecar)
        except Exception as e:
            logger.warning(f"[pas] bucket sidecar load failed ({type(e).__name__}: {e}); using static demos.")
            return None
        if len(router) == 0:
            logger.warning(f"[pas] bucket sidecar at {sidecar} has 0 demos; using static demos.")
            return None
        logger.info(
            f"[pas] DemoRouter active: {len(router)} demos across "
            f"{len(router.bucket_counts())} buckets, top_n={self._router_top_n}"
        )
        return router

    def _prepare_inputs(
        self,
        *,
        track_ids: list[str],
        user_query: str,
        chat_history: list[dict],
        user_profile: dict | None,
        conversation_goal: dict | None,
        goal_progress: list[dict] | None,
        user_id: str | None = None,
    ) -> dict:
        # Context kwargs forwarded to query-embedding call sites so they hit
        # the same `query:embedding:{model_id}:{content_hash}` cache entry
        # that the qemb retrieval pool writes. `thought=None` mirrors the
        # qemb pool's hard-coded default (see strategies/pool/qemb.py).
        emb_ctx = dict(
            user_profile=user_profile,
            conversation_goal=conversation_goal,
            goal_progress_assessments=goal_progress,
            thought=None,
        )

        # PROPOSE — deterministic evidence-typed intent groups
        intent_groups = classify_intents(
            track_ids, user_query, chat_history, self._kv, top_n=self._overview_n, **emb_ctx
        )

        # Selective chat-expand: compute the relevant-prior set ONLY when the
        # active expand mode is `selective`. Otherwise pass None (no overhead).
        selective_relevant_ids: set[str] | None = None
        if self._chat_expand_mode == "selective":
            selective_relevant_ids = _compute_relevant_priors(
                user_query=user_query or "",
                chat_history=chat_history,
                candidate_track_ids=track_ids[: self._top_tracks],
                kv=self._kv,
                threshold_md=self._selective_threshold_md,
                threshold_bpr=self._selective_threshold_bpr,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress_assessments=goal_progress,
                thought=None,
            )

        return {
            # CRSResponse InputFields
            "user_query": user_query or "",
            "listener_goal": fmt_conversation_goal(conversation_goal),
            "response_style_notes": _format_response_style_notes(
                user_query,
                chat_history,
                conversation_goal,
                track_ids=track_ids,
                kv=self._kv,
            ),
            "chat_history": fmt_chat_history(chat_history, kv=self._kv, selective_relevant_ids=selective_relevant_ids),
            "user_profile": _format_user_profile_with_history(
                user_profile,
                user_id,
                self._kv,
                user_query,
                enabled=self._user_history_enabled,
                top_k=self._user_history_top_k,
                sim_threshold=self._user_history_sim_threshold,
            ),
            "recommended_tracks_overview": fmt_tracks_overview(track_ids, self._overview_n, self._kv),
            "recommended_tracks_detailed": fmt_recommended_tracks(
                track_ids, self._top_tracks, self._kv, with_crawl=self._with_crawl
            ),
            "recommended_titles_pool": fmt_recommended_titles_pool(track_ids, self._overview_n, self._kv),
            "intent_groups": format_intent_groups_for_prompt(intent_groups),
            "overused_words": ", ".join(_get_overused_words(self._lexdiv_topn, self._lexdiv_min_count)),
            "overused_bigrams": _format_overused_bigrams_for_prompt(
                _get_overused_bigrams(self._lexdiv_bigram_topn, self._lexdiv_bigram_min_count)
            ),
            # All three similarity hints use `graded=True` so the LLM reads
            # 3-tier band labels ("strong" / "moderate" / "weak") — same shape
            # as compile (`optimize.py`). Thresholds rely on helper defaults
            # (track 0.45 / query 0.30 / lyric 0.25); pass overrides only if
            # an experiment specifically requires tighter floors.
            "track_similarity_hints": fmt_track_similarity_hints(
                track_ids, self._top_tracks, chat_history, self._kv, graded=True
            ),
            "query_similarity_hints": fmt_query_similarity_hints(
                track_ids,
                self._top_tracks,
                user_query,
                self._kv,
                graded=True,
                chat_history=chat_history,
                **emb_ctx,
            ),
            "lyric_similarity_hints": fmt_lyric_similarity_hints(
                track_ids,
                self._top_tracks,
                user_query,
                self._kv,
                graded=True,
                chat_history=chat_history,
                **emb_ctx,
            ),
            # Side-channel for _postprocess (underscore prefix = stripped before
            # Predict call by the parent template method)
            "_intent_groups_raw": intent_groups,
            "_seed": f"{user_query}:{len(chat_history)}",
            # Pool for citation resolution — top-20 to match `recommended_titles_pool`.
            # Wider than the top-5 explanation block so swap fallback can land on any
            # legal pool track, not just the highlighted few.
            "_track_ids_pool": track_ids[: self._overview_n],
        }

    def _postprocess(self, raw_out: Any, *, predict_kwargs: dict, track_ids: list[str]) -> str:
        clean_kwargs = {k: v for k, v in predict_kwargs.items() if not k.startswith("_")}
        text = validate_and_repair(
            predictor=self.predictor,
            predict_kwargs=clean_kwargs,
            raw_out=raw_out,
            intent_groups=predict_kwargs["_intent_groups_raw"],
            max_retries=int(os.getenv("MYMODULE_PAS_REGEX_RETRY", "1")),
            seed=predict_kwargs["_seed"],
            track_ids=predict_kwargs["_track_ids_pool"],
            kv=self._kv,
        )
        _maybe_debug_dump(clean_kwargs, raw_out, text, track_ids)
        return text

    # ---- conditional demo retrieval ----------------------------------------

    @staticmethod
    def _demo_dict_to_example(demo: dict) -> Any:
        """Convert a bucket-sidecar demo dict into a `dspy.Example` for predictor.demos."""
        import dspy

        inputs = demo.get("inputs", {}) or {}
        outputs = demo.get("outputs", {}) or {}
        ex = dspy.Example(**inputs, **outputs)
        if inputs:
            ex = ex.with_inputs(*inputs.keys())
        return ex

    def _knn_select_demos(
        self,
        *,
        user_query: str,
        chat_history: list[dict] | None,
        user_profile: dict | None,
        conversation_goal: dict | None,
        goal_progress: list[dict] | None,
    ) -> list[dict]:
        """Embed current session's query and ask the router for top-K demos.

        Uses `InstructedQueryEmbedder.embed_query(...)` — same signature as at
        compile, so the embedding hits the same KV cache namespace and matches
        the pool's stored vectors. Returns `[]` on embedding failure so the
        caller transparently falls back to bucket routing.
        """
        try:
            from mymodule.feature.ollama_embed import get_shared_instructed_embedder

            embedder = get_shared_instructed_embedder()
            vec = embedder.embed_query(
                user_query or "",
                chat_history=chat_history,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress_assessments=goal_progress,
                thought=None,
            )
        except Exception as e:
            logger.warning(
                f"[pas] KNN embed_query failed ({type(e).__name__}: {e}); falling back to bucket routing for this call."
            )
            return []
        spec = (conversation_goal or {}).get("specificity") or ""
        category = (conversation_goal or {}).get("category") or ""
        turn_pos = turn_position_from_index(turn_number_from_chat_history(chat_history))
        return self._router.retrieve_knn(
            vec,
            top_k=self._router_top_n,
            fetch_m=self._knn_fetch_m,
            alpha=self._knn_alpha,
            specificity=spec,
            category=category,
            turn_position=turn_pos,
            category_bonus=self._knn_category_bonus,
            specificity_bonus=self._knn_specificity_bonus,
            turn_bonus=self._knn_turn_bonus,
        )

    def generate(
        self,
        track_ids: list[str],
        user_query: str,
        chat_history: list[dict],
        user_profile: dict | None = None,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_id: str | None = None,
    ) -> str:
        # No router → standard parent template (static demos from ckpt or zero-shot).
        if self._router is None:
            text = super().generate(
                track_ids=track_ids,
                user_query=user_query,
                chat_history=chat_history,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress=goal_progress,
                user_id=user_id,
            )
            _update_word_freq(text)
            _update_bigram_freq(text)
            return text

        # Demo selection: KNN+score blend when enabled, hierarchical bucket fallback otherwise.
        # Cold-start turns often have no history and can be harmed by a demo's
        # over-specific conversation pattern. Keep this few-shot-only knob out
        # of the zero-shot path: setting min_turn=2 gives turn 1 an empty demo
        # list while later turns still use the compiled router.
        demo_dicts: list[dict] = []
        current_turn = turn_number_from_chat_history(chat_history)
        use_fewshot = current_turn >= max(1, self._fewshot_min_turn)
        if use_fewshot and self._knn_enabled:
            demo_dicts = self._knn_select_demos(
                user_query=user_query,
                chat_history=chat_history,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress=goal_progress,
            )
        if use_fewshot and not demo_dicts:
            # KNN disabled, or KNN embed failed → bucket-mode fallback.
            spec = (conversation_goal or {}).get("specificity") or ""
            category = (conversation_goal or {}).get("category") or ""
            turn_pos = turn_position_from_index(current_turn)
            demo_dicts = self._router.retrieve(spec, turn_pos, top_n=self._router_top_n, category=category)
        demos = [self._demo_dict_to_example(d) for d in demo_dicts]

        # Build a per-call predictor by shallow-copying the inner dspy.Predict and
        # setting `.demos` on the copy. This avoids mutating shared state — earlier
        # version used a lock around `self.predictor.demos = ...` which serialized
        # all generate() calls (effective workers=1 with concurrent ThreadPool).
        # Shallow copy shares lm/signature (stateless) but isolates demos per call.
        import copy as _copy

        from mymodule.utils.common_dspy import apply_rate_limit, min_interval_sec

        inner = getattr(self.predictor, "_inner", self.predictor)
        local_predictor = _copy.copy(inner)
        local_predictor.demos = demos

        try:
            predict_kwargs = self._prepare_inputs(
                track_ids=track_ids,
                user_query=user_query,
                chat_history=chat_history,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
                goal_progress=goal_progress,
                user_id=user_id,
            )
            clean_kwargs = {k: v for k, v in predict_kwargs.items() if not k.startswith("_")}
            if min_interval_sec() > 0:
                apply_rate_limit(self.provider)
            raw_out = local_predictor(**clean_kwargs)
            # Custom postprocess uses the local predictor for soft-ban retry so the
            # retry call also sees the routed demos (consistent with the first call).
            text = validate_and_repair(
                predictor=local_predictor,
                predict_kwargs=clean_kwargs,
                raw_out=raw_out,
                intent_groups=predict_kwargs["_intent_groups_raw"],
                max_retries=int(os.getenv("MYMODULE_PAS_REGEX_RETRY", "1")),
                seed=predict_kwargs["_seed"],
                track_ids=predict_kwargs["_track_ids_pool"],
                kv=self._kv,
            )
            _maybe_debug_dump(clean_kwargs, raw_out, text, track_ids)
            stripped = strip_dspy_markers(text)
            _update_word_freq(stripped)
            _update_bigram_freq(stripped)
            return stripped
        except Exception as e:
            hint = track_ids[0][:8] if track_ids else "-"
            logger.warning(f"[response-gen] failed for top={hint}: {type(e).__name__}: {e}")
            return ""
