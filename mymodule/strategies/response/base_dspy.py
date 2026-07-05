"""DSPy-backed response generator — abstract parent + response-domain formatters.

Concrete variants (`pas/generator.py`, `auto/generator.py`) subclass
`DspyResponseGenerator` and declare:
    signature: type[dspy.Signature]          # required
    ckpt_basename: str | None = None          # optional — ckpt/{basename}_{provider}.json

plus override `_prepare_inputs(...)` and (optionally) `_postprocess(...)`.

LM plumbing lives in `mymodule/utils/common_dspy.py`
-----------------------------------------------------
Provider switch, `ensure_lm_configured`, rate limiter, and generic
conversation formatters (`fmt_chat_history`, `fmt_user_profile`,
`fmt_conversation_goal`) are shared with the query-instruction generator in
`mymodule/feature/ollama_embed.py`. Import them from `common_dspy` directly.

This module still owns **response-domain** formatters that depend on track
metadata / KVStore (`fmt_recommended_tracks`, `fmt_tracks_overview`,
`try_open_kvstore`) and the `DspyResponseGenerator` template class itself.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from mymodule.strategies.response.base import BaseResponseGenerator
from mymodule.utils.common_dspy import (
    Provider,
    RateLimitedPredictor,
    _first,
    ensure_lm_configured,
    min_interval_sec,
)

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


# DSPy delimits structured-output fields with `[[ ## field_name ## ]]` markers
# and closes the whole block with `[[ ## completed ]]`. Normally the parser
# strips these, but when a trailing field (like PAS `response`) runs close
# to `max_tokens` the closing marker can leak into the field text. Strip
# any leftover markers from the final response so they don't surface in
# the submission.
_DSPY_MARKER_RE = re.compile(r"\s*\[\[\s*##\s*[\w-]+\s*(?:##\s*)?\]\]\s*")


def strip_dspy_markers(text: str) -> str:
    """Remove any residual DSPy structured-output delimiters from `text`.

    Idempotent. Preserves the body text otherwise; only strips the
    `[[ ## ... ]]` / `[[ ## ... ## ]]` bracketed markers that should have
    been consumed by the parser.
    """
    if not text:
        return text
    return _DSPY_MARKER_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Track-metadata formatters (response-domain; stay here, NOT in common_dspy).
# ---------------------------------------------------------------------------


_SECTION_HEADER_RE = re.compile(r"^\s*[\[(].*[\])]\s*$")
# CJK ranges: Hiragana/Katakana, CJK Extension A + Unified Ideographs, Hangul syllables.
_NON_LATIN_RE = re.compile("[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")
_LANG_MARKER_RE = re.compile(
    r"^\s*(?:romaji|japanese|english|korean|chinese|mandarin|cantonese|spanish|french|"
    r"portuguese|italian|german|russian|jpn|kor|eng|esp|fra|deu|rus)\b.*$",
    re.IGNORECASE,
)


def _extract_lyric_hook(lyrics_text: str, max_lines: int = 2, max_len: int = 110) -> str:
    """Return the first `max_lines` content lines of `lyrics_text` as a hook.

    Filters out lines that aren't usable as an English-prose conversation hook:
    section markers (`[Verse 1]`, `(Chorus)`), parenthetical-only lines,
    language identifier headers (`Romaji`, `日本語 / JAPANESE`, `[KOR]`), and any
    line containing CJK characters. The English-prose response LLM benefits
    from a clean romanized/English snippet; non-Latin script lines confuse it.
    Joins picked lines with ` / ` and trims to `max_len` chars.
    """
    picked: list[str] = []
    for raw in lyrics_text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if _SECTION_HEADER_RE.match(ln):
            continue
        if _LANG_MARKER_RE.match(ln):
            continue
        if _NON_LATIN_RE.search(ln):
            continue
        picked.append(ln)
        if len(picked) >= max_lines:
            break
    if not picked:
        return ""
    hook = " / ".join(picked)
    if len(hook) > max_len:
        hook = hook[: max_len - 3].rsplit(" ", 1)[0] + "..."
    return hook


def _format_crawl_segment(tid: str, kv: Any) -> str:
    """Compact `{caption; lyric: "..."; key: ...; tempo: ...}` segment from crawl.

    Only fields useful for response explanation are pulled — `caption` (text
    description of the music piece), the first-2-line lyric hook, `key`, and
    `tempo`. `chord`, `mb_release_date`, `label`, `country`, `mbid` etc. are
    omitted: they don't help the LLM explain WHY this track fits the user
    while bloating the prompt. Returns `""` when no crawl record or no useful
    field is present.
    """
    if kv is None:
        return ""
    try:
        cr = kv.get_track_crawl(tid, fields=["caption", "lyrics", "key", "tempo"]) or {}
    except Exception:
        return ""
    if not cr:
        return ""

    bits: list[str] = []
    cap = (cr.get("caption") or "").strip()
    if cap:
        # First sentence (cap at 160 chars) — captions are descriptive and dense
        first_sentence = cap.split(". ")[0].strip()
        if len(first_sentence) > 160:
            first_sentence = first_sentence[:157].rsplit(" ", 1)[0] + "..."
        bits.append(f"caption: {first_sentence}")

    lyr = (cr.get("lyrics") or "").strip()
    if lyr:
        hook = _extract_lyric_hook(lyr)
        if hook:
            bits.append(f'lyric: "{hook}"')

    key = (cr.get("key") or "").strip()
    if key:
        bits.append(f"key: {key}")

    tempo_raw = cr.get("tempo")
    if tempo_raw not in (None, "", 0):
        try:
            t = float(tempo_raw)
            if t > 30:  # filter out obvious sentinel values like 0
                bits.append(f"tempo: {t:.0f}bpm")
        except (TypeError, ValueError):
            pass

    if not bits:
        return ""
    return " {" + "; ".join(bits) + "}"


def fmt_recommended_tracks(
    track_ids: list[str],
    top_n: int,
    kv: Any,
    *,
    with_popularity: bool = False,
    with_crawl: bool = False,
) -> str:
    """Render top-N tracks with a 1-based index the LLM can reference.

    Format (default): `N. <title> — <artist> (album: <album>, <year>) [tags]`.
    With `with_popularity=True`: appends `, pop: <score>` inside the parens.
    With `with_crawl=True`: appends `{caption: ...; lyric: "..."; key: ...; tempo: ...bpm}`
    drawn from `kv.get_track_crawl()`. Only the fields useful for explaining
    WHY a track fits the listener are pulled (caption, lyric hook, key, tempo)
    — see `_format_crawl_segment`. Designed to enrich the response LLM's
    grounding without touching retrieval. Token-budget aware (caption capped
    at 160 chars, lyric hook at ~110 chars).

    The popularity field is the same numeric column used by the rich BM25 doc
    build (`mymodule.feature.store --build-metadata-rich`). Passing it here
    lets the response LLM distinguish deep-cut discovery from crowd-pleaser
    picks; off by default to keep the baseline prompt token-budget unchanged.
    """
    lines: list[str] = []
    for i, tid in enumerate(track_ids[:top_n], 1):
        meta = kv.get_track_meta(tid) if kv is not None else None
        if not meta:
            lines.append(f"{i}. <track {tid[:8]}... (metadata unavailable)>")
            continue
        title = _first(meta.get("track_name"))
        artist = _first(meta.get("artist_name"))
        album = _first(meta.get("album_name"))
        release_date = _first(meta.get("release_date"))
        release_year = (
            release_date[:4] if release_date and len(release_date) >= 4 and release_date[:4].isdigit() else ""
        )
        tags = meta.get("tag_list") or []
        tag_str = ", ".join(str(t) for t in tags[:8]) if tags else ""

        pop_segment = ""
        if with_popularity:
            pop_raw = meta.get("popularity")
            if pop_raw is not None:
                try:
                    pop_segment = f", pop: {int(round(float(pop_raw)))}"
                except (TypeError, ValueError):
                    pop_segment = ""

        parts = [f"{i}. {title} — {artist}"]
        if album and album != "unknown":
            parts.append(f"(album: {album}")
            if release_year:
                parts[-1] += f", {release_year}{pop_segment})"
            else:
                parts[-1] += f"{pop_segment})"
        elif release_year:
            parts.append(f"({release_year}{pop_segment})")
        elif pop_segment:
            # Strip leading ", " when popularity is the only paren content.
            parts.append(f"({pop_segment.lstrip(', ')})")
        if tag_str:
            parts.append(f"[{tag_str}]")
        line = " ".join(parts)
        if with_crawl:
            line += _format_crawl_segment(tid, kv)
        lines.append(line)
    return "\n".join(lines) if lines else ""


def fmt_goal_progress(goal_progress: list[dict] | None, max_items: int = 3) -> str:
    """Render the most recent `goal_progress_assessments` so the response LLM
    can acknowledge multi-turn progress (e.g. "you've already explored era X").

    Surfacing this helps the personalization stage anchor on what the user
    has *already* achieved within the conversation.
    """
    if not goal_progress:
        return ""
    lines: list[str] = []
    for entry in list(goal_progress)[-max_items:]:
        if not isinstance(entry, dict):
            continue
        turn = entry.get("turn_number")
        assessment = (entry.get("assessment") or entry.get("content") or "").strip()
        if not assessment:
            continue
        prefix = f"turn {turn}: " if turn is not None else ""
        lines.append(f"{prefix}{assessment}")
    return "\n".join(lines)


def fmt_recommended_titles_pool(track_ids: list[str], top_n: int, kv: Any) -> str:
    """Render top-N tracks as a flat closed citation pool: `N. "Title" — Artist`.

    Distinct from `fmt_recommended_tracks` (which carries album/year/tags/crawl
    for explanation grounding): this is the EXACT enumeration that bounds what
    `cited_titles` may emit. Mention parity with the post-processor's
    `_build_title_pool` (same metadata source, same 1-based index, same
    `unknown`-skip semantics) — both sides agree on what counts as in-pool.

    Empty string when no resolvable metadata (caller decides whether to skip
    the InputField entirely or pass empty).
    """
    if not track_ids or kv is None:
        return ""
    lines: list[str] = []
    for i, tid in enumerate(track_ids[:top_n], 1):
        meta = kv.get_track_meta(tid)
        if not meta:
            continue
        title = _first(meta.get("track_name"))
        if not title or title == "unknown":
            continue
        artist = _first(meta.get("artist_name"))
        if artist and artist != "unknown":
            lines.append(f'{i}. "{title}" — {artist}')
        else:
            lines.append(f'{i}. "{title}"')
    return "\n".join(lines)


def fmt_tracks_overview(track_ids: list[str], top_n: int, kv: Any) -> str:
    """Aggregate summary of top-N tracks (artist / year / tag / album distribution).

    Compact multi-line summary so the LLM can identify sub-themes without
    reading every row.
    """
    if not track_ids or kv is None:
        return f"tracks: {len(track_ids)} (metadata unavailable)"

    artist_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    album_set: set[str] = set()
    years: list[int] = []
    valid = 0

    for tid in track_ids[:top_n]:
        meta = kv.get_track_meta(tid)
        if not meta:
            continue
        valid += 1
        artist = _first(meta.get("artist_name"))
        if artist and artist != "unknown":
            artist_counts[artist] += 1
        album = _first(meta.get("album_name"))
        if album and album != "unknown":
            album_set.add(album)
        release_date = _first(meta.get("release_date"))
        if release_date and len(release_date) >= 4 and release_date[:4].isdigit():
            years.append(int(release_date[:4]))
        for tag in (meta.get("tag_list") or [])[:10]:
            if tag:
                tag_counts[str(tag)] += 1

    lines: list[str] = [f"tracks: {valid}"]

    if artist_counts:
        top_artists = [f"{a}×{c}" for a, c in artist_counts.most_common(3) if c >= 2]
        artist_line = f"artists: {len(artist_counts)} unique"
        if top_artists:
            artist_line += f" (top: {', '.join(top_artists)})"
        lines[-1] += f" | {artist_line}"

    if years:
        years_sorted = sorted(years)
        median_year = years_sorted[len(years_sorted) // 2]
        lines.append(f"years: {min(years)}-{max(years)} (median {median_year})")

    if tag_counts:
        top_tags = [f"{t} ({c})" for t, c in tag_counts.most_common(5)]
        lines.append(f"top tags: {', '.join(top_tags)}")

    if album_set:
        lines.append(f"albums: {len(album_set)} unique")

    return "\n".join(lines)


def try_open_kvstore() -> Any:
    """Open the repo KVStore or return None (gracefully degraded)."""
    try:
        from mymodule.feature.kvdb import KVStore

        return KVStore.open()
    except Exception as e:
        logger.warning(
            f"[response-gen] KVStore unavailable ({type(e).__name__}: {e}). "
            f"Track metadata lookup will fall back to id prefixes."
        )
        return None


# ---------------------------------------------------------------------------
# Abstract parent
# ---------------------------------------------------------------------------


class DspyResponseGenerator(BaseResponseGenerator):
    """Abstract parent for DSPy-powered response generators.

    Concrete subclasses (e.g. `PasResponseGenerator`) set `signature` and
    override `_prepare_inputs`. `_postprocess` defaults to returning
    `raw_out.response`. `_build_predictor` can be overridden if the variant
    needs a non-`dspy.Predict` module (e.g. `ChainOfThought`).

    Not a registry entry itself — `response/__init__.py` registers only
    concrete subclasses explicitly.
    """

    # Subclass contract
    signature: ClassVar[type]  # `type[dspy.Signature]`; set by concrete subclass
    ckpt_basename: ClassVar[str | None] = None

    def __init__(self, provider: Provider = "ollama", **kwargs: Any) -> None:
        if dspy is None:
            raise RuntimeError("dspy is required for DSPy response generators. Install: `uv add dspy`.")
        if not hasattr(type(self), "signature"):
            raise TypeError(
                f"{type(self).__name__} must set a class attribute `signature` "
                f"(a dspy.Signature subclass) before instantiation."
            )
        self.provider: Provider = provider
        ensure_lm_configured(provider)
        self.predictor = self._build_predictor()

    # ---- checkpoint loading ------------------------------------------------
    def _ckpt_dir(self) -> Path:
        return Path(__file__).parent / "ckpt"

    def _ckpt_path(self) -> Path | None:
        if not self.ckpt_basename:
            return None
        p = self._ckpt_dir() / f"{self.ckpt_basename}_{self.provider}.json"
        return p if p.exists() else None

    def _build_predictor(self) -> Any:
        pred = dspy.Predict(self.signature)
        ckpt = self._ckpt_path()
        if ckpt is not None:
            try:
                pred.load(str(ckpt))
            except Exception as e:
                logger.warning(f"[response-gen] compiled ckpt load failed ({e}), using uncompiled.")
        if min_interval_sec() > 0:
            # Zero-overhead when no rate limit is configured.
            return RateLimitedPredictor(pred, self.provider)
        return pred

    # ---- template method ---------------------------------------------------
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
            raw_out = self.predictor(**clean_kwargs)
            text = self._postprocess(raw_out, predict_kwargs=predict_kwargs, track_ids=track_ids)
            # Strip any residual DSPy `[[ ## field ## ]]` delimiters that
            # leaked past the parser. Applied uniformly after every
            # subclass's _postprocess so pas/auto/noop all stay clean.
            return strip_dspy_markers(text)
        except Exception as e:
            hint = track_ids[0][:8] if track_ids else "-"
            logger.warning(f"[response-gen] failed for top={hint}: {type(e).__name__}: {e}")
            return ""

    # ---- subclass hooks ----------------------------------------------------
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
        raise NotImplementedError

    def _postprocess(self, raw_out: Any, *, predict_kwargs: dict, track_ids: list[str]) -> str:
        return (getattr(raw_out, "response", "") or "").strip()


__all__ = [
    "fmt_recommended_tracks",
    "fmt_tracks_overview",
    "fmt_goal_progress",
    "try_open_kvstore",
    "DspyResponseGenerator",
]
