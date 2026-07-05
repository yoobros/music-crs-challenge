"""LLM-generated synthetic doc enrichment for the two-tower retrieval pool.

Given each track's metadata (HF + crawl) we call an LLM to emit three fields
that drive the doc-side `compose_doc` enrichment:

- ``mood``       — 2-4 mood adjectives                  (Mood: line)
- ``themes``     — 2-4 thematic keywords / phrases     (Themes: line)
- ``use_cases``  — 5 short search-use-case phrases     (Suitable for: line, ★ EOS anchor)

Provider selection is the same shared abstraction used by
``mymodule/strategies/response`` (``mymodule.utils.common_dspy``). For
reproducibility we force ``temperature=0`` and a fixed ``seed`` so the same
prompt deterministically reproduces.

Output is appended to a JSONL cache at
``mymodule/strategies/twotower/data/synth_use_cases.jsonl`` with the schema::

    {"track_id": "<tid>", "version": "<stamp>", "mood": "...", "themes": "...",
     "use_cases": ["phrase1", ..., "phrase5"]}

The CLI is resumable: tracks already cached under the same ``version`` are
skipped. ``--force`` re-generates everything.

CLI
---

    # Default — OpenAI-compat endpoint, google/gemma-4-26b-a4b-it, T=0, seed=42
    uv run python -m mymodule.strategies.twotower.synth_doc

    # Smaller sanity sample
    uv run python -m mymodule.strategies.twotower.synth_doc --max-tracks 200

    # Local Ollama provider override
    uv run python -m mymodule.strategies.twotower.synth_doc \\
        --provider ollama --model gemma4:e2b
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from mymodule.utils.common_dspy import (
    Provider,
    RateLimitedPredictor,
    ensure_lm_configured,
    min_interval_sec,
)

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = _MODULE_DIR / "data" / "synth_use_cases.jsonl"
SYNTH_VERSION_PREFIX = "v1"
SUITABLE_N = 5

# CLI defaults — primary recipe is OpenAI-compat endpoint with a 26B Gemma
# variant. Override via env / flags for cheaper iteration.
DEFAULT_PROVIDER: Provider = "openai"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0


if dspy is not None:

    class TrackSynthCard(dspy.Signature):
        """Generate retrieval-aligned enrichment for a music track.

        Return concise, retrieval-friendly phrases. Avoid generic filler
        ("a great song", "amazing music"). Use natural language that a real
        user would type when searching for music in this style or context.
        Do NOT use the exact track title or artist name in the use_cases —
        those are already encoded elsewhere; use_cases should describe the
        situation, mood, or audience.
        """

        title: str = dspy.InputField(desc="track title")
        artist: str = dspy.InputField(desc="artist name")
        album: str = dspy.InputField(desc="album name (may be empty)")
        year: str = dspy.InputField(desc="release year e.g. 1993 (may be empty)")
        tags: str = dspy.InputField(desc="user-generated genre/mood tags, comma-separated")
        mb_tags: str = dspy.InputField(desc="curated MusicBrainz tags, comma-separated (may be empty)")
        caption: str = dspy.InputField(desc="audio description, natural language (may be empty)")
        lyrics_excerpt: str = dspy.InputField(desc="short lyrics excerpt, may be empty")

        mood: str = dspy.OutputField(
            desc=(
                "2-4 mood adjectives, comma-separated. "
                "Examples: 'brooding, raw, melancholic' | 'energetic, uplifting, danceable' | "
                "'introspective, dreamy'."
            )
        )
        themes: str = dspy.OutputField(
            desc=(
                "2-4 thematic keywords or short phrases, comma-separated. "
                "Examples: 'emotional volatility, vulnerability' | "
                "'late-night drive, freedom, escapism' | 'unrequited love, regret'."
            )
        )
        use_cases: str = dspy.OutputField(
            desc=(
                f"Exactly {SUITABLE_N} short search-use-case phrases (4-10 words each), "
                "semicolon-separated, NO numbering or bullets. These should mirror how a "
                "real listener would describe when they'd seek this kind of track. "
                "Examples: 'late-night introspection; processing heavy emotions; "
                "90s alt-rock nostalgia; cathartic angst listening; deep cuts from grunge era'."
            )
        )

else:  # pragma: no cover
    TrackSynthCard = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _version_stamp(provider: str, model: str, seed: int) -> str:
    """Deterministic version identifier — keeps cache entries for different
    provider/model/seed combinations from leaking into each other."""
    model_slug = model.replace("/", "_").replace(":", "_").replace(".", "_")
    return f"{SYNTH_VERSION_PREFIX}_{provider}_{model_slug}_seed{seed}"


def load_synth_use_cases(
    path: Path = DEFAULT_OUT,
    *,
    version: str | None = None,
) -> dict[str, dict]:
    """Load JSONL → ``{track_id: {mood, themes, use_cases}}``.

    Entries whose ``version`` field does not match the requested ``version``
    (when supplied) are treated as stale and skipped. Pass ``version=None``
    to accept all rows (useful when consumers don't care about staleness).
    """
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("track_id")
            if not tid:
                continue
            if version is not None and obj.get("version") != version:
                continue
            uc = obj.get("use_cases") or []
            if isinstance(uc, str):
                uc = [p.strip() for p in uc.split(";") if p.strip()]
            out[tid] = {
                "mood": (obj.get("mood") or "").strip(),
                "themes": (obj.get("themes") or "").strip(),
                "use_cases": [str(p).strip() for p in uc if str(p).strip()],
            }
    return out


# ---------------------------------------------------------------------------
# DSPy predictor wiring
# ---------------------------------------------------------------------------


def _build_predictor(provider: Provider, seed: int, temperature: float) -> Any:
    if dspy is None or TrackSynthCard is None:
        raise RuntimeError("dspy is required for synth_doc. Install: `uv add dspy`.")
    ensure_lm_configured(provider, seed=seed, temperature=temperature)
    pred = dspy.Predict(TrackSynthCard)
    if min_interval_sec() > 0:
        return RateLimitedPredictor(pred, provider)
    return pred


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _coerce_use_cases(text: str) -> list[str]:
    """Parse ';'-separated output → up to 5 deduped, cleaned phrases."""
    if not text:
        return []
    parts = [p.strip().lstrip("0123456789.-)* ").rstrip(".;: ") for p in text.split(";")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out[:SUITABLE_N]


def generate_one(
    predictor: Any,
    *,
    title: str,
    artist: str,
    album: str,
    year: str,
    tags: str,
    mb_tags: str,
    caption: str,
    lyrics_excerpt: str,
) -> dict:
    raw = predictor(
        title=title or "(unknown)",
        artist=artist or "(unknown)",
        album=album,
        year=year,
        tags=tags,
        mb_tags=mb_tags,
        caption=caption,
        lyrics_excerpt=lyrics_excerpt,
    )
    mood = (getattr(raw, "mood", "") or "").strip().rstrip(".")
    themes = (getattr(raw, "themes", "") or "").strip().rstrip(".")
    use_cases = _coerce_use_cases(getattr(raw, "use_cases", "") or "")
    return {"mood": mood, "themes": themes, "use_cases": use_cases}


# ---------------------------------------------------------------------------
# Bulk pipeline
# ---------------------------------------------------------------------------


def _track_inputs(
    row: dict,
    crawl: dict | None,
    *,
    lyrics_excerpt_chars: int = 180,
    caption_chars: int = 220,
) -> dict[str, str]:
    """Pack HF row + KV crawl record into DSPy input strings."""
    # Local imports to avoid a circular dependency at module load time
    # (text_compose pulls in some shared helpers that we want available here).
    from mymodule.strategies.twotower.text_compose import (
        _first_str,
        _join_strs,
        _truncate_at_word_boundary,
        _year,
    )

    title = _first_str(row.get("track_name"), "")
    artist = _join_strs(row.get("artist_name"))
    album = _join_strs(row.get("album_name"))
    year = _year(row.get("release_date"))

    tags_list = row.get("tag_list") or []
    if not isinstance(tags_list, list):
        tags_list = [tags_list]
    tags = ", ".join(str(t).strip() for t in tags_list[:20] if t)

    mb_tags_str = ""
    caption = ""
    lyrics_excerpt = ""
    if crawl:
        mb = crawl.get("mb_tags") or []
        if isinstance(mb, list):
            mb_tags_str = ", ".join(str(t).strip() for t in mb if t)
        elif isinstance(mb, str):
            mb_tags_str = mb
        caption_raw = (crawl.get("caption") or "").strip()
        caption = _truncate_at_word_boundary(caption_raw, caption_chars)
        lyrics_raw = (crawl.get("lyrics") or "").strip().replace("\n", " ")
        lyrics_excerpt = _truncate_at_word_boundary(lyrics_raw, lyrics_excerpt_chars)

    return {
        "title": title,
        "artist": artist,
        "album": album,
        "year": year,
        "tags": tags,
        "mb_tags": mb_tags_str,
        "caption": caption,
        "lyrics_excerpt": lyrics_excerpt,
    }


def bulk_generate(
    *,
    out_path: Path = DEFAULT_OUT,
    provider: Provider = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tracks: int | None = None,
    force: bool = False,
    concurrency: int = 1,
) -> None:
    """Iterate the full track catalog, call the LLM per track, append JSONL.

    Resumable: rows already present in ``out_path`` under the same ``version``
    stamp are skipped. ``--force`` re-generates every row (does NOT truncate
    the file — manually delete it first to bypass the resume cache).

    ``concurrency > 1`` uses a ThreadPoolExecutor — useful when the LLM
    endpoint can handle parallel requests (vLLM / SGLang / managed APIs).
    The DSPy predictor + LiteLLM stack is thread-safe; the JSONL writer is
    protected by a lock and flushed per-record so the resume cache stays
    consistent on interruption.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from datasets import load_dataset

    # Route the requested model to whichever env var the chosen provider reads.
    # common_dspy reads MYMODULE_LLM_OPENAI_MODEL for openai and MYMODULE_LLM_MODEL
    # for ollama. We override here so the same --model flag works for both.
    if provider == "openai":
        os.environ["MYMODULE_LLM_OPENAI_MODEL"] = model
    else:
        os.environ["MYMODULE_LLM_MODEL"] = model

    version = _version_stamp(provider, model, seed)
    logger.info(
        f"[synth-doc] version={version} provider={provider} model={model} seed={seed} T={temperature} "
        f"concurrency={concurrency}"
    )

    existing: dict[str, dict] = {} if force else load_synth_use_cases(out_path, version=version)
    if existing:
        logger.info(f"[synth-doc] resuming — {len(existing)} tracks already cached for this version")

    kv = None
    try:
        from mymodule.feature.kvdb import KVStore

        kv = KVStore.open(read_only=True)
    except Exception as e:
        logger.warning(f"[synth-doc] KV unavailable ({type(e).__name__}: {e}); crawl fields will be empty.")

    logger.info("[synth-doc] loading talkpl Track-Metadata (split=all_tracks)")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    n = len(ds) if max_tracks is None else min(max_tracks, len(ds))

    predictor = _build_predictor(provider, seed=seed, temperature=temperature)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    failures = 0
    write_lock = threading.Lock()

    def _process(tid: str, inputs: dict[str, str]) -> dict | None:
        try:
            return generate_one(predictor, **inputs)
        except Exception as e:
            nonlocal failures
            with write_lock:
                failures += 1
                if failures <= 5:
                    logger.warning(f"[synth-doc] failed tid={tid[:8]}: {type(e).__name__}: {e}")
            return None

    # Build the list of (tid, inputs) pairs first — pre-fetch KV crawl on
    # the main thread so workers only do LLM calls. Resume-skip happens here.
    pending_inputs: list[tuple[str, dict[str, str]]] = []
    for i in range(n):
        row = ds[int(i)]
        tid = row["track_id"]
        if tid in existing:
            skipped += 1
            continue
        crawl = None
        if kv is not None:
            try:
                crawl = kv.get_track_crawl(tid)
            except Exception:
                crawl = None
        pending_inputs.append((tid, _track_inputs(row, crawl)))
    logger.info(f"[synth-doc] {len(pending_inputs)} tracks queued for generation ({skipped} already cached, skipping)")

    with out_path.open("w" if force else "a") as f:
        if concurrency <= 1:
            # Sequential path (legacy behavior).
            for tid, inputs in pending_inputs:
                result = _process(tid, inputs)
                if result is None:
                    continue
                obj = {"track_id": tid, "version": version, **result}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if written % 100 == 0:
                    logger.info(f"[synth-doc] progress: {written} written, {skipped} skipped, {failures} failed")
        else:
            # Parallel path — submit all, drain via as_completed.
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                future_to_tid: dict = {ex.submit(_process, tid, inputs): tid for tid, inputs in pending_inputs}
                for fut in as_completed(future_to_tid):
                    tid = future_to_tid[fut]
                    result = fut.result()
                    if result is None:
                        continue
                    obj = {"track_id": tid, "version": version, **result}
                    with write_lock:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                        log_now = written % 100 == 0
                    if log_now:
                        logger.info(f"[synth-doc] progress: {written} written, {skipped} skipped, {failures} failed")
    logger.success(f"[synth-doc] done: {written} written, {skipped} skipped, {failures} failed → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Bulk-generate synthetic doc enrichment for two-tower.")
    p.add_argument("--provider", choices=["ollama", "openai"], default=DEFAULT_PROVIDER)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model tag. Default = google/gemma-4-26b-a4b-it (OpenAI-compat endpoint). "
            "Ollama recipe: --provider ollama --model gemma4:e2b."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after N tracks (smoke test). Default: full catalog (~47k).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore the resume cache and re-generate every row.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Worker threads. >1 parallelizes the LLM calls via ThreadPoolExecutor. "
            "Safe with vLLM / SGLang / managed OpenAI-compat endpoints. "
            "Recommended range: 4-10 depending on endpoint capacity."
        ),
    )
    args = p.parse_args()
    bulk_generate(
        out_path=args.out,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        temperature=args.temperature,
        max_tracks=args.max_tracks,
        force=args.force,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
