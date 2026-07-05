"""LLM-generated 'Session so far:' summaries for two-tower query composition.

Given a single (session, turn) — i.e. the chat history up to (but not
including) the cutoff user message plus the conversation goal — we call an
LLM once to emit a one-sentence summary of the user's evolving intent /
preference trajectory. That sentence fills the ``Session so far:`` line of
``compose_query``.

Why LLM? Lemma / keyword extraction (the legacy ``_summarise_queries``)
compresses to disjoint tokens that don't carry conversational nuance ("user
shifted from indie to grunge", "preference toward heavier textures"). The
embedding model picks up much more from a fluent natural-language sentence
near the EOS than from a tag-cloud of nouns.

Pattern mirrors ``synth_doc.py``:

- DSPy ``Signature`` with a single ``session_summary`` output
- provider switch (``ollama`` / ``openai``) via ``mymodule.utils.common_dspy``
- ``temperature=0`` + fixed seed → reproducible
- JSONL cache at ``mymodule/strategies/twotower/data/query_summaries.jsonl``

Cache key
---------
A SHA-256 hex of ``user_msg | chat_history (turn_number:role:content)``,
truncated to 16 chars. We do NOT use ``session_id`` because the runtime pool
gets ``chat_history`` but not the dataset's session_id — hashing the
content yields the same key as long as the history is unchanged.

CLI
---
The bulk builder iterates a dataset (``devset`` / ``blindset_A``) and
pre-bakes summaries for every (session, turn) that has at least one prior
turn (turn 1 cutoff = no history → skipped).

    uv run python -m mymodule.strategies.twotower.query_summary \\
        --dataset devset --provider openai --model google/gemma-4-26b-a4b-it
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUT = _MODULE_DIR / "data" / "query_summaries.jsonl"
SUMMARY_VERSION_PREFIX = "v1"

DEFAULT_PROVIDER: Provider = "openai"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0


if dspy is not None:

    class SessionSummary(dspy.Signature):
        """Compress a music-recommendation chat history into ONE sentence
        capturing the user's evolving intent and preference trajectory.

        Focus on concrete patterns: genre shifts, mood progressions, era
        preferences, artist/album anchors. Avoid bullets, numbering, or
        meta-commentary. The output is consumed verbatim as the body of a
        retrieval query's ``Session so far:`` line — write it so it reads
        naturally after that label. Max 25 words.
        """

        chat_history: str = dspy.InputField(
            desc=(
                "Prior user / music turns, line-separated. Most recent last. "
                "Music turns are rendered as 'music: \"Title\" by Artist'."
            )
        )
        goal_text: str = dspy.InputField(desc="The listener's stated goal as natural language (may be empty).")
        current_message: str = dspy.InputField(
            desc="The user's current-turn message — provided for context only. Do NOT echo it in the output."
        )
        session_summary: str = dspy.OutputField(
            desc=(
                "ONE sentence (≤25 words) describing the user's evolving preference / intent. "
                "Examples: 'user gravitated from 2000s indie alt-rock toward 90s grunge, hinting at "
                "a preference for heavier raw textures' | 'user is exploring upbeat Korean pop and "
                "consistently favors mid-2010s hits'."
            )
        )

else:  # pragma: no cover
    SessionSummary = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Cache key + IO
# ---------------------------------------------------------------------------


def summary_key(chat_history: list[dict] | None, user_msg: str) -> str:
    """SHA-256 hex (16 chars) over user_msg + chat_history content."""
    parts: list[str] = [user_msg or ""]
    for msg in chat_history or []:
        parts.append(f"{msg.get('turn_number', '')}:{msg.get('role', '')}:{msg.get('content', '')}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _version_stamp(provider: str, model: str, seed: int) -> str:
    slug = model.replace("/", "_").replace(":", "_").replace(".", "_")
    return f"{SUMMARY_VERSION_PREFIX}_{provider}_{slug}_seed{seed}"


def load_query_summaries(
    path: Path = DEFAULT_OUT,
    *,
    version: str | None = None,
) -> dict[str, str]:
    """Load JSONL → ``{cache_key: session_summary}``. Stale versions skipped."""
    out: dict[str, str] = {}
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
            k = obj.get("key")
            if not k:
                continue
            if version is not None and obj.get("version") != version:
                continue
            text = (obj.get("session_summary") or "").strip()
            if text:
                out[k] = text
    return out


# ---------------------------------------------------------------------------
# DSPy predictor
# ---------------------------------------------------------------------------


def _build_predictor(provider: Provider, seed: int, temperature: float) -> Any:
    if dspy is None or SessionSummary is None:
        raise RuntimeError("dspy required for query_summary. `uv add dspy`.")
    ensure_lm_configured(provider, seed=seed, temperature=temperature)
    pred = dspy.Predict(SessionSummary)
    if min_interval_sec() > 0:
        return RateLimitedPredictor(pred, provider)
    return pred


# ---------------------------------------------------------------------------
# Input formatting
# ---------------------------------------------------------------------------


def _render_chat_history(chat_history: list[dict] | None, kv: Any | None) -> str:
    """Render chat_history as line-based 'role: content' with music turns resolved."""
    if not chat_history:
        return ""
    from mymodule.strategies.twotower.text_compose import _first_str

    lines: list[str] = []
    for msg in chat_history:
        role = msg.get("role") or "?"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "music" and kv is not None:
            try:
                meta = kv.get_track_meta(content)
            except Exception:
                meta = None
            if meta:
                title = _first_str(meta.get("track_name"), "")
                artist = _first_str(meta.get("artist_name"), "")
                if title and artist:
                    content = f'"{title}" by {artist}'
                elif title:
                    content = f'"{title}"'
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def generate_one(
    predictor: Any,
    *,
    chat_history: list[dict] | None,
    user_msg: str,
    conversation_goal: dict | None,
    kv: Any | None,
) -> str:
    """One DSPy call → cleaned 1-sentence session summary."""
    rendered = _render_chat_history(chat_history, kv)
    goal_text = ""
    if conversation_goal and isinstance(conversation_goal, dict):
        goal_text = (conversation_goal.get("listener_goal") or "").strip()
    raw = predictor(
        chat_history=rendered or "(no prior turns)",
        goal_text=goal_text or "(none)",
        current_message=(user_msg or "").strip() or "(none)",
    )
    text = (getattr(raw, "session_summary", "") or "").strip()
    # Single-line normalize.
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Bulk pipeline
# ---------------------------------------------------------------------------


def _iter_dataset_turns(dataset: str, *, limit: int | None = None):
    """Yield (chat_history, user_msg, conversation_goal) for every cutoff turn.

    ``train`` (talkpl `train`) / ``devset`` (talkpl `test`): emit every turn
    (1..8) as a separate cutoff with its history slice — these feed the
    ``Session so far:`` line of both the training-pair queries (``extract_pairs``)
    and the devset eval queries, which must share the same summary cache.
    ``blindset_A`` / ``blindset_B``: emits the single cutoff per session.
    Turn 1 with no prior turns is skipped — there's nothing to summarize
    (Blind-B has many cold-start turn-1 sessions, so most are skipped).
    """
    from datasets import load_dataset

    path = {
        "train": ("talkpl-ai/TalkPlayData-Challenge-Dataset", "train"),
        "devset": ("talkpl-ai/TalkPlayData-Challenge-Dataset", "test"),
        "blindset_A": ("talkpl-ai/TalkPlayData-Challenge-Blind-A", "test"),
        "blindset_B": ("talkpl-ai/TalkPlayData-Challenge-Blind-B", "test"),
    }[dataset]
    ds = load_dataset(path[0], split=path[1])
    items = list(ds)
    if limit is not None:
        items = items[:limit]

    for item in items:
        conv = item.get("conversations") or []
        goal = item.get("conversation_goal")
        if dataset in ("blindset_A", "blindset_B"):
            if not conv:
                continue
            current = conv[-1]
            user_msg = (current.get("content") or "").strip()
            history = [
                {
                    "turn_number": int(m.get("turn_number", 0)),
                    "role": m.get("role"),
                    "content": m.get("content"),
                }
                for m in conv[:-1]
            ]
            if not history:
                continue
            yield history, user_msg, goal
        else:
            # devset — every turn becomes a cutoff candidate.
            by_turn: dict[int, list[dict]] = {}
            for row in conv:
                by_turn.setdefault(int(row.get("turn_number", 0)), []).append(row)
            turn_nums = sorted(by_turn.keys())
            for tn in turn_nums:
                user_msg = ""
                for m in by_turn[tn]:
                    if m.get("role") == "user":
                        user_msg = (m.get("content") or "").strip()
                        break
                if not user_msg:
                    continue
                history = [
                    {
                        "turn_number": int(m.get("turn_number", 0)),
                        "role": m.get("role"),
                        "content": m.get("content"),
                    }
                    for m in conv
                    if int(m.get("turn_number", 0)) < tn
                ]
                if not history:
                    continue
                yield history, user_msg, goal


def bulk_generate(
    *,
    dataset: str = "devset",
    out_path: Path = DEFAULT_OUT,
    provider: Provider = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE,
    limit: int | None = None,
    force: bool = False,
    concurrency: int = 1,
) -> None:
    """Pre-bake session summaries for every cutoff turn in ``dataset``.

    ``concurrency > 1`` uses a ThreadPoolExecutor (mirrors ``synth_doc``) — the
    DSPy/LiteLLM predictor is thread-safe and the JSONL writer is lock-guarded +
    flushed per record so the resume cache stays consistent on interruption.
    """
    if provider == "openai":
        os.environ["MYMODULE_LLM_OPENAI_MODEL"] = model
    else:
        os.environ["MYMODULE_LLM_MODEL"] = model

    version = _version_stamp(provider, model, seed)
    logger.info(f"[query-summary] version={version} dataset={dataset} provider={provider} model={model} seed={seed}")

    existing: dict[str, str] = {} if force else load_query_summaries(out_path, version=version)
    if existing:
        logger.info(f"[query-summary] resuming — {len(existing)} keys already cached")

    kv = None
    try:
        from mymodule.feature.kvdb import KVStore

        kv = KVStore.open(read_only=True)
    except Exception as e:
        logger.warning(f"[query-summary] KV unavailable ({type(e).__name__}: {e}); music turns render as raw tid.")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    predictor = _build_predictor(provider, seed=seed, temperature=temperature)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    failures = 0
    write_lock = threading.Lock()

    def _process(key: str, history, user_msg, goal) -> str | None:
        try:
            text = generate_one(
                predictor,
                chat_history=history,
                user_msg=user_msg,
                conversation_goal=goal,
                kv=kv,
            )
        except Exception as e:
            nonlocal failures
            with write_lock:
                failures += 1
                if failures <= 5:
                    logger.warning(f"[query-summary] failed key={key}: {type(e).__name__}: {e}")
            return None
        return text or None

    # Materialize pending turns on the main thread (resume-skip + intra-run dedup)
    # so workers only issue LLM calls.
    pending: list[tuple[str, list, str, dict]] = []
    seen_keys: set[str] = set()
    for history, user_msg, goal in _iter_dataset_turns(dataset, limit=limit):
        key = summary_key(history, user_msg)
        if key in existing or key in seen_keys:
            skipped += 1
            continue
        seen_keys.add(key)
        pending.append((key, history, user_msg, goal))
    logger.info(f"[query-summary] {len(pending)} turns queued ({skipped} already cached/dup, skipping)")

    def _emit(f, key: str, text: str | None) -> None:
        nonlocal written
        if not text:
            return
        obj = {"key": key, "version": version, "session_summary": text}
        with write_lock:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            log_now = written % 100 == 0
        if log_now:
            logger.info(f"[query-summary] progress: {written} written, {skipped} skipped, {failures} failed")

    with out_path.open("w" if force else "a") as f:
        if concurrency <= 1:
            for key, history, user_msg, goal in pending:
                _emit(f, key, _process(key, history, user_msg, goal))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                fut_to_key = {
                    ex.submit(_process, key, history, user_msg, goal): key for key, history, user_msg, goal in pending
                }
                for fut in as_completed(fut_to_key):
                    _emit(f, fut_to_key[fut], fut.result())

    logger.success(f"[query-summary] done: {written} written, {skipped} skipped, {failures} failed → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Bulk-generate 'Session so far:' summaries.")
    p.add_argument("--dataset", choices=["train", "devset", "blindset_A", "blindset_B"], default="devset")
    p.add_argument("--provider", choices=["ollama", "openai"], default=DEFAULT_PROVIDER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None, help="Process at most N sessions.")
    p.add_argument("--force", action="store_true", help="Ignore the resume cache.")
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel LLM workers (ThreadPoolExecutor). >1 for managed endpoints (vLLM/SGLang/MaaS).",
    )
    args = p.parse_args()
    bulk_generate(
        dataset=args.dataset,
        out_path=args.out,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        temperature=args.temperature,
        limit=args.limit,
        force=args.force,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
