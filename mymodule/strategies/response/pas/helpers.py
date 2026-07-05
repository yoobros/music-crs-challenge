"""PAS-specific input formatters — similarity hints consumed by `CRSResponse`.

Each hint function corresponds 1:1 to an `InputField` on the PAS signature
(`track_similarity_hints`, `query_similarity_hints`, `lyric_similarity_hints`).
Other variants may or may not use similar signals, so these live under the
PAS subpackage rather than the shared `base_dspy.py`.

All formatters gracefully return `""` when their dependencies (Ollama, KV
store, prior listens) are unavailable.
"""

from __future__ import annotations

from typing import Any


def _grade_label(sim: float) -> str:
    """Map cosine similarity to a 3-tier adjective so the LLM sees a coarse
    coarse relevance-strength signal instead of a bare float.

    Bands are calibrated for `metadata_rich-qwen3` cosine distributions:
      - strong   ≥ 0.65  (clear semantic alignment)
      - moderate ≥ 0.45  (partial overlap, plausible match)
      - weak     ≥ 0.25  (above noise floor but tenuous)
      - <0.25 returns "weak" too — by the time `_grade_label` is called we
        already gated by the helper's `sim_threshold`, so anything reaching
        here is at least worth surfacing as weak.
    """
    if sim >= 0.65:
        return "strong"
    if sim >= 0.45:
        return "moderate"
    return "weak"


def fmt_track_similarity_hints(
    track_ids: list[str],
    top_n: int,
    chat_history: list[dict] | None,
    kv: Any,
    sim_threshold: float = 0.45,
    *,
    graded: bool = False,
    target_vector_type: str = "metadata_rich-qwen3_embedding_0.6b",
) -> str:
    """For each top-N recommended track, find the most similar previously played track.

    Default format: `1: similar to your previous listen "PrevTitle"`.
    With `graded=True`: `1: strong continuity with prior listen "PrevTitle"` —
    a 3-tier band label (`strong` / `moderate` / `weak`) replaces the bare
    cosine score so the LLM can see relevance strength at a glance.
    The numeric score is NOT exposed; raw floats encourage spurious precision.

    Vector space: `metadata_rich-qwen3_embedding_0.6b` (1024d) — same space the
    qemb retrieval pool uses, so prior↔candidate similarity reflects semantic
    track metadata closeness (artist/album/era/tags) rather than pure collab
    co-listen pattern. Override via `target_vector_type` if needed.

    Returns `""` if no chat history / KV / qualifying matches.
    """
    if not chat_history or kv is None:
        return ""
    try:
        import numpy as np

        prev_ids = [m["content"] for m in chat_history if m.get("role") == "music"]
        if not prev_ids:
            return ""

        def _load(tid: str) -> "np.ndarray | None":  # type: ignore[name-defined]
            try:
                return kv.get_track_embedding(tid, target_vector_type)
            except Exception:
                return None

        prev_vecs = [(pid, _load(pid)) for pid in prev_ids]
        prev_vecs = [(pid, v) for pid, v in prev_vecs if v is not None]
        if not prev_vecs:
            return ""

        hints: list[str] = []
        for i, tid in enumerate(track_ids[:top_n], 1):
            rec_vec = _load(tid)
            if rec_vec is None:
                continue
            rec_norm = rec_vec / (np.linalg.norm(rec_vec) + 1e-9)
            best_sim, best_pid = 0.0, None
            for pid, pv in prev_vecs:
                pv_norm = pv / (np.linalg.norm(pv) + 1e-9)
                sim = float(np.dot(rec_norm, pv_norm))
                if sim > best_sim:
                    best_sim, best_pid = sim, pid
            if best_sim >= sim_threshold and best_pid:
                prev_meta = kv.get_track_meta(best_pid)
                prev_title = _first(prev_meta.get("track_name")) if prev_meta else best_pid[:8]
                if graded:
                    hints.append(f'{i}: {_grade_label(best_sim)} continuity with prior listen "{prev_title}"')
                else:
                    hints.append(f'{i}: similar to your previous listen "{prev_title}"')
        return "\n".join(hints)
    except Exception:
        return ""


def fmt_query_similarity_hints(
    track_ids: list[str],
    top_n: int,
    user_query: str,
    kv: Any,
    sim_threshold: float = 0.30,
    *,
    chat_history: list[dict] | None = None,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    goal_progress_assessments: Any = None,
    thought: str | None = None,
    graded: bool = False,
    target_vector_type: str = "metadata_rich-qwen3_embedding_0.6b",
) -> str:
    """For each top-N recommended track, find semantic relevance to user's query.

    Embeds the query via the shared `InstructedQueryEmbedder` (qwen3-embedding,
    1024d). The query is composed in a doc-style metadata format that aligns
    naturally with `metadata_rich-qwen3_embedding_0.6b` track embeddings —
    same vector space the qemb retrieval pool searches against. Override via
    `target_vector_type` if needed.

    With `graded=True`: emits `1: strong relevance to your query` — a 3-tier
    band (`strong` / `moderate` / `weak`) so the LLM sees relevance strength
    without a bare cosine score. Default sim_threshold lowered to 0.30
    (was 0.55) — query-vs-metadata cosine peaks lower than track-vs-track and
    the prior threshold left this field empty on most turn-1 demos.
    """
    if not user_query or kv is None:
        return ""
    try:
        import numpy as np

        from mymodule.feature.ollama_embed import get_shared_instructed_embedder

        query_vec = get_shared_instructed_embedder().embed_query(
            user_query,
            chat_history=chat_history,
            user_profile=user_profile,
            conversation_goal=conversation_goal,
            goal_progress_assessments=goal_progress_assessments,
            thought=thought,
        )
        if query_vec is None or len(query_vec) == 0:
            return ""

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        hints: list[str] = []

        for i, tid in enumerate(track_ids[:top_n], 1):
            try:
                track_vec = kv.get_track_embedding(tid, target_vector_type)
                if track_vec is None or track_vec.shape[-1] != query_norm.shape[-1]:
                    continue
                track_norm = track_vec / (np.linalg.norm(track_vec) + 1e-9)
                sim = float(np.dot(query_norm, track_norm))
                if sim >= sim_threshold:
                    if graded:
                        hints.append(f"{i}: {_grade_label(sim)} relevance to your query")
                    else:
                        hints.append(f"{i}: relevant to your query")
            except Exception:
                continue

        return "\n".join(hints)
    except Exception:
        return ""


def fmt_lyric_similarity_hints(
    track_ids: list[str],
    top_n: int,
    user_query: str,
    kv: Any,
    sim_threshold: float = 0.25,
    *,
    chat_history: list[dict] | None = None,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    goal_progress_assessments: Any = None,
    thought: str | None = None,
    graded: bool = False,
) -> str:
    """For each top-N track, find lyric relevance to user's query.

    Embeds the query via the shared `InstructedQueryEmbedder` so the same
    vector is reused across retrieval pool and response helpers (cache hit
    when qemb pool already ran for this turn). Target space is
    lyrics-qwen3_embedding_0.6b (1024d) — dimensionally aligned with the
    instructed query vector.

    With `graded=True`: emits `1: strong lyrical resonance` — 3-tier label
    (strong / moderate / weak) instead of a numeric score. Default threshold
    lowered to 0.25 (was 0.40) since lyric-vs-query cosine sits lower than
    metadata-vs-query, and the prior floor left this field empty on most demos.
    """
    if not user_query or kv is None:
        return ""
    try:
        import numpy as np

        from mymodule.feature.ollama_embed import get_shared_instructed_embedder

        query_vec = get_shared_instructed_embedder().embed_query(
            user_query,
            chat_history=chat_history,
            user_profile=user_profile,
            conversation_goal=conversation_goal,
            goal_progress_assessments=goal_progress_assessments,
            thought=thought,
        )
        if query_vec is None or len(query_vec) == 0:
            return ""

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        hints: list[str] = []

        for i, tid in enumerate(track_ids[:top_n], 1):
            try:
                lyric_vec = kv.get_track_embedding(tid, "lyrics-qwen3_embedding_0.6b")
                if lyric_vec is None:
                    continue
                lyric_norm = lyric_vec / (np.linalg.norm(lyric_vec) + 1e-9)
                sim = float(np.dot(query_norm, lyric_norm))
                if sim >= sim_threshold:
                    if graded:
                        hints.append(f"{i}: {_grade_label(sim)} lyrical resonance with your request")
                    else:
                        hints.append(f"{i}: lyrically similar to your request")
            except Exception:
                continue

        return "\n".join(hints)
    except Exception:
        return ""


def _first(values: Any, fallback: str = "unknown") -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str) and values:
        return values
    return fallback
