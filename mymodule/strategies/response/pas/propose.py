"""Deterministic PROPOSE stage — evidence-typed intent classification.

CRS framing: each recommended track in the top-20 is assigned to exactly ONE intent
group based on which explanation channel most strongly justifies it *on this turn*.
Priority (highest wins): query_aligned > lyric_resonant > taste_continuous >
pool_coherent > discovery.

The structured output is consumed by the DSPy signature as `intent_groups` input —
the LLM's ASSIGN+SELECT stages write per-group rationales grounded in that group's
evidence channel, which gives LLM-Judge (Explanation Quality) verifiable signal and
naturally diversifies vocabulary across groups (Distinct-2).

All embedding reads reuse the existing KVStore + OllamaEmbedder that power the
similarity-hint helpers in `_dspy_common.py`. No new precompute required.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


EVIDENCE_ORDER: tuple[str, ...] = (
    "query_aligned",
    "lyric_resonant",
    "taste_continuous",
    "pool_coherent",
    "discovery",
)


def _norm(vec: "np.ndarray") -> "np.ndarray":
    return vec / (np.linalg.norm(vec) + 1e-9)


def _first(values: Any, fallback: str = "unknown") -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str) and values:
        return values
    return fallback


def _maybe_embed_query(
    user_query: str,
    *,
    chat_history: list[dict] | None = None,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    goal_progress_assessments: Any = None,
    thought: str | None = None,
) -> "np.ndarray | None":
    """Return a normalized instructed query embedding, or None on failure.

    Uses the shared `InstructedQueryEmbedder` so the vector matches qemb
    pool's cached entry when available (cache key derives from the context
    kwargs — pass them through from `classify_intents`).
    """
    if not user_query or np is None:
        return None
    try:
        from mymodule.feature.ollama_embed import get_shared_instructed_embedder

        vec = get_shared_instructed_embedder().embed_query(
            user_query,
            chat_history=chat_history,
            user_profile=user_profile,
            conversation_goal=conversation_goal,
            goal_progress_assessments=goal_progress_assessments,
            thought=thought,
        )
        if vec is None or len(vec) == 0:
            return None
        return _norm(vec)
    except Exception:
        return None


def _derive_cluster_descriptor(track_ids: list[str], kv: Any, max_tracks: int = 20) -> str:
    """Compact descriptor of a cluster: 'era 1993-1996 | tags: grunge, rock'."""
    tag_counts: Counter[str] = Counter()
    years: list[int] = []
    artist_counts: Counter[str] = Counter()

    for tid in track_ids[:max_tracks]:
        meta = kv.get_track_meta(tid) if kv is not None else None
        if not meta:
            continue
        release_date = _first(meta.get("release_date"))
        if release_date and len(release_date) >= 4 and release_date[:4].isdigit():
            years.append(int(release_date[:4]))
        for tag in (meta.get("tag_list") or [])[:10]:
            if tag:
                tag_counts[str(tag)] += 1
        artist = _first(meta.get("artist_name"))
        if artist and artist != "unknown":
            artist_counts[artist] += 1

    parts: list[str] = []
    if years:
        parts.append(f"era {min(years)}-{max(years)}")
    if tag_counts:
        top_tags = [t for t, _ in tag_counts.most_common(3)]
        parts.append("tags: " + ", ".join(top_tags))
    if artist_counts:
        dominant = [f"{a}×{c}" for a, c in artist_counts.most_common(2) if c >= 2]
        if dominant:
            parts.append("dominant: " + ", ".join(dominant))
    return " | ".join(parts) if parts else "stylistically adjacent"


def _best_prior_title(prev_match: dict[int, str], kv: Any) -> str | None:
    """Of the prior tracks used as taste anchors, name the most-referenced one."""
    if not prev_match or kv is None:
        return None
    counter = Counter(prev_match.values())
    top_pid, _ = counter.most_common(1)[0]
    meta = kv.get_track_meta(top_pid)
    if not meta:
        return None
    title = _first(meta.get("track_name"), fallback="")
    artist = _first(meta.get("artist_name"), fallback="")
    if title and artist and artist != "unknown":
        return f"{title} by {artist}"
    return title or None


def classify_intents(
    track_ids: list[str],
    user_query: str,
    chat_history: list[dict] | None,
    kv: Any,
    top_n: int = 20,
    query_threshold: float | None = None,
    lyric_threshold: float | None = None,
    taste_threshold: float | None = None,
    pool_threshold: float | None = None,
    *,
    user_profile: dict | None = None,
    conversation_goal: dict | None = None,
    goal_progress_assessments: Any = None,
    thought: str | None = None,
) -> dict[str, dict]:
    """Classify each of the top-`top_n` tracks into an evidence-typed intent group.

    Returns: ``{evidence_type: {"indices": [0-based ints], "evidence_summary": str}}``.
    Only groups with at least 1 assigned index are present. At most 5 groups.
    """
    if np is None or kv is None or not track_ids:
        return {}

    # Thresholds calibrated on the observed empirical distributions:
    #   query (qwen3 1024d) vs track metadata-qwen3: median ~0.60 on BM25 pool
    #   query vs lyrics-qwen3 (1024d):                max ~0.54 on BM25 pool
    #   track vs track cf-bpr (128d): p90 ~0.13, max ~0.23 — centered near 0
    query_threshold = (
        query_threshold if query_threshold is not None else float(os.getenv("MYMODULE_PAS_QUERY_THRESHOLD", "0.60"))
    )
    lyric_threshold = (
        lyric_threshold if lyric_threshold is not None else float(os.getenv("MYMODULE_PAS_LYRIC_THRESHOLD", "0.45"))
    )
    taste_threshold = (
        taste_threshold if taste_threshold is not None else float(os.getenv("MYMODULE_PAS_TASTE_THRESHOLD", "0.20"))
    )
    pool_threshold = (
        pool_threshold if pool_threshold is not None else float(os.getenv("MYMODULE_PAS_POOL_THRESHOLD", "0.15"))
    )

    tracks = track_ids[:top_n]
    n = len(tracks)

    # Load embeddings once (may be None individually)
    def _load(tid: str, emb_type: str) -> "np.ndarray | None":
        try:
            v = kv.get_track_embedding(tid, emb_type)
            return _norm(v) if v is not None else None
        except Exception:
            return None

    # cf-bpr (128d) for track↔track similarity (taste + pool clustering).
    # metadata-qwen3 & lyrics-qwen3 (1024d) match the query embedder space → query↔track.
    cfbpr = [_load(t, "cf-bpr") for t in tracks]
    metadata = [_load(t, "metadata-qwen3_embedding_0.6b") for t in tracks]
    lyrics = [_load(t, "lyrics-qwen3_embedding_0.6b") for t in tracks]
    q = _maybe_embed_query(
        user_query,
        chat_history=chat_history,
        user_profile=user_profile,
        conversation_goal=conversation_goal,
        goal_progress_assessments=goal_progress_assessments,
        thought=thought,
    )

    assigned: dict[int, str] = {}
    groups: dict[str, dict] = {}

    # 1. query_aligned — user's current-turn intent, strongest conversational signal.
    # Compared against track metadata embedding (title + artist + tags in qwen3 space).
    if q is not None:
        idxs, sims = [], []
        for i, v in enumerate(metadata):
            if v is None:
                continue
            s = float(np.dot(q, v))
            if s >= query_threshold:
                idxs.append(i)
                sims.append(s)
        if idxs:
            for i in idxs:
                assigned[i] = "query_aligned"
            avg_sim = sum(sims) / len(sims)
            qsnippet = user_query.strip().replace("\n", " ")
            qsnippet = qsnippet[:48] + "…" if len(qsnippet) > 48 else qsnippet
            groups["query_aligned"] = {
                "indices": idxs,
                "evidence_summary": f'embed close to your request "{qsnippet}" (avg sim {avg_sim:.2f})',
            }

    # 2. lyric_resonant — thematic match via lyrics embedding
    if q is not None:
        idxs, sims = [], []
        for i, v in enumerate(lyrics):
            if i in assigned or v is None:
                continue
            s = float(np.dot(q, v))
            if s >= lyric_threshold:
                idxs.append(i)
                sims.append(s)
        if idxs:
            for i in idxs:
                assigned[i] = "lyric_resonant"
            avg_sim = sum(sims) / len(sims)
            groups["lyric_resonant"] = {
                "indices": idxs,
                "evidence_summary": f"lyrics echo your phrasing (avg sim {avg_sim:.2f})",
            }

    # 3. taste_continuous — close to what the user already played in this conversation
    prev_ids: list[str] = []
    if chat_history:
        prev_ids = [m["content"] for m in chat_history if m.get("role") == "music" and m.get("content")]
    if prev_ids:
        prev_vecs: list[tuple[str, "np.ndarray"]] = []
        for pid in prev_ids:
            pv = _load(pid, "cf-bpr")
            if pv is not None:
                prev_vecs.append((pid, pv))
        if prev_vecs:
            idxs: list[int] = []
            best_ref: dict[int, str] = {}
            for i, v in enumerate(cfbpr):
                if i in assigned or v is None:
                    continue
                best_s, best_pid = 0.0, None
                for pid, pv in prev_vecs:
                    s = float(np.dot(v, pv))
                    if s > best_s:
                        best_s, best_pid = s, pid
                if best_pid is not None and best_s >= taste_threshold:
                    idxs.append(i)
                    best_ref[i] = best_pid
            if idxs:
                for i in idxs:
                    assigned[i] = "taste_continuous"
                anchor = _best_prior_title(best_ref, kv)
                anchor_str = f' anchored on "{anchor}"' if anchor else ""
                groups["taste_continuous"] = {
                    "indices": idxs,
                    "evidence_summary": f"cf-bpr-close to your earlier picks{anchor_str}",
                }

    # 4. pool_coherent — connected components on remaining (unclassified) tracks
    remaining = [i for i in range(n) if i not in assigned]
    if len(remaining) >= 2:
        adj: dict[int, set[int]] = {i: set() for i in remaining}
        for a_idx, i in enumerate(remaining):
            vi = cfbpr[i]
            if vi is None:
                continue
            for j in remaining[a_idx + 1 :]:
                vj = cfbpr[j]
                if vj is None:
                    continue
                if float(np.dot(vi, vj)) >= pool_threshold:
                    adj[i].add(j)
                    adj[j].add(i)
        visited: set[int] = set()
        components: list[list[int]] = []
        for i in remaining:
            if i in visited:
                continue
            comp: list[int] = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                stack.extend(nb for nb in adj[node] if nb not in visited)
            components.append(sorted(comp))
        pool_idxs: list[int] = []
        for comp in components:
            if len(comp) >= 2:
                pool_idxs.extend(comp)
        if pool_idxs:
            pool_idxs.sort()
            for i in pool_idxs:
                assigned[i] = "pool_coherent"
            descriptor = _derive_cluster_descriptor([tracks[i] for i in pool_idxs], kv)
            groups["pool_coherent"] = {
                "indices": pool_idxs,
                "evidence_summary": f"cf-bpr-adjacent cluster — {descriptor}",
            }

    # 5. discovery — everything still unclassified
    discovery_idxs = sorted(i for i in range(n) if i not in assigned)
    if discovery_idxs:
        groups["discovery"] = {
            "indices": discovery_idxs,
            "evidence_summary": "lateral picks outside the anchored groups",
        }

    return groups


def greedy_coverage(groups: dict[str, dict], max_groups: int = 5) -> dict[str, dict]:
    """Enforce ≤ max_groups groups, keeping the ones covering the most tracks.

    Ties broken by EVIDENCE_ORDER priority (conversational relevance). Any tracks
    whose owning group is dropped are re-absorbed into `discovery` so the partition
    stays exhaustive (every track still belongs to a group).
    """
    if len(groups) <= max_groups:
        return groups

    scored = sorted(
        groups.items(),
        key=lambda kv: (
            -len(kv[1].get("indices", [])),
            EVIDENCE_ORDER.index(kv[0]) if kv[0] in EVIDENCE_ORDER else 99,
        ),
    )
    keep_keys = {k for k, _ in scored[:max_groups]}
    dropped: list[int] = []
    kept: dict[str, dict] = {}
    for k, v in groups.items():
        if k in keep_keys:
            kept[k] = v
        else:
            dropped.extend(v.get("indices", []))

    if dropped:
        if "discovery" in kept:
            merged = sorted(set(kept["discovery"]["indices"]) | set(dropped))
            kept["discovery"] = {
                **kept["discovery"],
                "indices": merged,
            }
        else:
            # promote dropped → discovery; rare edge case if discovery wasn't in top-max
            kept["discovery"] = {
                "indices": sorted(set(dropped)),
                "evidence_summary": "lateral picks outside the anchored groups",
            }
    return kept


def format_intent_groups_for_prompt(groups: dict[str, dict]) -> str:
    """Render groups as a newline-delimited string for the DSPy input field.

    Track indices are 1-based here to match `fmt_recommended_tracks*` numbering.
    """
    if not groups:
        return "(no groups — empty recommendation pool)"
    lines: list[str] = []
    for etype in EVIDENCE_ORDER:
        if etype not in groups:
            continue
        g = groups[etype]
        idx_str = ", ".join(str(i + 1) for i in g["indices"])
        lines.append(f"{etype} ({len(g['indices'])} tracks): tracks [{idx_str}]")
        ev = g.get("evidence_summary", "")
        if ev:
            lines.append(f"  → evidence: {ev}")
    return "\n".join(lines)


def evidence_type_of_track(groups: dict[str, dict], track_index_0based: int) -> str | None:
    """Reverse lookup: which group does a given 0-based track index belong to?"""
    for etype, g in groups.items():
        if track_index_0based in g.get("indices", []):
            return etype
    return None
