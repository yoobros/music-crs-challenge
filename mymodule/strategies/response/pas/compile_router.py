"""Hierarchical few-shot demo retrieval for PAS conditional compilation.

`DemoRouter` is loaded from a sidecar JSON written by `pas/optimize.py` next to
the standard DSPy ckpt (`pas_{provider}.buckets.json`). At inference time, the
PAS generator uses it to pick demos that match the current session's
`(category, specificity, turn_position)` bucket — falling back through a
quality-preserving chain
when an exact bucket has too few demos.

Bucket axes (kept narrow on purpose — see plan.md):
- `category` — from `conversation_goal["category"]`, preserved as a soft route
  key. Sparse categories fall back immediately to specificity/turn routing.
- `specificity` ∈ {"HH","HL","LH","LL"} — from `conversation_goal["specificity"]`
- `turn_position` ∈ {"first","mid","final"} — derived from current turn index
  (1 → "first", 8 → "final", else "mid")

Fallback chain (first hit wins, subsequent layers de-dup against earlier picks):
  1. exact `(category, spec, turn)` match
  2. `(category, spec, *)`
  3. `(category, *, turn)`
  4. `(category, *, *)`
  5. exact `(spec, turn)` match
  6. `(spec, *)`
  7. `(*, turn)`
  8. global top-N (by judge score, descending)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mymodule.utils.similarity import cosine_topk, l2_normalize, stack_normalized

_VALID_SPECIFICITIES = frozenset({"HH", "HL", "LH", "LL"})
_VALID_TURNS = frozenset({"first", "mid", "final"})


def turn_position_from_index(turn_number: int | None) -> str:
    """Map 1-based turn index to coarse position bucket. None → 'mid'."""
    if turn_number is None:
        return "mid"
    if turn_number <= 1:
        return "first"
    if turn_number >= 8:
        return "final"
    return "mid"


def turn_number_from_chat_history(chat_history: list[dict] | None) -> int:
    """Estimate the *current* (in-progress) turn index from the chat_history list.

    The chat_history passed to a generator contains messages with `turn_number < target`.
    Current turn is therefore `max(turn_number) + 1`, clamped to [1, 8].
    Empty chat_history → turn 1 (cold-start).
    """
    if not chat_history:
        return 1
    nums: list[int] = []
    for m in chat_history:
        if not isinstance(m, dict):
            continue
        try:
            nums.append(int(m.get("turn_number", 0) or 0))
        except (TypeError, ValueError):
            continue
    if not nums:
        return 1
    return max(1, min(8, max(nums) + 1))


class DemoRouter:
    """In-memory hierarchical demo store keyed by (specificity, turn_position).

    Demos are stored as simple dicts (`inputs`, `outputs`, `meta`) so the router
    is decoupled from `dspy.Example` — conversion to Example happens in the
    generator at runtime.

    Two retrieval modes are supported, depending on what the bucket sidecar
    carries:

    1. **Bucket mode** (`retrieve(...)`) — hierarchical fallback by
       `(specificity, turn_position)`. Always available; no embeddings needed.

    2. **KNN+score mode** (`retrieve_knn(...)`) — available only when every
       demo's `meta` carries `query_embedding: list[float]`. Caller embeds the
       inference-time query with the same embedder used at compile, then
       receives the top-K demos ranked by a blend of cosine similarity AND
       judge/lexdiv score (`alpha * sim + (1 - alpha) * norm_score`). Combines
       relevance (kNN narrows to similar sessions) with quality (metric breaks
       ties among similar candidates).
    """

    def __init__(self, demos: list[dict[str, Any]]) -> None:
        # Sort once by score desc so every retrieval slice respects ordering.
        self._all: list[dict[str, Any]] = sorted(demos, key=lambda d: -float((d.get("meta") or {}).get("score", 0.0)))
        self._cat_spec_turn: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self._cat_spec: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._cat_turn: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._cat_only: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._spec_only: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._turn_only: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in self._all:
            meta = d.get("meta") or {}
            cat = str(meta.get("category", "") or "")
            spec = str(meta.get("specificity", "") or "")
            turn = str(meta.get("turn_position", "") or "")
            if cat:
                self._cat_spec_turn[(cat, spec, turn)].append(d)
                self._cat_spec[(cat, spec)].append(d)
                self._cat_turn[(cat, turn)].append(d)
                self._cat_only[cat].append(d)
            self._exact[(spec, turn)].append(d)
            self._spec_only[spec].append(d)
            self._turn_only[turn].append(d)
        # KNN side state — built lazily on first call so a router with NO
        # embeddings stays cheap.
        self._embeddings: np.ndarray | None = None
        self._embedding_dim: int | None = None
        self._embedding_model: str | None = None
        self._scores_norm: np.ndarray | None = None
        self._knn_built: bool = False

    @classmethod
    def empty(cls) -> "DemoRouter":
        return cls(demos=[])

    @classmethod
    def from_sidecar(cls, path: str | Path) -> "DemoRouter":
        raw = json.loads(Path(path).read_text())
        demos = raw.get("demos", []) if isinstance(raw, dict) else []
        return cls(demos=demos)

    def __len__(self) -> int:
        return len(self._all)

    def bucket_counts(self) -> dict[tuple[str, str], int]:
        return {k: len(v) for k, v in self._exact.items()}

    def retrieve(
        self,
        specificity: str,
        turn_position: str,
        top_n: int = 3,
        *,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to `top_n` demos ranked by fallback chain.

        Layers later in the chain only contribute demos not already returned by
        earlier layers (de-duplicated by object identity, so the same demo never
        appears twice).
        """
        if top_n <= 0 or not self._all:
            return []
        cat = str(category or "")
        spec = str(specificity or "")
        turn = str(turn_position or "")
        chain: list[list[dict[str, Any]]] = []
        if cat:
            chain.extend(
                [
                    self._cat_spec_turn.get((cat, spec, turn), []),
                    self._cat_spec.get((cat, spec), []),
                    self._cat_turn.get((cat, turn), []),
                    self._cat_only.get(cat, []),
                ]
            )
        chain.extend(
            [
                self._exact.get((spec, turn), []),
                self._spec_only.get(spec, []),
                self._turn_only.get(turn, []),
                self._all,
            ]
        )
        out: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for src in chain:
            for d in src:
                key = id(d)
                if key in seen_ids:
                    continue
                out.append(d)
                seen_ids.add(key)
                if len(out) >= top_n:
                    return out
        return out

    # ---- KNN+score retrieval ------------------------------------------------

    def _build_knn_index(self) -> None:
        """Stack per-demo query embeddings into an L2-normalized matrix.

        Sets `self._embeddings` to None when any demo is missing
        `meta.query_embedding`, signaling that KNN retrieval is unavailable —
        callers should fall back to `retrieve(...)`. Idempotent.
        """
        if self._knn_built:
            return
        self._knn_built = True
        if not self._all:
            return
        vecs: list[np.ndarray] = []
        models: set[str] = set()
        for d in self._all:
            meta = d.get("meta") or {}
            emb = meta.get("query_embedding")
            if not emb:
                self._embeddings = None
                return
            try:
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                self._embeddings = None
                return
            vecs.append(vec)
            mid = meta.get("embedding_model")
            if mid:
                models.add(str(mid))
        if len({v.shape[0] for v in vecs}) != 1:
            # Dim mismatch — sidecar is inconsistent; treat as unavailable.
            self._embeddings = None
            return
        self._embeddings = stack_normalized(vecs)
        self._embedding_dim = int(self._embeddings.shape[1])
        self._embedding_model = next(iter(models)) if len(models) == 1 else None
        # Normalize judge/lexdiv scores to [0, 1] for the re-rank blend. Scores
        # come from `meta.score` (compile metric), already roughly in [0, 1].
        raw = np.asarray(
            [float((d.get("meta") or {}).get("score", 0.0)) for d in self._all],
            dtype=np.float32,
        )
        if raw.size > 0:
            lo, hi = float(raw.min()), float(raw.max())
            if hi > lo:
                self._scores_norm = (raw - lo) / (hi - lo)
            else:
                self._scores_norm = np.zeros_like(raw)
        else:
            self._scores_norm = np.zeros(0, dtype=np.float32)

    def has_knn_index(self) -> bool:
        """True iff every demo carries a `meta.query_embedding` (post lazy build)."""
        self._build_knn_index()
        return self._embeddings is not None and self._embeddings.size > 0

    @property
    def embedding_dim(self) -> int | None:
        self._build_knn_index()
        return self._embedding_dim

    @property
    def embedding_model(self) -> str | None:
        self._build_knn_index()
        return self._embedding_model

    def retrieve_knn(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        fetch_m: int = 10,
        alpha: float = 0.7,
        *,
        specificity: str | None = None,
        turn_position: str | None = None,
        category: str | None = None,
        category_bonus: float = 0.06,
        specificity_bonus: float = 0.08,
        turn_bonus: float = 0.03,
    ) -> list[dict[str, Any]]:
        """Return up to `top_k` demos using kNN + score re-rank.

        Pipeline:
        1. Cosine-similarity top-`fetch_m` against the demo pool (caller must
           supply an embedding produced by the same embedder used at compile).
        2. Re-rank those `fetch_m` by `alpha * cosine + (1 - alpha) * norm_score`
           plus small metadata-match bonuses, then take top `top_k`.

        `alpha=1.0` collapses to pure kNN; `alpha=0.0` collapses to pure
        score-ranking *within* the kNN-fetched window (still narrower than the
        whole pool). Default `alpha=0.7` leans on relevance while letting the
        metric break ties among similar candidates.

        Returns `[]` when no KNN index is available — callers should fall back
        to `retrieve(...)`.
        """
        if top_k <= 0:
            return []
        self._build_knn_index()
        if self._embeddings is None or self._scores_norm is None:
            return []
        n = self._embeddings.shape[0]
        if n == 0:
            return []
        m = min(max(fetch_m, top_k), n)
        q = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(-1))
        if q.shape[0] != self._embeddings.shape[1]:
            # Dim mismatch between query and pool — fall back, caller will use bucket mode.
            return []
        idx, sims = cosine_topk(q, self._embeddings, m)
        if idx.size == 0:
            return []
        a = float(np.clip(alpha, 0.0, 1.0))
        # Re-rank the fetched window by blended score. Metadata bonuses are
        # intentionally small: they break ties toward same category/spec/turn
        # without defeating semantic nearest-neighbor relevance.
        blended = a * sims + (1.0 - a) * self._scores_norm[idx]
        cat = str(category or "")
        spec = str(specificity or "")
        turn = str(turn_position or "")
        if cat or spec or turn:
            bonuses = np.zeros_like(blended, dtype=np.float32)
            for pos, demo_idx in enumerate(idx.tolist()):
                meta = self._all[demo_idx].get("meta") or {}
                if cat and str(meta.get("category", "") or "") == cat:
                    bonuses[pos] += float(category_bonus)
                if spec and str(meta.get("specificity", "") or "") == spec:
                    bonuses[pos] += float(specificity_bonus)
                if turn and str(meta.get("turn_position", "") or "") == turn:
                    bonuses[pos] += float(turn_bonus)
            blended = blended + bonuses
        order_within = np.argsort(-blended)
        picked = idx[order_within][: min(top_k, idx.size)]
        return [self._all[i] for i in picked.tolist()]


__all__ = [
    "DemoRouter",
    "turn_position_from_index",
    "turn_number_from_chat_history",
]
