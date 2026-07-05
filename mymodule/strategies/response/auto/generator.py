"""AutoResponseGenerator — multi-stage DSPy pipeline tuned for the Gemini
LLM-as-a-Judge evaluator.

Composite Score (from evaluation framework screenshot):
    0.50·nDCG@20/20 + 0.10·CatalogDiv + 0.10·Distinct-2 + 0.30·LLM-Judge

This generator affects the last two terms. It exists alongside `pas/` so
we can compare head-to-head; logic is selected via
`--response-gen {noop,pas,auto}`.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from mymodule.strategies.response.auto.module import AutoResponseModule
from mymodule.strategies.response.auto.signature import ResponseComposition
from mymodule.strategies.response.base_dspy import (
    DspyResponseGenerator,
    fmt_goal_progress,
    fmt_recommended_tracks,
    fmt_tracks_overview,
    try_open_kvstore,
)
from mymodule.strategies.response.pas.helpers import (
    fmt_query_similarity_hints,
    fmt_track_similarity_hints,
)
from mymodule.strategies.response.pas.select import apply_hard_bans, resolve_title_citations
from mymodule.utils.common_dspy import (
    Provider,
    fmt_chat_history,
    fmt_conversation_goal,
    fmt_user_profile,
)


def _env_flag(name: str) -> bool:
    """Truthy parser for env toggles: `1`, `true`, `yes`, `on` (case-insensitive)."""
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


class AutoResponseGenerator(DspyResponseGenerator):
    """3-stage response generator focused on LLM-Judge + Distinct-2.

    Pipeline (see `module.py`):
      1. PersonalizationAnalysis (ChainOfThought)
      2. TrackExplanationPlan     (ChainOfThought)
      3. ResponseComposition      (Predict)

    Post-processing reuses PAS's hard-ban substitution and title citation
    resolver. Soft-ban retry is deliberately skipped — the retry would
    re-run the entire 3-stage pipeline (3 more LM calls), so we trust the
    in-signature `response_excluded_patterns` pre-commit to keep the first
    draft clean and let hard bans mop up any globally-overused phrases.

    Rich-input toggles (default off — keep baseline prompt unchanged)
    ----------------------------------------------------------------
    - `MYMODULE_RESPONSE_RICH=1` enables popularity in detailed-track
      lines, graded similarity hints, and goal_progress in `listener_goal`.
      Mirrors the "rich metadata" approach on the response side.
    - `MYMODULE_RESPONSE_CHAT_MUSIC_ONLY=1` drops natural-language assistant
      messages from `chat_history`, keeping only `user` queries and `music`
      track lists. Mirrors the BM25 query-composition approach.
    """

    # `signature` satisfies the parent contract (existence check in __init__),
    # even though the actual predictor is a multi-stage Module.
    signature = ResponseComposition
    # Compiled by `mymodule.strategies.response.auto.optimize` →
    # `ckpt/auto_{provider}.json`. Loaded lazily by `_build_predictor`.
    ckpt_basename = "auto"

    def __init__(self, provider: Provider = "ollama", **kwargs: Any) -> None:
        super().__init__(provider=provider, **kwargs)
        self._kv = try_open_kvstore()
        self._top_tracks = int(os.getenv("MYMODULE_LLM_TOP_TRACKS", "5"))
        self._overview_n = int(os.getenv("MYMODULE_LLM_OVERVIEW_TRACKS", "20"))
        self._rich = _env_flag("MYMODULE_RESPONSE_RICH")
        self._music_only_chat = _env_flag("MYMODULE_RESPONSE_CHAT_MUSIC_ONLY")

    def _build_predictor(self) -> Any:
        # Override parent: return the multi-stage Module directly, WITHOUT
        # the external `RateLimitedPredictor` wrapper — the Module applies
        # `apply_rate_limit` itself between stages so each LM sub-call is
        # individually rate-limited.
        module = AutoResponseModule(provider=self.provider)
        ckpt = self._ckpt_path()
        if ckpt is not None:
            try:
                module.load(str(ckpt))
                logger.info(f"[auto-gen] loaded compiled module from {ckpt}")
            except Exception as e:
                logger.warning(f"[auto-gen] ckpt load failed ({type(e).__name__}: {e}); using uncompiled.")
        return module

    def _prepare_inputs(
        self,
        *,
        track_ids: list[str],
        user_query: str,
        chat_history: list[dict],
        user_profile: dict | None,
        conversation_goal: dict | None,
        goal_progress: list[dict] | None,
    ) -> dict:
        # Forward context so query-embedding helpers hit the shared qemb
        # cache. `thought=None` mirrors the qemb pool hard-coded default.
        emb_ctx = dict(
            user_profile=user_profile,
            conversation_goal=conversation_goal,
            goal_progress_assessments=goal_progress,
            thought=None,
        )

        # Stitch goal_progress onto the listener_goal block when rich mode
        # is on — preserves signature shape (no new InputField needed).
        listener_goal = fmt_conversation_goal(conversation_goal)
        if self._rich:
            progress_str = fmt_goal_progress(goal_progress)
            if progress_str:
                listener_goal = f"{listener_goal}\nprogress so far:\n{progress_str}"

        return {
            "user_query": user_query or "",
            "listener_goal": listener_goal,
            "chat_history": fmt_chat_history(chat_history, music_only=self._music_only_chat, kv=self._kv),
            "user_profile": fmt_user_profile(user_profile),
            "recommended_tracks_overview": fmt_tracks_overview(track_ids, self._overview_n, self._kv),
            "recommended_tracks_detailed": fmt_recommended_tracks(
                track_ids, self._top_tracks, self._kv, with_popularity=self._rich
            ),
            "query_similarity_hints": fmt_query_similarity_hints(
                track_ids,
                self._top_tracks,
                user_query,
                self._kv,
                chat_history=chat_history,
                graded=self._rich,
                **emb_ctx,
            ),
            "track_similarity_hints": fmt_track_similarity_hints(
                track_ids, self._top_tracks, chat_history, self._kv, graded=self._rich
            ),
            # Side-channel for _postprocess (underscore-prefix stripped before
            # the Module call by the parent template method).
            "_seed": f"{user_query}:{len(chat_history)}",
            "_track_ids_top": track_ids[: self._top_tracks],
        }

    def _postprocess(self, raw_out: Any, *, predict_kwargs: dict, track_ids: list[str]) -> str:
        text = (getattr(raw_out, "response", "") or "").strip()
        if not text:
            return ""

        # Hard bans (globally repeated phrases) — applied deterministically.
        text = apply_hard_bans(text, seed=predict_kwargs["_seed"])

        # Canonicalize / strip verified title citations.
        if predict_kwargs["_track_ids_top"] and self._kv is not None:
            cited_raw = getattr(raw_out, "cited_titles", "") or ""
            text, _report = resolve_title_citations(text, cited_raw, predict_kwargs["_track_ids_top"], self._kv)

        return text
