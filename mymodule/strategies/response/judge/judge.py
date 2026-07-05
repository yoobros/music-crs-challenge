"""`Judge` — local LLM-as-Judge that scores response candidates on
Personalization (1-5) and Explanation Quality (1-5), mimicking the official
Gemini judge described in `assets/eval_rule.png`.

Configuration
-------------
Judge LM is configured via dedicated env vars (separate from the generator
LM) so the same generator+judge run can use different models — important to
avoid self-grading bias:

    MYMODULE_JUDGE_PROVIDER  Judge LM provider: "openai" (default) | "ollama"
    MYMODULE_JUDGE_MODEL     Provider-specific model id. Defaults:
                                openai → "zai-org/glm-5.1-fp8"
                                ollama → "gemma4:e2b"
    MYMODULE_JUDGE_URL       OpenAI-compat base URL. Falls back to
                              MYMODULE_LLM_OPENAI_URL.
    MYMODULE_JUDGE_API_KEY   API key. Falls back to MYMODULE_LLM_OPENAI_API_KEY,
                              then a dummy value.
    MYMODULE_JUDGE_MAX_TOKENS  Max generated tokens (default 2048).
    MYMODULE_JUDGE_ADAPTER     "auto" (default) | "dspy" | "raw_json".
    MYMODULE_JUDGE_ENABLE_THINKING
                              Enable model reasoning mode when supported
                              (default false for stable judge parsing).
    MYMODULE_JUDGE_TIMEOUT     Per-call seconds (default 120).

Internal vs official judge
--------------------------
The official Gemini judge prompt is private. We mimic with the rubric
documented in `judge/signature.py::JudgeRubric`, which mirrors the rubric
that `auto/signature.py` uses to generate responses. This alignment buys
*relative* ranking stability across variants but NOT absolute calibration —
always cross-check blind composite deltas before trusting an internal-judge
verdict for submission selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from loguru import logger

from mymodule.strategies.response.judge.signature import JudgeRubric

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Score dataclass
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeScore:
    """Result of a single judge invocation.

    `mean_normalized` follows the official composite formula:
        score_norm = (score - 1) / 4
        mean_normalized = (P_norm + E_norm) / 2  ∈ [0, 1]

    Multiplied by 0.30 in the official composite (per `assets/eval_rule.png`).
    """

    p_score: int
    e_score: int
    p_reason: str
    e_reason: str

    @property
    def mean_normalized(self) -> float:
        p = (self.p_score - 1) / 4.0
        e = (self.e_score - 1) / 4.0
        return (p + e) / 2.0

    def to_dict(self) -> dict:
        return {
            "personalization_score": self.p_score,
            "explanation_quality_score": self.e_score,
            "personalization_reason": self.p_reason,
            "explanation_quality_reason": self.e_reason,
            "mean_normalized": self.mean_normalized,
        }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _clamp_score(raw: Any) -> int:
    """Coerce LLM output to int in [1, 5]. Defaults to 1 on parse failure."""
    if isinstance(raw, int):
        return max(1, min(5, raw))
    if isinstance(raw, float):
        return max(1, min(5, int(round(raw))))
    if isinstance(raw, str):
        s = raw.strip()
        # Handle responses like "4" or "4 (Strong)".
        for tok in (s, s.split()[0] if s.split() else ""):
            try:
                return max(1, min(5, int(tok)))
            except (ValueError, IndexError):
                continue
    return 1


def _input_fingerprint(payload: dict) -> bytes:
    """Stable hash of judge inputs for the in-memory cache key."""
    serialized = "\n".join(f"{k}={payload.get(k, '')}" for k in sorted(payload))
    return hashlib.sha256(serialized.encode()).digest()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clip_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# --------------------------------------------------------------------------
# Judge class
# --------------------------------------------------------------------------


class Judge:
    """LLM-as-Judge runner.

    Constructs its own `dspy.LM` so that it does NOT clobber the
    process-global LM that generators have configured via
    `ensure_lm_configured()`. All calls happen inside a `dspy.context(lm=...)`
    scope.
    """

    _DEFAULT_MODELS: dict[str, str] = {
        "openai": "zai-org/glm-5.1-fp8",
        "ollama": "gemma4:e2b",
    }

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        adapter: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        if dspy is None:
            raise RuntimeError("dspy is required for Judge. Install: `uv add dspy`.")
        self.provider = (provider or os.getenv("MYMODULE_JUDGE_PROVIDER") or "openai").strip()
        self.model = (model or os.getenv("MYMODULE_JUDGE_MODEL") or self._DEFAULT_MODELS.get(self.provider, "")).strip()
        if not self.model:
            raise ValueError(f"No judge model resolved for provider={self.provider!r}")
        self.adapter = self._resolve_adapter(adapter)
        self.enable_thinking = (
            _env_flag("MYMODULE_JUDGE_ENABLE_THINKING", False) if enable_thinking is None else bool(enable_thinking)
        )
        self.max_tokens = int(max_tokens or os.getenv("MYMODULE_JUDGE_MAX_TOKENS", "2048"))
        self.timeout = int(timeout or os.getenv("MYMODULE_JUDGE_TIMEOUT", "120"))

        self.lm: Any = self._build_lm()
        self._predictor = dspy.Predict(JudgeRubric)
        self._lock = threading.Lock()
        self._cache: dict[bytes, JudgeScore] = {}

        self._warn_self_grading_if_collision()

    # ---- LM construction --------------------------------------------------

    def _resolve_adapter(self, adapter: str | None) -> str:
        resolved = (adapter or os.getenv("MYMODULE_JUDGE_ADAPTER") or "auto").strip().lower()
        if resolved == "auto":
            if self.provider == "openai" and "kimi" in self.model.lower():
                return "raw_json"
            return "dspy"
        if resolved not in {"dspy", "raw_json"}:
            raise ValueError(f"Unknown judge adapter {resolved!r}. Expected 'auto', 'dspy', or 'raw_json'.")
        return resolved

    def _build_lm(self) -> Any:
        if self.provider == "openai":
            url = (
                os.getenv("MYMODULE_JUDGE_URL")
                or os.getenv("MYMODULE_LLM_OPENAI_URL")
                or "https://models.github.ai/inference"
            ).rstrip("/")
            api_key = (
                os.getenv("MYMODULE_JUDGE_API_KEY")
                or os.getenv("MYMODULE_LLM_OPENAI_API_KEY")
                or "dummy"
            )
            model = self.model if self.model.startswith("openai/") else f"openai/{self.model}"
            return dspy.LM(
                model=model,
                api_base=url,
                api_key=api_key,
                max_tokens=self.max_tokens,
                temperature=0.0,
                timeout=self.timeout,
                enable_thinking=self.enable_thinking,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
        if self.provider == "ollama":
            url = (
                os.getenv("MYMODULE_JUDGE_URL")
                or os.getenv("MYMODULE_LLM_OLLAMA_URL")
                or os.getenv("MYMODULE_EMB_OLLAMA_URL")
                or "http://localhost:11434"
            ).rstrip("/")
            return dspy.LM(
                model=f"ollama_chat/{self.model}",
                api_base=url,
                api_key="ollama",
                max_tokens=self.max_tokens,
                temperature=0.0,
                timeout=self.timeout,
                think=self.enable_thinking,
            )
        raise ValueError(f"Unknown judge provider {self.provider!r}. Expected 'ollama' or 'openai'.")

    def _warn_self_grading_if_collision(self) -> None:
        """If the active dspy LM (generator) shares the judge's model name,
        warn — relative ranking is still useful but bias is elevated."""
        active = getattr(getattr(dspy, "settings", None), "lm", None)
        active_model = getattr(active, "model", "") or ""
        if active_model and (active_model == self.model or active_model.endswith(f"/{self.model}")):
            logger.warning(
                f"[judge] Judge model '{self.model}' matches the currently configured "
                f"generator LM ('{active_model}'). This elevates self-grading bias — "
                f"prefer a different MYMODULE_JUDGE_MODEL."
            )

    # ---- public API -------------------------------------------------------

    def score(
        self,
        *,
        user_query: str,
        chat_history: str,
        user_profile: str,
        listener_goal: str,
        recommended_tracks_detailed: str,
        predicted_response: str,
        gt_response: str = "",
    ) -> JudgeScore:
        """Score one (response, context) pair. Returns a `JudgeScore`.

        Empty `predicted_response` shortcuts to (1, 1) — the official
        evaluator's worst case for missing text. No LM call is made.
        """
        if not (predicted_response or "").strip():
            return JudgeScore(p_score=1, e_score=1, p_reason="empty response", e_reason="empty response")

        payload = {
            "user_query": user_query or "",
            "chat_history": chat_history or "",
            "user_profile": user_profile or "",
            "listener_goal": listener_goal or "",
            "recommended_tracks_detailed": recommended_tracks_detailed or "",
            "predicted_response": predicted_response,
            "gt_response": gt_response or "",
        }
        key = _input_fingerprint(payload)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        try:
            if self.adapter == "raw_json":
                score = self._score_raw_json(payload)
            else:
                with dspy.context(lm=self.lm):
                    out = self._predictor(**payload)
                score = JudgeScore(
                    p_score=_clamp_score(getattr(out, "personalization_score", 1)),
                    e_score=_clamp_score(getattr(out, "explanation_quality_score", 1)),
                    p_reason=str(getattr(out, "personalization_reason", "") or "").strip(),
                    e_reason=str(getattr(out, "explanation_quality_reason", "") or "").strip(),
                )
        except Exception as e:
            logger.warning(f"[judge] scoring failed ({type(e).__name__}: {e}); returning (1,1)")
            score = JudgeScore(p_score=1, e_score=1, p_reason=f"judge error: {e}", e_reason=f"judge error: {e}")

        with self._lock:
            self._cache[key] = score
        return score

    def _score_raw_json(self, payload: dict[str, str]) -> JudgeScore:
        prompt = f"""Grade this recommendation response. Return ONLY one JSON object, no prose.
Scores: 4=strong personalized+concrete, 3=one gap, 2=generic/malformed, 1=empty, 5=exceptional.
Penalize malformed citations, stock verdicts, unexplained second pick, numeric-only reasons, generic bridge.
Schema fields: personalization_score, personalization_reason, explanation_quality_score, explanation_quality_reason.
Query: {_clip_text(payload.get("user_query", ""), 260)}
History: {_clip_text(payload.get("chat_history", ""), 260)}
Goal: {_clip_text(payload.get("listener_goal", ""), 220)}
Tracks: {_clip_text(payload.get("recommended_tracks_detailed", ""), 360)}
Response: {_clip_text(payload.get("predicted_response", ""), 700)}
JSON only."""
        raw = self.lm(prompt, enable_thinking=self.enable_thinking)
        text = ""
        if isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, dict):
                text = str(first.get("text") or "")
            else:
                text = str(first or "")
        else:
            text = str(raw or "")
        obj = _extract_json_object(text)
        if not obj:
            return JudgeScore(p_score=1, e_score=1, p_reason="judge parse failed", e_reason="judge parse failed")
        return JudgeScore(
            p_score=_clamp_score(obj.get("personalization_score")),
            e_score=_clamp_score(obj.get("explanation_quality_score")),
            p_reason=str(obj.get("personalization_reason") or "").strip(),
            e_reason=str(obj.get("explanation_quality_reason") or "").strip(),
        )


# --------------------------------------------------------------------------
# Module-level singleton (lazy)
# --------------------------------------------------------------------------

_DEFAULT_JUDGE: Judge | None = None
_DEFAULT_JUDGE_LOCK = threading.Lock()


def get_default_judge() -> Judge:
    """Return a process-singleton `Judge` configured from env vars.

    Lazy-built so importing `mymodule.strategies.response.judge` does not
    require LM configuration at module-load time.
    """
    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is not None:
        return _DEFAULT_JUDGE
    with _DEFAULT_JUDGE_LOCK:
        if _DEFAULT_JUDGE is None:
            _DEFAULT_JUDGE = Judge()
    return _DEFAULT_JUDGE


__all__ = ["Judge", "JudgeScore", "get_default_judge"]
