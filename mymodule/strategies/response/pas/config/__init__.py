"""Single-source-of-truth rule loader for PAS response generation.

`rule.yaml` declares the constraint knobs that signature.py / optimize.py /
select.py all need to agree on (word-budget bands, judge threshold, track
count policy). Module-level cache so first import pays the YAML parse once.

Public API
----------
- ``load_rules()`` — returns the cached `PasRules` (re-read with ``force=True``)
- ``PasRules`` dataclass + nested dataclasses

The intent is **co-location of constraints**: editing `rule.yaml` should ripple
to (a) what the LLM sees in its prompt, (b) what the compile metric rejects,
and (c) what runtime repair enforces — without each consumer carrying its own
hardcoded copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_RULE_YAML_PATH = Path(__file__).parent / "rule.yaml"
_CACHED: PasRules | None = None  # type: ignore[name-defined]


@dataclass(frozen=True)
class SpecBand:
    """Word-budget band for one specificity code (HH / HL / LH / LL)."""

    lower_words: int
    upper_words: int
    gt_avg_words: int
    notes: str = ""

    def contains(self, word_count: int) -> bool:
        return self.lower_words <= word_count <= self.upper_words


@dataclass(frozen=True)
class ResponseLengthRules:
    """Length budget — spec-aware bands + global hard cap + compile margin."""

    specificity: dict[str, SpecBand]
    hard_cap_words: int
    compile_floor_words: int
    compile_ceiling_words: int

    def band_for(self, specificity: str) -> SpecBand | None:
        return self.specificity.get(specificity)

    def spec_summary(self) -> str:
        """Compact one-line summary for embedding in LLM prompts.

        Example: ``HH/HL 25-45w, LH/LL 35-55w. Hard cap 120w.``
        """
        groups: dict[tuple[int, int], list[str]] = {}
        for code, band in self.specificity.items():
            key = (band.lower_words, band.upper_words)
            groups.setdefault(key, []).append(code)
        parts = []
        for (lo, hi), codes in groups.items():
            parts.append(f"{'/'.join(sorted(codes))} {lo}-{hi}w")
        return f"{', '.join(parts)}. Hard cap {self.hard_cap_words}w."


@dataclass(frozen=True)
class TrackCountRules:
    """Citation count policy."""

    default: int
    max: int
    multi_keywords: tuple[str, ...]


@dataclass(frozen=True)
class CompileMetricRules:
    """Compile-time accept/reject metric knobs."""

    threshold: float
    novelty_weight: float


@dataclass(frozen=True)
class PasRules:
    """Top-level rule bundle. Frozen so consumers can rely on it being stable."""

    version: int
    response_length: ResponseLengthRules
    track_count: TrackCountRules
    compile_metric: CompileMetricRules


def load_rules(path: Path | None = None, *, force: bool = False) -> PasRules:
    """Return the cached PAS rule bundle (parsed from ``rule.yaml``).

    ``path`` overrides the default location (useful for tests). When ``path``
    is provided the result is NOT cached. ``force=True`` invalidates the
    module-level cache.
    """
    global _CACHED
    if path is None and _CACHED is not None and not force:
        return _CACHED
    target = path or _RULE_YAML_PATH
    with target.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    rules = _parse(raw)
    if path is None:
        _CACHED = rules
    return rules


def _parse(raw: dict[str, Any]) -> PasRules:
    rl = raw.get("response_length") or {}
    spec_raw = rl.get("specificity") or {}
    spec = {
        code: SpecBand(
            lower_words=int(v["lower_words"]),
            upper_words=int(v["upper_words"]),
            gt_avg_words=int(v.get("gt_avg_words", (v["lower_words"] + v["upper_words"]) // 2)),
            notes=str(v.get("notes", "")),
        )
        for code, v in spec_raw.items()
    }
    response_length = ResponseLengthRules(
        specificity=spec,
        hard_cap_words=int(rl.get("hard_cap_words", 120)),
        compile_floor_words=int(rl.get("compile_floor_words", 15)),
        compile_ceiling_words=int(rl.get("compile_ceiling_words", 130)),
    )
    tc = raw.get("track_count") or {}
    track_count = TrackCountRules(
        default=int(tc.get("default", 1)),
        max=int(tc.get("max", 2)),
        multi_keywords=tuple(tc.get("multi_keywords") or ()),
    )
    cm = raw.get("compile_metric") or {}
    compile_metric = CompileMetricRules(
        threshold=float(cm.get("threshold", 0.5)),
        novelty_weight=float(cm.get("novelty_weight", 0.0)),
    )
    return PasRules(
        version=int(raw.get("version", 1)),
        response_length=response_length,
        track_count=track_count,
        compile_metric=compile_metric,
    )


__all__ = [
    "CompileMetricRules",
    "PasRules",
    "ResponseLengthRules",
    "SpecBand",
    "TrackCountRules",
    "load_rules",
]
