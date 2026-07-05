"""Helpers for aggregating multiple in-house LLM-as-Judge runs."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record.get("session_id") or ""), int(record.get("turn_number") or 0)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize judge records with the compact shape used by blind diagnostics."""
    if not records:
        return {"n": 0}
    p = [float(r.get("personalization_score", 1)) for r in records]
    e = [float(r.get("explanation_quality_score", 1)) for r in records]
    norm = [float(r.get("mean_normalized", 0.0)) for r in records]
    return {
        "n": len(records),
        "p_mean": statistics.mean(p),
        "e_mean": statistics.mean(e),
        "mean_normalized": statistics.mean(norm),
        "p_dist": dict(sorted(Counter(int(round(x)) for x in p).items())),
        "e_dist": dict(sorted(Counter(int(round(x)) for x in e).items())),
    }


def aggregate_model_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average matching sample records across judge-model runs.

    All runs must score the same `(session_id, turn_number)` set. This keeps
    model averaging honest when using a sampled blindset subset.
    """
    if not runs:
        raise ValueError("at least one model run is required")

    key_sets = [set(_record_key(r) for r in run.get("records", [])) for run in runs]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("judge sample sets differ; use the same seed/sample for every model")

    models = [str(run.get("judge_model") or "") for run in runs]
    records_by_model = [{_record_key(r): r for r in run.get("records", [])} for run in runs]
    averaged: list[dict[str, Any]] = []
    for key in sorted(key_sets[0]):
        per_model = [by_key[key] for by_key in records_by_model]
        p_scores = [float(r.get("personalization_score", 1)) for r in per_model]
        e_scores = [float(r.get("explanation_quality_score", 1)) for r in per_model]
        norms = [float(r.get("mean_normalized", 0.0)) for r in per_model]
        first = per_model[0]
        averaged.append(
            {
                "session_id": key[0],
                "turn_number": key[1],
                "predicted_response": first.get("predicted_response") or first.get("response") or "",
                "personalization_score": statistics.mean(p_scores),
                "explanation_quality_score": statistics.mean(e_scores),
                "mean_normalized": statistics.mean(norms),
                "judge_count": len(per_model),
                "per_model": {
                    model: {
                        "personalization_score": r.get("personalization_score"),
                        "explanation_quality_score": r.get("explanation_quality_score"),
                        "mean_normalized": r.get("mean_normalized"),
                        "personalization_reason": r.get("personalization_reason", ""),
                        "explanation_quality_reason": r.get("explanation_quality_reason", ""),
                    }
                    for model, r in zip(models, per_model, strict=True)
                },
            }
        )

    return {
        "models": models,
        "summary": summarize_records(averaged),
        "records": averaged,
    }
