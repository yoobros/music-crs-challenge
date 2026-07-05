"""Centralized path management for project submodules and experiment artifacts."""

import shutil
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Submodule directories
EVALUATOR_DIR = PROJECT_ROOT / "music-crs-evaluator"
BASELINES_DIR = PROJECT_ROOT / "music-crs-baselines"
MYMODULE_DIR = PROJECT_ROOT / "mymodule"

# mymodule experiment directories (primary)
MYMODULE_INFERENCE_DIR = MYMODULE_DIR / "exp" / "inference"
MYMODULE_SCORES_DIR = MYMODULE_DIR / "exp" / "scores"

# Evaluator experiment directories (copy destination)
EVALUATOR_INFERENCE_DIR = EVALUATOR_DIR / "exp" / "inference"
EVALUATOR_SCORES_DIR = EVALUATOR_DIR / "exp" / "scores"

# Ground truth (evaluator only)
GROUND_TRUTH_DIR = EVALUATOR_DIR / "exp" / "ground_truth"

# Baselines config
BASELINES_CONFIG_DIR = BASELINES_DIR / "config"


def inference_path(tid: str, dataset: str = "devset") -> Path:
    """Path to a prediction JSON file (under mymodule/exp/)."""
    return MYMODULE_INFERENCE_DIR / dataset / f"{tid}.json"


def evaluator_inference_path(tid: str, dataset: str = "devset") -> Path:
    """Path to a prediction JSON file in the evaluator directory."""
    return EVALUATOR_INFERENCE_DIR / dataset / f"{tid}.json"


def save_inference(inference_results: list, tid: str, dataset: str = "devset") -> Path:
    """Save inference results to mymodule/exp/ and copy to evaluator/exp/.

    Returns the primary (mymodule) output path.
    """
    import json

    primary = inference_path(tid, dataset)
    primary.parent.mkdir(parents=True, exist_ok=True)
    with open(primary, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)

    evaluator_dst = evaluator_inference_path(tid, dataset)
    evaluator_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary, evaluator_dst)

    logger.info(f"Saved {len(inference_results)} predictions → {primary}")
    logger.info(f"Copied → {evaluator_dst}")
    return primary


def scores_path(tid: str, dataset: str = "devset") -> Path:
    """Path to an evaluation scores JSON file."""
    return MYMODULE_SCORES_DIR / dataset / f"{tid}.json"


def ground_truth_path(dataset: str = "devset") -> Path:
    """Path to a ground truth JSON file."""
    return GROUND_TRUTH_DIR / f"{dataset}.json"


def baselines_inference_path(tid: str, dataset: str = "devset") -> Path:
    """Path where baselines submodule writes inference output (before copy)."""
    return BASELINES_DIR / "exp" / "inference" / dataset / f"{tid}.json"
