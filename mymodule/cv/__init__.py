"""N-fold cross-validation infrastructure.

Shared, model-agnostic K-fold splitter + CLI config. Each trainable model
imports `CVConfig` (for CLI wiring) and `get_fold` / `make_folds` (for split
computation) instead of re-implementing the logic.

Convention: ``fold{i}`` names the run where **fold i is held out** (i.e.
trained on the K-1 remaining folds). See ``mymodule/cv/README.md`` and
``.claude/rules/cv.md`` for the full convention and usage.
"""

from mymodule.cv.config import CVConfig
from mymodule.cv.fold import get_fold, make_folds, split_items

__all__ = ["CVConfig", "get_fold", "make_folds", "split_items"]
