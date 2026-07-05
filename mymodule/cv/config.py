"""CV configuration + argparse wiring shared by every trainable model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CVConfig:
    """K-fold CV settings.

    ``fold_idx is None`` disables CV; :func:`mymodule.cv.split_items` then
    performs a random ``val_ratio`` split using the caller-supplied seed
    instead of a K-fold slice.
    """

    num_folds: int = 5
    fold_idx: int | None = None
    split_seed: int = 42

    def __post_init__(self) -> None:
        if self.num_folds < 2:
            raise ValueError(f"num_folds must be >= 2 (got {self.num_folds})")
        if self.fold_idx is not None and not (0 <= self.fold_idx < self.num_folds):
            raise ValueError(f"fold_idx must be in [0, {self.num_folds}) (got {self.fold_idx})")

    @property
    def is_active(self) -> bool:
        return self.fold_idx is not None

    def run_suffix(self) -> str:
        """String to append to a model's ``run_name`` for checkpoint/output paths.

        Empty when CV is inactive (no suffix appended to ``run_name``).
        """
        return f"_fold{self.fold_idx}" if self.is_active else ""

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Attach ``--num-folds / --fold-idx / --split-seed`` to a trainer parser.

        Convention: every trainable model calls this in its CLI setup so the
        three flag names are identical across models.
        """
        group = parser.add_argument_group("cross-validation (optional)")
        group.add_argument(
            "--num-folds",
            type=int,
            default=5,
            help="Total number of folds. Ignored when --fold-idx is not set.",
        )
        group.add_argument(
            "--fold-idx",
            type=int,
            default=None,
            help=(
                "Held-out fold for this run (0..num_folds-1). Unset → "
                "group-aware random val_ratio split is used instead of K-fold."
            ),
        )
        group.add_argument(
            "--split-seed",
            type=int,
            default=42,
            help="Seed for deterministic fold assignment (independent of model --seed).",
        )

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> CVConfig:
        return cls(
            num_folds=args.num_folds,
            fold_idx=args.fold_idx,
            split_seed=args.split_seed,
        )
