"""Group-aware deterministic splitter (both K-fold and simple val_ratio).

Items are grouped by ``group_fn(item)``; every group lands in exactly one
partition (no group leakage). For the default SASRec case the group key is the
session id, but callers can pass ``group_fn=lambda s: s[1]`` for user-level
splits when cold-user generalisation or OOF feature generation is the target.

Core function: :func:`split_items`. Delegates to :func:`get_fold` (K-fold) when
``CVConfig`` is active, otherwise performs a group-aware random val_ratio
split. Both paths share group bucketing + deterministic group sort + shuffle
so they are consistent under the same ``group_fn``.

Algorithm (K-fold):

1. Bucket item indices by group id.
2. Sort group ids (stable, deterministic ordering independent of input order).
3. Shuffle the sorted list with ``split_seed``.
4. Round-robin assign groups to folds (``group j → fold j % K``).
5. For a given fold i, validation = items in groups assigned to fold i.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

from mymodule.cv.config import CVConfig

T = TypeVar("T")


def _bucket_and_shuffle(
    items: Sequence[T],
    group_fn: Callable[[T], Hashable],
    seed: int,
) -> tuple[list[Hashable], dict[Hashable, list[int]]]:
    """Bucket item indices by group id; return (shuffled_group_ids, by_group)."""
    by_group: dict[Hashable, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        by_group[group_fn(item)].append(i)
    # Sort for deterministic order independent of item insertion order.
    sorted_groups = sorted(by_group.keys(), key=lambda g: (repr(type(g).__name__), repr(g)))
    rng = random.Random(seed)
    rng.shuffle(sorted_groups)
    return sorted_groups, dict(by_group)


def make_folds(
    items: Sequence[T],
    group_fn: Callable[[T], Hashable],
    config: CVConfig,
) -> list[tuple[list[int], list[int]]]:
    """Return all K ``(train_idx, val_idx)`` pairs.

    ``train_idx`` and ``val_idx`` are lists of indices into ``items``; their
    union is exactly ``range(len(items))`` and they are disjoint. Indices are
    returned in ascending order (stable, easy to diff).
    """
    if len(items) == 0:
        raise ValueError("items is empty")

    shuffled_groups, by_group = _bucket_and_shuffle(items, group_fn, config.split_seed)
    assignment = {g: (j % config.num_folds) for j, g in enumerate(shuffled_groups)}

    per_fold_val: list[list[int]] = [[] for _ in range(config.num_folds)]
    for g, fold in assignment.items():
        per_fold_val[fold].extend(by_group[g])
    for v in per_fold_val:
        v.sort()

    all_indices = set(range(len(items)))
    folds: list[tuple[list[int], list[int]]] = []
    for i in range(config.num_folds):
        val = per_fold_val[i]
        train = sorted(all_indices - set(val))
        folds.append((train, val))
    return folds


def get_fold(
    items: Sequence[T],
    group_fn: Callable[[T], Hashable],
    config: CVConfig,
) -> tuple[list[int], list[int]]:
    """Return ``(train_idx, val_idx)`` for the single fold ``config.fold_idx``.

    ``config.fold_idx`` must be set (``is_active == True``).
    """
    if not config.is_active:
        raise ValueError("get_fold requires config.fold_idx to be set (CV active)")
    return make_folds(items, group_fn, config)[config.fold_idx]


def split_items(
    items: Sequence[T],
    group_fn: Callable[[T], Hashable],
    config: CVConfig,
    *,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Unified train/val splitter. Group-aware in both branches.

    - ``config.is_active``: K-fold slice where ``config.fold_idx`` is held out.
      ``val_ratio`` and ``random_seed`` are ignored in this path.
    - Otherwise: random val_ratio split. ``val_ratio`` fraction of **groups**
      (not items) goes to val, deterministically shuffled by ``random_seed``.
      Equivalent to item-level when ``group_fn`` returns unique ids per item.

    Under the same ``group_fn`` both paths honour group boundaries — a group
    never spans train and val.
    """
    if len(items) == 0:
        raise ValueError("items is empty")
    if config.is_active:
        return get_fold(items, group_fn, config)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1) (got {val_ratio})")

    shuffled_groups, by_group = _bucket_and_shuffle(items, group_fn, random_seed)
    n_val_groups = max(1, int(len(shuffled_groups) * val_ratio))
    val_group_set = set(shuffled_groups[:n_val_groups])

    val_idx: list[int] = []
    train_idx: list[int] = []
    for g, indices in by_group.items():
        (val_idx if g in val_group_set else train_idx).extend(indices)
    val_idx.sort()
    train_idx.sort()
    return train_idx, val_idx
