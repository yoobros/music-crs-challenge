"""Compile a few-shot-optimized `AutoResponseModule` using the in-house
LLM-as-Judge as the metric.

Usage:
    uv run python -m mymodule.strategies.response.auto.optimize \
        --response-provider openai \
        --train-n 100 \
        --n-shot 4 \
        --metric-threshold 0.5

Defaults are conservative because each bootstrap trial fires the full
3-stage pipeline (3 LM calls) plus one judge call. With 100 trial
examples and 3-stage Module, expect ~300 generator + ~100 judge calls.
Cache hits compress the repeat-pass cost on subsequent runs.

Output: `mymodule/strategies/response/ckpt/auto_{provider}.json` —
loaded automatically by `AutoResponseGenerator._build_predictor()` once
present (subclass sets `ckpt_basename = "auto"`).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from loguru import logger

# Checkpoints live under response/ckpt/ (one level up from auto/).
CKPT_DIR = Path(__file__).parent.parent / "ckpt"


_UNKNOWN_RESPONSE = "Unknown message"  # 5.65% training noise; same filter as PAS


def _build_examples(train_n: int, kv: object) -> list:
    """Build dspy.Example objects matching `AutoResponseModule.forward` inputs.

    Mirrors `pas/optimize.py::_build_examples` but trims to the auto signature's
    input set (no `intent_groups`, no `lyric_similarity_hints`).
    """
    import dspy
    from datasets import load_dataset

    from mymodule.strategies.response.base_dspy import (
        fmt_recommended_tracks,
        fmt_tracks_overview,
    )
    from mymodule.strategies.response.pas.helpers import (
        fmt_query_similarity_hints,
        fmt_track_similarity_hints,
    )
    from mymodule.utils.common_dspy import (
        fmt_chat_history,
        fmt_conversation_goal,
        fmt_user_profile,
    )

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    indices = random.sample(range(len(ds)), min(train_n * 4, len(ds)))
    examples: list[dspy.Example] = []

    for idx in indices:
        item = ds[idx]
        convs = item["conversations"]
        music_turns = {m["turn_number"] for m in convs if m["role"] == "music"}
        assistant_turns = {m["turn_number"]: m["content"] for m in convs if m["role"] == "assistant"}

        for turn_num in sorted(music_turns & set(assistant_turns.keys())):
            gt_response = (assistant_turns[turn_num] or "").strip()
            if not gt_response or gt_response == _UNKNOWN_RESPONSE:
                continue

            chat_history = [m for m in convs if m["turn_number"] < turn_num]
            user_msgs = [m for m in convs if m["role"] == "user" and m["turn_number"] == turn_num]
            if not user_msgs:
                continue
            user_query = user_msgs[0]["content"]

            music_msgs = [m for m in convs if m["role"] == "music" and m["turn_number"] == turn_num]
            track_ids = [m["content"] for m in music_msgs if m.get("content")]
            if not track_ids:
                continue

            emb_ctx = dict(
                user_profile=item.get("user_profile"),
                conversation_goal=item.get("conversation_goal"),
                goal_progress_assessments=item.get("goal_progress_assessments"),
                thought=None,
            )

            ex = dspy.Example(
                user_query=user_query,
                listener_goal=fmt_conversation_goal(item.get("conversation_goal")),
                chat_history=fmt_chat_history(chat_history, kv=kv),
                user_profile=fmt_user_profile(item.get("user_profile")),
                recommended_tracks_overview=fmt_tracks_overview(track_ids, 20, kv),
                recommended_tracks_detailed=fmt_recommended_tracks(track_ids, 5, kv),
                query_similarity_hints=fmt_query_similarity_hints(
                    track_ids, 5, user_query, kv, chat_history=chat_history, **emb_ctx
                ),
                track_similarity_hints=fmt_track_similarity_hints(track_ids, 5, chat_history, kv),
                response=gt_response,
            ).with_inputs(
                "user_query",
                "listener_goal",
                "chat_history",
                "user_profile",
                "recommended_tracks_overview",
                "recommended_tracks_detailed",
                "query_similarity_hints",
                "track_similarity_hints",
            )
            examples.append(ex)
            if len(examples) >= train_n:
                break
        if len(examples) >= train_n:
            break

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile DSPy few-shot AutoResponseModule.")
    parser.add_argument(
        "--response-provider",
        choices=["ollama", "openai"],
        default="openai",
        help="LM provider for the AutoResponseModule (default: openai).",
    )
    parser.add_argument("--train-n", type=int, default=int(os.getenv("MYMODULE_LLM_OPTIM_TRAIN_N", "100")))
    parser.add_argument(
        "--n-shot",
        type=int,
        default=int(os.getenv("MYMODULE_LLM_FEW_SHOT_N", "4")),
        help="max_labeled_demos and max_bootstrapped_demos for BootstrapFewShot.",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=float(os.getenv("MYMODULE_LLM_METRIC_THRESHOLD", "0.5")),
        help="Minimum judge score (0-1) for a bootstrap trace to qualify as a demo. "
        "0.5 ≈ average dim score 3 (Adequate).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CKPT_DIR / f"auto_{args.response_provider}.json"

    import dspy

    from mymodule.strategies.response.auto.module import AutoResponseModule
    from mymodule.strategies.response.base_dspy import try_open_kvstore
    from mymodule.strategies.response.judge.metric import judge_metric
    from mymodule.utils.common_dspy import ensure_lm_configured

    ensure_lm_configured(args.response_provider)

    kv = try_open_kvstore()
    logger.info(f"Building up to {args.train_n} training examples …")
    trainset = _build_examples(args.train_n, kv)
    logger.info(f"Built {len(trainset)} examples (after filtering 'Unknown message' / empty turns).")

    if len(trainset) < args.n_shot:
        logger.warning(f"only {len(trainset)} examples, less than n_shot={args.n_shot}")

    program = AutoResponseModule(provider=args.response_provider)

    optimizer = dspy.BootstrapFewShot(
        metric=judge_metric,
        max_labeled_demos=args.n_shot,
        max_bootstrapped_demos=args.n_shot,
        metric_threshold=args.metric_threshold,
    )
    logger.info(
        f"Compiling with BootstrapFewShot(metric=judge_metric, "
        f"max_demos={args.n_shot}, metric_threshold={args.metric_threshold}) …"
    )
    compiled = optimizer.compile(program, trainset=trainset)

    compiled.save(str(out_path))
    logger.success(f"Saved compiled module to {out_path}")

    # Verify it loads on the same shape used at runtime.
    loaded = AutoResponseModule(provider=args.response_provider)
    loaded.load(str(out_path))
    logger.success("Load verification: OK")


if __name__ == "__main__":
    main()
