#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SMOKE_DIR="${SMOKE_DIR:-tmp/submission-smoke}"
SMOKE_LIMIT="${SMOKE_LIMIT:-2}"
OPENAI_MODEL="${OPENAI_MODEL:-${MYMODULE_LLM_OPENAI_MODEL:-glm-5.2}}"
OPENAI_URL="${MYMODULE_LLM_OPENAI_URL:-https://api.z.ai/api/paas/v4}"
mkdir -p "$SMOKE_DIR"

export MYMODULE_LLM_OPENAI_URL="$OPENAI_URL"
export MYMODULE_LLM_OPENAI_MODEL="$OPENAI_MODEL"
export MYMODULE_LLM_MAX_TOKENS="${MYMODULE_LLM_MAX_TOKENS:-32768}"
export MYMODULE_LLM_TIMEOUT="${MYMODULE_LLM_TIMEOUT:-180}"
export MYMODULE_LLM_WORKERS="${MYMODULE_LLM_WORKERS:-1}"
export MYMODULE_EMB_PROVIDER="${MYMODULE_EMB_PROVIDER:-ollama}"
export MYMODULE_EMB_MODEL="${MYMODULE_EMB_MODEL:-qwen3-embedding:0.6b}"

if [[ -z "${MYMODULE_LLM_OPENAI_API_KEY:-}" && -n "${ZAI_API_KEY:-}" ]]; then
  export MYMODULE_LLM_OPENAI_API_KEY="$ZAI_API_KEY"
fi
if [[ -z "${MYMODULE_LLM_OPENAI_API_KEY:-}" ]]; then
  echo "MYMODULE_LLM_OPENAI_API_KEY is required for GLM-5.2 smoke generation." >&2
  echo "Set it in .env or export it before running this script." >&2
  exit 2
fi

echo "[1/5] GLM query-summary smoke"
uv run python -m mymodule.strategies.twotower.query_summary \
  --dataset devset \
  --provider openai \
  --model "$OPENAI_MODEL" \
  --limit "$SMOKE_LIMIT" \
  --out "$SMOKE_DIR/query_summaries.jsonl" \
  --force \
  --concurrency 1

echo "[2/5] GLM doc-enrichment smoke"
uv run python -m mymodule.strategies.twotower.synth_doc \
  --provider openai \
  --model "$OPENAI_MODEL" \
  --max-tracks "$SMOKE_LIMIT" \
  --out "$SMOKE_DIR/synth_use_cases.jsonl" \
  --force \
  --concurrency 1

echo "[3/5] Ollama metadata-rich embedding smoke"
SMOKE_DIR="$SMOKE_DIR" uv run python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset

from mymodule.feature.kvdb import _compose_track_metadata_rich_text
from mymodule.feature.ollama_embed import get_embedder

out_dir = Path(os.environ["SMOKE_DIR"])
embedder = get_embedder("ollama")
if not embedder.health():
    raise SystemExit(f"Ollama embedding server is unreachable at {embedder.url}")
row = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")[0]
text = _compose_track_metadata_rich_text(row)
vec = embedder.embed(text)
np.savez(out_dir / "metadata_rich_ollama_vector.npz", track_id=row["track_id"], vector=vec.astype(np.float32))
(out_dir / "metadata_rich_ollama_vector.json").write_text(
    json.dumps({"track_id": row["track_id"], "dim": int(vec.shape[0])}, indent=2),
    encoding="utf-8",
)
print(f"encoded 1 metadata-rich track vector: dim={vec.shape[0]}")
PY

echo "[4/5] Training CLI import smoke"
uv run python -m mymodule.strategies.twotower.train --help > "$SMOKE_DIR/twotower_train_help.txt"
uv run python -m mymodule.strategies.twotower.query_cache --help > "$SMOKE_DIR/twotower_query_cache_help.txt"
uv run python -m mymodule.strategies.rerank.gbm.build_features --help > "$SMOKE_DIR/gbm_build_features_help.txt"
uv run python -m mymodule.strategies.rerank.gbm.train --help > "$SMOKE_DIR/gbm_train_help.txt"

if [[ "${RUN_GBM_OOF_SMOKE:-0}" == "1" ]]; then
  echo "[5/5] GBM OOF feature smoke (limit=${SMOKE_LIMIT})"
  echo "This opt-in step expects train-side 8B query caches or a GPU that can load the 8B encoder."
  uv run python -m mymodule.strategies.rerank.gbm.build_features \
    --tid ensemble__bm25_qmr-qemb_twotower_8b__gbm \
    --limit "$SMOKE_LIMIT" \
    --workers 1 \
    --out "$SMOKE_DIR/gbm_oof" \
    --no-resume
else
  echo "[5/5] GBM OOF feature smoke skipped"
  echo "      Set RUN_GBM_OOF_SMOKE=1 only after train-side 8B query caches exist, or on a GPU machine."
fi

echo "Stage 2 smoke completed: $SMOKE_DIR"
