#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TID="${TID:-ensemble__bm25_qmr-qemb_twotower_8b__gbm}"
EVAL_DATASET="${EVAL_DATASET:-blindset_B}"
RESPONSE_GEN="${RESPONSE_GEN:-pas}"
RESPONSE_PROVIDER="${RESPONSE_PROVIDER:-openai}"
PAD_LEXICAL_TARGET="${PAD_LEXICAL_TARGET:-0.98}"

export MYMODULE_TWOTOWER_ANN="${MYMODULE_TWOTOWER_ANN:-1}"
export MYMODULE_CHAT_PROVIDER="$RESPONSE_PROVIDER"
export MYMODULE_RESPONSE_GEN="$RESPONSE_GEN"
export MYMODULE_LLM_OPENAI_URL="${MYMODULE_LLM_OPENAI_URL:-https://api.z.ai/api/paas/v4}"
export MYMODULE_LLM_OPENAI_MODEL="${MYMODULE_LLM_OPENAI_MODEL:-glm-5.2}"
export MYMODULE_LLM_MAX_TOKENS="${MYMODULE_LLM_MAX_TOKENS:-32768}"
export MYMODULE_LLM_TIMEOUT="${MYMODULE_LLM_TIMEOUT:-180}"

if [[ -z "${MYMODULE_LLM_OPENAI_API_KEY:-}" && -n "${ZAI_API_KEY:-}" ]]; then
  export MYMODULE_LLM_OPENAI_API_KEY="$ZAI_API_KEY"
fi

if [[ "$RESPONSE_GEN" != "noop" && "$RESPONSE_PROVIDER" == "openai" && -z "${MYMODULE_LLM_OPENAI_API_KEY:-}" ]]; then
  echo "MYMODULE_LLM_OPENAI_API_KEY is required for OpenAI-compatible response generation." >&2
  echo "Set it in .env or export it before running this script." >&2
  exit 2
fi

args=(
  -m mymodule.run_inference_blindset
  --tid "$TID"
  --eval_dataset "$EVAL_DATASET"
  --response-gen "$RESPONSE_GEN"
  --response-provider "$RESPONSE_PROVIDER"
)

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "${NO_PACKAGE:-0}" == "1" ]]; then
  args+=(--no-package)
fi

uv run python "${args[@]}"

if [[ -z "${LIMIT:-}" && "${NO_PACKAGE:-0}" != "1" && "$RESPONSE_GEN" != "noop" && -n "$PAD_LEXICAL_TARGET" ]]; then
  uv run python scripts/package_submission.py \
    --tid "$TID" \
    --eval-dataset "$EVAL_DATASET" \
    --pad-lexical "$PAD_LEXICAL_TARGET"
fi
