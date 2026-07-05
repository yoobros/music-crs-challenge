"""HF Qwen3-Embedding-0.6B encoder with optional PEFT LoRA adapter.

Used by `train`, `eval`, `doc_cache`, and `pool` to encode queries / docs
with last-token pooling (Qwen3-Embedding's official pooling).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F  # noqa: N812
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MAX_LEN = 256
QWEN3_QUERY_INSTRUCTION = (
    "Given a user's listening history and stated context, retrieve music tracks matching their "
    "current mood and intent. Consider genre, era, mood, lyrical themes, and emotional tone."
)
_VALID_DEVICES = {"cpu", "cuda", "mps"}


def autodetect_device() -> str:
    for env_name in ("MYMODULE_TWOTOWER_DEVICE", "MYMODULE_TORCH_DEVICE"):
        value = os.getenv(env_name)
        if not value:
            continue
        device = value.strip().lower()
        if device not in _VALID_DEVICES:
            raise ValueError(f"{env_name} must be one of {sorted(_VALID_DEVICES)}, got {value!r}")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"{env_name}=cuda was requested but CUDA is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(f"{env_name}=mps was requested but MPS is not available")
        logger.info(f"[twotower-encoder] using device override from {env_name}: {device}")
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3-Embedding uses left-padding + last-token pooling.

    With right-padding (HF default), the last *real* token sits at index
    (seq_len - 1) shifted by left padding. We compute the per-row last index
    explicitly so it works regardless of padding side.
    """
    if attention_mask.dtype != torch.long:
        attention_mask = attention_mask.long()
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bsz = last_hidden.shape[0]
    return last_hidden[torch.arange(bsz, device=last_hidden.device), seq_lens]


def format_query(text: str, instruction: str = QWEN3_QUERY_INSTRUCTION) -> str:
    """Qwen3-Embedding query format: 'Instruct: {inst}\\nQuery: {q}'."""
    return f"Instruct: {instruction}\nQuery: {text}"


def format_doc(text: str) -> str:
    """Doc side gets raw text (matches qwen3-embedding doc-side recipe)."""
    return text


def get_lora_target_modules() -> list[str]:
    """LoRA target modules for Qwen3 family — attention QKV/output + MLP up/down/gate."""
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_max_body_tokens(tokenizer, max_len: int, instruction: str = QWEN3_QUERY_INSTRUCTION) -> int:
    """max_len minus the Instruct/Query prefix length, measured via tokenizer."""
    prefix = f"Instruct: {instruction}\nQuery: "
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    return max(32, max_len - prefix_tokens - 1)


class QwenEmbedderTorch:
    """HF Qwen3-Embedding-0.6B encoder with optional LoRA adapter.

    Wraps `transformers.AutoModel` + PEFT. Use `encode_queries` /
    `encode_docs` for batched float32 numpy outputs (L2-normalized).
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | Path | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        max_len: int = DEFAULT_MAX_LEN,
        attn_implementation: str = "eager",
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device or autodetect_device()
        self.dtype = dtype
        self.max_len = max_len

        logger.info(f"[twotower-encoder] loading tokenizer ({base_model}, padding=left, truncation=left)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, padding_side="left", truncation_side="left", trust_remote_code=True
        )
        # HF fast tokenizer's Rust internals are not thread-safe; concurrent
        # calls raise `RuntimeError: Already borrowed`. Serialize tokenization.
        self._tok_lock = threading.Lock()

        logger.info(f"[twotower-encoder] loading base model on {self.device} (dtype={dtype})")
        self.model = AutoModel.from_pretrained(
            base_model,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).to(self.device)

        if self.adapter_path is not None:
            from peft import PeftModel

            logger.info(f"[twotower-encoder] applying LoRA adapter from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
            self.model = self.model.to(self.device)

        self.model.eval()

    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        normalize: bool = True,
        desc: str = "embedding",
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.hidden_dim()), dtype=np.float32)
        out: list[np.ndarray] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        # Inference-time single-query calls produce noisy `1/1` tqdm bars per
        # call. Suppress when there's only one batch — progress is meaningful
        # only for bulk ops (training, doc-cache build).
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, total=n_batches, disable=n_batches <= 1):
            chunk = texts[i : i + batch_size]
            with self._tok_lock:
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
            enc = enc.to(self.device)
            with torch.autocast(device_type=("cuda" if self.device == "cuda" else "cpu"), enabled=False):
                hidden = self.model(**enc).last_hidden_state
            vec = last_token_pool(hidden, enc["attention_mask"])
            if normalize:
                vec = F.normalize(vec, dim=-1)
            out.append(vec.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def encode_queries(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        instruction: str = QWEN3_QUERY_INSTRUCTION,
        normalize: bool = True,
    ) -> np.ndarray:
        queries = [format_query(t, instruction) for t in texts]
        return self.encode(queries, batch_size=batch_size, normalize=normalize, desc="encode queries")

    def encode_docs(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        normalize: bool = True,
    ) -> np.ndarray:
        docs = [format_doc(t) for t in texts]
        return self.encode(docs, batch_size=batch_size, normalize=normalize, desc="encode docs")


def assert_torch_cuda_or_warn() -> str:
    dev = autodetect_device()
    if dev != "cuda":
        logger.warning(f"[twotower-encoder] CUDA not available — using {dev}. LoRA training will be slow on CPU/MPS.")
    else:
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        logger.info(f"[twotower-encoder] CUDA available: {n} device(s), primary={name}")
    return dev


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MAX_LEN",
    "QWEN3_QUERY_INSTRUCTION",
    "QwenEmbedderTorch",
    "assert_torch_cuda_or_warn",
    "autodetect_device",
    "ensure_dirs",
    "format_doc",
    "format_query",
    "get_lora_target_modules",
    "last_token_pool",
    "resolve_max_body_tokens",
]
