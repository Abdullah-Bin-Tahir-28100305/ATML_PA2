"""Small utilities shared by the Task 1-3 training loops (not part of any objective)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from common.data import render_prompt, repo_path


def disable_dropout(model) -> None:
    """Set every dropout probability to 0 (the convention TRL uses for DPO/PPO/GRPO).

    With dropout active, the log-probabilities of the same tokens differ between the
    rollout/reference pass and the training pass, so PPO ratios would deviate from 1 even
    before any update. Training mode is kept so gradient checkpointing stays active.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def upcast_trainable_params(model) -> None:
    """Keep trainable (LoRA / value-head) parameters in float32 for stable AdamW updates.

    The frozen backbone stays in the configured half precision.
    """
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def adapter_exists(path) -> bool:
    p = repo_path(path)
    return (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()


def device_of(model):
    return next(model.parameters()).device


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 2**30)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def grads_are_finite(params) -> bool:
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def finite_or_none(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def token_entropy_from_logits(logits: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Exact per-token entropy H = -sum_v p log p, computed in float32 in chunks to bound memory."""
    out = []
    for start in range(0, logits.shape[1], chunk):
        lg = logits[:, start:start + chunk].float()
        logp = torch.log_softmax(lg, dim=-1)
        out.append(-(logp.exp() * logp).sum(-1))
    return torch.cat(out, dim=1)


def describe(values) -> dict:
    """Mean / std / median / IQR summary used for every length statistic."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def prompt_token_length(tokenizer, messages) -> int:
    return len(tokenizer(render_prompt(tokenizer, messages), add_special_tokens=False)["input_ids"])


def rl_prompt_schedule(prompt_rows, tokenizer, max_prompt_length: int, seed: int):
    """Deterministic prompt order for PPO/GRPO continuations.

    Prompts whose rendered chat template exceeds `max_prompt_length` are dropped (the released
    batch_generate would otherwise truncate them from the right, cutting off the assistant
    header). The permutation depends only on the seed, so every fork sees the same prompts
    in the same order.
    """
    from common.data import prompt_messages

    kept, dropped = [], []
    for row in prompt_rows:
        n = prompt_token_length(tokenizer, prompt_messages(row))
        (kept if n <= max_prompt_length else dropped).append(row)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(kept))
    return [kept[i] for i in order], [str(r.get("prompt_id", r.get("source_index"))) for r in dropped]


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.start


def write_run_config(path: Path, cfg: dict, extra: dict | None = None) -> None:
    from common.logging_utils import save_json

    payload = {"config": cfg}
    if extra:
        payload.update(extra)
    save_json(path, payload)
