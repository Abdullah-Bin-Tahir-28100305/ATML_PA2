"""Diagnostic: how far is each LoRA adapter from the base model?

Reads adapter_model.safetensors directly (CPU, no model load, ~seconds) and reports, per adapter:
  * number of LoRA A/B tensors and a sample key (to spot key-name mismatches that would make
    PEFT silently fall back to B = 0, i.e. policy == reference);
  * ||B||_F summed over layers and ||scale * B @ A||_F (the actual weight change).
Used to check whether the near-zero exact KL of the supplied midpoints (~1.2e-5 nats/token)
reflects a genuinely tiny update or a loading problem.

    python -m scripts.check_adapters checkpoints/grpo_midpoint_policy outputs/task1_dpo/standard ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def describe(path: str) -> dict:
    p = Path(path)
    f = p / "adapter_model.safetensors"
    if not f.exists():
        return {"adapter": path, "error": "adapter_model.safetensors not found"}
    cfg = json.loads((p / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(f))
    a_keys = sorted(k for k in sd if "lora_A" in k)
    b_norm2, delta_norm2 = 0.0, 0.0
    for ka in a_keys:
        kb = ka.replace("lora_A", "lora_B")
        A, B = sd[ka].float(), sd[kb].float()
        b_norm2 += B.pow(2).sum().item()
        delta_norm2 += (scale * B @ A).pow(2).sum().item()
    return {
        "adapter": path,
        "r": cfg["r"], "alpha": cfg["lora_alpha"], "target_modules": cfg.get("target_modules"),
        "n_lora_A": len(a_keys), "n_lora_B": sum("lora_B" in k for k in sd),
        "sample_key": a_keys[0] if a_keys else None,
        "B_fro_total": b_norm2 ** 0.5,
        "deltaW_fro_total": delta_norm2 ** 0.5,
    }


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "checkpoints/grpo_midpoint_policy",
        "checkpoints/grpo_midpoint_policy_v3_masked512",
        "checkpoints/ppo_midpoint_policy",
        "outputs/task3_grpo/standard",
        "outputs/task1_dpo/standard",
    ]
    for pth in paths:
        if Path(pth).exists():
            print(json.dumps(describe(pth), indent=1))
        else:
            print(json.dumps({"adapter": pth, "error": "path not found"}))
