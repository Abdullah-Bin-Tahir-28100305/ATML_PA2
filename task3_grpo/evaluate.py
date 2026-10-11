"""Task 3 held-out evaluation: the same protocol as Task 2 (common/policy_eval.py).

Generation cap = configs/grpo.yaml `cache_generation_cap` (768 tokens), which equals the PPO
frozen-evaluation cap, so PPO and GRPO numbers are directly comparable.
Use `--adapter checkpoints/grpo_midpoint_policy --name midpoint` for the starting point.
"""

from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import evaluate_rl_policy


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_grpo_policy(config_path: str, adapter, name: str, batch_size: int = 16, max_prompts=None):
    cfg = load_yaml(config_path)
    return evaluate_rl_policy(
        config_path, adapter, name, cfg["results_dir"],
        max_new_tokens=int(cfg.get("cache_generation_cap", 768)),
        batch_size=batch_size, max_prompts=max_prompts,
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-prompts", type=int)
    args = ap.parse_args()
    evaluate_grpo_policy(args.config, args.adapter, args.name, args.batch_size, args.max_prompts)


if __name__ == "__main__":
    main()
