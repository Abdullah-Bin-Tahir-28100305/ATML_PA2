"""Task 2 held-out evaluation (common protocol shared with Task 3; see common/policy_eval.py).

Use `--adapter checkpoints/ppo_midpoint_policy --name midpoint` to evaluate the supplied starting
point, and `--adapter none --name sft` for the untouched instruct model.
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


def evaluate_ppo_policy(config_path: str, adapter, name: str, batch_size: int = 16, max_prompts=None):
    cfg = load_yaml(config_path)
    return evaluate_rl_policy(
        config_path, adapter, name, cfg["results_dir"],
        max_new_tokens=int(cfg["eval_max_response_length"]),
        batch_size=batch_size, max_prompts=max_prompts,
        reward_max_length=int(cfg["reward_max_length"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-prompts", type=int)
    args = ap.parse_args()
    evaluate_ppo_policy(args.config, args.adapter, args.name, args.batch_size, args.max_prompts)


if __name__ == "__main__":
    main()
