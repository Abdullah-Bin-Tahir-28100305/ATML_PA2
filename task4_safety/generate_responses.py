from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from common.training_utils import adapter_exists


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    """Greedy (deterministic) responses for every XSTest prompt, in the fixed CSV order."""
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
                "label": str(row["label"]),
                "focus": str(row.get("focus", "")),
            })
    del model
    clear_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=16,
                    help="same value for every policy (left-padding layout is kept identical)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    specs = policy_specs(cfg)
    for name in args.policies:
        adapter = specs[name]
        if adapter is not None and not adapter_exists(adapter):
            raise SystemExit(f"Policy '{name}' adapter missing at {adapter}. Finish the standard run of the corresponding task first.")
        path = outdir / f"generated_{name}.jsonl"
        if path.exists() and not args.force:
            print(f"[safety-gen] {name}: reusing {path}")
            continue
        print(f"[safety-gen] {name}: generating (greedy, max_new_tokens={cfg['safety_max_new_tokens']})", flush=True)
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)
        write_jsonl(path, records)
    save_json(outdir / "generation_config.json", {
        "policies": {k: specs[k] for k in args.policies},
        "decoding": "greedy (do_sample=False)",
        "max_new_tokens": int(cfg["safety_max_new_tokens"]),
        "max_prompt_length": 256,
        "batch_size": args.batch_size,
        "xstest_file": cfg["paths"]["xstest"],
    })


if __name__ == "__main__":
    main()
