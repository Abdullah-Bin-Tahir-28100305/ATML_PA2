"""Diagnostic: is the ~1.2e-5 nats/token exact KL of the midpoints real, or a float16 precision floor?

Teacher-forces the same fixed responses (the first N held-out generations of the GRPO midpoint, from
results/task3_grpo/generations_midpoint.jsonl) through the policy and through the reference
(adapter disabled), and reports the exact full-vocabulary KL per token for several adapters, in
float16 (the released config dtype) and in float32.

How to read it:
  * real signal  -> fp16 and fp32 agree, and doubling the adapter's B matrices (`*_x2`) multiplies
                    KL by ~4 (KL is quadratic in a small weight change);
  * precision floor -> fp32 is much smaller than fp16, and/or `_x2` barely changes the fp16 value.
The Task 1 DPO adapter is a positive control with a ~10x larger weight change.

    python -m scripts.kl_floor_check --n 24
Writes results/kl_floor_check.json.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from common.policy_eval import exact_token_kl


def build_examples(cfg, tok, n, max_resp):
    gens = {g["prompt_id"]: g for g in read_jsonl(repo_path("results/task3_grpo/generations_midpoint.jsonl"))}
    ex = []
    for i, row in enumerate(read_jsonl(repo_path(cfg["paths"]["rl_prompt_eval"]))):
        pid = str(row.get("prompt_id", row.get("source_index", i)))
        if pid not in gens:
            continue
        p = tok.apply_chat_template(prompt_messages(row), tokenize=True, add_generation_prompt=True)
        r = tok(gens[pid]["response"], add_special_tokens=False)["input_ids"][:max_resp]
        if not r:
            continue
        ex.append((torch.tensor([p + r]), len(p), len(r)))
        if len(ex) >= n:
            break
    return ex


@torch.no_grad()
def logits_for(model, ids, plen, rlen):
    out = model(input_ids=ids.to(DEV), use_cache=False).logits
    return out[:, plen - 1: plen - 1 + rlen, :]


@torch.no_grad()
def run_dtype(cfg, dtype, adapters, examples):
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=dtype, low_cpu_mem_usage=True)
    names = list(adapters)
    model = PeftModel.from_pretrained(base, str(repo_path(adapters[names[0]][0])), adapter_name=names[0])
    for nm in names[1:]:
        model.load_adapter(str(repo_path(adapters[nm][0])), adapter_name=nm)
    for nm, (_, scale) in adapters.items():
        if scale != 1.0:
            for pn, p in model.named_parameters():
                if f"lora_B.{nm}." in pn:
                    p.mul_(scale)
    model = model.to(DEV).eval()
    tot = {nm: 0.0 for nm in names}
    ntok = 0
    for ids, plen, rlen in examples:
        with model.disable_adapter():
            ref = logits_for(model, ids, plen, rlen)
        for nm in names:
            model.set_adapter(nm)
            pol = logits_for(model, ids, plen, rlen)
            tot[nm] += exact_token_kl(pol, ref).sum().item()
            del pol
        ntok += rlen
        del ref
    del model, base
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return {nm: tot[nm] / ntok for nm in names}, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-resp", type=int, default=512)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    examples = build_examples(cfg, tok, args.n, args.max_resp)
    adapters = {
        "grpo_midpoint": ("checkpoints/grpo_midpoint_policy", 1.0),
        "grpo_midpoint_x2": ("checkpoints/grpo_midpoint_policy", 2.0),
        "ppo_midpoint": ("checkpoints/ppo_midpoint_policy", 1.0),
        "task3_standard": ("outputs/task3_grpo/standard", 1.0),
        "task1_dpo_standard": ("outputs/task1_dpo/standard", 1.0),
    }
    adapters = {k: v for k, v in adapters.items() if (repo_path(v[0]) / "adapter_model.safetensors").exists()}
    res = {}
    for dname, dt in [("float16", torch.float16), ("float32", torch.float32)]:
        kl, ntok = run_dtype(cfg, dt, adapters, examples)
        res[dname] = kl
        print(f"[{dname}] {len(examples)} responses, {ntok} tokens", flush=True)
    print(f"\n{'adapter':22s} {'KL/token fp16':>15s} {'KL/token fp32':>15s} {'fp16/fp32':>10s}")
    for nm in adapters:
        a, b = res["float16"][nm], res["float32"][nm]
        print(f"{nm:22s} {a:15.3e} {b:15.3e} {a / b if b > 0 else float('nan'):10.2f}")
    for d in res:
        if "grpo_midpoint" in res[d] and "grpo_midpoint_x2" in res[d] and res[d]["grpo_midpoint"] > 0:
            print(f"[{d}] KL(x2) / KL(x1) = {res[d]['grpo_midpoint_x2'] / res[d]['grpo_midpoint']:.2f}  (real signal -> ~4)")
    save_json(repo_path("results/kl_floor_check.json"), {"n_responses": len(examples), "kl_per_token": res})


if __name__ == "__main__":
    main()
