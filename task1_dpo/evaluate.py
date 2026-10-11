"""Task 1 held-out evaluation (shared by the standard run, the beta forks and the length study).

Outputs (results/task1_dpo/):
  eval_<name>.json            aggregate metrics (DPO loss/accuracy, KL, reward, length stats)
  pairs_<name>.jsonl          per held-out pair: margins, accuracy, loss, length stratum
  generations_<name>.jsonl    one sampled response per held-out prompt with reward/KL/length
  examples_<name>.json        qualitative candidates (vs the SFT baseline if it was evaluated)
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import response_sequence_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.policy_eval import generate_and_score, qualitative_candidates
from common.training_utils import Timer, device_of, to_device
from task1_dpo.train import filter_pairs, make_collate


def is_base(adapter) -> bool:
    return adapter is None or str(adapter).lower() in {"", "none", "sft", "base"}


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=None if is_base(adapter) else adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def run_beta(cfg, adapter, beta_override=None) -> float:
    if beta_override is not None:
        return float(beta_override)
    if not is_base(adapter):
        rc = repo_path(adapter) / "run_config.json"
        if rc.exists():
            return float(load_json(rc)["beta"])
    return float(cfg["beta"])


@torch.no_grad()
def pair_metrics(policy, tokenizer, rows, max_length: int, beta: float, batch_size: int = 4):
    """Held-out DPO margin m_theta, preference accuracy (m>0) and DPO loss for every pair."""
    rows, dropped = filter_pairs(rows, tokenizer, max_length)
    collate = make_collate(tokenizer, max_length)
    device = device_of(policy)
    is_peft = isinstance(policy, PeftModel)
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        chosen, rejected = collate(chunk)
        chosen, rejected = to_device(chosen, device), to_device(rejected, device)
        pc, _, mc = response_sequence_logprobs(policy, chosen)
        pr, _, mr = response_sequence_logprobs(policy, rejected)
        if is_peft:
            with reference_mode(policy):
                rc, _, _ = response_sequence_logprobs(policy, chosen)
                rr, _, _ = response_sequence_logprobs(policy, rejected)
        else:
            rc, rr = pc.clone(), pr.clone()
        margin = (pc - rc) - (pr - rr)
        loss = -F.logsigmoid(beta * margin)
        for i, row in enumerate(chunk):
            records.append({
                "prompt_id": str(row.get("prompt_id")),
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "margin": float(margin[i].item()),
                "correct": bool(margin[i].item() > 0),
                "dpo_loss": float(loss[i].item()),
                "policy_chosen_logp": float(pc[i].item()),
                "policy_rejected_logp": float(pr[i].item()),
                "ref_chosen_logp": float(rc[i].item()),
                "ref_rejected_logp": float(rr[i].item()),
                "chosen_response_tokens": int(mc[i].sum().item()),
                "rejected_response_tokens": int(mr[i].sum().item()),
                "score_chosen": row.get("score_chosen"),
                "score_rejected": row.get("score_rejected"),
            })
    return records, dropped


def summarize_pairs(records):
    def agg(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "preference_accuracy": float(np.mean([r["correct"] for r in rs])),
            "tie_rate": float(np.mean([abs(r["margin"]) < 1e-9 for r in rs])),
            "dpo_loss": float(np.mean([r["dpo_loss"] for r in rs])),
            "margin_mean": float(np.mean([r["margin"] for r in rs])),
            "margin_std": float(np.std([r["margin"] for r in rs])),
            "chosen_logratio_mean": float(np.mean([r["policy_chosen_logp"] - r["ref_chosen_logp"] for r in rs])),
            "rejected_logratio_mean": float(np.mean([r["policy_rejected_logp"] - r["ref_rejected_logp"] for r in rs])),
        }

    out = {"overall": agg(records)}
    strata = defaultdict(list)
    for r in records:
        if r.get("length_stratum"):
            strata[r["length_stratum"]].append(r)
    if strata:
        out["by_stratum"] = {k: agg(v) for k, v in sorted(strata.items())}
    return out


def evaluate_dpo_policy(
    config_path: str,
    adapter: str | None,
    name: str,
    beta: float | None = None,
    eval_path: str | None = None,
    generate: bool = True,
    max_prompts: int | None = None,
    batch_size: int = 16,
    reward=None,
    tag: str = "",
):
    """Evaluate one policy; saves results under results/task1_dpo/ and returns the summary dict."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    results = repo_path(cfg["results_dir"])
    suffix = f"{name}{tag}"
    timer = Timer()
    rows = read_jsonl(eval_path or cfg["paths"]["dpo_standard_eval"])
    beta_used = run_beta(cfg, adapter, beta)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=None if is_base(adapter) else adapter, trainable=False)

    print(f"[DPO-eval:{suffix}] pair metrics on {len(rows)} held-out pairs (beta={beta_used})", flush=True)
    pairs, dropped = pair_metrics(policy, tokenizer, rows, int(cfg["max_sequence_length"]), beta_used, batch_size=4)
    write_jsonl(results / f"pairs_{suffix}.jsonl", pairs)
    summary = {
        "name": name,
        "adapter": adapter,
        "beta": beta_used,
        "eval_file": eval_path or cfg["paths"]["dpo_standard_eval"],
        "dropped_long_prompt_ids": dropped,
        "pairs": summarize_pairs(pairs),
    }
    if is_base(adapter):
        # pi_theta == pi_ref, so every margin is exactly 0 (all ties): accuracy (m > 0) is undefined, not 0.
        summary["pairs"]["overall"]["preference_accuracy"] = None
        summary["pairs"]["note"] = "starting policy: policy == reference, all margins are 0; accuracy undefined (chance = 0.5)"

    if generate:
        own_reward = reward is None
        if own_reward:
            reward = load_reward_model(cfg)
        gen_rows = rows if max_prompts is None else rows[: int(max_prompts)]
        print(f"[DPO-eval:{suffix}] sampling responses for {len(gen_rows)} held-out prompts", flush=True)
        records, agg = generate_and_score(
            policy,
            tokenizer,
            gen_rows,
            max_new_tokens=int(cfg["max_generation_tokens"]),
            gen_cfg=cfg["generation"],
            reward=reward,
            batch_size=batch_size,
            seed=int(cfg["seed"]),
        )
        write_jsonl(results / f"generations_{suffix}.jsonl", records)
        summary["generation"] = agg
        summary["generation"]["decoding"] = {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"])}
        base_file = results / f"generations_sft{tag}.jsonl"
        baseline = read_jsonl(base_file) if (base_file.exists() and name != "sft") else None
        save_json(results / f"examples_{suffix}.json", qualitative_candidates(records, baseline))
        if own_reward:
            clear_gpu(*reward)
            reward = None

    summary["eval_wall_clock_sec"] = timer()
    save_json(results / f"eval_{suffix}.json", summary)
    p = summary["pairs"]["overall"]
    g = summary.get("generation", {})
    print(f"[DPO-eval:{suffix}] acc={p.get('preference_accuracy')} loss={p.get('dpo_loss')} "
          f"reward={g.get('reward_mean')} kl_tok={g.get('kl_token_mean')} len={g.get('length', {}).get('mean')}", flush=True)
    del policy
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT starting policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: the run's beta)")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--max-prompts", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    evaluate_dpo_policy(args.config, args.adapter, args.name, args.beta, generate=not args.no_generate,
                        max_prompts=args.max_prompts, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
