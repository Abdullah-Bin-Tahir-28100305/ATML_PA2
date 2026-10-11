"""Shared held-out generation + scoring protocol for Tasks 1-3.

For each prompt we sample one response from the evaluated policy and record:
  * reward-model score (fixed course RM),
  * sampled-response KL to the frozen reference policy: sum over response tokens of
    log pi(y_t) - log pi_ref(y_t)  (the released `sampled_kl` estimator),
  * exact token entropy of the policy at every response position, plus the sampled estimate,
  * response length (generated tokens after the prompt, excluding padding), EOS / truncation.

Because evaluation responses are sampled with temperature 0.7 / top-p 0.9 (the fixed decoding), the
sampled estimator is NOT guaranteed to be >= 0: its expectation is under the tempered/truncated
sampler, not under pi. When the policy is close to the reference it can come out slightly negative.
We therefore also report the exact per-position KL over the full vocabulary,
  KL_t = sum_v pi(v|s_t) [log pi(v|s_t) - log pi_ref(v|s_t)]  >= 0 for every t,
evaluated at the same sampled prefixes (`kl_exact_*`).

Aggregation convention (identical for every condition):
  * `kl_token_mean`  = sum of per-token log-ratios / number of valid response tokens
                       (== common.metrics.sampled_kl applied to the whole evaluation set);
  * `kl_seq_mean`    = mean over responses of the per-response summed log-ratio.
  Both are always reported; the report should pick one and use it everywhere.
"""

from __future__ import annotations

import numpy as np
import torch
from peft import PeftModel

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.models import reference_mode
from common.training_utils import describe, token_entropy_from_logits


def eval_prompt_rows(rows: list[dict]) -> list[dict]:
    """Normalise evaluation rows to {'id', 'messages', 'source_index'}."""
    out = []
    for i, row in enumerate(rows):
        out.append({
            "id": str(row.get("prompt_id", row.get("source_index", i))),
            "source_index": row.get("source_index", i),
            "messages": prompt_messages(row),
        })
    return out


@torch.no_grad()
def _logprob_stats(policy, gen: dict, micro_batch: int = 2):
    """Policy/reference log-probs of sampled tokens, exact entropies and exact per-token KL, in micro-batches."""
    seq, attn, width = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
    resp, rmask = gen["response_ids"], gen["response_mask"]
    is_peft = isinstance(policy, PeftModel)
    pol_all, ref_all, ent_all, kl_all = [], [], [], []
    for s in range(0, seq.shape[0], micro_batch):
        sl = slice(s, s + micro_batch)
        # Trim trailing positions that are padding for every row of this micro-batch.
        r_len = int(rmask[sl].sum(-1).max().item())
        r_len = max(r_len, 1)
        sub_seq = seq[sl, : width + r_len]
        sub_attn = attn[sl, : width + r_len]
        sub_resp = resp[sl, :r_len]
        pol, logits = response_token_logprobs(policy, sub_seq, sub_attn, width, sub_resp)
        ent = token_entropy_from_logits(logits)
        if is_peft:
            with reference_mode(policy):
                ref, ref_logits = response_token_logprobs(policy, sub_seq, sub_attn, width, sub_resp)
            kl = exact_token_kl(logits, ref_logits)
            del ref_logits
        else:
            ref = pol.clone()
            kl = torch.zeros_like(pol)
        del logits
        pad = resp.shape[1] - r_len
        if pad > 0:
            z = torch.zeros(pol.shape[0], pad, device=pol.device)
            pol, ref, ent, kl = (torch.cat([x, z], 1) for x in (pol, ref, ent, kl))
        pol_all.append(pol.float().cpu())
        ref_all.append(ref.float().cpu())
        ent_all.append(ent.float().cpu())
        kl_all.append(kl.float().cpu())
    return torch.cat(pol_all), torch.cat(ref_all), torch.cat(ent_all), torch.cat(kl_all)


@torch.no_grad()
def exact_token_kl(logits_p, logits_q, chunk: int = 128):
    """KL(p || q) over the full vocabulary at every position, in float32 chunks."""
    out = []
    for start in range(0, logits_p.shape[1], chunk):
        lp = torch.log_softmax(logits_p[:, start:start + chunk].float(), dim=-1)
        lq = torch.log_softmax(logits_q[:, start:start + chunk].float(), dim=-1)
        out.append((lp.exp() * (lp - lq)).sum(-1).clamp_min(0.0))
    return torch.cat(out, dim=1)


@torch.no_grad()
def generate_and_score(
    policy,
    tokenizer,
    rows: list[dict],
    *,
    max_new_tokens: int,
    gen_cfg: dict,
    reward=None,
    reward_max_length: int = 1280,
    max_prompt_length: int = 2048,
    batch_size: int = 16,
    seed: int = 6304,
    do_sample: bool | None = None,
    logprob_micro_batch: int = 2,
    progress: bool = True,
):
    """Generate one response per prompt and score it. Returns (records, aggregate)."""
    prompts = eval_prompt_rows(rows)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    sample = bool(gen_cfg.get("do_sample", True)) if do_sample is None else bool(do_sample)
    records = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(prompts), batch_size)):
        chunk = prompts[start:start + batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            [p["messages"] for p in chunk],
            max_prompt_length=max_prompt_length,
            max_new_tokens=int(max_new_tokens),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=sample,
        )
        pol, ref, ent, klx = _logprob_stats(policy, gen, micro_batch=logprob_micro_batch)
        mask = gen["response_mask"].float().cpu()
        if reward is not None:
            rm, rm_tok = reward
            scores = score_reward_pairs(
                rm, rm_tok, [p["messages"] for p in chunk], gen["responses"], max_length=reward_max_length
            ).cpu().tolist()
        else:
            scores = [None] * len(chunk)
        for i, p in enumerate(chunk):
            m = mask[i]
            n_tok = float(m.sum().item())
            log_ratio = ((pol[i] - ref[i]) * m).sum().item()
            records.append({
                "prompt_id": p["id"],
                "source_index": p["source_index"],
                "response": gen["responses"][i],
                "response_tokens": int(gen["response_lengths"][i]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "truncated": bool(gen["truncated"][i]),
                "reward": None if scores[i] is None else float(scores[i]),
                "kl_seq": float(log_ratio),
                "kl_tokens": n_tok,
                "kl_exact_seq": float((klx[i] * m).sum().item()),
                "entropy_sum": float((ent[i] * m).sum().item()),
                "sampled_logp_sum": float((pol[i] * m).sum().item()),
            })
        if progress:
            print(f"  generated batch {b + 1}/{n_batches}", flush=True)
    return records, aggregate_records(records)


def aggregate_records(records: list[dict]) -> dict:
    tok = float(sum(r["kl_tokens"] for r in records)) or 1.0
    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    out = {
        "n": len(records),
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "reward_std": float(np.std(rewards)) if rewards else None,
        "kl_token_mean": float(sum(r["kl_seq"] for r in records) / tok),
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])) if records else None,
        "kl_seq_se": float(np.std([r["kl_seq"] for r in records]) / np.sqrt(len(records))) if records else None,
        "kl_exact_token_mean": float(sum(r.get("kl_exact_seq", 0.0) for r in records) / tok),
        "kl_exact_seq_mean": float(np.mean([r.get("kl_exact_seq", 0.0) for r in records])) if records else None,
        "entropy_token_mean": float(sum(r["entropy_sum"] for r in records) / tok),
        "sampled_entropy_token_mean": float(-sum(r["sampled_logp_sum"] for r in records) / tok),
        "length": describe([r["response_tokens"] for r in records]),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])) if records else None,
        "truncation_rate": float(np.mean([r["truncated"] for r in records])) if records else None,
    }
    return out


def qualitative_candidates(records: list[dict], baseline: list[dict] | None = None, k: int = 5) -> dict:
    """Pick candidate examples for the report's qualitative evidence (to be read by a human)."""
    by_reward = sorted([r for r in records if r.get("reward") is not None], key=lambda r: r["reward"])
    out = {
        "highest_reward": by_reward[-k:][::-1],
        "lowest_reward": by_reward[:k],
        "longest": sorted(records, key=lambda r: r["response_tokens"])[-k:][::-1],
        "truncated": [r for r in records if r["truncated"]][:k],
    }
    if baseline:
        base = {r["prompt_id"]: r for r in baseline}
        paired = []
        for r in records:
            b = base.get(r["prompt_id"])
            if b is None or r.get("reward") is None or b.get("reward") is None:
                continue
            paired.append({
                "prompt_id": r["prompt_id"],
                "reward_gain": r["reward"] - b["reward"],
                "length_gain": r["response_tokens"] - b["response_tokens"],
                "policy_response": r["response"],
                "baseline_response": b["response"],
                "policy_reward": r["reward"],
                "baseline_reward": b["reward"],
                "policy_tokens": r["response_tokens"],
                "baseline_tokens": b["response_tokens"],
            })
        paired.sort(key=lambda x: x["reward_gain"])
        out["largest_reward_gain_vs_baseline"] = paired[-k:][::-1]
        out["largest_reward_loss_vs_baseline"] = paired[:k]
        out["largest_length_gain_vs_baseline"] = sorted(paired, key=lambda x: x["length_gain"])[-k:][::-1]
    return out


def evaluate_rl_policy(
    config_path: str,
    adapter: str | None,
    name: str,
    results_dir: str,
    max_new_tokens: int,
    batch_size: int = 16,
    max_prompts: int | None = None,
    reward_max_length: int = 1280,
):
    """Common held-out protocol for the PPO/GRPO policies (Tasks 2-3).

    Fixed prompt set (rl_prompt_pool_eval.jsonl), fixed decoding (configs/base.yaml generation
    block: temperature 0.7, top-p 0.9), fixed seed, fixed generation cap, same reward model.
    Writes eval_<name>.json, generations_<name>.jsonl and examples_<name>.json.
    """
    from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
    from common.logging_utils import save_json, set_seed
    from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
    from common.training_utils import Timer

    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    out_dir = repo_path(results_dir)
    timer = Timer()
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_prompts is not None:
        rows = rows[: int(max_prompts)]
    base = adapter is None or str(adapter).lower() in {"", "none", "sft", "base"}
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=None if base else adapter, trainable=False)
    reward = load_reward_model(cfg)
    print(f"[eval:{name}] {len(rows)} held-out prompts, cap {max_new_tokens} tokens", flush=True)
    records, agg = generate_and_score(
        policy, tokenizer, rows,
        max_new_tokens=int(max_new_tokens),
        gen_cfg=cfg["generation"],
        reward=reward,
        reward_max_length=int(reward_max_length),
        batch_size=batch_size,
        seed=int(cfg["seed"]),
    )
    write_jsonl(out_dir / f"generations_{name}.jsonl", records)
    baseline_name = "midpoint" if name != "midpoint" else None
    baseline = None
    if baseline_name and (out_dir / f"generations_{baseline_name}.jsonl").exists():
        baseline = read_jsonl(out_dir / f"generations_{baseline_name}.jsonl")
    save_json(out_dir / f"examples_{name}.json", qualitative_candidates(records, baseline))
    summary = {
        "name": name,
        "adapter": adapter,
        "eval_prompts": cfg["paths"]["rl_prompt_eval"],
        "n_prompts": len(rows),
        "decoding": {**cfg["generation"], "max_new_tokens": int(max_new_tokens), "seed": int(cfg["seed"])},
        **agg,
        "eval_wall_clock_sec": timer(),
    }
    save_json(out_dir / f"eval_{name}.json", summary)
    print(f"[eval:{name}] reward={agg['reward_mean']:.4f} kl_tok={agg['kl_token_mean']:.5f} "
          f"len={agg['length']['mean']:.1f} entropy={agg['entropy_token_mean']:.3f}", flush=True)
    del policy
    clear_gpu(*reward)
    clear_gpu()
    return summary
