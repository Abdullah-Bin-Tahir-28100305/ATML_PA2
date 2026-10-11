"""Matched short PPO forks shared by the clipping and KL-pressure studies.

A fork is identified by (clip_epsilon, kl_beta). Every fork:
  * starts from the identical supplied midpoint policy + critic,
  * runs `fork_updates` updates with the same seed, prompt order and generation settings,
  * is evaluated with the common held-out protocol.
The (eps=0.20, beta=0.10) fork is shared by both studies, so it is trained once.
"""

from __future__ import annotations

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.training_utils import adapter_exists
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo_policy


def fork_name(eps: float, beta: float) -> str:
    return f"fork_eps{eps:g}_kl{beta:g}"


def ensure_fork(config_path: str, eps: float, beta: float, batch_size: int = 16, max_prompts=None, force=False):
    cfg = load_yaml(config_path)
    name = fork_name(eps, beta)
    out = f"outputs/task2_ppo/forks/{name}"
    if force or not adapter_exists(out):
        run_ppo(config_path, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name)
    else:
        print(f"[fork] reusing {out}")
    ev = repo_path(cfg["results_dir"]) / f"eval_{name}.json"
    if force or not ev.exists():
        evaluate_ppo_policy(config_path, out, name, batch_size=batch_size, max_prompts=max_prompts)
    return name


def ensure_midpoint_eval(config_path: str, batch_size: int = 16, max_prompts=None):
    cfg = load_yaml(config_path)
    ev = repo_path(cfg["results_dir"]) / "eval_midpoint.json"
    if not ev.exists():
        evaluate_ppo_policy(config_path, cfg["paths"]["ppo_midpoint_policy"], "midpoint", batch_size, max_prompts)


def training_stability(config_path: str, name: str) -> dict:
    """Stability statistics computed from a run's per-update training log.

    Primary statistic (defined once, used for every fork): `ratio_abs_dev_mean` = mean over
    updates of E_t|rho_t - 1| after the last PPO epoch, i.e. how far a single update moves the
    sampled tokens' probabilities. Also reported: std of per-update reward, max/min ratios,
    KL growth over the fork, policy grad-norm statistics and skipped (non-finite) steps.
    """
    cfg = load_yaml(config_path)
    log = read_jsonl(repo_path(cfg["results_dir"]) / "train_logs" / f"{name}.jsonl")
    get = lambda k: np.array([r[k] for r in log if r.get(k) is not None], dtype=float)
    kl = get("kl_token_mean")
    gn = get("grad_norm_policy")
    return {
        "ratio_abs_dev_mean": float(get("ratio_abs_dev_last_epoch").mean()),
        "ratio_max": float(get("ratio_max").max()),
        "ratio_min": float(get("ratio_min").min()),
        "clip_fraction_mean": float(get("clip_fraction_last_epoch").mean()),
        "reward_std_over_updates": float(get("reward_raw").std()),
        "reward_mean_over_updates": float(get("reward_raw").mean()),
        "kl_first": float(kl[0]) if kl.size else None,
        "kl_last": float(kl[-1]) if kl.size else None,
        "kl_max": float(kl.max()) if kl.size else None,
        "kl_exact_last": float(get("kl_exact_token_mean")[-1]) if get("kl_exact_token_mean").size else None,
        "policy_grad_norm_mean": float(gn.mean()) if gn.size else None,
        "policy_grad_norm_max": float(gn.max()) if gn.size else None,
        "value_loss_mean": float(get("value_loss").mean()),
        "entropy_first": float(get("entropy")[0]),
        "entropy_last": float(get("entropy")[-1]),
        "length_mean": float(get("response_length").mean()),
        "skipped_steps": int(sum(r.get("skipped_steps", 0) for r in log)),
    }


def eval_row(config_path: str, name: str) -> dict:
    cfg = load_yaml(config_path)
    e = load_json(repo_path(cfg["results_dir"]) / f"eval_{name}.json")
    return {
        "heldout_reward_mean": e["reward_mean"],
        "heldout_reward_std": e["reward_std"],
        "heldout_kl_token_mean": e["kl_token_mean"],
        "heldout_kl_seq_mean": e["kl_seq_mean"],
        "heldout_entropy": e["entropy_token_mean"],
        "heldout_len_mean": e["length"]["mean"],
        "heldout_len_std": e["length"]["std"],
        "heldout_len_iqr": e["length"]["iqr"],
        "heldout_truncation_rate": e["truncation_rate"],
    }
