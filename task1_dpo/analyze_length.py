"""Task 1 Step 3: length-confounding study.

1. Dataset diagnostics (a property of the data, not of any policy): how often the preferred
   response is longer in the standard training file and the balanced file.
2. Train one DPO model on the supplied length-balanced file (same config as the standard run).
3. Evaluate standard vs length-balanced DPO on the length-stratified held-out set, separately for
   preferred-longer / length-matched / rejected-longer pairs.
4. Generated length + explicit word-limit compliance on the common word-limit prompt set for the
   SFT policy, standard DPO and length-balanced DPO (greedy + a few seeded samples per prompt).
5. Generated length / reward on the standard held-out prompts (the standard eval protocol).

Resumable. Writes results/task1_dpo/length_analysis.json (+ CSVs and per-response files).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, preference_responses, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import parse_word_limit, safe_corr, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_tokenizer
from common.training_utils import adapter_exists, describe
from task1_dpo.evaluate import evaluate_dpo_policy, is_base, pair_metrics, summarize_pairs
from task1_dpo.train import run_training


def dataset_length_stats(rows, tokenizer, name):
    """Chosen-vs-rejected length structure of a preference file."""
    cl, rl, sd = [], [], []
    for row in rows:
        yc, yr = preference_responses(row)
        c = row.get("chosen_tokens")
        r = row.get("rejected_tokens")
        if c is None or r is None:
            c = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
            r = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        cl.append(int(c))
        rl.append(int(r))
        if row.get("score_chosen") is not None and row.get("score_rejected") is not None:
            sd.append(float(row["score_chosen"]) - float(row["score_rejected"]))
    cl, rl = np.array(cl), np.array(rl)
    diff = cl - rl
    out = {
        "file": name,
        "n": int(len(cl)),
        "frac_chosen_longer": float((diff > 0).mean()),
        "frac_rejected_longer": float((diff < 0).mean()),
        "frac_equal_length": float((diff == 0).mean()),
        "chosen_tokens": describe(cl.tolist()),
        "rejected_tokens": describe(rl.tolist()),
        "length_difference": describe(diff.tolist()),
        "corr_length_diff_vs_score_diff": safe_corr(diff[: len(sd)], sd) if sd else None,
    }
    if rows and "length_stratum" in rows[0]:
        out["stratum_counts"] = pd.Series([r["length_stratum"] for r in rows]).value_counts().to_dict()
    return out


@torch.no_grad()
def word_limit_generations(cfg, adapter, name, tokenizer, prompts, n_samples=4):
    policy = load_policy(cfg, adapter_path=None if is_base(adapter) else adapter, trainable=False)
    msgs = [prompt_messages(p) for p in prompts]
    records = []
    modes = [("greedy", False, 0)] + [(f"sample{s}", True, s) for s in range(n_samples)]
    for mode, sample, s in modes:
        set_seed(int(cfg["seed"]) + s)
        gen = batch_generate(policy, tokenizer, msgs, max_prompt_length=512,
                             max_new_tokens=int(cfg["max_generation_tokens"]),
                             temperature=float(cfg["generation"]["temperature"]),
                             top_p=float(cfg["generation"]["top_p"]), do_sample=sample)
        for p, m, text, n, trunc in zip(prompts, msgs, gen["responses"], gen["response_lengths"], gen["truncated"]):
            prompt_text = m[-1]["content"]
            records.append({
                "policy": name,
                "prompt_id": p.get("prompt_id"),
                "decoding": mode,
                "word_limit": parse_word_limit(prompt_text),
                "response_words": word_count(text),
                "response_tokens": int(n),
                "truncated": bool(trunc),
                "compliant": word_limit_compliance(prompt_text, text),
                "response": text,
            })
    del policy
    clear_gpu()
    return records


def summarize_word_limit(records):
    df = pd.DataFrame(records)
    out = {}
    for (pol, dec), g in df.groupby(["policy", df["decoding"].where(df["decoding"] == "greedy", "sampled")]):
        comp = g["compliant"].dropna()
        out.setdefault(pol, {})[dec] = {
            "n": int(len(g)),
            "compliance_rate": float(comp.mean()) if len(comp) else None,
            "words": describe(g["response_words"].tolist()),
            "tokens": describe(g["response_tokens"].tolist()),
            "words_over_limit_mean": float((g["response_words"] - g["word_limit"]).clip(lower=0).mean()),
            "truncation_rate": float(g["truncated"].mean()),
        }
    return out


def normalized_margin_analysis(records):
    """Summed vs per-token (length-normalised) DPO margins on already-computed pair records.

    summed margin      m      = [log pi/pi_ref](y+) - [log pi/pi_ref](y-)           (what DPO optimises)
    per-token margin   m_norm = [log pi/pi_ref](y+)/|y+| - [log pi/pi_ref](y-)/|y-|
    If m_norm prefers y+ where m does not, the summed objective's length sensitivity (not the content
    preference) is what flips the decision.
    """
    def one(rs):
        if not rs:
            return {"n": 0}
        cr = np.array([r["policy_chosen_logp"] - r["ref_chosen_logp"] for r in rs])
        rr = np.array([r["policy_rejected_logp"] - r["ref_rejected_logp"] for r in rs])
        ct = np.array([max(r["chosen_response_tokens"], 1) for r in rs], float)
        rt = np.array([max(r["rejected_response_tokens"], 1) for r in rs], float)
        m, mn = cr - rr, cr / ct - rr / rt
        return {
            "n": len(rs),
            "summed_margin_accuracy": float((m > 0).mean()),
            "per_token_margin_accuracy": float((mn > 0).mean()),
            "chosen_logratio_per_token": float((cr / ct).mean()),
            "rejected_logratio_per_token": float((rr / rt).mean()),
            "chosen_tokens_mean": float(ct.mean()),
            "rejected_tokens_mean": float(rt.mean()),
            "corr_summed_margin_vs_length_diff": safe_corr(m, ct - rt),
            "corr_per_token_margin_vs_length_diff": safe_corr(mn, ct - rt),
        }
    out = {"overall": one(records)}
    strata = sorted({r.get("length_stratum") for r in records if r.get("length_stratum")})
    for st in strata:
        out[st] = one([r for r in records if r.get("length_stratum") == st])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--samples", type=int, default=4, help="sampled responses per word-limit prompt (plus greedy)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    tokenizer = load_tokenizer(cfg["base_model"])

    # 1. Dataset structure.
    data_stats = {
        "standard_train": dataset_length_stats(read_jsonl(cfg["paths"]["dpo_standard_train"]), tokenizer, cfg["paths"]["dpo_standard_train"]),
        "standard_eval": dataset_length_stats(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, cfg["paths"]["dpo_standard_eval"]),
        "length_balanced_train": dataset_length_stats(balanced, tokenizer, cfg["paths"]["dpo_length_train"]),
        "length_stratified_eval": dataset_length_stats(stratified, tokenizer, cfg["paths"]["dpo_length_eval"]),
    }

    # 2. Train the length-balanced condition (identical recipe to the standard run).
    std_out, bal_out = cfg["standard_output"], cfg["length_output"]
    if not adapter_exists(std_out):
        raise SystemExit(f"Standard DPO adapter not found at {std_out}; run task1_dpo.train first.")
    if args.force or not adapter_exists(bal_out):
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=bal_out)

    # 3. Per-stratum preference fitting on the stratified held-out set.
    strata = {}
    for name, adapter in [("standard", std_out), ("length_balanced", bal_out)]:
        path = results / f"pairs_{name}_stratified.jsonl"
        if args.force or not path.exists():
            policy = load_policy(cfg, adapter_path=adapter, trainable=False)
            recs, _ = pair_metrics(policy, tokenizer, stratified, int(cfg["max_sequence_length"]), float(cfg["beta"]))
            write_jsonl(path, recs)
            del policy
            clear_gpu()
        strata[name] = summarize_pairs(read_jsonl(path))

    # 4. Word-limit prompts.
    wl_prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    wl_path = results / "word_limit_generations.jsonl"
    if args.force or not wl_path.exists():
        recs = []
        for name, adapter in [("sft", None), ("standard", std_out), ("length_balanced", bal_out)]:
            print(f"[length] word-limit generations for {name}")
            recs += word_limit_generations(cfg, adapter, name, tokenizer, wl_prompts, args.samples)
        write_jsonl(wl_path, recs)
    wl_summary = summarize_word_limit(read_jsonl(wl_path))

    # 5. Standard held-out generation protocol for all three policies.
    gen_summary = {}
    for name, adapter in [("sft", None), ("standard", std_out), ("length_balanced", bal_out)]:
        ev = results / f"eval_{name}.json"
        if args.force or not ev.exists():
            evaluate_dpo_policy(args.config, adapter, name, batch_size=args.batch_size)
        e = load_json(ev)
        gen_summary[name] = {
            "heldout_pref_accuracy": e["pairs"]["overall"].get("preference_accuracy"),
            "reward_mean": e.get("generation", {}).get("reward_mean"),
            "kl_token_mean": e.get("generation", {}).get("kl_token_mean"),
            "length": e.get("generation", {}).get("length"),
        }

    normalized = {name: normalized_margin_analysis(read_jsonl(results / f"pairs_{name}_stratified.jsonl"))
                  for name in ["standard", "length_balanced"]}
    std_pairs = results / "pairs_standard.jsonl"
    if std_pairs.exists():
        normalized["standard_on_standard_eval"] = normalized_margin_analysis(read_jsonl(std_pairs))
    norm_rows = [{"policy": pol, "stratum": st, **m} for pol, d in normalized.items() for st, m in d.items()]
    pd.DataFrame(norm_rows).to_csv(results / "length_normalized_margins.csv", index=False)

    out = {
        "dataset_length_structure": data_stats,
        "summed_vs_per_token_margins": normalized,
        "stratified_preference_fitting": strata,
        "word_limit": wl_summary,
        "standard_heldout_generation": gen_summary,
    }
    save_json(results / "length_analysis.json", out)

    rows = []
    for name, s in strata.items():
        for stratum, m in s.get("by_stratum", {}).items():
            rows.append({"policy": name, "stratum": stratum, **m})
    pd.DataFrame(rows).to_csv(results / "length_strata_summary.csv", index=False)
    wl_rows = [{"policy": p, "decoding": d, **{k: v for k, v in m.items() if not isinstance(v, dict)},
                "words_mean": m["words"]["mean"], "tokens_mean": m["tokens"]["mean"]}
               for p, dd in wl_summary.items() for d, m in dd.items()]
    pd.DataFrame(wl_rows).to_csv(results / "word_limit_summary.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pd.DataFrame(rows))
        print(pd.DataFrame(wl_rows))
        print(pd.DataFrame(norm_rows)[["policy", "stratum", "n", "summed_margin_accuracy", "per_token_margin_accuracy",
                                       "corr_summed_margin_vs_length_diff", "corr_per_token_margin_vs_length_diff"]])


if __name__ == "__main__":
    main()
