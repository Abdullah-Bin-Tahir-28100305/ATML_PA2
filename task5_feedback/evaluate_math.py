"""Task 5 Steps 1 and 3: in-domain (GSM8K) and out-of-domain (SVAMP) evaluation of SFT, RLVR, RLAIF.

Phase 1 - generation: one greedy (deterministic) response per problem for each policy, identical
          prompts (`messages` field), generation cap `math_max_new_tokens`.
Phase 2 - exact verifier: final-answer accuracy, format compliance ('#### <number>' parsed),
          response length, failure type (correct / wrong_final / no_final_format / truncated_no_final).
Phase 3 - fixed pairwise AI judge (cached, orientation-balanced): RLVR vs SFT and RLAIF vs SFT
          (plus RLVR vs RLAIF). Win = 1, tie = 0.5, loss = 0; ties counted separately.
          Verifier-judge agreement: on each pair, the verifier "prefers" the response whose
          exact reward is higher (tie if equal); agreement is measured on all pairs and on the
          pairs where the verifier is decisive.

Resumable. Writes results/task5_feedback/<dataset>/{generations_<policy>.jsonl, judgments.jsonl,
metrics.json, metrics.csv}.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from common.training_utils import describe
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

COMPARISONS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def row_id(row, i):
    return str(row.get("prompt_id", row.get("source_index", i)))


def failure_type(response, gold, truncated):
    pred = extract_designated_final(response)
    if pred is None:
        return "truncated_no_final" if truncated else "no_final_format"
    return "correct" if exact_reward(response, gold) == 1.0 else "wrong_final"


def generate_policy(cfg, rows, tokenizer, name, batch_size):
    policy = load_frozen_policy(cfg, name)
    set_seed(int(cfg["seed"]))
    out = []
    cap = int(cfg["math_max_new_tokens"])
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        gen = batch_generate(policy, tokenizer, [r["messages"] for r in chunk], max_prompt_length=1024,
                             max_new_tokens=cap, temperature=0.0, top_p=1.0, do_sample=False)
        for j, r in enumerate(chunk):
            resp = gen["responses"][j]
            out.append({
                "id": row_id(r, start + j),
                "source_index": r.get("source_index"),
                "policy": name,
                "question": r["question"],
                "gold_final": str(r["gold_final"]),
                "response": resp,
                "response_tokens": int(gen["response_lengths"][j]),
                "truncated": bool(gen["truncated"][j]),
                "predicted_final": extract_designated_final(resp),
                "exact_reward": exact_reward(resp, r["gold_final"]),
                "format_ok": extract_designated_final(resp) is not None,
                "failure_type": failure_type(resp, r["gold_final"], bool(gen["truncated"][j])),
            })
        print(f"  [{name}] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    del policy
    clear_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset

    # Phase 1-2: generation + exact verification.
    gens = {}
    for name in policy_specs(cfg):
        path = outdir / f"generations_{name}.jsonl"
        if args.force or not path.exists():
            print(f"[math:{args.dataset}] generating with {name}", flush=True)
            write_jsonl(path, generate_policy(cfg, rows, tokenizer, name, args.batch_size))
        gens[name] = {r["id"]: r for r in read_jsonl(path)}

    metrics = {}
    for name, g in gens.items():
        recs = list(g.values())
        metrics[name] = {
            "n": len(recs),
            "exact_accuracy": float(np.mean([r["exact_reward"] for r in recs])),
            "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
            "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
            "response_tokens": describe([r["response_tokens"] for r in recs]),
            "failure_types": pd.Series([r["failure_type"] for r in recs]).value_counts().to_dict(),
        }

    # Phase 3: pairwise AI judge.
    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
        jpath = outdir / "judgments.jsonl"
        done = set()
        if jpath.exists() and not args.force:
            done = {(r["id"], r["policy_a"], r["policy_b"]) for r in read_jsonl(jpath)}
        elif jpath.exists():
            jpath.unlink()
        ids = [row_id(r, i) for i, r in enumerate(rows)]
        for a, b in COMPARISONS:
            todo = [i for i in ids if (i, a, b) not in done]
            print(f"[math:{args.dataset}] judging {a} vs {b}: {len(todo)} pairs", flush=True)
            for n, i in enumerate(todo):
                ra, rb = gens[a][i], gens[b][i]
                t0 = time.perf_counter()
                pref = judge.compare(ra["question"], ra["response"], rb["response"])
                dt = time.perf_counter() - t0
                score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
                va, vb = ra["exact_reward"], rb["exact_reward"]
                verifier_pref = "A" if va > vb else ("B" if vb > va else "TIE")
                append_jsonl(jpath, {"id": i, "policy_a": a, "policy_b": b, "judge_pref": pref, "judge_score_a": score,
                                     "verifier_a": va, "verifier_b": vb, "verifier_pref": verifier_pref,
                                     "agree": pref == verifier_pref, "judge_seconds": dt})
                if (n + 1) % 50 == 0:
                    print(f"    {n + 1}/{len(todo)}", flush=True)
        judg = pd.DataFrame(read_jsonl(jpath))
        pairwise = {}
        for a, b in COMPARISONS:
            d = judg[(judg["policy_a"] == a) & (judg["policy_b"] == b)]
            decisive = d[d["verifier_pref"] != "TIE"]
            pairwise[f"{a}_vs_{b}"] = {
                "n": int(len(d)),
                "win_rate": float(d["judge_score_a"].mean()),
                "wins": int((d["judge_pref"] == "A").sum()),
                "ties": int((d["judge_pref"] == "TIE").sum()),
                "losses": int((d["judge_pref"] == "B").sum()),
                "verifier_judge_agreement_all": float(d["agree"].mean()),
                "verifier_judge_agreement_when_verifier_decisive": float(decisive["agree"].mean()) if len(decisive) else None,
                "n_verifier_decisive": int(len(decisive)),
                "judge_tie_when_verifier_decisive": float((decisive["judge_pref"] == "TIE").mean()) if len(decisive) else None,
                "judge_decisive_when_verifier_tie": float((d[d["verifier_pref"] == "TIE"]["judge_pref"] != "TIE").mean()) if (d["verifier_pref"] == "TIE").any() else None,
                "mean_judge_seconds_uncached": float(d.loc[d["judge_seconds"] > 0.05, "judge_seconds"].mean()) if (d["judge_seconds"] > 0.05).any() else None,
            }
        metrics["pairwise_vs_judge"] = pairwise
        for name in ["rlvr", "rlaif"]:
            metrics[name]["ai_win_rate_vs_sft"] = pairwise[f"{name}_vs_sft"]["win_rate"]
            metrics[name]["verifier_judge_agreement_vs_sft"] = pairwise[f"{name}_vs_sft"]["verifier_judge_agreement_all"]

    metrics["meta"] = {"dataset": args.dataset, "file": dataset_path(cfg, args.dataset),
                       "decoding": "greedy", "max_new_tokens": int(cfg["math_max_new_tokens"]),
                       "judge_model": cfg["ai_judge_model"]}
    save_json(outdir / "metrics.json", metrics)
    table = [{"policy": n, "exact_accuracy": m["exact_accuracy"], "format_compliance": m["format_compliance"],
              "mean_tokens": m["response_tokens"]["mean"], "std_tokens": m["response_tokens"]["std"],
              "truncation_rate": m["truncation_rate"], "ai_win_rate_vs_sft": m.get("ai_win_rate_vs_sft"),
              "verifier_judge_agreement_vs_sft": m.get("verifier_judge_agreement_vs_sft")}
             for n, m in metrics.items() if n in policy_specs(cfg)]
    pd.DataFrame(table).to_csv(outdir / "metrics.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pd.DataFrame(table))


if __name__ == "__main__":
    main()
