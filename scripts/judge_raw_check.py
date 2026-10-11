"""Diagnostic: is the RLAIF judge's near-universal TIE a real judgment or a parse failure?

The supplied PairwiseAIJudge decodes 4 tokens, searches for A/B/TIE, and silently maps anything
unparseable to TIE; its cache stores only the parsed result. This script re-asks the judge (no
cache) on fixed diagnostic pairs whose correct answer is unambiguous (clean_correct vs a wrong
final answer), in both A/B orders, and records the RAW decoded text with the released 4-token
budget and with a 32-token budget.

    python -m scripts.judge_raw_check --n-problems 10
Writes results/task5_feedback/judge_raw_check.jsonl and prints a tally.
"""

from __future__ import annotations

import argparse
import collections
import re

import torch

from common.data import load_yaml, repo_path, write_jsonl
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.score_perturbations import load_diagnostic_groups


@torch.no_grad()
def raw(judge, problem, a, b, max_new_tokens):
    text = PAIRWISE_RUBRIC.format(problem=problem, a=a, b=b)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id)
    return judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def parse(decoded):
    m = re.search(r"\b(A|B|TIE)\b", decoded.strip().upper())
    return m.group(1) if m else "UNPARSEABLE(->TIE)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--n-problems", type=int, default=10)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    judge = PairwiseAIJudge(cfg, "results/task5_feedback/_raw_check_unused_cache.json")
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    rows, tally = [], collections.Counter()
    for pid in list(groups)[: args.n_problems]:
        v = groups[pid]
        q = v["clean_correct"]["question"]
        for worse in ["good_reasoning_wrong_final", "gold_distractor_wrong_final"]:
            good, bad = v["clean_correct"]["response"], v[worse]["response"]
            for order, (a, b, correct) in {"good_as_A": (good, bad, "A"), "good_as_B": (bad, good, "B")}.items():
                r4, r32 = raw(judge, q, a, b, 4), raw(judge, q, a, b, 32)
                p4 = parse(r4)
                tally[("correct" if p4 == correct else p4 if p4.startswith("UNP") or p4 == "TIE" else "wrong")] += 1
                rows.append({"problem_id": pid, "worse_variant": worse, "order": order, "correct_choice": correct,
                             "raw_4_tokens": r4, "parsed_4": p4, "raw_32_tokens": r32, "parsed_32": parse(r32)})
                print(f"[{pid} {worse[:12]} {order}] 4tok={r4!r} -> {p4} | 32tok={r32[:80]!r}", flush=True)
    write_jsonl(repo_path("results/task5_feedback/judge_raw_check.jsonl"), rows)
    print("\nTally (4-token budget, as released):", dict(tally))
    print("Unparseable with 32 tokens:", sum(r["parsed_32"].startswith("UNP") for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
