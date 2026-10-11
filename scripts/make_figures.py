"""Build report figures from the saved results (never from notebook state).

python -m scripts.make_figures            -> report/figures/*.png (skips any figure whose inputs are missing)

Every figure reads machine-readable files under results/, so each plotted value traces to a file.
Single y-axis per panel; multi-measure trajectories are small multiples; categorical colours are
assigned in a fixed order and bars carry value labels (the CSVs in results/ are the table views).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path

OUT = repo_path("report/figures")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})


def _exists(*paths):
    return all(repo_path(p).exists() for p in paths)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def _trajectory_grid(log, panels, title, name):
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.7 * rows), squeeze=False)
    x = [r.get("update", i + 1) for i, r in enumerate(log)]
    for ax, (key, label) in zip(axes.flat, panels):
        y = [r.get(key) for r in log]
        ax.plot(x, [np.nan if v is None else v for v in y], color=COLORS[0], marker="o", markersize=2.5)
        ax.set_title(label, color=INK)
        ax.set_xlabel("update")
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, color=INK, fontsize=9)
    fig.tight_layout()
    _save(fig, name)


def _multi_trajectory(logs: dict, panels, title, name):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 1.9), squeeze=False)
    for ax, (key, label) in zip(axes.flat, panels):
        for i, (lab, log) in enumerate(logs.items()):
            ax.plot([r["update"] for r in log], [r.get(key) for r in log], color=COLORS[i], marker="o", markersize=2.5, label=lab)
        ax.set_title(label, color=INK)
        ax.set_xlabel("update")
        ax.ticklabel_format(axis="y", useOffset=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=7, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, color=INK, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def _bar_panels(df, label_col, panels, title, name, fmt="{:.3g}"):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.0), squeeze=False)
    labels = df[label_col].astype(str).tolist()
    for ax, (key, lab) in zip(axes.flat, panels):
        vals = df[key].astype(float).to_numpy()
        bars = ax.bar(range(len(vals)), vals, color=[COLORS[i % len(COLORS)] for i in range(len(vals))],
                      edgecolor="white", linewidth=1.0, width=0.7)
        for b, v in zip(bars, vals):
            if np.isfinite(v) and v != 0:
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", fontsize=6.5, color=INK, xytext=(0, 1), textcoords="offset points")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(lab, color=INK)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, color=INK, fontsize=9)
    fig.tight_layout()
    _save(fig, name)


def _grouped_bars(ax, categories, series: dict, fmt="{:.2f}", rate=False):
    k = len(series)
    w = 0.8 / k
    for i, (lab, vals) in enumerate(series.items()):
        xs = np.arange(len(categories)) + (i - (k - 1) / 2) * w
        bars = ax.bar(xs, vals, w * 0.92, color=COLORS[i], label=lab, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            # Value labels only where they fit; dense panels rely on the results/*.csv table view.
            if v is not None and np.isfinite(v) and v != 0 and len(categories) * k <= 8:
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", fontsize=6, color=INK, xytext=(0, 1), textcoords="offset points")
    if rate:
        ax.set_ylim(0, 1.12)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=7)
    ax.grid(axis="x", visible=False)


def _header(fig, ax, title):
    """Left-aligned title with the legend in the figure header (never over the data)."""
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(title, color=INK, fontsize=9, x=0.01, ha="left")
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=7, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def fig_dpo():
    if _exists("results/task1_dpo/beta_ablation_summary.csv"):
        df = pd.read_csv(repo_path("results/task1_dpo/beta_ablation_summary.csv"))
        # KL is shown in kl_fp32_drift.png (float32); the fp16 values here carry a ~1e-5/token precision floor.
        _bar_panels(df, "condition", [("heldout_pref_accuracy", "held-out pref. accuracy"),
                                      ("reward_mean", "RM score"), ("gen_len_mean", "generated tokens (mean)")],
                    "Task 1: DPO beta forks (short budget) vs standard run", "task1_dpo_beta.png")
    if _exists("results/task1_dpo/length_strata_summary.csv"):
        df = pd.read_csv(repo_path("results/task1_dpo/length_strata_summary.csv"))
        strata = sorted(df["stratum"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.1))
        for ax, (key, lab) in zip(axes, [("preference_accuracy", "held-out preference accuracy"), ("margin_mean", "mean DPO margin")]):
            _grouped_bars(ax, strata, {p: [float(df[(df.policy == p) & (df.stratum == s)][key].iloc[0]) for s in strata]
                                       for p in df["policy"].unique()})
            ax.set_title(lab, color=INK)
        _header(fig, axes[0], "Task 1: standard vs length-balanced DPO by length stratum")
        _save(fig, "task1_dpo_length_strata.png")
    if _exists("results/task1_dpo/word_limit_summary.csv"):
        df = pd.read_csv(repo_path("results/task1_dpo/word_limit_summary.csv"))
        df = df[df["decoding"] == "sampled"].reset_index(drop=True)
        _bar_panels(df, "policy", [("compliance_rate", "word-limit compliance"), ("words_mean", "response words (mean)")],
                    "Task 1: word-limit prompts (sampled responses)", "task1_dpo_word_limit.png")


def fig_ppo():
    if _exists("results/task2_ppo/train_logs/standard.jsonl"):
        log = read_jsonl(repo_path("results/task2_ppo/train_logs/standard.jsonl"))
        # KL is omitted: in the released float16 setup it sits on a ~1e-5 precision floor (see fig_drift).
        _trajectory_grid(log, [("reward_raw", "learned reward"),
                               ("policy_loss", "policy loss"), ("value_loss", "value loss"), ("entropy", "entropy"),
                               ("clip_fraction_last_epoch", "clip fraction"), ("grad_norm_policy", "policy grad norm"),
                               ("response_length", "response tokens")],
                         "Task 2: standard PPO continuation from the supplied midpoint", "task2_ppo_standard.png")
    kl_logs = {}
    for beta in ["0", "0.1", "0.2"]:
        p = f"results/task2_ppo/train_logs/fork_eps0.2_kl{beta}.jsonl"
        if _exists(p):
            kl_logs[f"beta_KL={beta}"] = read_jsonl(repo_path(p))
    if kl_logs:
        _multi_trajectory(kl_logs, [("reward_raw", "learned reward"), ("entropy", "entropy"), ("policy_loss", "policy loss"),
                                    ("response_length", "response tokens")], "Task 2: KL-pressure forks", "task2_ppo_kl_forks.png")
    clip_logs = {}
    for eps in ["0.05", "0.2", "0.5"]:
        p = f"results/task2_ppo/train_logs/fork_eps{eps}_kl0.1.jsonl"
        if _exists(p):
            clip_logs[f"eps={eps}"] = read_jsonl(repo_path(p))
    if clip_logs:
        _multi_trajectory(clip_logs, [("reward_raw", "learned reward"), ("policy_loss", "policy loss"),
                                      ("clip_fraction_last_epoch", "clip fraction"), ("ratio_abs_dev_last_epoch", "mean |ratio - 1|")],
                          "Task 2: clipping forks", "task2_ppo_clip_forks.png")
    if _exists("results/task2_ppo/clipping_cached.json"):
        d = json.loads(repo_path("results/task2_ppo/clipping_cached.json").read_text())
        eps = list(d["after_updates"].keys())
        fig, ax = plt.subplots(figsize=(3.6, 2.1))
        _grouped_bars(ax, [f"eps={e}" for e in eps], {
            "clip fraction": [d["after_updates"][e]["final_full_batch"]["clip_fraction"] for e in eps],
            "affected-token fraction": [d["after_updates"][e]["final_full_batch"]["affected_fraction"] for e in eps]}, fmt="{:.3f}", rate=True)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("Task 2: cached batch after PPO epochs", color=INK)
        _save(fig, "task2_ppo_cached_clipping.png")


def fig_grpo():
    if _exists("results/task3_grpo/train_logs/standard.jsonl"):
        log = read_jsonl(repo_path("results/task3_grpo/train_logs/standard.jsonl"))
        _trajectory_grid(log, [("reward_mean", "reward"), ("truncated_fraction", "truncated (masked) fraction"),
                               ("group_reward_std_mean", "within-group reward std"), ("frac_zero_std_groups", "zero-std group fraction"),
                               ("policy_loss", "policy loss"), ("grad_norm", "grad norm"), ("entropy", "entropy"),
                               ("response_length", "response tokens")],
                         "Task 3: standard GRPO continuation from the supplied midpoint", "task3_grpo_standard.png")
    if _exists("results/task3_grpo/group_size_summary.csv"):
        df = pd.read_csv(repo_path("results/task3_grpo/group_size_summary.csv"))
        bins = [b for b in ["all", "hard", "medium", "easy"] if b in set(df["bin"])]
        ks = sorted(df["K"].unique())
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
        for ax, (key, lab) in zip(axes, [("informative_rate_std_gt_0.1", "P(group reward std > 0.1)"),
                                         ("informative_rate_std_gt_0.25", "P(group reward std > 0.25)"),
                                         ("mean_group_reward_std", "mean group std (biased low at small K)")]):
            _grouped_bars(ax, bins, {f"K={k}": [float(df[(df.K == k) & (df.bin == b)][key].iloc[0]) for b in bins] for k in ks})
            ax.set_title(lab, color=INK)
        _header(fig, axes[0], "Task 3: equal-generation group-size study")
        _save(fig, "task3_grpo_group_size.png")
    logs = {}
    for lt in ["grpo", "dr_grpo"]:
        p = f"results/task3_grpo/train_logs/fork_{lt}.jsonl"
        if _exists(p):
            logs[lt] = read_jsonl(repo_path(p))
    if logs:
        _multi_trajectory(logs, [("reward_mean", "reward"), ("response_length", "response tokens"), ("policy_loss", "policy loss"),
                                 ("grad_norm", "grad norm")], "Task 3: canonical vs Dr. GRPO normalisation", "task3_grpo_normalization.png")
        fig, ax = plt.subplots(figsize=(3.6, 2.3))
        for i, (lt, log) in enumerate(logs.items()):
            comps = [c for r in log for c in r["completions"] if not c["masked"] and c.get("grad_norm") is not None]
            ax.scatter([c["length"] for c in comps], [c["grad_norm"] for c in comps], s=14, color=COLORS[i], label=lt,
                       edgecolor="white", linewidth=0.5)
        ax.set_xlabel("completion tokens")
        ax.set_ylabel("per-completion grad norm")
        ax.set_title("Task 3: gradient allocation vs length", color=INK)
        ax.legend(fontsize=7)
        _save(fig, "task3_grpo_gradnorm_vs_length.png")


def fig_safety():
    if _exists("results/task4_safety/safety_summary.csv"):
        df = pd.read_csv(repo_path("results/task4_safety/safety_summary.csv"))
        # The judge never emits OVER_REFUSAL; refusals of safe prompts appear as JUSTIFIED_REFUSAL, so that
        # column is the judge's effective over-refusal signal.
        metrics = [("safe_answer_rate", "safe: answered"), ("safe_prompts_labelled_justified_refusal", "safe: refused\n(judge said JUST._REF.)"),
                   ("unsafe_compliance_rate", "unsafe: complied"), ("unsafe_justified_refusal_rate", "unsafe: refused")]
        fig, ax = plt.subplots(figsize=(7.2, 2.2))
        _grouped_bars(ax, [m[1] for m in metrics], {p: [float(df[df.policy == p][m[0]].iloc[0]) for m in metrics] for p in df["policy"]}, rate=True)
        _header(fig, ax, "Task 4: XSTest safety calibration (AI judge labels)")
        _save(fig, "task4_safety_rates.png")
    if _exists("results/task4_safety/audit_confusion.csv"):
        conf = pd.read_csv(repo_path("results/task4_safety/audit_confusion.csv"), index_col=0)
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.imshow(conf.to_numpy(), cmap="Blues")
        short = [c.replace("_", "\n") for c in conf.columns]
        ax.set_xticks(range(len(conf.columns)), short, fontsize=5.5)
        ax.set_yticks(range(len(conf.index)), [c.replace("_", " ") for c in conf.index], fontsize=6)
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                v = int(conf.iat[i, j])
                if v:
                    ax.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if v > conf.to_numpy().max() / 2 else INK)
        ax.set_xlabel("AI judge label")
        ax.set_ylabel("manual label")
        ax.grid(False)
        ax.set_title("Task 4: manual audit vs AI judge (n=%d)" % int(conf.to_numpy().sum()), color=INK)
        _save(fig, "task4_audit_confusion.png")


def fig_feedback():
    p = "results/task5_feedback/perturbations/summary.csv"
    if _exists(p):
        df = pd.read_csv(repo_path(p))
        df = df[df["kind"] != "conflict"]
        cats = list(dict.fromkeys(df["category"]))
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
        for ax, (key, lab) in zip(axes, [("better_rate", "better-response rate"), ("tie_rate", "tie rate"), ("wrong_preference_rate", "wrong-preference rate")]):
            _grouped_bars(ax, [c.replace("_", "\n") for c in cats],
                          {m: [float(df[(df.category == c) & (df.mechanism == m)][key].iloc[0]) if ((df.category == c) & (df.mechanism == m)).any() else np.nan for c in cats]
                           for m in ["rlvr", "rlaif"]}, rate=True)
            ax.set_title(lab, color=INK)
            ax.tick_params(axis="x", labelsize=6)
        _header(fig, axes[0], "Task 5: controlled reward diagnostics")
        _save(fig, "task5_perturbations.png")
    p = "results/task5_feedback/feedback_comparison.csv"
    if _exists(p):
        df = pd.read_csv(repo_path(p))
        fig, ax = plt.subplots(figsize=(3.6, 2.1))
        pols = df["policy"].tolist()
        series = {"GSM8K strict": df["gsm_exact_accuracy"].tolist(), "SVAMP strict": df["svamp_exact_accuracy"].tolist()}
        for ds, lab in [("gsm", "GSM8K lenient"), ("transfer", "SVAMP lenient")]:
            vals = []
            for pol in pols:
                f = repo_path(f"results/task5_feedback/{ds}/generations_{pol}.jsonl")
                rows = read_jsonl(f) if f.exists() else []
                vals.append(np.mean([_lenient_correct(r["response"], r["gold_final"]) for r in rows]) if rows else np.nan)
            series[lab] = vals
        fig, ax = plt.subplots(figsize=(7.2, 2.2))
        _grouped_bars(ax, pols, series, rate=True)
        ax.set_title("Task 5: accuracy under the official '#### n' verifier (strict) vs lenient answer extraction (sensitivity only)", color=INK)
        ax.legend(fontsize=6.5, ncol=4, loc="upper left")
        _save(fig, "task5_accuracy.png")


_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"


def _lenient_correct(response, gold):
    """Sensitivity-only extraction: last \\boxed{}, else last '####' number, else last number in the text."""
    import re
    text = str(response)
    cand = None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        cand = boxed[-1]
    else:
        hashes = re.findall(r"####[^\d\-+]*(" + _NUM + ")", text)
        cand = hashes[-1] if hashes else None
        if cand is None:
            nums = re.findall(_NUM, text)
            cand = nums[-1] if nums else None
    nums = re.findall(_NUM, cand or "")
    if not nums:
        return 0.0
    try:
        return float(abs(float(nums[-1].replace(",", "")) - float(str(gold).replace(",", ""))) < 1e-6)
    except ValueError:
        return 0.0


def fig_drift():
    """Float32 drift from the reference (scripts/kl_fp32_drift.py), with paired SEs."""
    if not _exists("results/kl_fp32_drift.csv"):
        return
    df = pd.read_csv(repo_path("results/kl_fp32_drift.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    t1 = df[df.adapter.str.startswith("task1/")]
    ax = axes[0]
    ax.bar(range(len(t1)), t1.kl_per_token_fp32, yerr=t1.kl_per_token_se, color=COLORS[0], edgecolor="white", capsize=2)
    ax.set_xticks(range(len(t1)), [a.split("/", 1)[1] for a in t1.adapter], rotation=30, ha="right", fontsize=6.5)
    ax.set_yscale("log")
    lo = float(t1.kl_per_token_fp32.min()) * 0.4
    ax.set_ylim(bottom=lo)
    for x, v in enumerate(t1.kl_per_token_fp32):
        ax.text(x, lo * 1.5, f"{v:.1e}", ha="center", va="center", fontsize=6, color="white")
    ax.set_ylabel("nats / token (log scale)")
    ax.set_title("Task 1: KL to reference (fp32, per token)", color=INK)
    ax.grid(axis="x", visible=False)
    rl = df[df.adapter.str.startswith(("task2/", "task3/"))].dropna(subset=["diff_vs_midpoint"])
    ax = axes[1]
    cols = [COLORS[1] if a.startswith("task2/") else COLORS[2] for a in rl.adapter]
    ax.bar(range(len(rl)), rl.diff_vs_midpoint, yerr=rl.diff_vs_midpoint_se, color=cols, edgecolor="white", capsize=2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(rl)), [a.replace("task2/", "PPO ").replace("task3/", "GRPO ").replace("forks/", "").replace("fork_", "") for a in rl.adapter], rotation=30, ha="right", fontsize=6.5)
    ax.set_title("Tasks 2-3: change in KL vs midpoint (fp32, paired SE)", color=INK)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "kl_fp32_drift.png")


def main():
    for f in (fig_dpo, fig_ppo, fig_grpo, fig_safety, fig_feedback, fig_drift):
        try:
            f()
        except Exception as exc:  # one missing/partial result must not block the other figures
            print(f"[figures] {f.__name__} skipped: {exc}")


if __name__ == "__main__":
    main()
