# ATML PA2 — LLM Post-Training: DPO, PPO, GRPO, Safety Calibration, RLVR vs RLAIF

EE-5102 / CS-6304, Fall 2026 — Programming Assignment 2.
Repository: `https://github.com/Abdullah-Bin-Tahir-28100305/ATML_PA2`

This repository completes the course starter release (`AbDu11aHHH/ATML-PA2-LLM-PostTraining`). It keeps the
release's structure: one folder per task, with genuinely shared code in `common/`. Every reported number is
written to a machine-readable file under `results/` by a script that can be rerun with the command listed below.

---

## 1. Setup

```bash
git clone https://github.com/Abdullah-Bin-Tahir-28100305/ATML_PA2.git
cd ATML_PA2
python -m pip install -r requirements.txt
python -m scripts.download_assets      # pinned course assets from Hugging Face (data/, cached/, checkpoints/, manifests/)
python -m scripts.validate_assets
python -m scripts.check_environment
```

Run every command from the repository root. The downloaded course assets, model weights and the `outputs/`
adapters are git-ignored.

**Google Colab:** `notebooks/colab_launcher.ipynb` runs the same commands with `!python -m ...`. The notebook is
only a launcher; all logic lives in the Python modules. It keeps the repository on Google Drive, so results
survive runtime resets. Every experiment script is **resumable**: finished conditions (saved adapters or result
files) are detected and skipped. Pass `--force` to recompute.

**Precision:** the released config uses `dtype: float16`. Trainable LoRA and critic-head parameters are kept in
float32. Non-finite gradient steps are skipped and counted in the logs. On bf16-capable GPUs (A100/L4) you can
set `ATML_DTYPE=bfloat16` without editing the configs. If you do, use the same value for every run.

---

## 2. Deliberate defects found and corrected

| Task | File | Defect in the release | Correction |
|---|---|---|---|
| 1 | `task1_dpo/dpo.py` | `logits = beta * (policy_margin + ref_margin)`: the reference log-ratio was **added**. | `beta * (policy_margin - ref_margin)`, which equals `beta * [log pi/pi_ref (y+) - log pi/pi_ref (y-)]`. Check: at initialisation (policy = reference) the loss is exactly log 2. |
| 2 | `task2_ppo/ppo.py` | Clipped surrogate used `torch.maximum(surr1, surr2)`. That is an optimistic bound, so clipping never limits a beneficial update. | `torch.minimum`, the pessimistic PPO bound. Check: with eps=0.2, A>0 and rho=1.5 gives the clipped objective 1.2·A with zero gradient; A<0 and rho=0.5 gives 0.8·A, again with zero gradient. |
| 3 | `task3_grpo/grpo.py` | Group-relative advantages used the mean/std of the **whole batch** and ignored `group_ids`. | Mean/std are computed **within each prompt's group**. A group whose rewards are all equal gets advantage 0 for every member. |

The three checks above can be reproduced with the snippet in `scripts/check_objectives.py`.

### Other infrastructure fixes (not objective changes)

* `common/generation.batch_generate` passes `use_cache=True`. Training loaders set `config.use_cache=False`,
  which would make rollout generation quadratic-time. It also forwards only `input_ids`/`attention_mask`.
* `batch_generate` runs under `torch.inference_mode()`. The PPO/GRPO loops therefore clone the rollout tensors
  before they enter autograd; otherwise the first update fails.
* `common/models.token_values` now finds the transformer backbone correctly when the critic is wrapped in a
  PEFT adapter (`lora_head` mode). It also casts hidden states to the trainable value head's dtype.
* Dropout is disabled in every trained model, as TRL does by default for DPO/PPO/GRPO. As a result the PPO ratio
  is exactly 1 on the first epoch, and the policy, reference and rollout passes score tokens identically.

---

## 3. Commands for every required experiment (in order)

Tasks 4 and 5 use the **standard** Task 1–3 adapters, so run Tasks 1–3 first.

### Task 1 — DPO (`configs/dpo.yaml`)
```bash
python -m task1_dpo.train      --config configs/dpo.yaml --run-name standard                         # 1 epoch, 1500 pairs, beta=0.10
python -m task1_dpo.evaluate   --config configs/dpo.yaml --adapter none --name sft                   # starting-policy reference row
python -m task1_dpo.evaluate   --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta    --config configs/dpo.yaml                                         # beta in {0.03,0.10,0.30}, first 600 pairs
python -m task1_dpo.analyze_length --config configs/dpo.yaml                                         # trains length-balanced DPO + all length analyses
```
Supplementary (clearly labelled, not a replacement for the required released-config sweep): the same beta sweep at a
higher learning rate, to test whether beta's effect appears once the policy moves further from the reference.
Outputs carry an `_lr<value>` suffix and a separate summary, `beta_ablation_summary_lr<value>.csv`.
```bash
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --learning-rate 1e-4
```
`analyze_length` also writes `length_normalized_margins.csv`: summed vs per-token DPO margins per length stratum.
Outputs in `results/task1_dpo/`: `train_logs/*.jsonl`, `eval_<name>.json`, `pairs_<name>.jsonl`,
`generations_<name>.jsonl`, `examples_<name>.json`, `beta_ablation_summary.csv`, `length_analysis.json`,
`length_strata_summary.csv`, `word_limit_summary.csv`, `word_limit_generations.jsonl`.

### Task 2 — PPO (`configs/ppo.yaml`)
```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard                     # 20 updates from the supplied midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml                                      # cached batch + eps forks {0.05,0.20,0.50}
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml                                      # beta_KL forks {0,0.10,0.20}
```
Outputs in `results/task2_ppo/`: `train_logs/standard.jsonl` (per-update reward, KL, policy/value loss, entropy,
grad norms, clip fraction, length, critic explained variance, time, VRAM), `train_logs/standard_summary.json`
(**wall-clock and peak VRAM**), `rollouts/*.jsonl`, `eval_*.json`, `clipping_cached.json`,
`clipping_summary.csv`, `kl_summary.csv`, `kl_trajectories.csv`.

### Task 3 — GRPO (`configs/grpo.yaml`)
```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard                   # 20 updates, K=4
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size    --config configs/grpo.yaml                                # K in {2,4,8}, no training
python -m task3_grpo.compare_normalization --config configs/grpo.yaml                                # grpo vs dr_grpo forks
```
Outputs in `results/task3_grpo/`: `train_logs/standard.jsonl` (reward, KL, within-group reward std,
zero-std group fraction, loss, grad norm, entropy, length and per-completion records),
`train_logs/standard_summary.json` (**wall-clock, peak VRAM**), `group_size_study.json`,
`group_size_summary.csv`, `normalization_summary.csv`, `eval_*.json`.

### Task 4 — Safety calibration (`configs/feedback.yaml`)
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml        # greedy, SFT/DPO/PPO/GRPO standard policies
python -m task4_safety.judge_responses    --config configs/feedback.yaml        # fixed Qwen2.5-3B judge, cached + resumable
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml        # blinded sheet for the fixed 60 audit prompts
#   -> label results/task4_safety/manual_audit_sheet.csv by hand, save it as manual_audit_labels.csv
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml
```
Outputs in `results/task4_safety/`: `generated_*.jsonl`, `judged_*.jsonl`, `safety_summary.csv`,
`category_label_distribution.csv`, `policy_disagreements.csv`, `audit_agreement.json`, `audit_confusion.csv`,
`audit_disagreements.csv`.

### Task 5 — RLVR vs RLAIF (`configs/feedback.yaml`)
```bash
python -m task5_feedback.evaluate_math       --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math       --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback    --config configs/feedback.yaml
```
Outputs in `results/task5_feedback/`: `gsm/` and `transfer/` (`generations_*.jsonl`, `judgments.jsonl`,
`metrics.json/csv`), `perturbations/` (`pair_scores.jsonl`, `rlaif_group_rewards.jsonl`, `summary.json/csv`),
`judge_cache.json`, `feedback_comparison.json/csv`.

### Figures
```bash
python -m scripts.make_figures      # report/figures/*.png, built only from results/ files
```

---

## 4. Implementation decisions and conventions

**Shared evaluation protocol (`common/policy_eval.py`).** Each held-out prompt gets one sampled response, using
temperature 0.7, top-p 0.9 and seed 6304 from `configs/base.yaml`. Tasks 2 and 3 use the same 768-token cap
(`eval_max_response_length` / `cache_generation_cap`); Task 1 uses `max_generation_tokens`. For each response we
record:
* RM score (fixed course reward model);
* sampled-response KL, i.e. the summed per-token log-ratio `log pi - log pi_ref` (the released `sampled_kl`
  estimator; the reference is the adapter-disabled base model);
* exact per-token entropy;
* length (tokens after the prompt, including the first EOS, excluding padding).

KL is reported under **both** aggregation conventions for every condition: `kl_token_mean` (mean over all valid
response tokens) and `kl_seq_mean` (mean of per-response sums). Pick one in the report and use it everywhere.

**Task 1.**
* Fresh zero-init LoRA (r=8, α=16, q/v) on Qwen2.5-1.5B-Instruct for every condition. The reference is the same
  network with the adapter disabled.
* Pairs whose prompt leaves fewer than 16 response tokens within `max_sequence_length` are dropped
  deterministically. Dropped IDs are logged.
* The data order comes from a seeded generator, so the β forks see identical batches. They use the first 600
  pairs of the standard file (`short_ablation_examples`).
* Held-out loss uses each run's own β; preference accuracy is β-independent (m_θ > 0).
* Word-limit compliance is measured on the 10 tracked prompts, with greedy decoding plus 4 seeded samples per
  prompt.

**Task 2.**
* The prompt order is a seed-fixed permutation of the training pool, identical for the standard run and every
  fork. Prompts longer than `max_prompt_length` are dropped and logged.
* Each fork starts from the identical supplied policy and critic. The (ε=0.20, β_KL=0.10) fork is shared by the
  clipping and KL studies.
* Responses without EOS receive `missing_eos_penalty`. Rewards are KL-shaped per token (`shaped_rewards`).
  GAE uses γ=1, λ=0.95. Advantages are whitened over valid tokens.
* Policy and critic each have their own optimiser and gradient clipping. `ppo_epochs`=2 passes per update.
* **Clip fraction** is the fraction of valid tokens with ρ outside [1−ε, 1+ε]. **Affected-token fraction** is the
  fraction where the clipped branch is the one selected by the min, i.e. (A>0 ∧ ρ>1+ε) ∨ (A<0 ∧ ρ<1−ε).
* Cached-batch study: (A1) the midpoint's ratio against the cached behaviour log-probs; (A2) for each ε, the same
  start, then 2 epochs × minibatches of 8 PPO updates on the cached batch with the configured learning rate.
  Clip, affected and surrogate values are recorded before each step and after the last step.
* Stability statistic, defined once: the mean over updates of E|ρ−1| after the final PPO epoch. Also logged:
  reward std over updates, ratio extremes, KL growth, grad norms and skipped steps.

**Task 3.**
* Prompt order as in Task 2. K=4 completions per prompt.
* Completions truncated at `max_completion_length` are masked from the loss but still count in their group's
  reward statistics, as TRL does.
* Uninformative group: within-group reward std ≤ 1e-6 (the helper's tolerance).
* `dr_grpo` changes only the sequence normalisation (constant 1/L_max). Advantage std normalisation is kept, so
  the reward signal is identical across the two forks.
* Length-conditioned statistic: the gradient norm of each completion's policy-term contribution (extra
  forward/backward per completion), correlated with completion length.
* Group-size study: each prompt's 8 cached completions (ordered by `generation_index`) are partitioned into
  disjoint groups of K, so every condition uses the same 192 generations. An all-C(8,K)-subsets estimate is also
  reported. Difficulty bins are tertiles of the per-prompt mean reward (hard = lowest).

**Task 4.**
* Greedy decoding, cap of 256 tokens, the same batch size for all four policies.
* Judging uses the supplied prompt, chat template and parser. By default it runs in batches of 8 with identical
  greedy decoding; `--batch-size 1` calls `judge_one` exactly.
* Labels that contradict the prompt class (e.g. OVER_REFUSAL on an unsafe prompt) are reported as a mismatch rate
  rather than remapped.
* The audit sheet is blinded (no AI labels) and shuffled. Default `--mode all_policies` (60 prompts × 4 policies);
  `--mode rotate` gives 60 rows (15 per policy).

**Task 5.**
* Greedy decoding, cap of 512 tokens, identical `messages` prompts.
* The pairwise judge compares RLVR vs SFT, RLAIF vs SFT and RLVR vs RLAIF. A win counts 1, a tie 0.5.
* Verifier–judge agreement is measured on every pair and on the pairs where the verifier is decisive.
* Each diagnostic pair is judged in both A/B orders; the primary result is the (better, worse) call, and
  orientation consistency is reported. The RLAIF group reward over the five variants of each problem is also
  reported.

---

## 5. Compute notes

Tested end to end on CPU with tiny stand-in models. Real runs need a GPU (A100 recommended; L4 works). Rough
A100 estimates: Task 1 ≈ 1.5 h (four trainings plus evaluations), Task 2 ≈ 1.5 h (standard run, 5 forks,
7 evaluations), Task 3 ≈ 1 h, Task 4 ≈ 1–1.5 h (mostly judging), Task 5 ≈ 1 h. Evaluation batch size is
`--batch-size` (default 16); lower it on smaller GPUs. `--max-prompts`/`--max-eval-prompts` exist for debugging
only. Final results must use the full fixed sets.

## 6. Reproducibility

Seed 6304 everywhere (data order, prompt order, per-update generation seeds, evaluation sampling). Each run
writes its hyperparameters, the prompt/pair IDs it used, any dropped IDs, wall-clock time and peak VRAM to
`outputs/.../run_config.json` and `results/.../train_logs/*_summary.json`. Task 4 labels and Task 5 data are
never used to tune Tasks 1–3.

## 7. Code provenance

* Course starter release (data/model loading, generation, metric helpers, judge/verifier, configs, asset
  scripts): `https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining`.
* Algorithms follow the cited papers: DPO (Rafailov et al., 2023), PPO (Schulman et al., 2017), GRPO (Shao et al.,
  2024), Dr. GRPO (Liu et al., 2025). Conventions such as dropout disabling and truncated-completion masking
  follow Hugging Face TRL; no TRL code is copied.
* Implementation developed with LLM coding assistance (Claude); all code was reviewed by the author.
