from __future__ import annotations

import argparse
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.training_utils import (
    Timer,
    device_of,
    disable_dropout,
    finite_or_none,
    gpu_name,
    grads_are_finite,
    peak_vram_gib,
    prompt_token_length,
    reset_peak_vram,
    to_device,
    upcast_trainable_params,
)
from task1_dpo.dpo import dpo_loss

# A pair is kept only if at least this many response tokens fit after the prompt.
MIN_RESPONSE_TOKENS = 16


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_pairs(rows, tokenizer, max_length):
    """Drop pairs whose prompt alone (almost) fills the sequence budget.

    `encode_prompt_response` raises for prompts >= max_length; the filter is deterministic, so
    every condition trained or evaluated on the same file sees exactly the same pairs.
    """
    kept, dropped = [], []
    for row in rows:
        n = prompt_token_length(tokenizer, prompt_messages_from_preference(row))
        if n + MIN_RESPONSE_TOKENS <= max_length:
            kept.append(row)
        else:
            dropped.append(str(row.get("prompt_id", row.get("source_index"))))
    return kept, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None, learning_rate: float | None = None):
    cfg = load_yaml(config_path)
    if learning_rate is not None:
        # Supplementary runs only; the required experiments use the released learning rate.
        cfg["learning_rate"] = float(learning_rate)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_pairs(rows, tokenizer, int(cfg["max_sequence_length"]))

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    disable_dropout(model)
    upcast_trainable_params(model)

    # Explicit generator: the data order depends only on the seed, so beta forks see identical batches.
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_prompt_ids": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def dpo_batch_terms(model, chosen, rejected, beta):
    """Policy and frozen-reference sequence log-probs (summed over response tokens) + DPO loss."""
    with torch.no_grad(), reference_mode(model):
        ref_c, _, _ = response_sequence_logprobs(model, chosen)
        ref_r, _, _ = response_sequence_logprobs(model, rejected)
    pol_c, _, _ = response_sequence_logprobs(model, chosen)
    pol_r, _, _ = response_sequence_logprobs(model, rejected)
    loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
    return loss, diag


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, learning_rate: float | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples, learning_rate)
    cfg, model, tokenizer = bundle["cfg"], bundle["model"], bundle["tokenizer"]
    loader, optimizer, beta = bundle["loader"], bundle["optimizer"], bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / "train_logs" / f"{run_name}.jsonl"
    if log_path.exists():
        log_path.unlink()

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    params = trainable_parameters(model)
    device = device_of(model)
    micro_per_epoch = len(loader)
    total_steps = epochs * math.ceil(micro_per_epoch / accum)

    print(f"[DPO:{run_name}] pairs={len(bundle['rows'])} dropped_long_prompts={len(bundle['dropped_prompt_ids'])} "
          f"beta={beta} micro_batches/epoch={micro_per_epoch} optimizer_steps={total_steps}", flush=True)

    timer = Timer()
    reset_peak_vram()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    skipped_steps = 0
    window = []
    for epoch in range(epochs):
        for micro, (chosen, rejected) in enumerate(loader):
            chosen, rejected = to_device(chosen, device), to_device(rejected, device)
            loss, diag = dpo_batch_terms(model, chosen, rejected, beta)
            (loss / accum).backward()
            window.append({"loss": float(loss.item()), **{k: float(v.item()) for k, v in diag.items()}})

            last_micro = micro + 1 == micro_per_epoch
            if (micro + 1) % accum == 0 or last_micro:
                # A trailing partial accumulation window is rescaled to the same average.
                n_in_window = len(window)
                if n_in_window != accum:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(accum / n_in_window)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
                finite = grads_are_finite(params) and math.isfinite(float(grad_norm))
                if finite:
                    optimizer.step()
                else:
                    skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                step += 1
                rec = {
                    "run": run_name,
                    "step": step,
                    "epoch": epoch,
                    "examples_seen": min((micro + 1 + epoch * micro_per_epoch) * int(cfg["batch_size"]), len(bundle["rows"]) * epochs),
                    "grad_norm": finite_or_none(grad_norm),
                    "skipped_nonfinite": not finite,
                    "elapsed_sec": timer(),
                }
                for key in window[0]:
                    rec[key] = float(sum(w[key] for w in window) / n_in_window)
                append_jsonl(log_path, rec)
                window = []
                if step % 5 == 0 or step == total_steps:
                    print(f"[DPO:{run_name}] step {step}/{total_steps} loss={rec['loss']:.4f} "
                          f"acc={rec['preference_accuracy']:.3f} margin={rec['preference_margin_mean']:.3f} "
                          f"gnorm={rec['grad_norm']} t={rec['elapsed_sec']:.0f}s", flush=True)

    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "beta": beta,
        "dataset": bundle["dataset_path"],
        "max_examples": max_examples,
        "n_pairs": len(bundle["rows"]),
        "dropped_long_prompt_ids": bundle["dropped_prompt_ids"],
        "train_prompt_ids": [str(r.get("prompt_id")) for r in bundle["rows"]],
        "optimizer_steps": step,
        "skipped_nonfinite_steps": skipped_steps,
        "wall_clock_sec": timer(),
        "peak_vram_gib": peak_vram_gib(),
        "gpu": gpu_name(),
        "output": str(output),
        "hyperparameters": {k: cfg[k] for k in ["learning_rate", "weight_decay", "batch_size", "grad_accum_steps", "epochs", "max_sequence_length", "max_grad_norm", "seed"]},
        "lora": cfg["lora"],
        "dropout_disabled": True,
    }
    save_json(output / "run_config.json", summary)
    save_json(results_dir / "train_logs" / f"{run_name}_summary.json", summary)
    print(f"[DPO:{run_name}] saved adapter to {output} ({summary['wall_clock_sec']:.0f}s, peak {summary['peak_vram_gib']:.1f} GiB)", flush=True)
    clear_gpu(model, optimizer)
    del bundle, model, optimizer
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--learning-rate", type=float, help="supplementary runs only (default: released config)")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.learning_rate)


if __name__ == "__main__":
    main()
