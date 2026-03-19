"""
Single-batch overfit sanity check.

Proves gradients flow, tensor shapes match, and losses decrease for all 4 tasks.
Run: python -m train.overfit_test [--data-dir data/training]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from train.dataset import SinciputDataset, collate_fn
from train.model import MultiTaskModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def overfit_single_task(
    task: str,
    batch: dict,
    device: torch.device,
    steps: int = 200,
    lr: float = 1e-3,
    threshold: float = 0.1,
) -> bool:
    """Train a fresh model on a single batch for one task. Returns True if loss drops below threshold."""
    model = MultiTaskModel(freeze_layers=0).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move batch to device
    dev_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "ner_labels": batch["ner_labels"].to(device),
        "parsing_arcs": batch["parsing_arcs"].to(device),
        "domain_class": batch["domain_class"].to(device),
        "coref_clusters": batch["coref_clusters"],
        "seq_lens": batch["seq_lens"].to(device),
    }

    # Build task kwargs
    task_kwargs = {"task": task}
    if task == "domain":
        task_kwargs["domain_class"] = dev_batch["domain_class"]
    elif task == "ner":
        task_kwargs["ner_labels"] = dev_batch["ner_labels"]
    elif task == "parsing":
        task_kwargs["parsing_arcs"] = dev_batch["parsing_arcs"]
    elif task == "coref":
        task_kwargs["coref_clusters"] = dev_batch["coref_clusters"]
        task_kwargs["seq_lens"] = dev_batch["seq_lens"]

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()} | threshold: {threshold}")
    print(f"{'='*60}")

    final_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        out = model(
            input_ids=dev_batch["input_ids"],
            attention_mask=dev_batch["attention_mask"],
            **task_kwargs,
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        final_loss = loss.item()
        if (step + 1) % 20 == 0:
            print(f"  step {step+1:>4d} | loss = {final_loss:.6f}")

    passed = final_loss < threshold
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] final loss = {final_loss:.6f} (threshold = {threshold})")
    return passed


def overfit_interleaved(batch: dict, device: torch.device, steps: int = 200, lr: float = 1e-3) -> bool:
    """Train a single model on all 4 tasks interleaved."""
    model = MultiTaskModel(freeze_layers=0).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dev_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "ner_labels": batch["ner_labels"].to(device),
        "parsing_arcs": batch["parsing_arcs"].to(device),
        "domain_class": batch["domain_class"].to(device),
        "coref_clusters": batch["coref_clusters"],
        "seq_lens": batch["seq_lens"].to(device),
    }

    tasks = ["domain", "ner", "parsing", "coref"]
    task_kwargs_map = {
        "domain": {"domain_class": dev_batch["domain_class"]},
        "ner": {"ner_labels": dev_batch["ner_labels"]},
        "parsing": {"parsing_arcs": dev_batch["parsing_arcs"]},
        "coref": {"coref_clusters": dev_batch["coref_clusters"], "seq_lens": dev_batch["seq_lens"]},
    }

    print(f"\n{'='*60}")
    print("INTERLEAVED: all 4 tasks, round-robin")
    print(f"{'='*60}")

    task_losses = {t: 0.0 for t in tasks}
    for step in range(steps):
        task = tasks[step % len(tasks)]
        optimizer.zero_grad()
        out = model(
            input_ids=dev_batch["input_ids"],
            attention_mask=dev_batch["attention_mask"],
            task=task,
            **task_kwargs_map[task],
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        task_losses[task] = loss.item()

        if (step + 1) % 40 == 0:
            loss_str = " | ".join(f"{t}: {task_losses[t]:.4f}" for t in tasks)
            print(f"  step {step+1:>4d} | {loss_str}")

    print("  Final losses:")
    for t in tasks:
        print(f"    {t}: {task_losses[t]:.6f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Overfit sanity check")
    parser.add_argument("--data-dir", default="data/training", help="Training data directory")
    parser.add_argument("--num-examples", type=int, default=8, help="Examples per batch")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps per task")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load a small batch
    dataset = SinciputDataset(args.data_dir, max_examples=args.num_examples)
    print(f"Loaded {len(dataset)} examples")

    loader = DataLoader(dataset, batch_size=args.num_examples, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))

    # Per-task config: (threshold, lr_multiplier, step_multiplier)
    # Parsing needs lower LR (large T-class output causes instability) and more steps
    task_config = {
        "domain": (0.01, 1.0, 1),
        "ner": (0.05, 1.0, 1),
        "parsing": (0.1, 0.1, 3),
        "coref": (0.5, 1.0, 1),
    }

    # Run single-task overfit for each task
    results = {}
    for task, (threshold, lr_mult, step_mult) in task_config.items():
        results[task] = overfit_single_task(
            task, batch, device,
            steps=args.steps * step_mult,
            lr=args.lr * lr_mult,
            threshold=threshold,
        )

    # Run interleaved test
    overfit_interleaved(batch, device, steps=args.steps, lr=args.lr)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for task, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {task:>10s}: [{status}]")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\nSome tasks FAILED. Check for shape mismatches, masking errors, or detached gradients.")
        sys.exit(1)
    else:
        print("\nAll tasks passed!")


if __name__ == "__main__":
    main()
