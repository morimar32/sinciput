"""
Full multi-task training loop with interleaved batching, differential LRs,
gradient clipping, TensorBoard logging, and checkpointing.

Run: python -m train.train --data-dir data/training --epochs 20
"""

import argparse
import itertools
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from train.dataset import SinciputDataset, collate_fn
from train.model import MultiTaskModel


TASK_BATCH_SIZES = {
    "domain": 64,
    "ner": 32,
    "parsing": 64,
    "coref": 16,
}

TASK_GRAD_CLIP = {
    "domain": 1.0,
    "ner": 1.0,
    "parsing": 5.0,
    "coref": 1.0,
}

TASKS = ["domain", "ner", "parsing", "coref"]


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: MultiTaskModel, base_lr: float) -> torch.optim.AdamW:
    """Build AdamW with differential learning rates and weight decay exclusion."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    # Group parameters
    encoder_params_decay = []
    encoder_params_no_decay = []
    head_groups = {
        "domain_head": {"decay": [], "no_decay": [], "lr": 1e-4},
        "ner_head": {"decay": [], "no_decay": [], "lr": 1e-4},
        "parsing_head": {"decay": [], "no_decay": [], "lr": 2e-4},
        "coref_head": {"decay": [], "no_decay": [], "lr": 3e-4},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine which group
        matched_head = None
        for head_name in head_groups:
            if name.startswith(head_name):
                matched_head = head_name
                break

        if matched_head:
            if any(nd in name for nd in no_decay):
                head_groups[matched_head]["no_decay"].append(param)
            else:
                head_groups[matched_head]["decay"].append(param)
        else:
            # Encoder params (trainable layers 3-5)
            if any(nd in name for nd in no_decay):
                encoder_params_no_decay.append(param)
            else:
                encoder_params_decay.append(param)

    param_groups = [
        {"params": encoder_params_decay, "lr": base_lr, "weight_decay": 0.01},
        {"params": encoder_params_no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]

    for head_name, group in head_groups.items():
        if group["decay"]:
            param_groups.append({"params": group["decay"], "lr": group["lr"], "weight_decay": 0.01})
        if group["no_decay"]:
            param_groups.append({"params": group["no_decay"], "lr": group["lr"], "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups)


def build_dataloaders(
    data_dir: str,
    domains: list[str] | None,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    """One DataLoader per task, all sharing the same dataset."""
    dataset = SinciputDataset(data_dir, domains=domains)
    print(f"Loaded {len(dataset)} training examples")

    loaders = {}
    for task in TASKS:
        bs = TASK_BATCH_SIZES[task]
        loaders[task] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return loaders


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "ner_labels": batch["ner_labels"].to(device),
        "parsing_arcs": batch["parsing_arcs"].to(device),
        "domain_class": batch["domain_class"].to(device),
        "coref_clusters": batch["coref_clusters"],
        "seq_lens": batch["seq_lens"].to(device),
    }


def get_task_kwargs(task: str, batch: dict) -> dict:
    """Extract task-specific keyword arguments from a batch."""
    kwargs = {"task": task}
    if task == "domain":
        kwargs["domain_class"] = batch["domain_class"]
    elif task == "ner":
        kwargs["ner_labels"] = batch["ner_labels"]
    elif task == "parsing":
        kwargs["parsing_arcs"] = batch["parsing_arcs"]
    elif task == "coref":
        kwargs["coref_clusters"] = batch["coref_clusters"]
        kwargs["seq_lens"] = batch["seq_lens"]
    return kwargs


def resolve_num_workers(device: torch.device, override: int | None) -> int:
    """Auto-set DataLoader num_workers based on device, or use override."""
    if override is not None:
        return override
    if device.type == "cuda":
        return 4
    return 0  # MPS and CPU


def cleanup_checkpoints(ckpt_dir: Path, keep_last_n: int):
    """Delete old checkpoints, always keeping epoch 1 and the latest N."""
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    if len(ckpts) <= keep_last_n:
        return

    # Always protect epoch 1 and the latest N
    protected = set()
    for c in ckpts:
        if c.name == "checkpoint_epoch1.pt":
            protected.add(c)
    for c in ckpts[-keep_last_n:]:
        protected.add(c)

    for c in ckpts:
        if c not in protected:
            c.unlink()
            print(f"  Removed old checkpoint: {c.name}")


def train(args):
    device = get_device(args.device)
    print(f"Device: {device}")

    # Output directory setup
    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader settings
    num_workers = resolve_num_workers(device, args.num_workers)
    pin_memory = device.type == "cuda"
    print(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}")

    # Data
    domains = args.domains.split(",") if args.domains else None
    loaders = build_dataloaders(args.data_dir, domains, num_workers, pin_memory)

    # Model
    model = MultiTaskModel(freeze_layers=3).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Optimizer & scheduler
    optimizer = build_optimizer(model, args.base_lr)

    # Total steps = sum of all loader lengths × epochs
    steps_per_epoch = sum(len(loader) for loader in loaders.values())
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")

    # Mixed precision
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        print(f"Warning: --fp16 requested but device is {device.type}, falling back to fp32")
    if use_fp16:
        print("Using mixed precision (fp16)")
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")

    # Determine how many epochs to run this session
    if args.stop_after is not None:
        end_epoch = min(start_epoch + args.stop_after, args.epochs)
        print(f"Will stop after epoch {end_epoch}/{args.epochs} this session (--stop-after {args.stop_after})")
    else:
        end_epoch = args.epochs

    # Logging
    writer = SummaryWriter(log_dir=str(log_dir))

    wall_start = time.time()
    final_losses = {}

    for epoch in range(start_epoch, end_epoch):
        model.train()

        # Create cycling iterators for each task
        task_iters = {task: itertools.cycle(loader) for task, loader in loaders.items()}

        # Interleave: cycle through tasks in order
        # Each "round" = one batch from each task
        rounds = max(len(loader) for loader in loaders.values())
        interleaved_schedule = []
        for r in range(rounds):
            for task in TASKS:
                if r < len(loaders[task]):
                    interleaved_schedule.append(task)

        pbar = tqdm(interleaved_schedule, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_losses = {task: 0.0 for task in TASKS}
        epoch_counts = {task: 0 for task in TASKS}

        for task in pbar:
            batch = next(task_iters[task])
            batch = move_batch(batch, device)
            task_kwargs = get_task_kwargs(task, batch)

            optimizer.zero_grad()

            if use_fp16:
                with torch.amp.autocast("cuda"):
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **task_kwargs,
                    )
                    loss = output["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                max_norm = TASK_GRAD_CLIP[task]
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **task_kwargs,
                )
                loss = output["loss"]
                loss.backward()
                max_norm = TASK_GRAD_CLIP[task]
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            epoch_losses[task] += loss_val
            epoch_counts[task] += 1
            global_step += 1

            # TensorBoard logging
            writer.add_scalar(f"loss/{task}", loss_val, global_step)
            writer.add_scalar(f"grad_norm/{task}", grad_norm, global_step)

            if global_step % args.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("lr/encoder", current_lr, global_step)

                avg_losses = {
                    t: epoch_losses[t] / max(epoch_counts[t], 1) for t in TASKS
                }
                loss_str = " | ".join(f"{t}: {avg_losses[t]:.4f}" for t in TASKS)
                pbar.set_postfix_str(loss_str)

        # Epoch summary
        print(f"\nEpoch {epoch+1} avg losses:")
        for task in TASKS:
            avg = epoch_losses[task] / max(epoch_counts[task], 1)
            final_losses[task] = avg
            print(f"  {task:>10s}: {avg:.6f}")
            writer.add_scalar(f"epoch_loss/{task}", avg, epoch + 1)

        # Checkpoint
        ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Checkpoint retention
        cleanup_checkpoints(ckpt_dir, args.keep_last_n)

    # Save final model weights only when all epochs are done
    completed = end_epoch >= args.epochs
    final_model_path = output_dir / "model_final.pt"
    if completed:
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model: {final_model_path}")

    writer.close()

    # Training summary
    wall_time = time.time() - wall_start
    hours, remainder = divmod(int(wall_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    latest_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))

    print("\n" + "=" * 60)
    if completed:
        print("TRAINING COMPLETE")
    else:
        print(f"SESSION PAUSED after epoch {end_epoch}/{args.epochs}")
    print("=" * 60)
    print(f"  Wall-clock time: {hours}h {minutes}m {seconds}s")
    print(f"  Final per-task losses:")
    for task in TASKS:
        print(f"    {task:>10s}: {final_losses.get(task, float('nan')):.6f}")
    if completed:
        print(f"  Final model:     {final_model_path}")
    if latest_ckpts:
        print(f"  Latest checkpoint: {latest_ckpts[-1]}")
    if not completed and latest_ckpts:
        print(f"\n  Resume with:")
        print(f"    python -m train.train --resume {latest_ckpts[-1]}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Multi-task training for sinciput")
    parser.add_argument("--data-dir", default="data/training", help="Training data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--base-lr", type=float, default=1e-5, help="Base learning rate for encoder")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--domains", default=None, help="Comma-separated domain list")
    parser.add_argument("--output-dir", default="output", help="Base output directory")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory (default: <output-dir>/checkpoints)")
    parser.add_argument("--log-dir", default=None, help="TensorBoard log directory (default: <output-dir>/runs)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (CUDA only)")
    parser.add_argument("--keep-last-n", type=int, default=3, help="Keep only the last N checkpoints (plus epoch 1)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader num_workers (auto if not set)")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after N epochs this session (resumes later with --resume)")
    args = parser.parse_args()

    # Resolve defaults that depend on output-dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    if args.log_dir is None:
        args.log_dir = str(Path(args.output_dir) / "runs")

    train(args)


if __name__ == "__main__":
    main()
