"""
Training helpers: one epoch loops, evaluation, checkpoint utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    return images, labels


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / max(1, target.size(0))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    current_epoch: int = 1,  # ADD THIS
    log_every: int = 50,     # ADD THIS - print every 50 batches
) -> Dict[str, float]:
    model.train()
    device = torch.device(device)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    total_batches = len(dataloader)  # ADD THIS - so we know total

    for batch_idx, batch in enumerate(dataloader, start=1):  # CHANGE: add enumerate
        optimizer.zero_grad()
        images, labels = _move_batch_to_device(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size
        
        # ADD THIS BLOCK - print progress every N batches
        if log_every > 0 and batch_idx % log_every == 0:
            current_acc = (logits.argmax(dim=1) == labels).float().mean().item()
            running_loss = total_loss / max(1, total_examples)
            running_acc = total_correct / max(1, total_examples)
            print(
                f"[epoch {current_epoch}] batch {batch_idx}/{total_batches} | "
                f"loss={loss.item():.4f} (avg={running_loss:.4f}) | "
                f"acc={current_acc:.4f} (avg={running_acc:.4f})"
            )

    avg_loss = total_loss / max(1, total_examples)
    acc = total_correct / max(1, total_examples)
    return {"loss": avg_loss, "accuracy": acc}


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    model.eval()
    device = torch.device(device)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        images, labels = _move_batch_to_device(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    avg_loss = total_loss / max(1, total_examples)
    acc = total_correct / max(1, total_examples)
    return {"loss": avg_loss, "accuracy": acc}


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path | str,
    **extra: Any,
) -> Path:
    """
    Save a checkpoint with model + optimizer state and extra metadata.
    """

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        ckpt["extra"] = extra

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def load_model_weights(model: nn.Module, checkpoint_path: Path | str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Load model (and optimizer if available) state_dict from checkpoint.
    Returns the loaded checkpoint dict.
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


__all__ = ["train_one_epoch", "eval_one_epoch", "save_checkpoint", "load_model_weights"]

