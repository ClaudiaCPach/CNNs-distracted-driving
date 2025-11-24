"""
Loss helpers for the AUC distracted driver project.

Currently supports CrossEntropy with optional label smoothing. More losses can
be registered later if needed.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

LossName = Literal["cross_entropy", "ce"]


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.

    smoothing=0 behaves like vanilla CrossEntropyLoss. smoothing should be in [0, 1).
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("LabelSmoothingCrossEntropy expects 2D logits [N, C]")
        log_probs = F.log_softmax(logits, dim=-1)
        n_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        return loss.mean()


def build_loss(
    name: LossName = "cross_entropy",
    *,
    label_smoothing: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    Build a loss function by name.

    Args:
        name: Currently only "cross_entropy"/"ce" is supported.
        label_smoothing: if >0, returns the smoothing variant.
        kwargs: forwarded to the underlying nn.CrossEntropyLoss if applicable.
    """

    normalized = name.strip().lower()
    if normalized not in {"cross_entropy", "ce"}:
        raise ValueError(f"Unsupported loss '{name}'. Only 'cross_entropy' is available.")

    if label_smoothing and label_smoothing > 0.0:
        return LabelSmoothingCrossEntropy(label_smoothing)
    return nn.CrossEntropyLoss(**kwargs)


__all__ = ["LabelSmoothingCrossEntropy", "build_loss"]

