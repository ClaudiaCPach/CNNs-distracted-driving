"""
Smoke tests for ddriver.models.losses.

Run with:
    python -m src.ddriver.test_losses
"""

from __future__ import annotations

import torch

from ddriver.models import losses


def test_cross_entropy():
    crit = losses.build_loss("cross_entropy")
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 1])
    loss = crit(logits, target)
    loss.backward()
    assert loss.item() > 0


def test_label_smoothing():
    crit = losses.build_loss("cross_entropy", label_smoothing=0.1)
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 1])
    loss = crit(logits, target)
    loss.backward()
    assert loss.item() > 0


def main() -> None:
    test_cross_entropy()
    test_label_smoothing()
    print("✅ losses tests passed")


if __name__ == "__main__":
    main()

