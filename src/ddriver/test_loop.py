"""
Smoke test for the high-level training loop.

Run with:
    python -m src.ddriver.test_loop
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ddriver.models import registry
from ddriver.train.loop import TrainLoopConfig, run_training


class ToyDataset(Dataset):
    def __init__(self, size: int = 16):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        x = torch.randn(3, 8, 8)
        y = torch.tensor(idx % 3)
        return {"image": x, "label": y}


@registry.register_model("toy_mlp")
def build_toy_mlp(num_classes: int = 3):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


def run_smoke_test() -> None:
    train_loader = DataLoader(ToyDataset(12), batch_size=4)
    val_loader = DataLoader(ToyDataset(6), batch_size=3)

    cfg = TrainLoopConfig(
        model_name="toy_mlp",
        epochs=1,
        lr=1e-2,
        out_tag="unit_test",
        batch_size=4,
        output_dir=Path("test_runs"),
    )

    result = run_training(cfg, dataloaders={"train": train_loader, "val": val_loader})
    assert "run_dir" in result
    print("✅ train loop test passed:", result["run_dir"])


if __name__ == "__main__":
    run_smoke_test()

