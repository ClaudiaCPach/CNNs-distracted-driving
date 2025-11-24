"""
Smoke tests for train.trainer helpers.

Run with:
    python -m src.ddriver.test_trainer
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ddriver.train import trainer


class TinyDataset(Dataset):
    def __init__(self, num_samples: int = 8):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = torch.randn(3, 8, 8)
        y = torch.tensor(idx % 3, dtype=torch.long)
        return {"image": x, "label": y}


def _make_model(num_classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


def run_smoke_test() -> None:
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=4)

    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_metrics = trainer.train_one_epoch(model, loader, criterion, optimizer, device="cpu")
    val_metrics = trainer.eval_one_epoch(model, loader, criterion, device="cpu")

    ckpt_path = trainer.save_checkpoint(model=model, optimizer=optimizer, epoch=1, path="checkpoint_test.pt")
    loaded = trainer.load_model_weights(model, ckpt_path)

    assert "epoch" in loaded
    assert train_metrics["loss"] >= 0
    assert val_metrics["loss"] >= 0
    print("✅ trainer tests passed")


if __name__ == "__main__":
    run_smoke_test()

