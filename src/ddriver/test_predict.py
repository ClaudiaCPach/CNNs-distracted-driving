"""
Smoke test for infer.predict helper.

Run with:
    python -m src.ddriver.test_predict
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ddriver.models import registry
from ddriver.train.loop import TrainLoopConfig, run_training
from ddriver.infer.predict import PredictConfig, run_prediction


class PredictDataset(Dataset):
    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        x = torch.randn(3, 8, 8)
        y = torch.tensor(idx % 3)
        return {"image": x, "label": y, "path": f"sample_{idx}.jpg"}


@registry.register_model("predict_toy")
def build_predict_toy(num_classes: int = 3):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


def run_smoke_test() -> None:
    train_loader = DataLoader(PredictDataset(12), batch_size=4)
    val_loader = DataLoader(PredictDataset(6), batch_size=3)

    cfg = TrainLoopConfig(
        model_name="predict_toy",
        epochs=1,
        lr=1e-2,
        out_tag="predict_test",
        output_dir=Path("test_runs"),
    )
    summary = run_training(cfg, dataloaders={"train": train_loader, "val": val_loader})
    run_dir = Path(summary["run_dir"])
    checkpoint = sorted(run_dir.glob("epoch_*.pt"))[-1]

    predict_cfg = PredictConfig(
        model_name="predict_toy",
        checkpoint_path=str(checkpoint),
        split="val",
        out_csv=run_dir / "preds.csv",
        idx_to_class={0: "c0", 1: "c1", 2: "c2"},
    )
    out_csv = run_prediction(predict_cfg, dataloader=val_loader)
    assert Path(out_csv).exists()
    print("✅ predict helper test passed:", out_csv)


if __name__ == "__main__":
    run_smoke_test()

