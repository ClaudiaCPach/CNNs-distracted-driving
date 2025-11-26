"""
High-level training loop orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

import torch

from ddriver import config as project_config
from ddriver.data.datamod import DefaultCfg as DataCfg
from ddriver.data.datamod import build_dataloaders, make_cfg_from_config
from ddriver.models import registry as model_registry
from ddriver.models import losses as loss_registry
from ddriver.train.trainer import train_one_epoch, eval_one_epoch, save_checkpoint
from ddriver.utils.runs import new_run_dir, save_json


@dataclass
class TrainLoopConfig:
    model_name: str
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 32
    num_workers: int = 2
    image_size: int = 224
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    out_tag: str = "debug"
    device: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_cfg: Optional[DataCfg] = None
    output_dir: Optional[Path] = None


def _default_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    cfg: TrainLoopConfig,
    *,
    dataloaders: Optional[Dict[str, torch.utils.data.DataLoader]] = None,
) -> Dict[str, Any]:
    """
    High-level training loop. Returns metadata including the run directory.
    """

    device = _default_device(cfg.device)

    if dataloaders is None:
        data_cfg = cfg.data_cfg or make_cfg_from_config(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
        )
        dataloaders = build_dataloaders(data_cfg)

    if "train" not in dataloaders:
        raise ValueError("dataloaders must contain a 'train' split")

    val_loader = dataloaders.get("val")

    model = model_registry.build_model(cfg.model_name, **cfg.model_kwargs).to(device)
    criterion = loss_registry.build_loss(cfg.loss_name, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    base_dir = Path(cfg.output_dir) if cfg.output_dir else (project_config.CKPT_ROOT / "runs")
    run_dir = new_run_dir(base_dir, cfg.out_tag)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device=device, current_epoch=epoch, log_every=50)
        val_metrics = {}
        if val_loader is not None:
            val_metrics = eval_one_epoch(model, val_loader, criterion, device=device)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        ckpt_path = run_dir / f"epoch_{epoch}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=ckpt_path,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        save_json(run_dir / "history.json", {"history": history})
        print(f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f}")
        if val_metrics:
            print(f"            val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}")

    summary = {
        "run_dir": str(run_dir),
        "history": history,
        "model_name": cfg.model_name,
        "device": str(device),
    }
    save_json(run_dir / "summary.json", summary)
    return summary


__all__ = ["TrainLoopConfig", "run_training"]
# Training loop: takes model, data, config, runs epochs, logs metrics
# returns history, the diary of loss/accuracy per epoch. 
# Will call this from notebooks. 