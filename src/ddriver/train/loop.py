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
from ddriver.utils.seed import seed_everything


@dataclass
class TrainLoopConfig:
    model_name: str
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    lr_drop_epoch: Optional[int] = None  # epoch index (1-based) to drop LR
    lr_drop_factor: float = 0.1          # multiplier applied after drop
    batch_size: int = 32
    num_workers: int = 2
    image_size: int = 224
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    out_tag: str = "debug"
    device: Optional[str] = None
    seed: int = 42  # Random seed for reproducibility
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_cfg: Optional[DataCfg] = None
    output_dir: Optional[Path] = None
    save_every_epoch: bool = False
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    checkpoint_metric: str = "val.loss"
    checkpoint_mode: str = "min"


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
    # Seed all random number generators for reproducibility
    seed_everything(cfg.seed)
    print(f"[train] Seeded all RNGs with seed={cfg.seed}")

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
    opt_name = (cfg.optimizer or "adam").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay or 0.0,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay or 0.0,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Use 'adam' or 'adamw'.")
    scheduler = None
    if cfg.lr_drop_epoch is not None:
        drop_epoch = max(1, cfg.lr_drop_epoch)
        factor = cfg.lr_drop_factor or 0.1

        def lr_lambda(epoch_idx: int) -> float:
            # epoch_idx is 0-based inside LambdaLR
            return factor if (epoch_idx + 1) >= drop_epoch else 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    base_dir = Path(cfg.output_dir) if cfg.output_dir else (project_config.CKPT_ROOT / "runs")
    run_dir = new_run_dir(base_dir, cfg.out_tag)
    
    # Save seed for reproducibility tracking
    seed_path = run_dir / "seed.txt"
    seed_path.write_text(str(cfg.seed))

    history = []

    best_metric_value: Optional[float] = None
    best_metric_name: Optional[str] = None
    best_epoch: Optional[int] = None
    best_checkpoint_path: Optional[Path] = None
    last_checkpoint_path: Optional[Path] = None
    epoch_checkpoint_paths: list[str] = []

    checkpoint_mode = (cfg.checkpoint_mode or "min").lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be either 'min' or 'max'.")

    metric_candidates: list[str] = []
    if cfg.checkpoint_metric:
        metric_candidates.append(cfg.checkpoint_metric)
    if "train.loss" not in metric_candidates:
        metric_candidates.append("train.loss")

    metric_fallback_notified = False
    metric_missing_warning = False

    def _lookup_metric(record: Dict[str, Any], spec: Optional[str]) -> Optional[float]:
        if not spec or "." not in spec:
            return None
        scope, metric = spec.split(".", 1)
        scope_metrics = record.get(scope) or {}
        if not isinstance(scope_metrics, dict):
            return None
        return scope_metrics.get(metric)

    def _is_better(candidate: float) -> bool:
        if best_metric_value is None:
            return True
        if checkpoint_mode == "min":
            return candidate < best_metric_value
        return candidate > best_metric_value

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

        extra_common = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        if cfg.save_every_epoch:
            ckpt_path = run_dir / f"epoch_{epoch}.pt"
            per_epoch_extra = dict(extra_common)
            per_epoch_extra["checkpoint_type"] = "per_epoch"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=ckpt_path,
                **per_epoch_extra,
            )
            epoch_checkpoint_paths.append(str(ckpt_path))

        if cfg.save_last_checkpoint:
            last_checkpoint_path = run_dir / "last.pt"
            last_extra = dict(extra_common)
            last_extra["checkpoint_type"] = "last"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=last_checkpoint_path,
                **last_extra,
            )

        if cfg.save_best_checkpoint:
            metric_value = None
            metric_name_used = None
            for candidate in metric_candidates:
                value = _lookup_metric(epoch_record, candidate)
                if value is not None:
                    metric_value = value
                    metric_name_used = candidate
                    if (
                        cfg.checkpoint_metric
                        and candidate != cfg.checkpoint_metric
                        and not metric_fallback_notified
                    ):
                        print(
                            f"[train] Falling back to checkpoint metric '{candidate}' "
                            f"because '{cfg.checkpoint_metric}' is unavailable."
                        )
                        metric_fallback_notified = True
                    break

            if metric_value is None:
                if not metric_missing_warning:
                    print(
                        f"[train] Could not find any of the checkpoint metrics {metric_candidates}; "
                        "best checkpoint will be skipped."
                    )
                    metric_missing_warning = True
            elif _is_better(metric_value):
                best_metric_value = metric_value
                best_metric_name = metric_name_used
                best_epoch = epoch
                best_checkpoint_path = run_dir / "best.pt"
                best_extra = dict(extra_common)
                best_extra.update(
                    {
                        "checkpoint_type": "best",
                        "best_metric": {
                            "name": best_metric_name,
                            "value": best_metric_value,
                            "mode": checkpoint_mode,
                        },
                    }
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=best_checkpoint_path,
                    **best_extra,
                )
                print(
                    f"[train] Saved new best checkpoint at epoch {epoch} "
                    f"({best_metric_name}={best_metric_value:.4f})."
                )

        save_json(run_dir / "history.json", {"history": history})
        print(f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f}")
        if val_metrics:
            print(f"            val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}")
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"            lr adjusted to {current_lr:.6f}")

    summary = {
        "run_dir": str(run_dir),
        "history": history,
        "model_name": cfg.model_name,
        "device": str(device),
        "seed": cfg.seed,
    }
    if best_checkpoint_path:
        summary["best_checkpoint"] = str(best_checkpoint_path)
        summary["best_metric"] = {
            "name": best_metric_name,
            "value": best_metric_value,
            "epoch": best_epoch,
            "mode": checkpoint_mode,
        }
    if last_checkpoint_path:
        summary["last_checkpoint"] = str(last_checkpoint_path)
    if epoch_checkpoint_paths:
        summary["epoch_checkpoints"] = epoch_checkpoint_paths

    save_json(run_dir / "summary.json", summary)
    return summary


__all__ = ["TrainLoopConfig", "run_training"]
# Training loop: takes model, data, config, runs epochs, logs metrics
# returns history, the diary of loss/accuracy per epoch. 
# Will call this from notebooks. 