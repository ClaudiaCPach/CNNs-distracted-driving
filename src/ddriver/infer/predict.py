"""
Prediction helper to export CSVs from checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddriver import config as project_config
from ddriver.data.datamod import DefaultCfg as DataCfg
from ddriver.data.datamod import build_dataloaders, make_cfg_from_config
from ddriver.data.dataset import CLASS_TO_INDEX as DATA_CLASS_TO_INDEX
from ddriver.models import registry as model_registry
from ddriver.train.trainer import load_model_weights

DEFAULT_IDX_TO_CLASS = {idx: class_id for class_id, idx in DATA_CLASS_TO_INDEX.items()}


@dataclass
class PredictConfig:
    model_name: str
    checkpoint_path: str
    split: str = "val"
    batch_size: int = 32
    num_workers: int = 2
    image_size: int = 224
    out_tag: str = "preds"
    device: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_cfg: Optional[DataCfg] = None
    idx_to_class: Optional[Dict[int, str]] = None
    out_csv: Optional[Path | str] = None


def _default_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_prediction(
    cfg: PredictConfig,
    *,
    dataloader: Optional[DataLoader] = None,
) -> Path:
    """
    Generate predictions CSV for the requested split.
    """

    device = _default_device(cfg.device)

    if dataloader is None:
        data_cfg = cfg.data_cfg or make_cfg_from_config(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
        )
        dataloader = build_dataloaders(data_cfg)[cfg.split]

    idx_to_class = cfg.idx_to_class or DEFAULT_IDX_TO_CLASS

    model = model_registry.build_model(cfg.model_name, **cfg.model_kwargs).to(device)
    load_model_weights(model, cfg.checkpoint_path, map_location=device)
    model.eval()

    rows = []
    for batch in dataloader:
        images = batch["image"].to(device)
        paths = batch.get("path")
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        for idx, pred in enumerate(preds.tolist()):
            class_id = idx_to_class.get(pred, str(pred))
            rows.append({"path": paths[idx] if paths is not None else f"sample_{len(rows)}", "pred_class_id": class_id})

    pred_dir = Path(cfg.out_csv) if cfg.out_csv else (project_config.OUT_ROOT / "preds" / cfg.split)
    if pred_dir.suffix.lower() != ".csv":
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_csv = pred_dir / f"{cfg.out_tag}.csv"
    else:
        out_csv = pred_dir
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[✓] predictions saved to {out_csv}")
    return out_csv


__all__ = ["PredictConfig", "run_prediction"]

