'''
Example command to evaluate val set:

python3 -m src.ddriver.metrics \
  --manifest /Users/claudiapacheco/TFM/outputs/manifests/manifest.csv \
  --split-csv /Users/claudiapacheco/TFM/outputs/splits/val.csv \
  --predictions /Users/claudiapacheco/TFM/outputs/preds/val_preds_run1.csv \
  --out-tag val_run1 \
  --per-driver --per-camera

  Example command to evaluate test set: 

  python3 -m src.ddriver.metrics \
  --manifest /Users/claudiapacheco/TFM/outputs/manifests/manifest.csv \
  --split-csv /Users/claudiapacheco/TFM/outputs/splits/test.csv \
  --predictions /Users/claudiapacheco/TFM/outputs/preds/test_preds_run1.csv \
  --out-tag test_run1 \
  --per-camera
'''


# src/ddriver/metrics.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ddriver import config
from ddriver.utils.runs import new_run_dir, save_json, save_yaml, save_text

# friendly class names in reports:
CLASS_MAP: Dict[str, str] = {
    'c0': 'safe_driving',
    'c1': 'text_right',
    'c2': 'phone_right',
    'c3': 'text_left',
    'c4': 'phone_left',
    'c5': 'adjusting_radio',
    'c6': 'drinking',
    'c7': 'reaching_behind',
    'c8': 'hair_makeup',
    'c9': 'talking_to_passenger',
}
CLASS_IDS = list(CLASS_MAP.keys())

def _read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"path","class_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return df

def _read_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"path","class_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split CSV missing required columns: {missing}")
    return df

def _pick_split_rows(manifest: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in manifest.columns:
        raise ValueError("manifest has no 'split' column; pass --split-csv instead")
    sub = manifest[manifest["split"] == split_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for split '{split_name}' in manifest")
    return sub

def _read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimum: path + pred_class_id (string like c0..c9)
    if "path" not in df.columns:
        raise ValueError("predictions CSV must have column 'path'")
    if "pred_class_id" not in df.columns:
        raise ValueError("predictions CSV must have column 'pred_class_id' (e.g., c0..c9)")
    return df[["path","pred_class_id"]].copy()

def _safe_class_id_series(s: pd.Series) -> pd.Series:
    # Normalize to strings like 'c0'..'c9'
    return s.astype(str)

def evaluate(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_order: List[str],
    per_driver: bool,
    per_camera: bool
) -> Dict:
    # Join on path
    merged = truth_df.merge(pred_df, on="path", how="left", suffixes=("", "_pred"))
    missing = merged["pred_class_id"].isna().sum()
    if missing > 0:
        print(f"[warn] {missing} images had no prediction; they will be ignored.")
        merged = merged.dropna(subset=["pred_class_id"])
    if merged.empty:
        raise ValueError("After joining predictions, no rows remain to evaluate.")

    y_true = _safe_class_id_series(merged["class_id"])
    y_pred = _safe_class_id_series(merged["pred_class_id"])

    # Overall report
    report_dict = classification_report(
        y_true, y_pred, labels=label_order, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    result = {
        "num_examples": int(len(merged)),
        "labels": label_order,
        "label_to_name": {k: CLASS_MAP.get(k, k) for k in label_order},
        "overall": {
            "accuracy": float(report_dict["accuracy"]),
            "macro_avg": {
                "precision": float(report_dict["macro avg"]["precision"]),
                "recall":    float(report_dict["macro avg"]["recall"]),
                "f1":        float(report_dict["macro avg"]["f1-score"]),
            },
            "weighted_avg": {
                "precision": float(report_dict["weighted avg"]["precision"]),
                "recall":    float(report_dict["weighted avg"]["recall"]),
                "f1":        float(report_dict["weighted avg"]["f1-score"]),
            },
            "per_class": {
                lab: {
                    "precision": float(report_dict.get(lab, {}).get("precision", 0.0)),
                    "recall":    float(report_dict.get(lab, {}).get("recall", 0.0)),
                    "f1":        float(report_dict.get(lab, {}).get("f1-score", 0.0)),
                    "support":   int(report_dict.get(lab, {}).get("support", 0)),
                }
                for lab in label_order
            },
        },
        "confusion_matrix": {
            "shape": [int(x) for x in cm.shape],
            "rows_cols_labels": label_order,
            "matrix": [[int(v) for v in row] for row in cm],
        },
    }

    # Optional: per-camera
    if per_camera and "camera" in merged.columns:
        by_cam = {}
        for cam, sub in merged.groupby("camera"):
            rep = classification_report(
                _safe_class_id_series(sub["class_id"]),
                _safe_class_id_series(sub["pred_class_id"]),
                labels=label_order, output_dict=True, zero_division=0
            )
            by_cam[str(cam)] = {
                "accuracy": float(rep["accuracy"]),
                "macro_f1": float(rep["macro avg"]["f1-score"]),
                "support":  int(len(sub)),
            }
        result["by_camera"] = by_cam

    # Optional: per-driver (only meaningful where driver_id exists, e.g., your VAL)
    if per_driver and "driver_id" in merged.columns:
        by_drv = {}
        for drv, sub in merged.dropna(subset=["driver_id"]).groupby("driver_id"):
            rep = classification_report(
                _safe_class_id_series(sub["class_id"]),
                _safe_class_id_series(sub["pred_class_id"]),
                labels=label_order, output_dict=True, zero_division=0
            )
            by_drv[str(drv)] = {
                "accuracy": float(rep["accuracy"]),
                "macro_f1": float(rep["macro avg"]["f1-score"]),
                "support":  int(len(sub)),
            }
        result["by_driver"] = by_drv

    return result

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Compute classification metrics from predictions and manifest.")
    p.add_argument("--manifest", type=str, default=str(config.OUT_ROOT / "manifests" / "manifest.csv"),
                   help="Path to manifest.csv (truth).")
    # Choose split by either a csv file OR a split name found inside manifest['split']
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split-csv", type=str, help="CSV of the split to evaluate (e.g., OUT_ROOT/splits/val.csv).")
    g.add_argument("--split-name", type=str, choices=["train","val","test"],
                   help="Split name inside manifest['split'] if present.")
    p.add_argument("--predictions", type=str, required=True,
                   help="CSV with columns: path,pred_class_id")
    p.add_argument("--out-tag", type=str, default="eval",
                   help="Folder tag under OUT_ROOT/metrics/ (e.g., 'val', 'test', 'cam_mix').")
    p.add_argument("--params-json", type=str, default=None,
                   help="Optional: path to a params.json to copy next to metrics.")
    p.add_argument("--per-driver", action="store_true", help="Also compute per-driver metrics if driver_id exists.")
    p.add_argument("--per-camera", action="store_true", help="Also compute per-camera metrics.")
    args = p.parse_args(argv)

    manifest = _read_manifest(Path(args.manifest))

    # Pick the rows to evaluate
    if args.split_csv:
        split_df = _read_split_csv(Path(args.split_csv))
        # Join to keep only those paths that belong to the split
        truth = split_df[["path","class_id"]].merge(
            manifest.drop_duplicates("path"),
            on=["path","class_id"], how="left",
            suffixes=("","_mf")
        )
    else:
        truth = _pick_split_rows(manifest, args.split_name)

    preds = _read_predictions(Path(args.predictions))

    # Build run directory
    base = config.OUT_ROOT / "metrics"
    run_dir = new_run_dir(base, args.out_tag)
    print(f"[i] Run directory: {run_dir}")

    # Save a tiny run manifest (what we evaluated)
    tiny = {
        "manifest": str(Path(args.manifest).resolve()),
        "split_source": args.split_csv if args.split_csv else f"manifest[split=='{args.split_name}']",
        "predictions": str(Path(args.predictions).resolve()),
        "num_truth_rows": int(len(truth)),
        "tag": args.out_tag,
        "class_ids": CLASS_IDS,
    }
    save_json(run_dir / "inputs.json", tiny)

    # Evaluate
    metrics = evaluate(
        truth_df=truth,
        pred_df=preds,
        label_order=CLASS_IDS,
        per_driver=args.per_driver,
        per_camera=args.per_camera,
    )
    save_json(run_dir / "metrics.json", metrics)

    # Copy params.json if provided (hyperparams)
    if args.params_json:
        params = json.loads(Path(args.params_json).read_text())
        save_json(run_dir / "params.json", params)

    # A friendly summary.txt
    lines = []
    lines.append(f"Support: {metrics['num_examples']}")
    lines.append(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
    lines.append(f"Macro-F1: {metrics['overall']['macro_avg']['f1']:.4f}")
    for c in CLASS_IDS:
        pc = metrics["overall"]["per_class"][c]
        lines.append(f"{c} ({CLASS_MAP[c]}): P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")
    save_text(run_dir / "summary.txt", "\n".join(lines))

    print("[✓] Wrote:", run_dir / "metrics.json")
    print("[✓] Wrote:", run_dir / "summary.txt")
    if args.params_json:
        print("[✓] Copied params.json")

if __name__ == "__main__":
    main()
