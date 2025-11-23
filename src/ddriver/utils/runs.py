# src/ddriver/utils/runs.py
from __future__ import annotations
import json, yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

def new_run_dir(base: Path, tag: str) -> Path:
    """
    Make a folder like: base / tag / 2025-11-09_12-34-56
    Returns the created path.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / tag / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
