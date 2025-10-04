import argparse, csv, re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from ddriver import config
except Exception as e:
    raise RuntimeError("ddriver.config must be importable before using manifest tools") from e


# Friendly class names for c0..c9
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


# Edit these to hardcode your driver assignments and split choices.
# Each driver can appear across multiple classes and original train/test folders.
# Format: (start, end, driver_id, class_id, orig_split)
# Example:
# DRIVER_RANGES = [
#     (0, 99, "D001", "c0", "train"),    # D001 in c0 train photos 0-99
#     (100, 199, "D001", "c1", "train"), # D001 in c1 train photos 100-199
#     (0, 99, "D002", "c0", "test"),     # D002 in c0 test photos 0-99
# ]
# SPLIT_MAP = {
#     "D001": "train",  # D001 goes to your custom train split
#     "D002": "val",    # D002 goes to your custom val split
# }
DRIVER_RANGES: List[Tuple[int, int, str, str, str, str]] = [
    # Example: D001 appears across classes in original train folder
    (416, 550, "D001", "c0", "train", "Camera 1"),
    (272, 304, "D001", "c1", "train", "Camera 1"),
    (174, 204, "D001", "c2", "train", "Camera 1"),
    (196, 213, "D001", "c3", "train", "Camera 1"),
    (154, 170, "D001", "c4", "train", "Camera 1"),
    (164, 182, "D001", "c5", "train", "Camera 1"),
    (201, 229, "D001", "c6", "train", "Camera 1"),
    (146, 168, "D001", "c7", "train", "Camera 1"),
    (163, 195, "D001", "c8", "train", "Camera 1"),
    (262, 336, "D001", "c9", "train", "Camera 1"),
    (555, 633, "D002", "c0", "train", "Camera 1"),
    (317, 389, "D002", "c1", "train", "Camera 1"),
    (210, 238, "D002", "c2", "train", "Camera 1"),
    (220, 250, "D002", "c3", "train", "Camera 1"),
    (182, 223, "D002", "c4", "train", "Camera 1"),
    (193, 212, "D002", "c5", "train", "Camera 1"),
    (244, 273, "D002", "c6", "train", "Camera 1"),
    (174, 213, "D002", "c7", "train", "Camera 1"),
    (206, 218, "D002", "c8", "train", "Camera 1"),
    (392, 419, "D002", "c9", "train", "Camera 1"),
    (637, 710, "D003", "c0", "train", "Camera 1"),
    (399, 438, "D003", "c1", "train", "Camera 1"),
    (244, 273, "D003", "c2", "train", "Camera 1"),
    (256, 271, "D003", "c3", "train", "Camera 1"),
    (233, 261, "D003", "c4", "train", "Camera 1"),
    (222, 260, "D003", "c5", "train", "Camera 1"),
    (323, 346, "D003", "c6", "train", "Camera 1"),
    (219, 241, "D003", "c7", "train", "Camera 1"),
    (261, 282, "D003", "c8", "train", "Camera 1"),
    (450, 484, "D003", "c9", "train", "Camera 1"),
    (711, 805, "D004", "c0", "train", "Camera 1"),
    (449, 475, "D004", "c1", "train", "Camera 1"),
    (280, 304, "D004", "c2", "train", "Camera 1"),
    (282, 308, "D004", "c3", "train", "Camera 1"),
    (268, 289, "D004", "c4", "train", "Camera 1"),
    (272, 301, "D004", "c5", "train", "Camera 1"),
    (354, 371, "D004", "c6", "train", "Camera 1"),
    (249, 265, "D004", "c7", "train", "Camera 1"),
    (285, 301, "D004", "c8", "train", "Camera 1"),
    (489, 508, "D004", "c9", "train", "Camera 1"),
    ''' (811, 845, "D005", "c0", "train", "Camera 1"),
    (484, 533, "D005", "c1", "train", "Camera 1"),
    (310, 338, "D005", "c2", "train", "Camera 1"),
    (318, 345, "D005", "c3", "train", "Camera 1"),
    (301, 343, "D005", "c4", "train", "Camera 1"),
    (312, 359, "D005", "c5", "train", "Camera 1"),
    (389, 424, "D005", "c6", "train", "Camera 1"),
    (275, 293, "D005", "c7", "train", "Camera 1"),
    (307, 338, "D005", "c8", "train", "Camera 1"),
    (514, 722, "D005", "c9", "train", "Camera 1"),
    (852, 959, "D006", "c0", "train", "Camera 1"),
    (536, 623, "D006", "c1", "train", "Camera 1"),
    (358, 399, "D006", "c2", "train", "Camera 1"),
    (350, 396, "D006", "c3", "train", "Camera 1"),
    (352, 395, "D006", "c4", "train", "Camera 1"),
    (377, 391, "D006", "c5", "train", "Camera 1"),
    (447, 474, "D006", "c6", "train", "Camera 1"),
    (311, 327, "D006", "c7", "train", "Camera 1"),
    (357, 379, "D006", "c8", "train", "Camera 1"),
    # D006 is not represented in c9 (0, 0, "D006", "c9", "train", "Camera 1"),
    (961, 1138, "D007", "c0", "train", "Camera 1"),
    (638, 679, "D007", "c1", "train", "Camera 1"),
    (408, 446, "D007", "c2", "train", "Camera 1"),
    (402, 426, "D007", "c3", "train", "Camera 1"),
    (405, 445, "D007", "c4", "train", "Camera 1"),
    (402, 414, "D007", "c5", "train", "Camera 1"),
    (495, 508, "D007", "c6", "train", "Camera 1"),
    (336, 375, "D007", "c7", "train", "Camera 1"),
    (389, 442, "D007", "c8", "train", "Camera 1"),
    (772, 847, "D007", "c9", "train", "Camera 1"), '''
    
    (266, 266, "D008", "c0", "train", "Camera 2"),
    (2114, 2896, "D008", "c0", "train", "Camera 2"),
    (2898, 2901, "D008", "c1", "train", "Camera 2"),
    (21002, 21796, "D008", "c1", "train", "Camera 2"),
    (21798, 22658, "D008", "c2", "train", "Camera 2"),
    (22697, 23596, "D008", "c3", "train", "Camera 2"),
    (23598, 24490, "D008", "c4", "train", "Camera 2"),
    (24524, 25371, "D008", "c5", "train", "Camera 2"),
    (25419, 26235, "D008", "c6", "train", "Camera 2"),
    (26283, 27035, "D008", "c7", "train", "Camera 2"),
    (27184, 27959, "D008", "c8", "train", "Camera 2"),  
    (28065, 28817, "D008", "c9", "train", "Camera 2"),
    (378, 378, "D009", "c0", "train", "Camera 2"),
    (3109, 3774, "D009", "c0", "train", "Camera 2"),
    (3897, 3960, "D009", "c1", "train", "Camera 2"),
    (31013, 31661, "D009", "c1", "train", "Camera 2"),
    (31795, 32663, "D009", "c2", "train", "Camera 2"),
    (32692, 33451, "D009", "c3", "train", "Camera 2"),
    (33589, 34444, "D009", "c4", "train", "Camera 2"),
    (34483, 35331, "D009", "c5", "train", "Camera 2"),
    (35465, 36270, "D009", "c6", "train", "Camera 2"),
    (36320, 36991, "D009", "c7", "train", "Camera 2"),
    (37373, 38049, "D009", "c8", "train", "Camera 2"),
    (38099, 38947, "D009", "c9", "train", "Camera 2"),
    (11122, 11897, "D010", "c0", "train", "Camera 2"),
    (111391, 111763, "D010", "c1", "train", "Camera 2"),
    (111798, 112502, "D010", "c2", "train", "Camera 2"),
    (112696, 113384, "D010", "c3", "train", "Camera 2"),
    (113593, 114325, "D010", "c4", "train", "Camera 2"),
    (114494, 115182, "D010", "c5", "train", "Camera 2"),
    (115392, 116244, "D010", "c6", "train", "Camera 2"),
    (116286, 116930, "D010", "c7", "train", "Camera 2"),
    (117167, 117953, "D010", "c8", "train", "Camera 2"),
    (118071, 118933, "D010", "c9", "train", "Camera 2")
    
    ''',
    (0, 0, "D011", "c0", "train", "Camera 1"),
    (0, 0, "D011", "c1", "train", "Camera 1"),
    (0, 0, "D011", "c2", "train", "Camera 1"),
    (0, 0, "D011", "c3", "train", "Camera 1"),
    (0, 0, "D011", "c4", "train", "Camera 1"),
    (0, 0, "D011", "c5", "train", "Camera 1"),
    (0, 0, "D011", "c6", "train", "Camera 1"),
    (0, 0, "D011", "c7", "train", "Camera 1"),
    (0, 0, "D011", "c8", "train", "Camera 1"),
    (0, 0, "D011", "c9", "train", "Camera 1"),
    (0, 0, "D012", "c0", "train", "Camera 1"),
    (0, 0, "D012", "c1", "train", "Camera 1"),
    (0, 0, "D012", "c2", "train", "Camera 1"),
    (0, 0, "D012", "c3", "train", "Camera 1"),
    (0, 0, "D012", "c4", "train", "Camera 1"),
    (0, 0, "D012", "c5", "train", "Camera 1"),
    (0, 0, "D012", "c6", "train", "Camera 1"),
    (0, 0, "D012", "c7", "train", "Camera 1"),
    (0, 0, "D012", "c8", "train", "Camera 1"),
    (0, 0, "D012", "c9", "train", "Camera 1"),
    (0, 0, "D013", "c0", "train", "Camera 1"),
    (0, 0, "D013", "c1", "train", "Camera 1"),
    (0, 0, "D013", "c2", "train", "Camera 1"),
    (0, 0, "D013", "c3", "train", "Camera 1"),
    (0, 0, "D013", "c4", "train", "Camera 1"),
    (0, 0, "D013", "c5", "train", "Camera 1"),
    (0, 0, "D013", "c6", "train", "Camera 1"),
    (0, 0, "D013", "c7", "train", "Camera 1"),
    (0, 0, "D013", "c8", "train", "Camera 1"),
    (0, 0, "D013", "c9", "train", "Camera 1"),
    # Add more ranges for D001 across other classes...
    
    # Example: D002 appears across classes in original test folder  
    (0, 99, "D002", "c0", "test"),
    (100, 199, "D002", "c1", "test"),
    (200, 299, "D002", "c2", "test"),
    # Add more ranges for D002 across other classes... '''
]

SPLIT_MAP: Dict[str, str] = {
    "D001": "train",  # D001 goes to your custom train split
    "D002": "val",    # D002 goes to your custom val split
    # Add more driver assignments...
}




def parse_img_num(filename: str) -> Optional[int]:
    """
    Extract a trailing number from a filename like "803.jpg" → 803.
    Returns None if not found.
    """
    m = re.search(r"(\d+)(?=\.(jpg|jpeg|png)$)", filename, re.I)
    return int(m.group(1)) if m else None


def _iter_v2_images(dataset_root: Path) -> Iterable[Dict[str, object]]:
    """
    Walk v2 layout:
    <root>/v2_cam1_cam2_split_by_driver/Camera X/{train,test}/c0..c9/*.jpg
    Yields dict rows with basic metadata.
    """
    v2 = dataset_root / "v2_cam1_cam2_split_by_driver"
    if not v2.exists():
        return []

    for camera_dir in ["Camera 1", "Camera 2"]:
        camera_path = v2 / camera_dir
        if not camera_path.exists():
            continue
        camera = "cam1" if camera_dir.endswith("1") else "cam2"
        for orig_split in ["train", "test"]:
            split_path = camera_path / orig_split
            if not split_path.exists():
                continue
            for class_id in CLASS_MAP.keys():
                class_path = split_path / class_id
                if not class_path.exists():
                    continue
                for img_path in class_path.iterdir():
                    if not img_path.is_file():
                        continue
                    img_num = parse_img_num(img_path.name)
                    yield {
                        "path": str(img_path),
                        "camera": camera,
                        "orig_split": orig_split,
                        "class_id": class_id,
                        "class_name": CLASS_MAP[class_id],
                        "img_num": img_num,
                    }




def build_manifest(dataset_root: Path) -> pd.DataFrame:
    """
    Create a manifest DataFrame with one row per image.
    Columns: path, camera, orig_split, class_id, class_name, img_num
    """
    rows = list(_iter_v2_images(dataset_root))
    if not rows:
        return pd.DataFrame(columns=[
            "path", "camera", "orig_split", "class_id", "class_name", "img_num",
        ])
    df = pd.DataFrame(rows)
    df.sort_values(["camera", "class_id", "img_num", "path"], inplace=True, ignore_index=True)
    return df


def _parse_driver_ranges_spec(spec: str) -> List[Tuple[int, int, str, str, str]]:
    """
    Parse a compact ranges string like: "0-99:D001:c0:train,100-199:D001:c1:train".
    Returns list of (start, end, driver_id, class_id, orig_split).
    """
    items = []
    if not spec:
        return items
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split(':')
        if len(parts) != 5:
            continue  # Skip malformed entries
        span, driver_id, class_id, orig_split = parts[0], parts[1], parts[2], parts[3]
        start_str, end_str = span.split('-', 1)
        items.append((int(start_str), int(end_str), driver_id.strip(), 
                     class_id.strip(), orig_split.strip()))
    return items


def _read_driver_ranges_csv(path: Path) -> List[Tuple[int, int, str, str, str]]:
    """
    CSV with columns: start,end,driver_id,class_id,orig_split
    """
    ranges: List[Tuple[int, int, str, str, str]] = []
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranges.append((int(row['start']), int(row['end']), str(row['driver_id']), 
                         str(row['class_id']), str(row['orig_split'])))
    return ranges


def assign_driver_ids(df: pd.DataFrame, ranges: List[Tuple[int, int, str, str, str]]) -> pd.DataFrame:
    """
    Assign driver_id based on img_num, class_id, and orig_split matching provided ranges.
    Format: (start, end, driver_id, class_id, orig_split)
    If no match, driver_id stays None.
    """
    if not ranges:
        df = df.copy()
        df["driver_id"] = None
        return df

    def find_driver(row) -> Optional[str]:
        img_num = row["img_num"]
        class_id = row["class_id"]
        orig_split = row["orig_split"]
        
        if img_num is None or pd.isna(img_num):
            return None
        val = int(img_num)
        
        for start, end, driver_id, target_class, target_split in ranges:
            if (start <= val <= end and 
                class_id == target_class and 
                orig_split == target_split):
                return driver_id
        return None

    df = df.copy()
    df["driver_id"] = df.apply(find_driver, axis=1)
    return df


def _parse_split_map_spec(spec: str) -> Dict[str, str]:
    """
    Parse spec like: "D001:train,D002:val,D003:test" → {driver_id: split}
    """
    mapping: Dict[str, str] = {}
    if not spec:
        return mapping
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        driver_id, split = token.split(':', 1)
        mapping[driver_id.strip()] = split.strip()
    return mapping


def _read_split_map_csv(path: Path) -> Dict[str, str]:
    """
    CSV with columns: driver_id,split
    """
    mapping: Dict[str, str] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[str(row['driver_id'])] = str(row['split'])
    return mapping


def assign_splits_by_driver(df: pd.DataFrame, split_map: Dict[str, str]) -> pd.DataFrame:
    """
    Assign split per driver_id. Rows with missing driver_id or missing mapping get split=None.
    Ensures a single driver does not straddle splits by using the mapping directly.
    """
    df = df.copy()
    df["split"] = df["driver_id"].map(lambda d: split_map.get(d) if d is not None else None)
    return df


def write_manifest_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_split_lists(df: pd.DataFrame) -> None:
    """
    Write per-split CSVs with columns: path,class_id,class_name,driver_id.
    Uses ddriver.config.split_csv to place them under OUT_ROOT/splits.
    """
    have_split = df["split"].notna()
    for split_name in sorted(df.loc[have_split, "split"].dropna().unique()):
        sub = df.loc[df["split"] == split_name, ["path", "class_id", "class_name", "driver_id"]].copy()
        out_path = config.split_csv(f"{split_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build image manifest with driver and split assignments.")
    parser.add_argument("--dataset-root", type=str, default=str(config.DATASET_ROOT), help="Dataset root (folder containing v2_cam1_cam2_split_by_driver)")
    parser.add_argument("--manifest-out", type=str, default=str(config.OUT_ROOT / "manifests" / "manifest.csv"), help="Output path for the manifest CSV")
    parser.add_argument("--write-split-lists", action="store_true", help="Also write OUT_ROOT/splits/{split}.csv files")

    # Expert escape hatches (optional overrides). If unused, hardcoded DRIVER_RANGES/SPLIT_MAP apply.
    parser.add_argument("--driver-ranges-file", type=str, default=None, help="CSV with columns: start,end,driver_id,class_id,orig_split (overrides hardcoded ranges)")
    parser.add_argument("--driver-ranges", type=str, default=None, help="Inline ranges, e.g. '0-99:D001:c0:train,100-199:D001:c1:train' (overrides hardcoded ranges)")
    parser.add_argument("--split-map-file", type=str, default=None, help="CSV with columns: driver_id,split (overrides hardcoded map)")
    parser.add_argument("--split-map", type=str, default=None, help="Inline map, e.g. 'D001:train,D002:val' (overrides hardcoded map)")

    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)

    df = build_manifest(dataset_root)

    # Choose driver ranges: CLI overrides > hardcoded > none
    ranges: List[Tuple[int, int, str, str, str]] = DRIVER_RANGES
    if args.driver_ranges_file:
        ranges = _read_driver_ranges_csv(Path(args.driver_ranges_file))
    elif args.driver_ranges:
        ranges = _parse_driver_ranges_spec(args.driver_ranges)

    df = assign_driver_ids(df, ranges)

    # Choose split map: CLI overrides > hardcoded > none
    split_map: Dict[str, str] = SPLIT_MAP
    if args.split_map_file:
        split_map = _read_split_map_csv(Path(args.split_map_file))
    elif args.split_map:
        split_map = _parse_split_map_spec(args.split_map)

    if split_map:
        df = assign_splits_by_driver(df, split_map)
    else:
        df["split"] = None

    # Write manifest
    manifest_out = Path(args.manifest_out)
    write_manifest_csv(df, manifest_out)

    # Optionally write per-split CSVs
    if args.write_split_lists:
        write_split_lists(df)

    print(f"Wrote manifest to: {manifest_out}")
    if args.write_split_lists:
        print(f"Per-split CSVs written under: {config.OUT_ROOT / 'splits'}")


if __name__ == "__main__":
    main()
