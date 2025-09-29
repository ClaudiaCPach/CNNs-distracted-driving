#quick text file
from ddriver.config import (
    DATASET_ROOT, OUT_ROOT, CKPT_ROOT, FAST_DATA,
    dataset_dir, split_csv
)

print("=== LOCAL MODE CHECK ===")
print("DATASET_ROOT:", DATASET_ROOT)
print("OUT_ROOT    :", OUT_ROOT)
print("CKPT_ROOT   :", CKPT_ROOT)
print("FAST_DATA   :", FAST_DATA)
print("dataset_dir(False):", dataset_dir(False))
print("split_csv('cam1/train.csv'):", split_csv('cam1/train.csv'))
