from __future__ import annotations

import argparse

from ddriver.infer.predict import PredictConfig, run_prediction
from ddriver.models import registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prediction CSV from a checkpoint.")
    parser.add_argument("--model-name", required=True, help="Registered model name.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--split", default="val", help="Which split to run on (val/test).")
    parser.add_argument("--manifest-csv", default=None, help="Override manifest CSV path.")
    parser.add_argument("--train-csv", default=None, help="Optional train CSV (for datamodule completeness).")
    parser.add_argument("--val-csv", default=None, help="Override val CSV path.")
    parser.add_argument("--test-csv", default=None, help="Override test CSV path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-tag", default="preds", help="Prefix for output CSV.")
    parser.add_argument("--out-csv", default=None, help="Optional path to write predictions CSV.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Mirror train CLI behavior: auto-register timm backbones if needed
    try:
        registry.register_timm_backbone(args.model_name)
    except ImportError:
        pass  # timm not available in this environment
    except ValueError:
        pass  # already registered or custom builder supplied

    data_cfg = make_cfg_from_config(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    if args.manifest_csv:
        data_cfg.manifest_csv = args.manifest_csv
    if args.train_csv:
        data_cfg.train_split_csv = args.train_csv
    if args.val_csv:
        data_cfg.val_split_csv = args.val_csv
    if args.test_csv:
        data_cfg.test_split_csv = args.test_csv

    cfg = PredictConfig(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        out_tag=args.out_tag,
        device=args.device,
        out_csv=args.out_csv,
        data_cfg=data_cfg,
    )
    run_prediction(cfg)


if __name__ == "__main__":
    main()

