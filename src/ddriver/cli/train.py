from __future__ import annotations

import argparse

from ddriver.data.datamod import make_cfg_from_config
from ddriver.train.loop import TrainLoopConfig, run_training

from ddriver.models import registry



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on the distracted driver dataset.")
    parser.add_argument("--model-name", required=True, help="Registered model name (e.g., resnet18).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--loss-name", default="cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--out-tag", default="experiment")
    parser.add_argument("--device", default=None, help="Optional device override, e.g., cuda:0")
    parser.add_argument("--manifest-csv", default=None, help="Override manifest CSV path.")
    parser.add_argument("--train-csv", default=None, help="Override train split CSV path (e.g., train_small.csv).")
    parser.add_argument("--val-csv", default=None, help="Override val split CSV path.")
    parser.add_argument("--test-csv", default=None, help="Override test split CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Auto-register timm backbones when requested
    try:
        registry.register_timm_backbone(args.model_name)
    except ImportError:
        pass  # timm not installed
    except ValueError:
        pass  # already registered or custom model
    
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

    cfg = TrainLoopConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        loss_name=args.loss_name,
        label_smoothing=args.label_smoothing,
        out_tag=args.out_tag,
        device=args.device,
        data_cfg=data_cfg,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()

