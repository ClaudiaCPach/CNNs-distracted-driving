from __future__ import annotations

import argparse

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
    )
    run_training(cfg)


if __name__ == "__main__":
    main()

