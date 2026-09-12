"""Train the YOLOv8 segmentation model.

Replaces the original 05_train_seg.py. Changes:
  - Fixes the model/weights mismatch (script referenced yolov8s-seg.pt while
    the repo shipped a committed yolov8m-seg.pt that no code ever used).
  - Explicit augmentation config (config.AUGMENTATION) tuned for road-surface
    imagery instead of silent Ultralytics defaults.
  - AdamW optimizer, cosine LR schedule, weight decay, early stopping
    (patience), close_mosaic for stable final-epoch convergence.
  - Warns (rather than silently proceeding) if the val or test split is
    missing/empty, since a meaningless val split would make every metric
    from this run unreliable.
  - --smoke-test flag for a fast 3-epoch run to confirm the pipeline works
    end-to-end before committing to a full training run.
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def verify_dataset():
    required = {
        "train images": config.FINAL_DIR / "images" / "train",
        "val images": config.FINAL_DIR / "images" / "val",
        "train labels": config.FINAL_DIR / "labels" / "train",
        "val labels": config.FINAL_DIR / "labels" / "val",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} directory: {path}")

    for split in ("train", "val", "test"):
        img_dir = config.FINAL_DIR / "images" / split
        n = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        if n == 0 and split in ("train", "val"):
            raise FileNotFoundError(
                f"'{split}' split has 0 images in {img_dir} — nothing to train/validate on. "
                f"Run scripts/01-03 first to populate dataset/final/."
            )
        elif n == 0:
            print(f"[WARN] '{split}' split has 0 images — metrics/behavior for this split will be meaningless.")


def print_device_info():
    print("=" * 60)
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"VRAM: {mem:.2f} GB")
    else:
        print("CUDA not available. Training on CPU (this will be slow).")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation on the pothole dataset")
    parser.add_argument("--model", default=config.MODEL_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--name", default="pothole_seg")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a fast 3-epoch pass to confirm the pipeline works end-to-end")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    verify_dataset()

    if not config.DATA_YAML.exists():
        raise FileNotFoundError(
            f"{config.DATA_YAML} not found. Run scripts/06_create_yaml.py first."
        )

    print_device_info()
    device = 0 if torch.cuda.is_available() else "cpu"

    epochs = 3 if args.smoke_test else args.epochs
    name = f"{args.name}_smoke" if args.smoke_test else args.name

    model = YOLO(args.model)

    model.train(
        data=str(config.DATA_YAML),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        amp=True,
        cache="disk",
        project=str(config.RUNS_DIR),
        name=name,
        exist_ok=True,
        verbose=True,
        pretrained=True,
        save=True,
        plots=True,
        resume=args.resume,
        optimizer=config.OPTIMIZER,
        lr0=config.LR0,
        weight_decay=config.WEIGHT_DECAY,
        cos_lr=config.COS_LR,
        patience=config.PATIENCE,
        close_mosaic=config.CLOSE_MOSAIC,
        **config.AUGMENTATION,
    )

    best = config.RUNS_DIR / name / "weights" / "best.pt"
    print("\nTraining completed.")
    print(f"Best model: {best}")
    if args.smoke_test:
        print("\nSmoke test finished. If this completed without errors, the pipeline works end-to-end.")
        print(f"Run without --smoke-test for a full training run (default {config.EPOCHS} epochs).")


if __name__ == "__main__":
    main()
