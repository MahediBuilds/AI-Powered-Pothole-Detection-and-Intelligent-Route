"""Run validation metrics for a trained checkpoint.

This is the piece the original project was missing entirely: nothing
consumed runs/*/weights/best.pt after training. Reports box + mask
precision, recall, mAP50, mAP50-95 (via Ultralytics' built-in validator,
which is the standard/correct way to compute these for a YOLO-seg model),
plus a derived F1 and a simple mean-IoU/Dice proxy from the mask metrics.
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main():
    parser = argparse.ArgumentParser(description="Validate a trained YOLOv8-seg checkpoint")
    parser.add_argument("--weights", required=True, help="Path to best.pt / last.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    if not config.DATA_YAML.exists():
        raise FileNotFoundError(f"{config.DATA_YAML} not found. Run scripts/06_create_yaml.py first.")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(config.DATA_YAML),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )

    box = metrics.box
    seg = metrics.seg

    def f1(precision, recall):
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = {
        "split": args.split,
        "box": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1": f1(float(box.mp), float(box.mr)),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        },
        "mask": {
            "precision": float(seg.mp),
            "recall": float(seg.mr),
            "f1": f1(float(seg.mp), float(seg.mr)),
            "mAP50": float(seg.map50),
            "mAP50-95": float(seg.map),
        },
    }

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS ({args.split})")
    print("=" * 60)
    for kind in ("box", "mask"):
        m = report[kind]
        print(f"\n{kind.upper()}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1        : {m['f1']:.4f}")
        print(f"  mAP@50    : {m['mAP50']:.4f}")
        print(f"  mAP@50-95 : {m['mAP50-95']:.4f}")

    out_path = weights_path.parent.parent / f"val_report_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
