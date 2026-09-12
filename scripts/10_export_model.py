"""Export a trained checkpoint to ONNX and TorchScript for deployment."""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main():
    parser = argparse.ArgumentParser(description="Export a trained YOLOv8-seg checkpoint")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"],
                         choices=["onnx", "torchscript"])
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = YOLO(str(weights_path))

    for fmt in args.formats:
        print(f"Exporting to {fmt}...")
        exported_path = model.export(format=fmt, imgsz=args.imgsz)
        print(f"  -> {exported_path}")


if __name__ == "__main__":
    main()
