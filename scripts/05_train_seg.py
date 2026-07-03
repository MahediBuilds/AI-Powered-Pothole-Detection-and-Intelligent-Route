from pathlib import Path
import multiprocessing
import torch
from ultralytics import YOLO
import yaml

# ==========================
# CONFIGURATION
# ==========================

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset" / "final"
DATA_YAML = ROOT / "data.yaml"

MODEL = "yolov8s-seg.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 8
WORKERS = 8

PROJECT = "runs"
NAME = "pothole_seg"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================


def verify():
    required = [
        DATASET / "images" / "train",
        DATASET / "images" / "val",
        DATASET / "labels" / "train",
        DATASET / "labels" / "val",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")


def create_yaml():
    data = {
        "path": str(DATASET),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "pothole"
        }
    }

    with open(DATA_YAML, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def print_gpu():

    print("=" * 60)

    if torch.cuda.is_available():

        print("GPU :", torch.cuda.get_device_name(0))

        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        print(f"VRAM: {mem:.2f} GB")

    else:

        print("CUDA not available. Using CPU.")

    print("=" * 60)


def main():

    verify()

    create_yaml()

    print_gpu()

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        amp=True,
        cache=True,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
        pretrained=True,
        save=True,
        plots=True
    )

    print("\nTraining completed.")
    print("Best model:")
    print(ROOT / PROJECT / NAME / "weights" / "best.pt")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
