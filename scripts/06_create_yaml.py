"""Generate the Ultralytics data.yaml for the final YOLO-seg dataset.

Was a 0-byte placeholder in the original project; the logic previously lived
inline in the training script. Pulled out as its own step so the dataset
config can be inspected/regenerated independently of training.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def verify_dataset_exists():
    required = [
        config.FINAL_DIR / "images" / "train",
        config.FINAL_DIR / "images" / "val",
        config.FINAL_DIR / "labels" / "train",
        config.FINAL_DIR / "labels" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset directories: {missing}")

    for split_dir in (config.FINAL_DIR / "images" / "train", config.FINAL_DIR / "images" / "val"):
        if not any(split_dir.glob("*.jpg")):
            raise FileNotFoundError(
                f"{split_dir} exists but contains no images. "
                f"Run scripts/01-03 first to populate dataset/final/."
            )


def create_yaml():
    data = {
        "path": str(config.FINAL_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": config.CLASS_NAMES,
    }

    with open(config.DATA_YAML, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return data


def main():
    verify_dataset_exists()
    data = create_yaml()
    print(f"Wrote {config.DATA_YAML}")
    print(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
