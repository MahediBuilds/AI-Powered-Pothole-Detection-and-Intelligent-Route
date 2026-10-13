"""Generate the Ultralytics data.yaml for the final YOLO-seg dataset.

Was a 0-byte placeholder in the original project; the logic previously lived
inline in the training script. Pulled out as its own step so the dataset
config can be inspected/regenerated independently of training.

The split paths are written relative to the repo root and no `path:` key is
set, so Ultralytics resolves them against data.yaml's own folder. That keeps
the committed file portable - an absolute path only works on the machine
that generated it.
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
    images_dir = (config.FINAL_DIR / "images").relative_to(config.DATA_YAML.parent).as_posix()
    data = {
        "train": f"{images_dir}/train",
        "val": f"{images_dir}/val",
        "test": f"{images_dir}/test",
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
