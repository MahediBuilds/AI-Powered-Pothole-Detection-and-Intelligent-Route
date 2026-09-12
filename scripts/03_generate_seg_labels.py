"""Convert pixel masks into YOLOv8 segmentation polygon labels.

Fixes vs. the original version:
  - Morphological open+close on the binarized mask before contour detection,
    to remove H.264 compression speckle/ringing around pothole edges instead
    of feeding raw video-compression noise directly into the polygons the
    model trains on.
  - Minimum contour area is a fraction of image area (not a fixed pixel
    count), so it scales correctly across different source resolutions.
  - Reports per-split object/negative-image counts to a JSON summary so class
    imbalance is visible instead of silent.
"""

import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SPLITS = config.SPLITS
MAX_WORKERS = os.cpu_count() or 1
_MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
)


def create_directories():
    for split in SPLITS:
        (config.FINAL_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.FINAL_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def clean_mask(mask):
    """Reduce video-compression noise before contour extraction."""
    _, binary = cv2.threshold(mask, config.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return binary


def process_image(task):
    image_path, mask_path, output_image, output_label = task
    try:
        if not output_image.exists():
            shutil.copyfile(image_path, output_image)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"objects": 0, "error": f"unreadable mask: {mask_path.name}"}

        h, w = mask.shape
        min_area = config.MIN_CONTOUR_AREA_FRACTION * (h * w)

        binary = clean_mask(mask)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = 0
        with open(output_label, "w") as f:
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                epsilon = config.POLYGON_EPSILON_FACTOR * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                pts = []
                for p in approx:
                    x, y = p[0]
                    pts.append(f"{x / w:.6f}")
                    pts.append(f"{y / h:.6f}")
                f.write(f"{config.CLASS_ID} {' '.join(pts)}\n")
                objects += 1

        return {"objects": objects}
    except Exception as e:
        return {"objects": 0, "error": f"{mask_path.name}: {e}"}


def process_split(split):
    image_dir = config.PROCESSED_DIR / "images" / split
    mask_dir = config.PROCESSED_DIR / "masks" / split
    out_image = config.FINAL_DIR / "images" / split
    out_label = config.FINAL_DIR / "labels" / split

    tasks = []
    for mask in sorted(mask_dir.glob("*.png")):
        image = image_dir / (mask.stem + config.RGB_EXTENSION)
        if image.exists():
            tasks.append((image, mask, out_image / image.name, out_label / (mask.stem + ".txt")))

    total_objects = 0
    negative_images = 0
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for result in tqdm(ex.map(process_image, tasks), total=len(tasks), desc=split.upper()):
            total_objects += result["objects"]
            if result["objects"] == 0:
                negative_images += 1
            if "error" in result:
                errors.append(result["error"])

    return {
        "images": len(tasks),
        "objects": total_objects,
        "negative_images": negative_images,
        "errors": errors,
    }


def main():
    create_directories()

    print("=" * 60)
    print("YOLOv8 SEGMENTATION LABEL GENERATION")
    print("=" * 60)
    print(f"Workers: {MAX_WORKERS}")

    summary = {}
    for split in SPLITS:
        stats = process_split(split)
        summary[split] = stats
        neg_pct = (stats["negative_images"] / stats["images"] * 100) if stats["images"] else 0
        print(f"\n{split.upper()} - Images: {stats['images']}  Objects: {stats['objects']}  "
              f"Negative (0-object) images: {stats['negative_images']} ({neg_pct:.1f}%)")
        if stats["errors"]:
            print(f"  {len(stats['errors'])} errors (see label_generation_report.json)")

    report_path = config.FINAL_DIR / "label_generation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
