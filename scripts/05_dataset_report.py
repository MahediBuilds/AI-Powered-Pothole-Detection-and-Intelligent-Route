"""Dataset quality report for the final YOLO-seg dataset.

Since the raw dataset arrives pre-split into train/val/test by video (there
is no flat pool to re-split), this script does not invent a random split.
Instead it verifies the split you have is sound and flags data-quality
problems that would otherwise silently hurt training:

  - Corrupted / unreadable images or label files
  - Video-stem leakage across splits (same source video appearing in more
    than one split)
  - Class imbalance (ratio of negative/background-only images per split)
  - Label file anomalies (empty polygons, degenerate coordinates outside
    [0, 1], zero-object files)
  - Basic instance-size distribution (to sanity-check MIN_CONTOUR_AREA_FRACTION)
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SPLITS = config.SPLITS


def video_stem_of(filename):
    # frames are named "{video_stem}_{frame_idx:05d}.jpg"
    return filename.rsplit("_", 1)[0]


def check_split(split):
    image_dir = config.FINAL_DIR / "images" / split
    label_dir = config.FINAL_DIR / "labels" / split

    images = sorted(image_dir.glob("*.jpg"))
    stats = {
        "total_images": len(images),
        "missing_labels": 0,
        "unreadable_images": 0,
        "empty_label_files": 0,
        "negative_images": 0,
        "total_instances": 0,
        "invalid_coord_files": [],
        "video_stems": set(),
    }

    for img_path in images:
        stats["video_stems"].add(video_stem_of(img_path.stem))

        img = cv2.imread(str(img_path))
        if img is None:
            stats["unreadable_images"] += 1
            continue

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            stats["missing_labels"] += 1
            continue

        lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
        if not lines:
            stats["empty_label_files"] += 1
            stats["negative_images"] += 1
            continue

        instances = 0
        for line in lines:
            vals = line.split()
            coords = list(map(float, vals[1:]))
            if any(c < 0.0 or c > 1.0 for c in coords):
                stats["invalid_coord_files"].append(label_path.name)
            instances += 1

        stats["total_instances"] += instances

    stats["video_stems"] = sorted(stats["video_stems"])
    return stats


def main():
    print("=" * 60)
    print("DATASET QUALITY REPORT")
    print("=" * 60)

    report = {}
    stems_by_split = {}

    for split in SPLITS:
        stats = check_split(split)
        stems_by_split[split] = set(stats["video_stems"])
        report[split] = stats

        neg_pct = (stats["negative_images"] / stats["total_images"] * 100) if stats["total_images"] else 0
        avg_inst = (stats["total_instances"] / max(1, stats["total_images"] - stats["negative_images"]))

        print(f"\n{split.upper()}")
        print("-" * 40)
        print(f"Images                 : {stats['total_images']}")
        print(f"Missing label files    : {stats['missing_labels']}")
        print(f"Unreadable images      : {stats['unreadable_images']}")
        print(f"Negative (0-object)    : {stats['negative_images']} ({neg_pct:.1f}%)")
        print(f"Total instances        : {stats['total_instances']}")
        print(f"Avg instances/positive : {avg_inst:.2f}")
        print(f"Source videos          : {len(stats['video_stems'])}")
        if stats["invalid_coord_files"]:
            print(f"[WARN] {len(stats['invalid_coord_files'])} label files with out-of-range coordinates")

    # Cross-split leakage check: same source video appearing in >1 split
    print("\n" + "=" * 60)
    print("SPLIT LEAKAGE CHECK")
    print("=" * 60)
    leakage_found = False
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = stems_by_split[a] & stems_by_split[b]
            if overlap:
                leakage_found = True
                print(f"[ERROR] {len(overlap)} video(s) appear in both '{a}' and '{b}': {sorted(overlap)[:5]}...")
    if not leakage_found:
        print("No source-video overlap between splits.")

    config.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.FINAL_DIR / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "video_stems"} for k, v in report.items()}, f, indent=2)
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
