"""Classify each detected pothole as Mild / Moderate / Severe.

Reads a predictions.json produced by 09_infer_seg.py (list of detections
with bbox_xyxy, mask_area_px, polygon) and writes a NEW file with three
extra geometric features and a severity label added to every detection.
The input file is never modified.

Features computed per detection:
  1. Pixel area       - reused from mask_area_px (already computed at
                         inference time from the smoothed mask), not
                         recomputed here.
  2. Circularity      - perimeter^2 / area, from the detection's polygon
                         contour via cv2.arcLength. Higher = more jagged /
                         fractured edge = worse damage.
  3. Aspect ratio     - max(w,h) / min(w,h) from bbox_xyxy.

Classification logic:
  Mild     -> small area AND not irregular
  Severe   -> large area OR (medium area AND irregular)
  Moderate -> everything else

No ground truth was found in this repo to calibrate against, so the
thresholds below are the defaults from the project spec. Adjust the
THRESHOLDS dict below if/when real calibration data becomes available.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Thresholds - kept as named constants so they're easy to find and adjust.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "area_small_max": 2000,       # px: area < this -> "small"
    "area_medium_max": 6000,      # px: small..this -> "medium", above -> "large"
    "circularity_irregular": 25,  # perimeter^2/area above this -> "irregular"
    "aspect_ratio_elongated": 2.5,  # max(w,h)/min(w,h) above this -> "elongated"
}


def bucket_area(area_px):
    if area_px < THRESHOLDS["area_small_max"]:
        return "small"
    if area_px <= THRESHOLDS["area_medium_max"]:
        return "medium"
    return "large"


def compute_circularity(polygon):
    """perimeter^2 / area via cv2.arcLength / cv2.contourArea on the polygon."""
    if not polygon or len(polygon) < 3:
        return 0.0
    contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, closed=True)
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    return float((perimeter ** 2) / area)


def compute_aspect_ratio(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return float(max(w, h) / min(w, h))


def classify(area_bucket, is_irregular):
    if area_bucket == "small" and not is_irregular:
        return "Mild"
    if area_bucket == "large" or (area_bucket == "medium" and is_irregular):
        return "Severe"
    return "Moderate"


def process(detections):
    print(f"Classifying severity for {len(detections)} detections "
          f"(thresholds: {THRESHOLDS})")

    counts = {"Mild": 0, "Moderate": 0, "Severe": 0}

    for det in detections:
        pixel_area = det.get("mask_area_px", 0)
        circularity = compute_circularity(det.get("polygon"))
        aspect_ratio = compute_aspect_ratio(det["bbox_xyxy"])

        area_bucket = bucket_area(pixel_area)
        is_irregular = circularity > THRESHOLDS["circularity_irregular"]
        is_elongated = aspect_ratio > THRESHOLDS["aspect_ratio_elongated"]

        severity = classify(area_bucket, is_irregular or is_elongated)

        det["pixel_area"] = pixel_area
        det["circularity"] = round(circularity, 3)
        det["aspect_ratio"] = round(aspect_ratio, 3)
        det["severity"] = severity

        counts[severity] += 1

    print(f"Done. Breakdown -> Mild: {counts['Mild']}, "
          f"Moderate: {counts['Moderate']}, Severe: {counts['Severe']}")
    return detections


def main():
    parser = argparse.ArgumentParser(description="Classify pothole severity from predictions.json")
    parser.add_argument(
        "--input",
        default=str(config.RESULTS_DIR / "inference" / "predictions.json"),
        help="Path to predictions.json produced by 09_infer_seg.py",
    )
    parser.add_argument(
        "--output",
        default=str(config.RESULTS_DIR / "inference" / "predictions_severity.json"),
        help="Path to write the new file with severity fields added (input is never modified)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {in_path}")

    print(f"Loading detections from {in_path}")
    with open(in_path) as f:
        detections = json.load(f)

    detections = process(detections)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"Wrote severity-classified detections to: {out_path}")


if __name__ == "__main__":
    main()
