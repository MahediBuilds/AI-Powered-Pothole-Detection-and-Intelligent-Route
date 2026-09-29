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

No human-graded ground truth was found in this repo to calibrate against.
The thresholds below were derived from a dry run of the actual inference
pipeline (09_infer_seg.py -> this script) against the only real images
present in the repository: the 99 ground-truth visualization overlays
under results/segmentation_visualization/{train,val,test} plus the single
image in test_images/ (100 images, 34 detections at the default confidence
threshold). That run showed the previous area thresholds (2000 / 6000 px)
were roughly two orders of magnitude too small for this dataset's actual
frame resolution (1080x1080, mask_area_px measured in full-frame pixel
space, not model input space) - 33 of 34 detections were classified
Severe, which does not discriminate. The area thresholds below are set at
the 33rd/67th percentile of the measured mask_area_px distribution from
that run (33rd pct ~= 307,971 px, 67th pct ~= 520,499 px), giving a roughly
even three-way split on the observed sample. Circularity and aspect-ratio
were left close to the original spec values, since the dry run showed the
old cutoffs (25, 2.5) already sit near the median/75th-percentile of the
observed distribution and discriminate reasonably well as-is; both were
nudged slightly to better match the measured 75th percentile.

Caveat: this is a same-source-as-original-spec heuristic recalibration
against N=34 detections drawn from GT-overlay images (the dataset itself
is absent from this repo), not a fit against human severity grades. Treat
these as a considerably better starting point than the previous defaults,
not as validated, production-ready thresholds - re-run
scripts/09_infer_seg.py + this script against the real dataset (once
available) and recompute percentiles, or better, calibrate against human
severity ratings, before relying on this for a real deployment decision.
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
    "area_small_max": 308000,     # px: area < this -> "small" (33rd pct of dry-run sample, N=34)
    "area_medium_max": 520000,    # px: small..this -> "medium", above -> "large" (67th pct)
    "circularity_irregular": 30,  # perimeter^2/area above this -> "irregular" (~median-to-75th pct)
    "aspect_ratio_elongated": 2.2,  # max(w,h)/min(w,h) above this -> "elongated" (~75th pct)
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
