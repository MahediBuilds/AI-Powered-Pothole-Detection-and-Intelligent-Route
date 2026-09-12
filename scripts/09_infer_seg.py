"""Run inference on images (or a video) with a trained checkpoint.

Post-processing applied on top of raw Ultralytics predictions:
  - Confidence threshold (config.CONF_THRESHOLD)
  - Small-instance filtering by mask area (config.MIN_MASK_AREA_PX) to drop
    noise predictions too small to be a real pothole
  - Morphological smoothing of predicted masks (config.MASK_SMOOTH_KERNEL)
  - Optional test-time augmentation via --tta (Ultralytics' built-in
    augmented inference)

Output: for every input image, an overlay jpg plus a row in
predictions.json with bbox, confidence, polygon and (if a geotag log is
supplied via --geotag) interpolated lat/lon. Without --geotag, lat/lon are
simply null - this is the extension point for the route-mapping feature,
not a required input right now.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.geo import load_geotag_log, interpolate_position

_SMOOTH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (config.MASK_SMOOTH_KERNEL, config.MASK_SMOOTH_KERNEL)
)


def smooth_mask(mask_uint8):
    mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, _SMOOTH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _SMOOTH_KERNEL)
    return mask


def mask_to_polygon(mask_uint8):
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).tolist()


def run(model, image_paths, args, geotag_log, video_stem):
    results_out = []
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] unreadable image, skipping: {img_path}")
            continue

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            augment=args.tta,
            verbose=False,
        )[0]

        overlay = img.copy()
        detections = []

        if results.masks is not None:
            h, w = img.shape[:2]
            for mask_t, box, conf in zip(results.masks.data, results.boxes.xyxy, results.boxes.conf):
                mask = mask_t.cpu().numpy().astype("uint8") * 255
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = smooth_mask(mask)

                area = int(np.count_nonzero(mask))
                if area < config.MIN_MASK_AREA_PX:
                    continue

                polygon = mask_to_polygon(mask)
                if polygon is None:
                    continue

                cv2.fillPoly(overlay, [np.array(polygon, dtype="int32")], (0, 255, 0))

                lat, lon = (None, None)
                if geotag_log:
                    fps = 30.0  # assumed; only used when caller passes frame timestamps directly
                    timestamp_sec = idx / fps
                    lat, lon = interpolate_position(timestamp_sec, geotag_log)

                detections.append({
                    "frame": img_path.name,
                    "source_video": video_stem,
                    "confidence": float(conf),
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "mask_area_px": area,
                    "polygon": polygon,
                    "lat": lat,
                    "lon": lon,
                })

        blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.imwrite(str(out_dir / img_path.name), blended)
        results_out.extend(detections)

    return results_out


def main():
    parser = argparse.ArgumentParser(description="Run pothole segmentation inference")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Directory of images")
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "inference"))
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--geotag", default=None, help="Optional CSV: timestamp_sec,lat,lon")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    source_dir = Path(args.source)
    image_paths = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    geotag_log = load_geotag_log(args.geotag)
    if args.geotag and not geotag_log:
        print(f"[WARN] --geotag given but no rows loaded from {args.geotag}")

    model = YOLO(str(weights_path))

    print(f"Running inference on {len(image_paths)} images (conf={args.conf}, iou={args.iou}, tta={args.tta})")
    detections = run(model, image_paths, args, geotag_log, video_stem=source_dir.name)

    out_json = Path(args.output) / "predictions.json"
    with open(out_json, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"\n{len(detections)} pothole instances detected across {len(image_paths)} images.")
    print(f"Overlays saved to: {args.output}")
    print(f"Structured predictions: {out_json}")


if __name__ == "__main__":
    main()
