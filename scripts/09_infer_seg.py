"""Run inference on images (or a video) with a trained checkpoint.

Post-processing applied on top of raw Ultralytics predictions:
  - Confidence threshold (config.CONF_THRESHOLD)
  - Small-instance filtering by mask area (config.MIN_MASK_AREA_PX) to drop
    noise predictions too small to be a real pothole
  - Morphological smoothing of predicted masks (config.MASK_SMOOTH_KERNEL)
  - Optional test-time augmentation via --tta (Ultralytics' built-in
    augmented inference)

Masks are predicted with retina_masks=True, so they come back at the
original frame resolution. An earlier version resized the model-resolution
mask straight to the frame size, which is only correct for square frames:
for e.g. a 1920x1080 dashcam frame the model input is a letterboxed 640x384,
and stretching that (padding included) shifted the mask and skewed its area.

Output: for every input image, an overlay jpg plus a row in
predictions.json with bbox, confidence, polygon, frame size and (if a
geotag log is supplied via --geotag) interpolated lat/lon. Without
--geotag, lat/lon are simply null - this is the extension point for the
route-mapping feature, not a required input right now.

Tracking (--track): consecutive frames of a video see the same pothole
several times. With --track, frames are run through Ultralytics' ByteTrack
tracker in filename order and every detection gets a track_id, which
12_build_detections.py uses to collapse each physical pothole to a single
record. Only detections that belong to a confirmed track are kept, so a
pothole has to appear in at least two sampled frames to be reported (except
in the first frame of a video). The tracker is reset whenever source_video
changes, so pass --frames-meta when --source mixes frames from several
videos; without it, all images are treated as one continuous video.

Geotagging here requires a real per-frame timestamp, which this script
does not know on its own - it only sees a directory of already-extracted
images, not the original video's fps or frame stride. Pass --frames-meta
pointing at the frames_meta.csv written by 02_extract_frames.py (or by
app.py's single-video extraction) to look up each frame's real
timestamp_sec by filename. Without --frames-meta, --geotag is ignored and
lat/lon are left null rather than guessed - an earlier version of this
script estimated timestamp_sec as (list position / assumed 30fps), which
silently produced wrong coordinates whenever the source video's real fps
differed from 30 or frames had been extracted at a stride > 1.
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
from utils.frames import load_frames_meta
from utils.geo import load_geotag_log, interpolate_position

_SMOOTH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (config.MASK_SMOOTH_KERNEL, config.MASK_SMOOTH_KERNEL)
)

# ByteTrack matches low-confidence detections in a second stage, so with
# --track it is fed detections down to this confidence; rows below --conf
# are dropped after tracking.
TRACK_DETECTION_CONF = 0.1


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


def reset_tracker(model):
    for tracker in getattr(model.predictor, "trackers", None) or []:
        tracker.reset()


def predict_frame(model, img_path, args, device):
    kwargs = dict(
        source=str(img_path),
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        augment=args.tta,
        retina_masks=True,
        verbose=False,
    )
    if args.track:
        return model.track(conf=min(args.conf, TRACK_DETECTION_CONF), persist=True,
                           tracker=args.tracker, **kwargs)[0]
    return model.predict(conf=args.conf, **kwargs)[0]


def run(model, image_paths, args, geotag_log, default_video, frames_meta):
    results_out = []
    unconfirmed = 0
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    prev_video = None

    for img_path in image_paths:
        meta = frames_meta.get(img_path.name, {})
        video = meta.get("source_video") or default_video
        if args.track and video != prev_video:
            reset_tracker(model)
        prev_video = video

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] unreadable image, skipping: {img_path}")
            continue

        results = predict_frame(model, img_path, args, device)

        overlay = img.copy()
        detections = []

        if results.masks is not None:
            h, w = img.shape[:2]
            track_ids = results.boxes.id
            if args.track and track_ids is None:
                # The tracker confirmed nothing in this frame, in which case
                # Ultralytics returns the raw untracked detections - skip them.
                unconfirmed += sum(float(c) >= args.conf for c in results.boxes.conf)
                track_ids = []
            elif track_ids is None:
                track_ids = [None] * len(results.boxes)

            for mask_t, box, conf, track_id in zip(
                results.masks.data, results.boxes.xyxy, results.boxes.conf, track_ids
            ):
                if float(conf) < args.conf:
                    continue

                mask = mask_t.cpu().numpy().astype("uint8") * 255
                if mask.shape != (h, w):
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
                    timestamp_sec = meta.get("timestamp_sec")
                    if timestamp_sec is not None:
                        lat, lon = interpolate_position(timestamp_sec, geotag_log)
                    # else: no real timestamp known for this frame (no
                    # --frames-meta, or this filename isn't in it) - leave
                    # lat/lon null rather than guessing one.

                detections.append({
                    "frame": img_path.name,
                    "source_video": video,
                    "track_id": int(track_id) if track_id is not None else None,
                    "confidence": float(conf),
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "mask_area_px": area,
                    "image_width": w,
                    "image_height": h,
                    "polygon": polygon,
                    "lat": lat,
                    "lon": lon,
                })

        blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        # Solid outline + confidence label on top of the translucent fill
        for det in detections:
            pts = np.array(det["polygon"], dtype="int32")
            cv2.polylines(blended, [pts], True, (0, 255, 0), 3)
            x1, y1 = int(det["bbox_xyxy"][0]), int(det["bbox_xyxy"][1])
            label, org = f"pothole {det['confidence']:.2f}", (x1 + 4, max(y1 - 8, 30))
            cv2.putText(blended, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(blended, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / img_path.name), blended)
        results_out.extend(detections)

    if args.track:
        print(f"Tracking: skipped {unconfirmed} detection(s) the tracker had not confirmed yet.")
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
    parser.add_argument(
        "--track",
        action="store_true",
        help="Treat --source as video frames (in filename order) and give each detection a "
             "track_id, so 12_build_detections.py can count each pothole once",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config used with --track")
    parser.add_argument("--geotag", default=None, help="Optional CSV: timestamp_sec,lat,lon")
    parser.add_argument(
        "--frames-meta",
        default=None,
        help=(
            "Path to the frames_meta.csv written by 02_extract_frames.py "
            "(or app.py's single-video extraction), mapping each frame "
            "filename to its source video and real timestamp_sec. Required "
            "for --geotag to do anything - without it, lat/lon stay null "
            "even if --geotag is given, since there is no reliable way to "
            "know a frame's real timestamp from its filename or position "
            "alone. With --track it also tells the tracker where one video "
            "ends and the next begins."
        ),
    )
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

    frames_meta = load_frames_meta(args.frames_meta)
    if args.geotag and not frames_meta:
        print(
            "[WARN] --geotag given but no --frames-meta (or it was empty) - "
            "lat/lon will be null for every detection, since there is no "
            "real per-frame timestamp to interpolate against."
        )
    if args.track and not frames_meta:
        print("[WARN] --track without --frames-meta - treating every image in --source as one continuous video.")

    model = YOLO(str(weights_path))

    print(f"Running inference on {len(image_paths)} images "
          f"(conf={args.conf}, iou={args.iou}, tta={args.tta}, track={args.track})")
    detections = run(model, image_paths, args, geotag_log, default_video=source_dir.name, frames_meta=frames_meta)

    out_json = Path(args.output) / "predictions.json"
    with open(out_json, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"\n{len(detections)} pothole instances detected across {len(image_paths)} images.")
    print(f"Overlays saved to: {args.output}")
    print(f"Structured predictions: {out_json}")


if __name__ == "__main__":
    main()
