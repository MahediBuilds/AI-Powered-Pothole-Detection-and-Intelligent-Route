"""Merge severity-classified predictions with real per-frame timestamps and
GPS positions into the final detections.json the UI (Part 4) reads.

Inputs:
  - predictions_severity.json  (from 11_classify_severity.py): confidence,
    bbox_xyxy, mask_area_px, polygon, severity, track_id etc. per detection.
  - frames_meta.csv (from 02_extract_frames.py): filename -> real
    timestamp_sec for every extracted frame. This replaces the fps=30
    guess that 09_infer_seg.py falls back to - if a frame is found here,
    its real timestamp is used instead.
  - a geotag log (CSV: timestamp_sec,lat,lon), OPTIONAL. Without one,
    latitude/longitude are written as null - this is expected until a real
    GPS log exists for a real video.

One pothole per track: when 09_infer_seg.py was run with --track, every
detection carries a track_id, and the same physical pothole shows up in
several consecutive frames under the same (source_video, track_id). Those
are collapsed to a single record - the highest-confidence observation -
with frames_seen recording how many frames it was detected in. Detections
without a track_id (inference run without --track) are kept one per row.

Output: detections.json, one record per pothole:
  { frame, latitude, longitude, severity, confidence, mask_coords, timestamp,
    pixel_area, image_url, source_video, track_id, frames_seen }
image_url is --image-url-prefix + frame (null without a prefix); app.py uses
it to show the annotated frame in the UI.

No GPS log or local video exists yet in this repo, so running this script
right now will produce timestamp=null and latitude/longitude=null for every
detection (since 0182_00007.jpg is a standalone test image, not part of an
extracted-frame video). That's expected - this script becomes fully live
the moment a real video is run through 02_extract_frames.py /
09_infer_seg.py / 11_classify_severity.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.frames import load_frames_meta
from utils.geo import load_geotag_log, interpolate_position


def collapse_tracks(predictions):
    """Keep one detection per (source_video, track_id): the highest-confidence
    one, annotated with frames_seen. Untracked detections pass through
    unchanged (frames_seen=1). Order follows each pothole's first sighting."""
    groups = {}
    for i, det in enumerate(predictions):
        track_id = det.get("track_id")
        key = ("track", det.get("source_video"), track_id) if track_id is not None else ("row", i)
        groups.setdefault(key, []).append(det)

    collapsed = []
    for observations in groups.values():
        best = max(observations, key=lambda d: d.get("confidence") or 0.0)
        collapsed.append({**best, "frames_seen": len(observations)})
    return collapsed


def build_detections(predictions, frames_meta, geotag_log, image_url_prefix=None):
    records = []
    matched_ts = 0
    matched_gps = 0

    potholes = collapse_tracks(predictions)

    for det in potholes:
        frame = det.get("frame")
        timestamp = frames_meta.get(frame, {}).get("timestamp_sec")
        if timestamp is not None:
            matched_ts += 1

        lat, lon = (None, None)
        if timestamp is not None and geotag_log:
            lat, lon = interpolate_position(timestamp, geotag_log)
            if lat is not None:
                matched_gps += 1

        records.append({
            "frame": frame,
            "latitude": lat,
            "longitude": lon,
            "severity": det.get("severity"),
            "confidence": det.get("confidence"),
            "mask_coords": det.get("polygon"),
            "timestamp": timestamp,
            # Extra field beyond the original spec schema - the Part 4 UI's
            # detections table has an "Area (px)" column that needs this.
            "pixel_area": det.get("pixel_area"),
            # URL of the annotated overlay frame, so the UI can show the
            # detection even when there is no GPS to place it on the map.
            "image_url": (image_url_prefix + frame) if (image_url_prefix and frame) else None,
            "source_video": det.get("source_video"),
            "track_id": det.get("track_id"),
            "frames_seen": det["frames_seen"],
        })

    print(f"Built {len(records)} pothole record(s) from {len(predictions)} frame-level detection(s) "
          f"({matched_ts} with a real timestamp, {matched_gps} with GPS)")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Merge severity-classified predictions + frame timestamps + GPS into detections.json"
    )
    parser.add_argument(
        "--predictions",
        default=str(config.RESULTS_DIR / "inference" / "predictions_severity.json"),
        help="Path to predictions_severity.json (output of 11_classify_severity.py)",
    )
    parser.add_argument(
        "--frames-meta",
        default=None,
        help="Path to a frames_meta.csv (written by 02_extract_frames.py) for real per-frame timestamps",
    )
    parser.add_argument(
        "--geotag",
        default=None,
        help="Optional CSV: timestamp_sec,lat,lon - without this, lat/lon are null",
    )
    parser.add_argument(
        "--output",
        default=str(config.RESULTS_DIR / "detections.json"),
        help="Where to write the merged detections.json",
    )
    parser.add_argument(
        "--image-url-prefix",
        default=None,
        help="Optional URL prefix for overlay images (frame filename is appended), used by the UI",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing detections.json instead of overwriting it",
    )
    args = parser.parse_args()

    in_path = Path(args.predictions)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {in_path}\n"
            f"Run scripts/11_classify_severity.py first."
        )

    print(f"Loading severity-classified predictions from {in_path}")
    with open(in_path) as f:
        predictions = json.load(f)

    frames_meta = load_frames_meta(args.frames_meta)
    geotag_log = load_geotag_log(args.geotag)
    if args.geotag and not geotag_log:
        print(f"[WARN] --geotag given but no rows loaded from {args.geotag}")

    new_records = build_detections(predictions, frames_meta, geotag_log, args.image_url_prefix)

    out_path = Path(args.output)
    existing = []
    if args.append and out_path.exists():
        print(f"Appending to existing {out_path}")
        with open(out_path) as f:
            existing = json.load(f)
    elif out_path.exists() and not args.append:
        print(f"[INFO] {out_path} already exists - overwriting (pass --append to add to it instead)")

    all_records = existing + new_records

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"Wrote {len(all_records)} total detection(s) to {out_path}")


if __name__ == "__main__":
    main()
