"""Shared helpers for extracting frames from a single video and reading the
frames_meta.csv files the pipeline writes alongside extracted frames.

Used by app.py, scripts/09_infer_seg.py, scripts/12_build_detections.py and
scripts/13_finetune_local.py. (scripts/02_extract_frames.py keeps its own
paired rgb+mask extraction loop, since it reads two synchronized streams.)

frames_meta.csv columns:
    filename,source_video,frame_index,timestamp_sec,lat,lon
"""

import csv
from pathlib import Path

import cv2

FRAMES_META_NAME = "frames_meta.csv"
FRAMES_META_HEADER = ["filename", "source_video", "frame_index", "timestamp_sec", "lat", "lon"]


def extract_frames(video_path, out_dir, stride, jpeg_quality=95, on_frame=None):
    """Save every `stride`-th frame of `video_path` to `out_dir` as
    frame_{frame_index:06d}.jpg and write out_dir/frames_meta.csv.

    `on_frame(frame_index, total_frames)` is called once per decoded frame,
    so callers can drive a progress bar.

    Returns (saved_paths, fps, total_frames).
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = []
    frame_idx = 0
    try:
        with open(out_dir / FRAMES_META_NAME, "w", newline="") as meta_file:
            writer = csv.writer(meta_file)
            writer.writerow(FRAMES_META_HEADER)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    # Phone recordings (ui/record.html) are variable frame rate, so
                    # frame_idx / fps drifts from real time; prefer the frame's own
                    # presentation timestamp when the decoder reports one.
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp = pos_ms / 1000 if pos_ms > 0 else frame_idx / fps
                    writer.writerow([out_path.name, video_path.stem, frame_idx, f"{timestamp:.3f}", "", ""])
                    saved.append(out_path)
                if on_frame is not None:
                    on_frame(frame_idx, total)
                frame_idx += 1
    finally:
        cap.release()

    return saved, fps, total


def load_frames_meta(path):
    """Load a frames_meta.csv as {filename: {"source_video", "timestamp_sec"}}.

    timestamp_sec is None for rows where it is missing or malformed. Returns
    an empty dict if `path` is None or does not exist.
    """
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        print(f"[WARN] frames_meta.csv not found: {path}")
        return {}

    lookup = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            filename = row.get("filename")
            if not filename:
                continue
            try:
                timestamp = float(row["timestamp_sec"])
            except (KeyError, TypeError, ValueError):
                timestamp = None
            lookup[filename] = {
                "source_video": row.get("source_video") or None,
                "timestamp_sec": timestamp,
            }
    return lookup
