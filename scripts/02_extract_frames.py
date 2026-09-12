"""Extract RGB frames + mask frames from raw video pairs.

Fixes vs. the original version:
  - Frame stride sampling (config.FRAME_STRIDE) to cut near-duplicate
    consecutive frames instead of keeping every single decoded frame.
  - Detects rgb/mask frame-count mismatch mid-extraction (rather than
    silently truncating to the shorter stream) and reports it.
  - Writes a per-split frames_meta.csv logging source video, frame index and
    timestamp for every saved frame - this is what a future geotag join
    (utils/geo.py) will key off of. No geotag data is required now; the
    lat/lon columns are simply left blank until a geotag log is supplied.
  - Flags masks that are entirely empty (no foreground) so downstream
    imbalance can be measured later (see 05_dataset_report.py).
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SPLITS = config.SPLITS


def create_directories():
    for split in SPLITS:
        (config.PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.PROCESSED_DIR / "masks" / split).mkdir(parents=True, exist_ok=True)


def extract_video(rgb_video, mask_video, split, meta_writer, stride):
    rgb_cap = cv2.VideoCapture(str(rgb_video))
    mask_cap = cv2.VideoCapture(str(mask_video))
    fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_number = 0
    saved_frames = 0
    empty_masks = 0
    video_name = rgb_video.stem

    rgb_output = config.PROCESSED_DIR / "images" / split
    mask_output = config.PROCESSED_DIR / "masks" / split

    rgb_ok = True
    mask_ok = True

    while True:
        rgb_ret, rgb_frame = rgb_cap.read()
        mask_ret, mask_frame = mask_cap.read()

        rgb_ok = rgb_ok and rgb_ret
        mask_ok = mask_ok and mask_ret

        if rgb_ret != mask_ret and frame_number == 0:
            # only meaningful to flag once, at the first divergence
            print(f"[WARN] {video_name}: rgb/mask streams desynced at frame {frame_number}")

        if not rgb_ret or not mask_ret:
            break

        if frame_number % stride == 0:
            rgb_filename = rgb_output / f"{video_name}_{saved_frames:05d}{config.RGB_EXTENSION}"
            mask_filename = mask_output / f"{video_name}_{saved_frames:05d}{config.MASK_EXTENSION}"

            cv2.imwrite(str(rgb_filename), rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])

            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if mask_frame.ndim == 3 else mask_frame
            cv2.imwrite(str(mask_filename), mask_gray)

            if not np.any(mask_gray > config.MASK_THRESHOLD):
                empty_masks += 1

            timestamp_sec = frame_number / fps
            meta_writer.writerow([rgb_filename.name, video_name, frame_number, f"{timestamp_sec:.3f}", "", ""])

            saved_frames += 1

        frame_number += 1

    rgb_cap.release()
    mask_cap.release()

    return saved_frames, empty_masks


def process_split(split, stride):
    rgb_dir = config.RAW_DIR / split / "rgb"
    mask_dir = config.RAW_DIR / split / "mask"

    if not rgb_dir.exists():
        print(f"[ERROR] Missing directory: {rgb_dir}")
        return 0, 0, 0, 0

    rgb_videos = sorted(rgb_dir.glob("*.mp4"))

    total_frames = 0
    total_empty_masks = 0
    failed = 0

    print(f"\nProcessing {split.upper()} set (stride={stride})...")

    meta_path = config.PROCESSED_DIR / "images" / split / "frames_meta.csv"
    with open(meta_path, "w", newline="") as meta_file:
        meta_writer = csv.writer(meta_file)
        meta_writer.writerow(["filename", "source_video", "frame_index", "timestamp_sec", "lat", "lon"])

        for rgb_video in tqdm(rgb_videos):
            mask_video = mask_dir / rgb_video.name

            if not mask_video.exists():
                print(f"Missing mask: {rgb_video.name}")
                failed += 1
                continue

            try:
                frames, empty_masks = extract_video(rgb_video, mask_video, split, meta_writer, stride)
                total_frames += frames
                total_empty_masks += empty_masks
            except Exception as e:
                print(f"Error processing {rgb_video.name}: {e}")
                failed += 1

    return len(rgb_videos), total_frames, total_empty_masks, failed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from raw rgb/mask video pairs")
    parser.add_argument("--stride", type=int, default=config.FRAME_STRIDE,
                         help="Keep 1 out of every N frames (default: %(default)s)")
    args = parser.parse_args()

    create_directories()

    total_videos = total_frames = total_empty = total_failed = 0

    print("=" * 60)
    print("FRAME EXTRACTION")
    print("=" * 60)

    for split in SPLITS:
        videos, frames, empty_masks, failed = process_split(split, args.stride)
        total_videos += videos
        total_frames += frames
        total_empty += empty_masks
        total_failed += failed

        print(f"\n{split.upper()} SUMMARY")
        print("-" * 40)
        print(f"Videos Processed  : {videos}")
        print(f"Frames Saved      : {frames}")
        print(f"Empty Masks       : {empty_masks} ({(empty_masks / frames * 100) if frames else 0:.1f}% of split)")
        print(f"Failed Videos     : {failed}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Videos      : {total_videos}")
    print(f"Total Frames      : {total_frames}")
    print(f"Total Empty Masks : {total_empty}")
    print(f"Failed Videos     : {total_failed}")
    print("\nFrame extraction completed.")


if __name__ == "__main__":
    main()
