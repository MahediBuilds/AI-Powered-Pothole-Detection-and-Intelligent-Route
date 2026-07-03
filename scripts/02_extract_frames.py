from pathlib import Path
import cv2
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = ROOT / "dataset" / "raw"
OUTPUT_DATASET = ROOT / "dataset" / "processed"

SPLITS = ["train", "val", "test"]

SAVE_EVERY_NTH_FRAME = 1

RGB_EXTENSION = ".jpg"
MASK_EXTENSION = ".png"

JPEG_QUALITY = 95

# ==========================================================


def create_directories():
    for split in SPLITS:
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "masks" / split).mkdir(parents=True, exist_ok=True)


def extract_video(rgb_video, mask_video, split):

    rgb_cap = cv2.VideoCapture(str(rgb_video))
    mask_cap = cv2.VideoCapture(str(mask_video))

    frame_number = 0
    saved_frames = 0

    video_name = rgb_video.stem

    rgb_output = OUTPUT_DATASET / "images" / split
    mask_output = OUTPUT_DATASET / "masks" / split

    while True:

        rgb_ret, rgb_frame = rgb_cap.read()
        mask_ret, mask_frame = mask_cap.read()

        if not rgb_ret or not mask_ret:
            break

        if frame_number % SAVE_EVERY_NTH_FRAME == 0:

            rgb_filename = rgb_output / f"{video_name}_{saved_frames:05d}{RGB_EXTENSION}"
            mask_filename = mask_output / f"{video_name}_{saved_frames:05d}{MASK_EXTENSION}"

            cv2.imwrite(
                str(rgb_filename),
                rgb_frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            cv2.imwrite(
                str(mask_filename),
                mask_frame,
            )

            saved_frames += 1

        frame_number += 1

    rgb_cap.release()
    mask_cap.release()

    return saved_frames


def process_split(split):

    rgb_dir = RAW_DATASET / split / "rgb"
    mask_dir = RAW_DATASET / split / "mask"

    rgb_videos = sorted(rgb_dir.glob("*.mp4"))

    total_frames = 0
    failed = 0

    print(f"\nProcessing {split.upper()} set...")

    for rgb_video in tqdm(rgb_videos):

        mask_video = mask_dir / rgb_video.name

        if not mask_video.exists():
            print(f"Missing mask: {rgb_video.name}")
            failed += 1
            continue

        try:
            frames = extract_video(rgb_video, mask_video, split)
            total_frames += frames

        except Exception as e:
            print(f"Error processing {rgb_video.name}: {e}")
            failed += 1

    return len(rgb_videos), total_frames, failed


def main():

    create_directories()

    total_videos = 0
    total_frames = 0
    total_failed = 0

    print("=" * 60)
    print("FRAME EXTRACTION")
    print("=" * 60)

    for split in SPLITS:

        videos, frames, failed = process_split(split)

        total_videos += videos
        total_frames += frames
        total_failed += failed

        print(f"\n{split.upper()} SUMMARY")
        print("-" * 40)
        print(f"Videos Processed : {videos}")
        print(f"Frames Saved     : {frames}")
        print(f"Failed Videos    : {failed}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Total Videos      : {total_videos}")
    print(f"Total Frames      : {total_frames}")
    print(f"Failed Videos     : {total_failed}")

    print("\nFrame extraction completed.")


if __name__ == "__main__":
    main()