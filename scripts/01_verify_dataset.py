from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "raw"

SPLITS = ["train", "val", "test"]


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    cap.release()
    return info


def verify_split(split):

    rgb_dir = DATASET / split / "rgb"
    mask_dir = DATASET / split / "mask"

    rgb_files = sorted(rgb_dir.glob("*.mp4"))

    print(f"\n{'='*60}")
    print(f"{split.upper()} SET")
    print(f"{'='*60}")

    total = 0
    passed = 0
    failed = 0

    for rgb_video in rgb_files:

        total += 1

        mask_video = mask_dir / rgb_video.name

        if not mask_video.exists():
            print(f"[ERROR] Missing mask: {rgb_video.name}")
            failed += 1
            continue

        rgb_info = get_video_info(rgb_video)
        mask_info = get_video_info(mask_video)

        if rgb_info is None:
            print(f"[ERROR] Cannot open RGB video: {rgb_video.name}")
            failed += 1
            continue

        if mask_info is None:
            print(f"[ERROR] Cannot open MASK video: {mask_video.name}")
            failed += 1
            continue

        errors = []

        if abs(rgb_info["fps"] - mask_info["fps"]) > 0.01:
            errors.append("FPS mismatch")

        if rgb_info["width"] != mask_info["width"]:
            errors.append("Width mismatch")

        if rgb_info["height"] != mask_info["height"]:
            errors.append("Height mismatch")

        if rgb_info["frames"] != mask_info["frames"]:
            errors.append("Frame count mismatch")

        if errors:
            failed += 1
            print(f"[ERROR] {rgb_video.name}")
            for e in errors:
                print(f"        - {e}")
        else:
            passed += 1

    print("\nSummary")
    print("-"*40)
    print(f"Videos Checked : {total}")
    print(f"Passed         : {passed}")
    print(f"Failed         : {failed}")

    return total, passed, failed


def main():

    total = 0
    passed = 0
    failed = 0

    for split in SPLITS:
        t, p, f = verify_split(split)

        total += t
        passed += p
        failed += f

    print("\n")
    print("="*60)
    print("FINAL REPORT")
    print("="*60)
    print(f"Total Videos : {total}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")

    if failed == 0:
        print("\nDataset verification successful.")
    else:
        print("\nDataset verification completed with errors.")


if __name__ == "__main__":
    main()