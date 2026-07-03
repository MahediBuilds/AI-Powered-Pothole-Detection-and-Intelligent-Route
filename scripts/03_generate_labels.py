from pathlib import Path
import cv2
import shutil
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / "dataset" / "processed"
FINAL = ROOT / "dataset" / "final"

SPLITS = ["train", "val", "test"]

CLASS_ID = 0

MIN_CONTOUR_AREA = 100

# ==========================================================


def create_directories():

    for split in SPLITS:

        (FINAL / "images" / split).mkdir(parents=True, exist_ok=True)
        (FINAL / "labels" / split).mkdir(parents=True, exist_ok=True)


def mask_to_yolo(mask_path, label_path):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return 0

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = mask.shape

    count = 0

    with open(label_path, "w") as f:

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_CONTOUR_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h

            bw /= w
            bh /= h

            f.write(
                f"{CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            )

            count += 1

    return count


def process_split(split):

    image_dir = PROCESSED / "images" / split
    mask_dir = PROCESSED / "masks" / split

    final_image_dir = FINAL / "images" / split
    final_label_dir = FINAL / "labels" / split

    masks = sorted(mask_dir.glob("*.png"))

    images_processed = 0
    potholes = 0

    for mask in tqdm(masks, desc=split.upper()):

        image = image_dir / (mask.stem + ".jpg")

        if not image.exists():
            continue

        shutil.copy2(
            image,
            final_image_dir / image.name,
        )

        label = final_label_dir / (mask.stem + ".txt")

        potholes += mask_to_yolo(mask, label)

        images_processed += 1

    return images_processed, potholes


def main():

    create_directories()

    total_images = 0
    total_potholes = 0

    print("=" * 60)
    print("GENERATING YOLO LABELS")
    print("=" * 60)

    for split in SPLITS:

        images, potholes = process_split(split)

        total_images += images
        total_potholes += potholes

        print(f"\n{split.upper()} SUMMARY")
        print("-" * 40)
        print(f"Images : {images}")
        print(f"Potholes : {potholes}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Images Processed : {total_images}")
    print(f"Total Potholes   : {total_potholes}")

    print("\nYOLO labels generated successfully.")


if __name__ == "__main__":
    main()