from pathlib import Path
import random
import cv2
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "dataset" / "final" / "images"
LABELS_DIR = ROOT / "dataset" / "final" / "labels"

OUTPUT_DIR = ROOT / "results" / "visualization"

SPLITS = ["train", "val", "test"]

NUM_IMAGES_PER_SPLIT = 50

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

SHOW_LABEL = True
FONT_SCALE = 0.5

# ==========================================================


def create_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_boxes(image_path, label_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    h, w = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                cls, xc, yc, bw, bh = map(float, values)

                xc *= w
                yc *= h
                bw *= w
                bh *= h

                x1 = int(xc - bw / 2)
                y1 = int(yc - bh / 2)
                x2 = int(xc + bw / 2)
                y2 = int(yc + bh / 2)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    BOX_COLOR,
                    BOX_THICKNESS,
                )

                if SHOW_LABEL:

                    cv2.putText(
                        image,
                        "Pothole",
                        (x1, max(y1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE,
                        BOX_COLOR,
                        2,
                    )

    return image


def process_split(split):

    image_dir = IMAGES_DIR / split
    label_dir = LABELS_DIR / split

    output_split = OUTPUT_DIR / split
    output_split.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))

    if len(images) == 0:
        print(f"No images found in {split}")
        return

    sample_size = min(NUM_IMAGES_PER_SPLIT, len(images))

    sampled = random.sample(images, sample_size)

    for image_path in tqdm(sampled, desc=split.upper()):

        label_path = label_dir / f"{image_path.stem}.txt"

        image = draw_boxes(image_path, label_path)

        if image is None:
            continue

        save_path = output_split / image_path.name

        cv2.imwrite(str(save_path), image)


def main():

    create_directories()

    print("=" * 60)
    print("VISUALIZING YOLO LABELS")
    print("=" * 60)

    random.seed(42)

    for split in SPLITS:

        process_split(split)

    print("\nVisualization completed.")

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()