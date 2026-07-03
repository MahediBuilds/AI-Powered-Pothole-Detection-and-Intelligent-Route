from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import cv2
from tqdm import tqdm
import multiprocessing

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "dataset" / "processed"
FINAL = ROOT / "dataset" / "final"

SPLITS = ["train", "val", "test"]
CLASS_ID = 0
MIN_CONTOUR_AREA = 100
EPSILON_FACTOR = 0.005
MAX_WORKERS = os.cpu_count() or 1

def create_directories():
    for split in SPLITS:
        (FINAL / "images" / split).mkdir(parents=True, exist_ok=True)
        (FINAL / "labels" / split).mkdir(parents=True, exist_ok=True)

def process_image(task):
    image_path, mask_path, output_image, output_label = task
    try:
        if not output_image.exists():
            shutil.copyfile(image_path, output_image)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        objects = 0
        with open(output_label, "w") as f:
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                epsilon = EPSILON_FACTOR * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                pts = []
                for p in approx:
                    x, y = p[0]
                    pts.append(f"{x / w:.6f}")
                    pts.append(f"{y / h:.6f}")
                f.write(f"{CLASS_ID} {' '.join(pts)}\n")
                objects += 1
        return objects
    except Exception as e:
        print(f"Error: {mask_path.name}: {e}")
        return 0

def process_split(split):
    image_dir = PROCESSED / "images" / split
    mask_dir = PROCESSED / "masks" / split
    out_image = FINAL / "images" / split
    out_label = FINAL / "labels" / split
    tasks = []
    for mask in sorted(mask_dir.glob("*.png")):
        image = image_dir / (mask.stem + ".jpg")
        if image.exists():
            tasks.append((image, mask, out_image / image.name, out_label / (mask.stem + ".txt")))
    total = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for c in tqdm(ex.map(process_image, tasks), total=len(tasks), desc=split.upper()):
            total += c
    return len(tasks), total

def main():
    create_directories()
    total_images = total_objects = 0
    print("="*60)
    print("YOLOv8 SEGMENTATION LABEL GENERATION")
    print("="*60)
    print(f"Workers: {MAX_WORKERS}")
    for split in SPLITS:
        imgs, objs = process_split(split)
        total_images += imgs
        total_objects += objs
        print(f"\n{split.upper()} - Images: {imgs} Objects: {objs}")
    print("\n" + "="*60)
    print(f"Images: {total_images}")
    print(f"Objects: {total_objects}")



if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()