from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import os
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "dataset" / "final" / "images"
LABELS_DIR = ROOT / "dataset" / "final" / "labels"
OUTPUT_DIR = ROOT / "results" / "segmentation_visualization"

SPLITS = ["train", "val", "test"]

TOTAL_SAMPLES = 100
MAX_WORKERS = os.cpu_count() or 1

COLOR = (0,255,0)
ALPHA = 0.35
THICKNESS = 2

def build_tasks():
    tasks=[]
    per_split=max(1,TOTAL_SAMPLES//len(SPLITS))
    random.seed(42)
    for split in SPLITS:
        imgs=list((IMAGES_DIR/split).glob("*.jpg"))
        random.shuffle(imgs)
        for img in imgs[:min(per_split,len(imgs))]:
            lbl=LABELS_DIR/split/(img.stem+".txt")
            out=OUTPUT_DIR/split/img.name
            tasks.append((img,lbl,out))
    return tasks

def process(task):
    img_path,lbl_path,out_path=task
    out_path.parent.mkdir(parents=True,exist_ok=True)
    img=cv2.imread(str(img_path))
    if img is None:
        return 0
    h,w=img.shape[:2]
    overlay=img.copy()
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                vals=line.strip().split()
                if len(vals)<7:
                    continue
                pts=[]
                coords=list(map(float,vals[1:]))
                for i in range(0,len(coords),2):
                    pts.append([int(coords[i]*w),int(coords[i+1]*h)])
                pts=cv2.convexHull(
                    __import__("numpy").array(pts,dtype="int32")
                )
                cv2.fillPoly(overlay,[pts],COLOR)
                cv2.polylines(img,[pts],True,COLOR,THICKNESS)
    img=cv2.addWeighted(overlay,ALPHA,img,1-ALPHA,0)
    cv2.imwrite(str(out_path),img)
    return 1

def main():
    tasks=build_tasks()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("="*60)
    print("SEGMENTATION VISUALIZATION")
    print("="*60)
    print(f"Images: {len(tasks)}")
    print(f"Workers: {MAX_WORKERS}")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(tqdm(ex.map(process,tasks),total=len(tasks),desc="Visualizing"))
    print("\nDone.")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
