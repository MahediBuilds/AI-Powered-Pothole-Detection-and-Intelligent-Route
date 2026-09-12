from pathlib import Path
import cv2

ROOT = Path.cwd()

print("=" * 70)
print("PROJECT INTEGRITY CHECK")
print("=" * 70)

required = [
    "scripts",
    "utils",
    "dataset",
    "config.py",
    "requirements.txt",
    "data.yaml"
]

print("\n[1] Required files/folders")
for item in required:
    p = ROOT / item
    print(f"{'✓' if p.exists() else '✗'} {item}")

print("\n[2] Raw Dataset")

for split in ["train", "val", "test"]:
    rgb = ROOT / "dataset" / "raw" / split / "rgb"
    mask = ROOT / "dataset" / "raw" / split / "mask"

    rgb_count = len(list(rgb.glob("*.mp4"))) if rgb.exists() else 0
    mask_count = len(list(mask.glob("*.mp4"))) if mask.exists() else 0

    print(f"{split.upper():5} RGB={rgb_count}  MASK={mask_count}")

print("\n[3] Processed Dataset")

for split in ["train", "val", "test"]:
    img = ROOT / "dataset" / "processed" / "images" / split
    mask = ROOT / "dataset" / "processed" / "masks" / split

    img_count = len(list(img.glob("*.jpg"))) if img.exists() else 0
    mask_count = len(list(mask.glob("*.png"))) if mask.exists() else 0

    print(f"{split.upper():5} Images={img_count}  Masks={mask_count}")

print("\n[4] Final Dataset")

for split in ["train", "val", "test"]:
    img_dir = ROOT / "dataset" / "final" / "images" / split
    lbl_dir = ROOT / "dataset" / "final" / "labels" / split

    images = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    missing = 0

    label_names = {x.stem for x in labels}

    for img in images:
        if img.stem not in label_names:
            missing += 1

    print(
        f"{split.upper():5} Images={len(images)} "
        f"Labels={len(labels)} Missing={missing}"
    )

print("\n[5] Checking sample images")

checked = 0
bad = 0

for img in (ROOT / "dataset" / "final" / "images").rglob("*.jpg"):
    im = cv2.imread(str(img))
    checked += 1

    if im is None:
        bad += 1

    if checked == 20:
        break

print(f"Checked {checked} sample images")
print(f"Corrupted images: {bad}")

print("\n[6] Models")

for model in ["yolov8s-seg.pt", "yolov8m-seg.pt"]:
    p = ROOT / model
    print(f"{'✓' if p.exists() else '✗'} {model}")

print("\n[7] data.yaml")

yaml = ROOT / "data.yaml"

if yaml.exists():
    print("✓ data.yaml found")
    print(yaml.read_text())
else:
    print("✗ data.yaml missing")

print("\n" + "=" * 70)
print("Integrity check complete.")
print("=" * 70)