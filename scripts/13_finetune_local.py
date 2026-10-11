"""Cascading / incremental fine-tuning of the trained base model on a new
local road video, WITHOUT touching the original model weights.

This is a separate, standalone script. It does not modify config.py,
utils/geo.py, best.pt, or any other existing script or model file.

Pipeline:
  1. Extract frames from the local video (every Nth frame, default 5th -
    same default as config.FRAME_STRIDE, used independently here).
  2. Run the base model on those frames to auto-generate pseudo-labels:
     any prediction with confidence >= --conf-threshold (default 0.6)
     becomes a YOLO-seg training label. No manual review step (by design,
     per project decision) - these pseudo-labels are treated as ground
     truth for fine-tuning.
  3. Fine-tune starting from the base .pt, with the backbone frozen
     (Ultralytics' built-in `freeze` argument - the documented, supported
     way to freeze layers in YOLOv8, rather than a hand-rolled training
     loop) so only the head adapts to the new footage.
  4. Save the result as a NEW versioned file under models/, e.g.
     models/pothole_local_v1.pt. The base .pt is never overwritten, and
     each run auto-increments the version if v1 already exists.

Usage example:
  python scripts/13_finetune_local.py \\
      --base-model runs/pothole_seg/weights/best.pt \\
      --video path/to/local_road.mp4
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Backbone/head split for YOLOv8-seg: layers 0-9 are the backbone
# (Ultralytics' own convention - see yolov8-seg.yaml module list), the rest
# is neck + detection/segmentation head. freeze=10 freezes indices 0..9.
# ---------------------------------------------------------------------------
BACKBONE_FREEZE_LAYERS = 10

WORK_DIR = config.ROOT / "local_finetune"          # scratch space for this script only
MODELS_DIR = config.ROOT / "models"                 # versioned fine-tuned outputs


def extract_frames(video_path, out_dir, stride):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name} | fps={fps:.2f} | total_frames={total} | stride={stride}")

    saved = []
    frame_idx = 0
    with tqdm(total=total, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                saved.append(out_path)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Extracted {len(saved)} frames to {out_dir}")
    return saved


def generate_pseudo_labels(model, frame_paths, labels_dir, conf_threshold):
    """Run the base model on extracted frames; keep predictions above
    conf_threshold and write them out as YOLO-seg label files
    (class_id x1 y1 x2 y2 ... normalized polygon points)."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    kept, dropped, empty = 0, 0, 0

    for frame_path in tqdm(frame_paths, desc="Generating pseudo-labels"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(source=str(frame_path), verbose=False)[0]
        label_lines = []

        if results.masks is not None:
            for mask_t, conf in zip(results.masks.xyn, results.boxes.conf):
                conf = float(conf)
                if conf < conf_threshold:
                    dropped += 1
                    continue
                # mask_t is already normalized (x,y) polygon points from Ultralytics
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in mask_t)
                label_lines.append(f"{config.CLASS_ID} {coords}")
                kept += 1

        label_path = labels_dir / f"{frame_path.stem}.txt"
        if label_lines:
            label_path.write_text("\n".join(label_lines) + "\n")
        else:
            empty += 1
            label_path.write_text("")  # empty label = background-only frame, valid for YOLO

    print(f"Pseudo-labels: {kept} kept (conf >= {conf_threshold}), "
          f"{dropped} dropped (below threshold), {empty} frames with no label")
    return kept


def next_version_path(models_dir, stem="pothole_local"):
    models_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while (models_dir / f"{stem}_v{version}.pt").exists():
        version += 1
    return models_dir / f"{stem}_v{version}.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Cascading fine-tune the base model on a new local video (frozen backbone)"
    )
    parser.add_argument("--base-model", required=True, help="Path to the existing trained .pt (never modified)")
    parser.add_argument("--video", required=True, help="Path to the local road video")
    parser.add_argument("--frame-stride", type=int, default=5, help="Keep every Nth frame")
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Min confidence for a pseudo-label")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--work-dir", default=str(WORK_DIR), help="Scratch dir for extracted frames + pseudo-labels")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Where the new versioned .pt is saved")
    args = parser.parse_args()

    from ultralytics import YOLO  # deferred: only needed once we actually run

    base_model_path = Path(args.base_model)
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir = Path(args.work_dir)
    frames_dir = work_dir / "images" / "train"
    labels_dir = work_dir / "labels" / "train"

    print(f"Base model (untouched): {base_model_path}")
    print(f"Fine-tune settings: epochs={args.epochs}, lr={args.lr}, batch={args.batch}, "
          f"frame_stride={args.frame_stride}, conf_threshold={args.conf_threshold}")

    # 1. Extract frames
    frame_paths = extract_frames(video_path, frames_dir, args.frame_stride)
    if not frame_paths:
        raise RuntimeError("No frames extracted - check the video file and --frame-stride")

    # 2. Auto-generate pseudo-labels with the base model
    print("Loading base model for pseudo-labeling...")
    base_model = YOLO(str(base_model_path))
    kept = generate_pseudo_labels(base_model, frame_paths, labels_dir, args.conf_threshold)
    if kept == 0:
        raise RuntimeError(
            f"No pseudo-labels survived the confidence threshold ({args.conf_threshold}). "
            "Lower --conf-threshold or check the video actually contains potholes the base model detects."
        )

    # Ultralytics needs a val split too; reuse the same frames (small local
    # video fine-tune, not a benchmark run - this is a known simplification).
    val_frames_dir = work_dir / "images" / "val"
    val_labels_dir = work_dir / "labels" / "val"
    val_frames_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    for img_path in frame_paths:
        shutil.copy(img_path, val_frames_dir / img_path.name)
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, val_labels_dir / lbl_path.name)

    # Write a throwaway data.yaml for this fine-tune run
    local_yaml = work_dir / "local_data.yaml"
    local_yaml.write_text(
        f"path: {work_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: pothole\n"
    )

    # 3. Fine-tune with the backbone frozen
    print(f"Fine-tuning from {base_model_path} with backbone frozen "
          f"(freeze={BACKBONE_FREEZE_LAYERS} layers)...")
    train_model = YOLO(str(base_model_path))
    train_model.train(
        data=str(local_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        freeze=BACKBONE_FREEZE_LAYERS,
        project=str(work_dir / "runs"),
        name="local_finetune",
        exist_ok=True,
    )

    # 4. Save as a new versioned file - never overwrite the base model
    trained_weights = work_dir / "runs" / "local_finetune" / "weights" / "best.pt"
    if not trained_weights.exists():
        raise FileNotFoundError(f"Expected trained weights not found at {trained_weights}")

    models_dir = Path(args.models_dir)
    out_path = next_version_path(models_dir)
    shutil.copy(trained_weights, out_path)

    print(f"\nDone. Fine-tuned model saved to: {out_path}")
    print(f"Base model left untouched at: {base_model_path}")


if __name__ == "__main__":
    main()
