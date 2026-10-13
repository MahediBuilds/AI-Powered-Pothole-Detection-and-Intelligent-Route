"""Cascading / incremental fine-tuning of the trained base model on a new
local road video, WITHOUT touching the original model weights.

This is a separate, standalone script. It does not modify config.py,
utils/geo.py, best.pt, or any other existing script or model file.

Pipeline:
  1. Extract frames from the local video (every Nth frame, default 5th -
    same default as config.FRAME_STRIDE, used independently here).
  2. Split the frames by time: the last --val-fraction of the video (default
     20%) is held out as one contiguous validation block. Neighbouring frames
     are near-duplicates, so a random split would leak training frames into
     validation; an earlier version validated on the training frames
     themselves, which made best.pt selection meaningless.
  3. Run the base model on those frames to auto-generate pseudo-labels:
     any prediction with confidence >= --conf-threshold (default 0.6)
     becomes a YOLO-seg training label. No manual review step (by design,
     per project decision) - these pseudo-labels are treated as ground
     truth for fine-tuning.
  4. Mix in a random sample of the original labelled training images
     (--replay-ratio x the number of local training frames, default 1:1) so
     the model keeps seeing the data it was trained on and does not forget
     it while adapting to the new footage.
  5. Fine-tune starting from the base .pt, with the backbone frozen
     (Ultralytics' built-in `freeze` argument - the documented, supported
     way to freeze layers in YOLOv8, rather than a hand-rolled training
     loop) so only the head adapts to the new footage.
  6. Save the result as a NEW versioned file under models/, e.g.
     models/pothole_local_v1.pt. The base .pt is never overwritten, and
     each run auto-increments the version if v1 already exists.

The frames, labels and replay copies under --work-dir are cleared at the
start of every run, so frames from a previous video never leak into this one.

Usage example:
  python scripts/13_finetune_local.py \\
      --base-model runs/pothole_seg/weights/best.pt \\
      --video path/to/local_road.mp4
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.frames import extract_frames

# ---------------------------------------------------------------------------
# Backbone/head split for YOLOv8-seg: layers 0-9 are the backbone
# (Ultralytics' own convention - see yolov8-seg.yaml module list), the rest
# is neck + detection/segmentation head. freeze=10 freezes indices 0..9.
# ---------------------------------------------------------------------------
BACKBONE_FREEZE_LAYERS = 10

WORK_DIR = config.ROOT / "local_finetune"          # scratch space for this script only
MODELS_DIR = config.ROOT / "models"                 # versioned fine-tuned outputs
REPLAY_SOURCE = config.FINAL_DIR / "images" / "train"  # original labelled training images


def split_train_val(frame_paths, val_fraction):
    """Split time-ordered frames into (train, val), holding out the last
    `val_fraction` of the video as one contiguous block."""
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames to make a train/val split, got {len(frame_paths)}")
    n_val = min(max(1, round(len(frame_paths) * val_fraction)), len(frame_paths) - 1)
    return frame_paths[:-n_val], frame_paths[-n_val:]


def sample_replay_images(images_dir, count, seed=0):
    """Reproducible random sample of up to `count` images from `images_dir`."""
    images = sorted(Path(images_dir).glob("*.jpg"))
    return random.Random(seed).sample(images, min(count, len(images)))


def label_path_for(image_path):
    """YOLO convention: <root>/images/<split>/x.jpg -> <root>/labels/<split>/x.txt"""
    return image_path.parent.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def generate_pseudo_labels(model, frame_paths, labels_dir, conf_threshold):
    """Run the base model on extracted frames; keep predictions above
    conf_threshold and write them out as YOLO-seg label files
    (class_id x1 y1 x2 y2 ... normalized polygon points)."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    kept, dropped, empty = 0, 0, 0

    for frame_path in tqdm(frame_paths, desc=f"Pseudo-labelling {labels_dir.name}"):
        if cv2.imread(str(frame_path)) is None:
            continue

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

    print(f"Pseudo-labels ({labels_dir.name}): {kept} kept (conf >= {conf_threshold}), "
          f"{dropped} dropped (below threshold), {empty} frames with no label")
    return kept


def add_replay_data(replay_source, count, images_dir, labels_dir, seed):
    """Copy a sample of original training images + labels into the fine-tune
    train split (prefixed replay_ so they never collide with frame names)."""
    if count <= 0:
        return 0
    if not Path(replay_source).is_dir():
        print(f"[WARN] Replay source {replay_source} not found - fine-tuning on local frames only, "
              "so the model may forget the original training data.")
        return 0

    copied = 0
    for img_path in sample_replay_images(replay_source, count, seed):
        shutil.copy(img_path, images_dir / f"replay_{img_path.name}")
        src_label = label_path_for(img_path)
        dst_label = labels_dir / f"replay_{img_path.stem}.txt"
        if src_label.exists():
            shutil.copy(src_label, dst_label)
        else:
            dst_label.write_text("")
        copied += 1
    print(f"Replay: mixed in {copied} original training image(s) from {replay_source}")
    return copied


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
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of the video, taken from its end, held out for validation")
    parser.add_argument("--replay-ratio", type=float, default=1.0,
                        help="Original training images to mix in, per local training frame (0 disables)")
    parser.add_argument("--replay-source", default=str(REPLAY_SOURCE),
                        help="Directory of original labelled training images to sample replay data from")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the replay sample")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--work-dir", default=str(WORK_DIR), help="Scratch dir for extracted frames + pseudo-labels")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Where the new versioned .pt is saved")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")

    from ultralytics import YOLO  # deferred: only needed once we actually run

    base_model_path = Path(args.base_model)
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir = Path(args.work_dir)
    frames_dir = work_dir / "frames"
    for stale in (frames_dir, work_dir / "images", work_dir / "labels"):
        if stale.exists():
            shutil.rmtree(stale)

    print(f"Base model (untouched): {base_model_path}")
    print(f"Fine-tune settings: epochs={args.epochs}, lr={args.lr}, batch={args.batch}, "
          f"frame_stride={args.frame_stride}, conf_threshold={args.conf_threshold}, "
          f"val_fraction={args.val_fraction}, replay_ratio={args.replay_ratio}")

    # 1. Extract frames
    with tqdm(desc="Extracting frames", unit="frame") as pbar:
        def on_frame(frame_idx, total):
            pbar.total = total
            pbar.update(1)

        frame_paths, fps, total = extract_frames(
            video_path, frames_dir, args.frame_stride, config.JPEG_QUALITY, on_frame=on_frame
        )
    print(f"Video: {video_path.name} | fps={fps:.2f} | total_frames={total} | stride={args.frame_stride}")
    print(f"Extracted {len(frame_paths)} frames to {frames_dir}")
    if not frame_paths:
        raise RuntimeError("No frames extracted - check the video file and --frame-stride")

    # 2. Hold out the end of the video for validation
    train_frames, val_frames = split_train_val(frame_paths, args.val_fraction)
    splits = {}
    for split, paths in (("train", train_frames), ("val", val_frames)):
        images_dir = work_dir / "images" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = [Path(shutil.move(str(p), str(images_dir / p.name))) for p in paths]
    print(f"Split: {len(train_frames)} train frames, {len(val_frames)} val frames (last "
          f"{args.val_fraction:.0%} of the video)")

    # 3. Auto-generate pseudo-labels with the base model
    print("Loading base model for pseudo-labeling...")
    base_model = YOLO(str(base_model_path))
    kept = generate_pseudo_labels(base_model, splits["train"], work_dir / "labels" / "train", args.conf_threshold)
    if kept == 0:
        raise RuntimeError(
            f"No pseudo-labels survived the confidence threshold ({args.conf_threshold}) in the training "
            "frames. Lower --conf-threshold or check the video actually contains potholes the base model detects."
        )
    val_kept = generate_pseudo_labels(base_model, splits["val"], work_dir / "labels" / "val", args.conf_threshold)
    if val_kept == 0:
        print("[WARN] The held-out validation frames have no pseudo-labels, so validation mAP will be 0 "
              "and best.pt selection will not be informative. Use a longer video or a larger --val-fraction.")

    # 4. Mix in original training data
    add_replay_data(
        args.replay_source,
        round(args.replay_ratio * len(train_frames)),
        work_dir / "images" / "train",
        work_dir / "labels" / "train",
        args.seed,
    )

    # Write a throwaway data.yaml for this fine-tune run
    local_yaml = work_dir / "local_data.yaml"
    with open(local_yaml, "w") as f:
        yaml.safe_dump({
            "path": str(work_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": config.CLASS_NAMES,
        }, f, sort_keys=False)

    # 5. Fine-tune with the backbone frozen
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

    # 6. Save as a new versioned file - never overwrite the base model
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
