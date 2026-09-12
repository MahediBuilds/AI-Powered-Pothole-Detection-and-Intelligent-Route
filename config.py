"""Central configuration for the pothole segmentation pipeline.

All scripts import paths and hyperparameters from here so there is a single
place to change them. Values can be overridden per-run via CLI flags on the
individual scripts (see each script's --help).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "dataset" / "raw"              # raw/{split}/{rgb,mask}/*.mp4
PROCESSED_DIR = ROOT / "dataset" / "processed"  # processed/{images,masks}/{split}/*
FINAL_DIR = ROOT / "dataset" / "final"          # final/{images,labels}/{split}/*  (YOLO-seg ready)

RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"
DATA_YAML = ROOT / "data.yaml"

SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

# Keep 1 out of every N decoded frames. Dashcam video at 24-30fps produces
# near-duplicate consecutive frames; sampling reduces redundancy and the risk
# of visually-identical frames splitting across train/val.
FRAME_STRIDE = 5

RGB_EXTENSION = ".jpg"
MASK_EXTENSION = ".png"
JPEG_QUALITY = 95

# ---------------------------------------------------------------------------
# Mask -> polygon label generation
# ---------------------------------------------------------------------------

CLASS_ID = 0
CLASS_NAMES = {0: "pothole"}

# Masks are decoded from lossy H.264 video, so edges carry compression noise.
# A morphological open+close pass before thresholding removes speckle and
# closes small gaps without eroding real pothole boundaries.
MORPH_KERNEL_SIZE = 5
MASK_THRESHOLD = 127

# Minimum contour area is expressed as a fraction of image area rather than a
# fixed pixel count, so it scales correctly across different source
# resolutions instead of over/under-filtering depending on video size.
MIN_CONTOUR_AREA_FRACTION = 0.0005  # 0.05% of image area
POLYGON_EPSILON_FACTOR = 0.004      # fraction of contour perimeter

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MODEL_WEIGHTS = "yolov8s-seg.pt"
EPOCHS = 150
IMG_SIZE = 640
BATCH = 16
WORKERS = 2
PATIENCE = 30          # early stopping patience (epochs with no val improvement)
OPTIMIZER = "AdamW"
LR0 = 1e-3
WEIGHT_DECAY = 5e-4
COS_LR = True
CLOSE_MOSAIC = 10      # disable mosaic for the last N epochs to stabilize convergence

# Augmentation - tuned for road-surface imagery. Ultralytics applies these
# internally during model.train(); listed explicitly here (rather than left
# as implicit defaults) so they're visible and tunable in one place.
AUGMENTATION = {
    "hsv_h": 0.015,     # slight hue jitter - lighting/camera variation
    "hsv_s": 0.5,       # saturation jitter - wet vs dry road surfaces
    "hsv_v": 0.4,       # brightness jitter - shadows, overcast vs sunny
    "degrees": 10.0,    # small rotations - camera roll, not upside-down roads
    "translate": 0.1,
    "scale": 0.5,       # scale jitter - potholes photographed at varying distance
    "shear": 2.0,
    "perspective": 0.0003,  # mild perspective warp - approximates viewing-angle change
    "flipud": 0.0,      # roads are never upside down
    "fliplr": 0.5,      # left-right flip is valid for road scenes
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,  # helps with rare/small pothole instances
    #"blur": 0.01,       # mild blur - approximates motion blur / low-quality video
}

# ---------------------------------------------------------------------------
# Inference / post-processing
# ---------------------------------------------------------------------------

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_MASK_AREA_PX = 150   # discard predicted instances smaller than this (noise)
MASK_SMOOTH_KERNEL = 5   # morphological smoothing applied to predicted masks

# ---------------------------------------------------------------------------
# Geotagging (future extension point)
# ---------------------------------------------------------------------------
# If a per-video geotag log (CSV: timestamp_sec,lat,lon) is supplied alongside
# a raw video, 02_extract_frames.py records a per-frame timestamp in
# frames_meta.csv and 09_infer_seg.py can join detections to interpolated
# GPS coordinates via utils/geo.py. Until geotag logs are available, the
# lat/lon fields in inference output are simply left null.
GEOTAG_ENABLED = False
