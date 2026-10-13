"""FastAPI backend for the Pothole Detection UI (Part 4).

This is a thin orchestration layer only - it does not reimplement any
detection/severity/geotag logic. It saves uploaded files, then runs the
existing pipeline as subprocesses in a background thread:

  frame extraction (utils/frames.py, single-video, no mask stream)
        -> scripts/09_infer_seg.py --track
        -> scripts/11_classify_severity.py
        -> scripts/12_build_detections.py

...and streams progress/log lines back to the browser via polling.

Run with:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000 in a browser.

All paths are variables at the top so they're easy to change.
"""

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from utils.frames import FRAMES_META_NAME, extract_frames

# ---------------------------------------------------------------------------
# Paths - change these if your layout differs
# ---------------------------------------------------------------------------
ROOT = config.ROOT
UI_DIR = ROOT / "ui"
UI_INDEX = UI_DIR / "index.html"
UPLOADS_DIR = ROOT / "uploads"
MODELS_DIR = ROOT / "models"
DETECTIONS_JSON = ROOT / "results" / "detections.json"
SCRIPTS_DIR = ROOT / "scripts"

DEFAULT_FRAME_STRIDE = 5
DEFAULT_CONF_THRESHOLD = config.CONF_THRESHOLD  # 0.35 - match the pipeline's validated inference threshold (was 0.6, apparently copied from script 13's unrelated pseudo-labeling threshold)

UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Pothole Detection UI")

# In-memory job store: {job_id: {"status", "log": [...], "summary": {...}}}
JOBS = {}


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------
@app.get("/")
def serve_index():
    return FileResponse(str(UI_INDEX))


# ---------------------------------------------------------------------------
# Models list (base model + anything in models/ + any runs/*/weights/best.pt)
# ---------------------------------------------------------------------------
@app.get("/api/models")
def list_models():
    # Pothole-trained models first so the dropdown defaults to one. The root
    # *.pt files are untrained COCO base weights (yolo26n.pt is detect-only,
    # no masks) and will not find potholes - listed last for completeness.
    found = []

    for p in sorted(ROOT.glob("runs/*/weights/best.pt")):
        found.append({"label": f"trained: {p.parent.parent.name}", "path": str(p)})

    for p in sorted(MODELS_DIR.glob("*.pt")):
        found.append({"label": f"fine-tuned: {p.name}", "path": str(p)})

    for p in sorted(ROOT.glob("*.pt")):
        found.append({"label": f"base (COCO, not pothole-trained): {p.name}", "path": str(p)})

    return {"models": found}


# ---------------------------------------------------------------------------
# Existing detections.json
# ---------------------------------------------------------------------------
@app.get("/api/detections")
def get_detections():
    if not DETECTIONS_JSON.exists():
        return JSONResponse({"detections": [], "exists": False})
    with open(DETECTIONS_JSON) as f:
        data = json.load(f)
    return {"detections": data, "exists": True}


# ---------------------------------------------------------------------------
# Video processing job
# ---------------------------------------------------------------------------
def log(job_id, message):
    print(f"[{job_id}] {message}")
    JOBS[job_id]["log"].append(message)


def extract_frames_single_video(video_path, out_dir, stride, job_id):
    """Single-video, no-mask frame extraction (this UI's video has no
    paired mask stream, unlike the training pipeline's 02_extract_frames.py).
    Writes a frames_meta.csv with the same columns 12_build_detections.py
    already knows how to read."""
    def on_frame(frame_idx, total):
        JOBS[job_id]["progress"] = min(30, int(30 * frame_idx / max(total, 1)))

    saved, fps, total = extract_frames(video_path, out_dir, stride, config.JPEG_QUALITY, on_frame=on_frame)
    log(job_id, f"Video opened: fps={fps:.2f}, total_frames={total}, stride={stride}")
    log(job_id, f"Extracted {len(saved)} frames -> {out_dir}")
    return len(saved)


def run_subprocess(cmd, job_id):
    log(job_id, "Running: " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        log(job_id, line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}")


def process_job(job_id, video_path, gps_path, model_path, frame_stride, conf_threshold, append):
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["progress"] = 0

        job_dir = UPLOADS_DIR / job_id
        frames_dir = job_dir / "frames"
        inference_dir = job_dir / "inference"

        log(job_id, "Step 1/4: Extracting frames...")
        extract_frames_single_video(Path(video_path), frames_dir, frame_stride, job_id)
        JOBS[job_id]["progress"] = 30

        log(job_id, "Step 2/4: Running inference...")
        predictions_path = inference_dir / "predictions.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "09_infer_seg.py"),
            "--weights", str(model_path),
            "--source", str(frames_dir),
            "--output", str(inference_dir),
            "--conf", str(conf_threshold),
            "--track",
            "--frames-meta", str(frames_dir / FRAMES_META_NAME),
        ]
        if gps_path:
            cmd += ["--geotag", str(gps_path)]
        run_subprocess(cmd, job_id)
        JOBS[job_id]["progress"] = 60

        log(job_id, "Step 3/4: Classifying severity...")
        severity_path = job_dir / "predictions_severity.json"
        run_subprocess([
            sys.executable, str(SCRIPTS_DIR / "11_classify_severity.py"),
            "--input", str(predictions_path),
            "--output", str(severity_path),
        ], job_id)
        JOBS[job_id]["progress"] = 80

        log(job_id, "Step 4/4: Building detections.json...")
        build_cmd = [
            sys.executable, str(SCRIPTS_DIR / "12_build_detections.py"),
            "--predictions", str(severity_path),
            "--frames-meta", str(frames_dir / FRAMES_META_NAME),
            "--output", str(DETECTIONS_JSON),
            "--image-url-prefix", f"/uploads/{job_id}/inference/",
        ]
        if gps_path:
            build_cmd += ["--geotag", str(gps_path)]
        if append:
            build_cmd += ["--append"]
        run_subprocess(build_cmd, job_id)
        JOBS[job_id]["progress"] = 100

        with open(DETECTIONS_JSON) as f:
            all_detections = json.load(f)
        counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for d in all_detections:
            if d.get("severity") in counts:
                counts[d["severity"]] += 1

        JOBS[job_id]["summary"] = {
            "total_detections": len(all_detections),
            "breakdown": counts,
        }
        JOBS[job_id]["status"] = "done"
        log(job_id, f"Done. {len(all_detections)} total detections in detections.json")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        log(job_id, f"ERROR: {e}")


@app.post("/api/process")
async def start_processing(
    video: UploadFile = File(...),
    gps: UploadFile = File(None),
    model_path: str = Form(...),
    frame_stride: int = Form(DEFAULT_FRAME_STRIDE),
    conf_threshold: float = Form(DEFAULT_CONF_THRESHOLD),
    append: bool = Form(False),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / video.filename
    with open(video_path, "wb") as f:
        f.write(await video.read())

    gps_path = None
    if gps is not None and gps.filename:
        gps_path = job_dir / gps.filename
        with open(gps_path, "wb") as f:
            f.write(await gps.read())

    JOBS[job_id] = {"status": "queued", "progress": 0, "log": [], "summary": None}

    thread = threading.Thread(
        target=process_job,
        args=(job_id, video_path, gps_path, model_path, frame_stride, conf_threshold, append),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/process/{job_id}/status")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job_id"}, status_code=404)
    return job


# Serve the ui/ folder (in case index.html references other static assets later)
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Serve job outputs so the UI can display annotated overlay frames
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
