"""End-to-end checks for scripts/09_infer_seg.py with the trained model.

Skipped when the trained weights are not present (they are not committed -
see "Model weights" in README.md).
"""

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from conftest import ROOT

WEIGHTS = ROOT / "runs" / "pothole_seg" / "weights" / "best.pt"
TEST_IMAGE = ROOT / "test_images" / "0182_00007.jpg"

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason=f"trained weights not found at {WEIGHTS}")


def run_inference(source, output, *extra):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "09_infer_seg.py"),
         "--weights", str(WEIGHTS), "--source", str(source), "--output", str(output), *extra],
        check=True, capture_output=True, text=True,
    )
    return json.loads((output / "predictions.json").read_text())


def widescreen_frame():
    """The square test image centred on a 1920x1080 canvas, like a dashcam frame."""
    canvas = np.full((1080, 1920, 3), 90, np.uint8)
    canvas[:, 420:1500] = cv2.imread(str(TEST_IMAGE))
    return canvas


def test_masks_line_up_with_boxes_on_non_square_frames(tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    cv2.imwrite(str(source / "wide.jpg"), widescreen_frame())

    detections = run_inference(source, tmp_path / "out")

    assert detections, "expected at least one pothole in the test image"
    for det in detections:
        assert (det["image_width"], det["image_height"]) == (1920, 1080)
        x1, y1, x2, y2 = det["bbox_xyxy"]
        xs = [p[0] for p in det["polygon"]]
        ys = [p[1] for p in det["polygon"]]
        # Before the fix the mask was stretched from the letterboxed model
        # input and sat ~30 px off its own box on this frame.
        tolerance = 10
        assert abs(min(xs) - x1) <= tolerance and abs(max(xs) - x2) <= tolerance
        assert abs(min(ys) - y1) <= tolerance and abs(max(ys) - y2) <= tolerance


def test_tracking_assigns_one_track_per_pothole_across_frames(tmp_path, build_detections):
    source = tmp_path / "frames"
    source.mkdir()
    frame = widescreen_frame()
    rows = ["filename,source_video,frame_index,timestamp_sec,lat,lon"]
    for i in range(4):
        name = f"frame_{i * 5:06d}.jpg"
        cv2.imwrite(str(source / name), frame)
        rows.append(f"{name},clip,{i * 5},{i * 5 / 30:.3f},,")
    (source / "frames_meta.csv").write_text("\n".join(rows) + "\n")

    detections = run_inference(source, tmp_path / "out", "--track",
                               "--frames-meta", str(source / "frames_meta.csv"))

    per_frame = {}
    for det in detections:
        assert det["track_id"] is not None and det["source_video"] == "clip"
        per_frame.setdefault(det["frame"], set()).add(det["track_id"])
    assert len(per_frame) == 4
    track_sets = list(per_frame.values())
    assert all(ids == track_sets[0] for ids in track_sets)

    potholes = build_detections.collapse_tracks(detections)
    assert len(potholes) == len(track_sets[0])
    assert all(p["frames_seen"] == 4 for p in potholes)
