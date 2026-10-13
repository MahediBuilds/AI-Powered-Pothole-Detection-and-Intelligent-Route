import cv2
import numpy as np
import pytest

from utils.frames import FRAMES_META_NAME, extract_frames, load_frames_meta


@pytest.fixture
def video(tmp_path):
    """12-frame, 10 fps synthetic clip."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    return path


def test_extract_frames_applies_stride_and_writes_meta(tmp_path, video):
    seen = []
    saved, fps, total = extract_frames(video, tmp_path / "frames", stride=5,
                                       on_frame=lambda idx, n: seen.append(idx))

    assert fps == pytest.approx(10.0)
    assert [p.name for p in saved] == ["frame_000000.jpg", "frame_000005.jpg", "frame_000010.jpg"]
    assert all(p.exists() for p in saved)
    assert seen == list(range(12))

    meta = load_frames_meta(tmp_path / "frames" / FRAMES_META_NAME)
    assert meta == {
        "frame_000000.jpg": {"source_video": "clip", "timestamp_sec": 0.0},
        "frame_000005.jpg": {"source_video": "clip", "timestamp_sec": 0.5},
        "frame_000010.jpg": {"source_video": "clip", "timestamp_sec": 1.0},
    }


def test_extract_frames_unreadable_video(tmp_path):
    with pytest.raises(RuntimeError):
        extract_frames(tmp_path / "missing.mp4", tmp_path / "frames", stride=1)


def test_load_frames_meta_missing_inputs_return_empty(tmp_path):
    assert load_frames_meta(None) == {}
    assert load_frames_meta(tmp_path / "missing.csv") == {}


def test_load_frames_meta_keeps_rows_with_bad_timestamps(tmp_path):
    path = tmp_path / FRAMES_META_NAME
    path.write_text(
        "filename,source_video,frame_index,timestamp_sec,lat,lon\n"
        "a.jpg,vid1,0,0.000,,\n"
        "b.jpg,vid1,5,oops,,\n"
        ",vid1,10,1.000,,\n"
    )
    assert load_frames_meta(path) == {
        "a.jpg": {"source_video": "vid1", "timestamp_sec": 0.0},
        "b.jpg": {"source_video": "vid1", "timestamp_sec": None},
    }
