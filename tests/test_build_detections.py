def obs(frame, conf, track_id=None, video="road", severity="Mild"):
    return {
        "frame": frame,
        "source_video": video,
        "track_id": track_id,
        "confidence": conf,
        "severity": severity,
        "polygon": [[0, 0], [1, 0], [1, 1]],
        "pixel_area": 100,
    }


def test_collapse_keeps_best_observation_per_track(build_detections):
    predictions = [
        obs("f0.jpg", 0.50, track_id=1, severity="Mild"),
        obs("f1.jpg", 0.90, track_id=1, severity="Moderate"),
        obs("f2.jpg", 0.70, track_id=1, severity="Mild"),
        obs("f1.jpg", 0.60, track_id=2),
    ]
    collapsed = build_detections.collapse_tracks(predictions)

    assert len(collapsed) == 2
    first, second = collapsed
    assert (first["frame"], first["confidence"], first["severity"], first["frames_seen"]) == \
        ("f1.jpg", 0.90, "Moderate", 3)
    assert (second["track_id"], second["frames_seen"]) == (2, 1)


def test_same_track_id_in_different_videos_is_not_merged(build_detections):
    predictions = [obs("f0.jpg", 0.5, track_id=1, video="a"), obs("f0.jpg", 0.5, track_id=1, video="b")]
    assert len(build_detections.collapse_tracks(predictions)) == 2


def test_untracked_detections_pass_through(build_detections):
    predictions = [obs("f0.jpg", 0.5), obs("f0.jpg", 0.6), obs("f1.jpg", 0.7, track_id=None)]
    collapsed = build_detections.collapse_tracks(predictions)
    assert [d["confidence"] for d in collapsed] == [0.5, 0.6, 0.7]
    assert all(d["frames_seen"] == 1 for d in collapsed)


def test_build_detections_uses_representative_frame_for_time_and_gps(build_detections):
    predictions = [obs("f0.jpg", 0.4, track_id=7), obs("f5.jpg", 0.8, track_id=7)]
    frames_meta = {
        "f0.jpg": {"source_video": "road", "timestamp_sec": 0.0},
        "f5.jpg": {"source_video": "road", "timestamp_sec": 1.0},
    }
    geotag_log = [(0.0, 10.0, 20.0), (2.0, 12.0, 22.0)]

    [record] = build_detections.build_detections(predictions, frames_meta, geotag_log)

    assert record["frame"] == "f5.jpg"
    assert record["timestamp"] == 1.0
    assert (record["latitude"], record["longitude"]) == (11.0, 21.0)
    assert (record["track_id"], record["frames_seen"]) == (7, 2)
    assert record["mask_coords"] == [[0, 0], [1, 0], [1, 1]]


def test_build_detections_without_meta_or_gps(build_detections):
    [record] = build_detections.build_detections([obs("x.jpg", 0.5)], {}, [])
    assert (record["timestamp"], record["latitude"], record["longitude"]) == (None, None, None)
    assert record["image_url"] is None


def test_image_url_points_at_the_representative_frame(build_detections):
    predictions = [obs("f0.jpg", 0.4, track_id=3), obs("f5.jpg", 0.8, track_id=3)]
    [record] = build_detections.build_detections(predictions, {}, [], image_url_prefix="/uploads/job1/inference/")
    assert record["image_url"] == "/uploads/job1/inference/f5.jpg"
