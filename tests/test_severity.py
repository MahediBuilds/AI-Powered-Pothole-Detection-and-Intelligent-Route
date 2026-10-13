import pytest


def detection(area_fraction, width, height, **extra):
    """A compact (not irregular, not elongated) square detection."""
    side = 100
    return {
        "mask_area_px": round(area_fraction * width * height),
        "image_width": width,
        "image_height": height,
        "bbox_xyxy": [0, 0, side, side],
        "polygon": [[0, 0], [side, 0], [side, side], [0, side]],
        **extra,
    }


@pytest.mark.parametrize("width,height", [(1080, 1080), (1280, 720), (1920, 1080), (3840, 2160)])
@pytest.mark.parametrize("fraction,expected", [(0.10, "Mild"), (0.35, "Moderate"), (0.60, "Severe")])
def test_severity_does_not_depend_on_resolution(severity, width, height, fraction, expected):
    [det] = severity.process([detection(fraction, width, height)])
    assert det["severity"] == expected
    assert det["area_fraction"] == pytest.approx(fraction, abs=1e-4)


def test_pixel_thresholds_preserved_on_1080_frames(severity):
    # The original thresholds were 308,000 / 520,000 px on 1080x1080 frames.
    assert severity.bucket_area(307_999 / (1080 * 1080)) == "small"
    assert severity.bucket_area(308_000 / (1080 * 1080)) == "medium"
    assert severity.bucket_area(520_000 / (1080 * 1080)) == "medium"
    assert severity.bucket_area(520_001 / (1080 * 1080)) == "large"


def test_missing_frame_size_falls_back_to_legacy_resolution(severity):
    det = {"mask_area_px": 1080 * 1080 // 2}
    fraction, used_legacy = severity.compute_area_fraction(det)
    assert used_legacy
    assert fraction == pytest.approx(0.5, abs=1e-6)


def test_process_keeps_pixel_area_and_adds_features(severity):
    [det] = severity.process([detection(0.1, 1920, 1080)])
    assert det["pixel_area"] == det["mask_area_px"]
    assert det["aspect_ratio"] == pytest.approx(1.0)
    assert det["circularity"] == pytest.approx(16.0)  # square: (4s)^2 / s^2


@pytest.mark.parametrize("bucket,irregular,expected", [
    ("small", False, "Mild"),
    ("small", True, "Moderate"),
    ("medium", False, "Moderate"),
    ("medium", True, "Severe"),
    ("large", False, "Severe"),
])
def test_classify_rules(severity, bucket, irregular, expected):
    assert severity.classify(bucket, irregular) == expected


def test_elongated_counts_as_irregular(severity):
    det = detection(0.35, 1920, 1080)
    det["bbox_xyxy"] = [0, 0, 300, 100]  # aspect ratio 3 > 2.2
    [det] = severity.process([det])
    assert det["severity"] == "Severe"
