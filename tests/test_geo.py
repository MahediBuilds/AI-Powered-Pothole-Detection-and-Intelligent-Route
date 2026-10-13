import pytest

from utils.geo import interpolate_position, load_geotag_log


def write_csv(path, text):
    path.write_text(text)
    return path


def test_load_geotag_log_missing_inputs_return_empty(tmp_path):
    assert load_geotag_log(None) == []
    assert load_geotag_log(tmp_path / "missing.csv") == []


def test_load_geotag_log_sorts_and_skips_malformed_rows(tmp_path):
    path = write_csv(tmp_path / "gps.csv", (
        "timestamp_sec,lat,lon\n"
        "2.0,12.2,77.2\n"
        "not-a-number,12.0,77.0\n"
        "0.0,12.0,77.0\n"
        "1.0,,77.1\n"
    ))
    assert load_geotag_log(path) == [(0.0, 12.0, 77.0), (2.0, 12.2, 77.2)]


def test_load_geotag_log_reads_android_recorder_csv(tmp_path):
    # Format written by android/ (GeoTrack.toCsv): extra columns, blank optional
    # values, and a pre-roll fix with a negative timestamp.
    path = write_csv(tmp_path / "road_20261013_143005_gps.csv", (
        "timestamp_sec,lat,lon,accuracy_m,speed_mps,bearing_deg,altitude_m,utc_ms\n"
        "-0.500,12.9716000,77.5946000,4.0,,,,1760345405000\n"
        "0.750,12.9717000,77.5947000,3.5,8.20,91.0,920.4,1760345406250\n"
    ))
    log = load_geotag_log(path)
    assert log == [(-0.5, 12.9716, 77.5946), (0.75, 12.9717, 77.5947)]
    lat, lon = interpolate_position(0.0, log)
    assert lat == pytest.approx(12.9716 + 0.0001 * 0.4)
    assert lon == pytest.approx(77.5946 + 0.0001 * 0.4)


def test_interpolate_position_empty_log():
    assert interpolate_position(1.0, []) == (None, None)


def test_interpolate_position_midpoint():
    log = [(0.0, 10.0, 20.0), (2.0, 12.0, 24.0)]
    lat, lon = interpolate_position(0.5, log)
    assert lat == pytest.approx(10.5)
    assert lon == pytest.approx(21.0)


def test_interpolate_position_clamps_outside_range():
    log = [(1.0, 10.0, 20.0), (2.0, 12.0, 24.0)]
    assert interpolate_position(0.0, log) == (10.0, 20.0)
    assert interpolate_position(5.0, log) == (12.0, 24.0)


def test_interpolate_position_exact_fix():
    log = [(0.0, 10.0, 20.0), (1.0, 11.0, 21.0), (2.0, 12.0, 22.0)]
    assert interpolate_position(1.0, log) == (11.0, 21.0)
