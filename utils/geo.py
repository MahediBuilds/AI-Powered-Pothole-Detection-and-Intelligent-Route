"""Geotagging helpers.

Not wired into the training pipeline yet - this module exists so that once
per-video GPS logs are available, inference output can be geotagged without
changing any other part of the pipeline.

Expected geotag log format (CSV, one row per GPS fix):
    timestamp_sec,lat,lon
where timestamp_sec is seconds since the start of the corresponding video.
"""

import csv
from bisect import bisect_left
from pathlib import Path


def load_geotag_log(path):
    """Load a (timestamp_sec, lat, lon) log, sorted by timestamp.

    Returns a list of tuples. Returns an empty list if `path` is None or
    does not exist, so callers can treat "no geotag available" uniformly.
    """
    if path is None:
        return []

    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((float(row["timestamp_sec"]), float(row["lat"]), float(row["lon"])))
            except (KeyError, ValueError, TypeError):
                continue  # malformed/blank row - skip rather than crash the run

    rows.sort(key=lambda r: r[0])
    return rows


def interpolate_position(timestamp_sec, log):
    """Linearly interpolate (lat, lon) at `timestamp_sec` from a sorted geotag log.

    Returns (None, None) if the log is empty. Clamps to the nearest endpoint
    if the timestamp falls outside the log's range.
    """
    if not log:
        return None, None

    timestamps = [r[0] for r in log]
    idx = bisect_left(timestamps, timestamp_sec)

    if idx == 0:
        _, lat, lon = log[0]
        return lat, lon
    if idx >= len(log):
        _, lat, lon = log[-1]
        return lat, lon

    t0, lat0, lon0 = log[idx - 1]
    t1, lat1, lon1 = log[idx]
    if t1 == t0:
        return lat0, lon0

    frac = (timestamp_sec - t0) / (t1 - t0)
    lat = lat0 + frac * (lat1 - lat0)
    lon = lon0 + frac * (lon1 - lon0)
    return lat, lon
