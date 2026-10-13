package com.potholes.roadrecorder;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * GPS fixes stamped with the device's monotonic clock (SystemClock.elapsedRealtimeNanos),
 * the same clock {@link VideoClock} uses, so each fix can be placed on the video timeline.
 *
 * Pure Java (no Android types) so it can be unit-tested on the JVM.
 */
public final class GeoTrack {

    /**
     * First three columns are what utils/geo.py reads (timestamp_sec = seconds since
     * the first video frame); the rest are extra context it ignores.
     */
    public static final String CSV_HEADER =
            "timestamp_sec,lat,lon,accuracy_m,speed_mps,bearing_deg,altitude_m,utc_ms";

    public static final class Fix {
        public final long elapsedNanos;
        public final long utcMillis;
        public final double lat;
        public final double lon;
        // NaN when the device didn't report the value
        public final double accuracyM;
        public final double speedMps;
        public final double bearingDeg;
        public final double altitudeM;

        public Fix(long elapsedNanos, long utcMillis, double lat, double lon,
                   double accuracyM, double speedMps, double bearingDeg, double altitudeM) {
            this.elapsedNanos = elapsedNanos;
            this.utcMillis = utcMillis;
            this.lat = lat;
            this.lon = lon;
            this.accuracyM = accuracyM;
            this.speedMps = speedMps;
            this.bearingDeg = bearingDeg;
            this.altitudeM = altitudeM;
        }
    }

    private final List<Fix> fixes = new ArrayList<>();

    /** Appends a fix; drops it if it isn't newer than the last one (batched duplicates). */
    public synchronized void add(Fix fix) {
        if (!fixes.isEmpty() && fix.elapsedNanos <= fixes.get(fixes.size() - 1).elapsedNanos) {
            return;
        }
        fixes.add(fix);
    }

    public synchronized int countSince(long fromNanos) {
        int n = 0;
        for (int i = fixes.size() - 1; i >= 0 && fixes.get(i).elapsedNanos >= fromNanos; i--) {
            n++;
        }
        return n;
    }

    /** Fixes with fromNanos <= elapsedNanos <= toNanos, oldest first. */
    public synchronized List<Fix> window(long fromNanos, long toNanos) {
        List<Fix> out = new ArrayList<>();
        for (Fix f : fixes) {
            if (f.elapsedNanos >= fromNanos && f.elapsedNanos <= toNanos) {
                out.add(f);
            }
        }
        return out;
    }

    /**
     * CSV with timestamp_sec relative to the first video frame. Fixes from just before
     * recording started get small negative timestamps, which lets utils/geo.py
     * interpolate (rather than clamp) the position of the very first frames.
     */
    public static String toCsv(List<Fix> fixes, long videoStartNanos) {
        StringBuilder sb = new StringBuilder(CSV_HEADER).append('\n');
        for (Fix f : fixes) {
            double t = (f.elapsedNanos - videoStartNanos) / 1e9;
            sb.append(String.format(Locale.US, "%.3f,%.7f,%.7f,", t, f.lat, f.lon))
                    .append(optional(f.accuracyM, 1)).append(',')
                    .append(optional(f.speedMps, 2)).append(',')
                    .append(optional(f.bearingDeg, 1)).append(',')
                    .append(optional(f.altitudeM, 1)).append(',')
                    .append(f.utcMillis).append('\n');
        }
        return sb.toString();
    }

    private static String optional(double value, int decimals) {
        return Double.isNaN(value) ? "" : String.format(Locale.US, "%." + decimals + "f", value);
    }

    /** Below this, a move is treated as GPS jitter even if the fix claims better accuracy. */
    static final double MIN_STEP_M = 5;

    /**
     * Distance travelled, ignoring jitter: a fix only counts once it is further from the
     * last counted fix than the GPS's own error margin. Summing every fix-to-fix hop
     * instead would add metres per minute to a phone that is standing still.
     */
    public static double distanceMeters(List<Fix> fixes) {
        if (fixes.isEmpty()) return 0;
        double total = 0;
        Fix anchor = fixes.get(0);
        for (int i = 1; i < fixes.size(); i++) {
            Fix f = fixes.get(i);
            double step = haversineMeters(anchor, f);
            double margin = Math.max(MIN_STEP_M, Double.isNaN(f.accuracyM) ? 0 : f.accuracyM);
            if (step > margin) {
                total += step;
                anchor = f;
            }
        }
        return total;
    }

    static double haversineMeters(Fix a, Fix b) {
        final double r = 6371000;
        double dLat = Math.toRadians(b.lat - a.lat);
        double dLon = Math.toRadians(b.lon - a.lon);
        double h = Math.pow(Math.sin(dLat / 2), 2)
                + Math.cos(Math.toRadians(a.lat)) * Math.cos(Math.toRadians(b.lat)) * Math.pow(Math.sin(dLon / 2), 2);
        return 2 * r * Math.asin(Math.sqrt(h));
    }

    /** Median reported accuracy in metres, or NaN if no fix reported one. */
    public static double medianAccuracy(List<Fix> fixes) {
        List<Double> values = new ArrayList<>();
        for (Fix f : fixes) {
            if (!Double.isNaN(f.accuracyM)) values.add(f.accuracyM);
        }
        if (values.isEmpty()) return Double.NaN;
        values.sort(null);
        int mid = values.size() / 2;
        return values.size() % 2 == 1 ? values.get(mid) : (values.get(mid - 1) + values.get(mid)) / 2;
    }
}
