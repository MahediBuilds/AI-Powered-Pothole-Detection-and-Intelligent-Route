package com.potholes.roadrecorder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class GeoTrackTest {

    private static final long SEC = 1_000_000_000L;

    private static GeoTrack.Fix fix(long nanos, double lat, double lon, double acc) {
        return new GeoTrack.Fix(nanos, 1_760_000_000_000L + nanos / 1_000_000, lat, lon,
                acc, Double.NaN, Double.NaN, Double.NaN);
    }

    @Test
    public void csvTimestampsAreRelativeToFirstVideoFrame() {
        long start = 100 * SEC;
        List<GeoTrack.Fix> fixes = Arrays.asList(
                fix(start - SEC / 2, 12.9716, 77.5946, 4.0),   // pre-roll
                fix(start + 1_250_000_000L, 12.9717, 77.5947, 5.5));

        String[] lines = GeoTrack.toCsv(fixes, start).split("\n");

        assertEquals(GeoTrack.CSV_HEADER, lines[0]);
        assertTrue(lines[1], lines[1].startsWith("-0.500,12.9716000,77.5946000,4.0,,,,"));
        assertTrue(lines[2], lines[2].startsWith("1.250,12.9717000,77.5947000,5.5,,,,"));
    }

    @Test
    public void csvUsesDotDecimalsWhateverTheLocale() {
        Locale saved = Locale.getDefault();
        try {
            Locale.setDefault(Locale.GERMANY);
            String csv = GeoTrack.toCsv(Arrays.asList(fix(SEC, 12.5, 77.25, 3.0)), 0);
            assertTrue(csv, csv.contains("1.000,12.5000000,77.2500000,3.0"));
        } finally {
            Locale.setDefault(saved);
        }
    }

    @Test
    public void addDropsFixesThatAreNotNewer() {
        GeoTrack track = new GeoTrack();
        track.add(fix(2 * SEC, 1, 1, 1));
        track.add(fix(2 * SEC, 2, 2, 1));
        track.add(fix(SEC, 3, 3, 1));
        track.add(fix(3 * SEC, 4, 4, 1));

        List<GeoTrack.Fix> all = track.window(0, Long.MAX_VALUE);
        assertEquals(2, all.size());
        assertEquals(1, all.get(0).lat, 0);
        assertEquals(4, all.get(1).lat, 0);
    }

    @Test
    public void windowAndCountSinceAreInclusive() {
        GeoTrack track = new GeoTrack();
        for (int i = 0; i < 10; i++) track.add(fix(i * SEC, i, 0, 1));

        assertEquals(4, track.window(3 * SEC, 6 * SEC).size());
        assertEquals(3, track.countSince(7 * SEC));
    }

    @Test
    public void distanceOfOneThousandthDegreeLatitudeIsAbout111m() {
        List<GeoTrack.Fix> fixes = Arrays.asList(fix(0, 12.000, 77.0, 1), fix(SEC, 12.001, 77.0, 1));
        assertEquals(111.2, GeoTrack.distanceMeters(fixes), 0.5);
    }

    @Test
    public void distanceIgnoresJitterOfAStationaryPhone() {
        // ~1-3 m wobble around one spot with 12 m reported accuracy (seen on a real device)
        List<GeoTrack.Fix> fixes = Arrays.asList(
                fix(0, 12.9398146, 77.5646092, 12), fix(SEC, 12.9398146, 77.5646067, 12),
                fix(2 * SEC, 12.9398129, 77.5646092, 12), fix(3 * SEC, 12.9398116, 77.5646129, 12),
                fix(4 * SEC, 12.9398155, 77.5645716, 12));
        assertEquals(0, GeoTrack.distanceMeters(fixes), 0);
    }

    @Test
    public void distanceCountsSlowSteadyMovementOverTime() {
        // 2 m per fix with 4 m accuracy: no single hop counts, but the drift does
        List<GeoTrack.Fix> fixes = new java.util.ArrayList<>();
        for (int i = 0; i <= 50; i++) fixes.add(fix(i * SEC, 12.0 + i * 0.000018, 77.0, 4));
        assertEquals(100, GeoTrack.distanceMeters(fixes), 6);
    }

    @Test
    public void medianAccuracyIgnoresMissingValues() {
        List<GeoTrack.Fix> fixes = Arrays.asList(
                fix(0, 0, 0, 3), fix(1, 0, 0, Double.NaN), fix(2, 0, 0, 9), fix(3, 0, 0, 5));
        assertEquals(5, GeoTrack.medianAccuracy(fixes), 0);
        assertTrue(Double.isNaN(GeoTrack.medianAccuracy(Arrays.asList(fix(0, 0, 0, Double.NaN)))));
    }
}
