package com.potholes.roadrecorder;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class VideoClockTest {

    private static final long MS = 1_000_000L;

    @Test
    public void usesTheLeastDelayedProgressReport() {
        VideoClock clock = new VideoClock();
        long firstFrame = 5_000 * MS;
        clock.onStart(firstFrame - 300 * MS);
        clock.onProgress(firstFrame + 1_000 * MS + 80 * MS, 1_000 * MS);  // 80 ms late
        clock.onProgress(firstFrame + 2_000 * MS + 15 * MS, 2_000 * MS);  // 15 ms late
        clock.onProgress(firstFrame + 3_000 * MS + 40 * MS, 3_000 * MS);  // 40 ms late

        assertEquals(firstFrame + 15 * MS, clock.videoStartNanos());
        assertEquals(3, clock.samples());
    }

    @Test
    public void ignoresReportsBeforeAnyFrameIsWritten() {
        VideoClock clock = new VideoClock();
        clock.onStart(1_000 * MS);
        clock.onProgress(900 * MS, 0);

        assertEquals(1_000 * MS, clock.videoStartNanos());
        assertEquals(0, clock.samples());
    }
}
