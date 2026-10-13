package com.potholes.roadrecorder;

/**
 * Works out when the first video frame was captured, on the elapsedRealtimeNanos clock.
 *
 * CameraX reports progress as "recorded duration so far" (time from the first frame to
 * the latest encoded frame). Each report arrives a little after that frame was captured,
 * so (arrival time - recorded duration) is the first frame's capture time plus that
 * report's delivery delay. The smallest value seen has the least delay, so it is the
 * best estimate; in practice it's within a few tens of milliseconds.
 *
 * Pure Java so it can be unit-tested on the JVM.
 */
public final class VideoClock {

    private long bestOffsetNanos = Long.MAX_VALUE;
    private long startEventNanos = -1;
    private int samples;

    /** Recording started (before any frame is written). Used only as a fallback. */
    public void onStart(long arrivalNanos) {
        startEventNanos = arrivalNanos;
    }

    public void onProgress(long arrivalNanos, long recordedDurationNanos) {
        if (recordedDurationNanos <= 0) return;  // no frame written yet, says nothing about frame timing
        long offset = arrivalNanos - recordedDurationNanos;
        if (offset < bestOffsetNanos) bestOffsetNanos = offset;
        samples++;
    }

    /** First video frame's capture time on the elapsedRealtimeNanos clock. */
    public long videoStartNanos() {
        return samples > 0 ? bestOffsetNanos : startEventNanos;
    }

    public int samples() {
        return samples;
    }
}
