package com.potholes.roadrecorder;

import java.util.Locale;

/** Display formatting shared by the recorder and the recordings list. */
final class Formats {

    private Formats() {}

    static String duration(double seconds) {
        long s = Math.max(0, Math.round(Math.floor(seconds)));
        return s >= 3600
                ? String.format(Locale.US, "%d:%02d:%02d", s / 3600, (s / 60) % 60, s % 60)
                : String.format(Locale.US, "%d:%02d", s / 60, s % 60);
    }

    static String distance(double meters) {
        return meters >= 1000
                ? String.format(Locale.US, "%.2f km", meters / 1000)
                : String.format(Locale.US, "%d m", Math.round(meters));
    }

    static String accuracy(double meters) {
        return Double.isNaN(meters) ? "no accuracy" : String.format(Locale.US, "±%d m", Math.round(meters));
    }
}
