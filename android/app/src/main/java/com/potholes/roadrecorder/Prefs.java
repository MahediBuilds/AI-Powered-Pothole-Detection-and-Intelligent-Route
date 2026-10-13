package com.potholes.roadrecorder;

import android.content.Context;
import android.content.SharedPreferences;

/** User settings: where the detection server is and what quality to record at. */
final class Prefs {

    static final String QUALITY_720 = "720p";
    static final String QUALITY_1080 = "1080p";
    private static final String DEFAULT_SERVER = "http://192.168.1.10:8000";

    private final SharedPreferences prefs;

    Prefs(Context context) {
        prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE);
    }

    /** Server base URL, normalised: has a scheme, no trailing slash. */
    String serverUrl() {
        return normalize(rawServerUrl());
    }

    String rawServerUrl() {
        return prefs.getString("server_url", DEFAULT_SERVER);
    }

    void setServerUrl(String url) {
        prefs.edit().putString("server_url", url.trim()).apply();
    }

    String quality() {
        return prefs.getString("quality", QUALITY_720);
    }

    void setQuality(String quality) {
        prefs.edit().putString("quality", quality).apply();
    }

    String lastModel() {
        return prefs.getString("last_model", "");
    }

    void setLastModel(String path) {
        prefs.edit().putString("last_model", path).apply();
    }

    static String normalize(String url) {
        String u = url.trim();
        if (!u.contains("://")) u = "http://" + u;
        while (u.endsWith("/")) u = u.substring(0, u.length() - 1);
        return u;
    }
}
