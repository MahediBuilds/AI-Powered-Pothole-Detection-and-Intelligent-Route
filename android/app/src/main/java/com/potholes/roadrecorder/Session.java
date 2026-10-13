package com.potholes.roadrecorder;

import android.content.Context;
import android.os.Build;
import android.os.SystemClock;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;

/**
 * One recording on disk, in the app's external files folder:
 *
 *   sessions/road_20261013_143005/
 *       road_20261013_143005.mp4        video (no audio)
 *       road_20261013_143005_gps.csv    GPS track on the video timeline (see GeoTrack.CSV_HEADER)
 *       road_20261013_143005.json       metadata: device, start time, stats, uploads
 *
 * The .json is written last, so a folder without one is an interrupted recording.
 */
final class Session {

    final String name;
    final File dir;

    private Session(File dir) {
        this.dir = dir;
        this.name = dir.getName();
    }

    static File root(Context context) {
        return new File(context.getExternalFilesDir(null), "sessions");
    }

    static Session create(Context context) {
        String name = "road_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File dir = new File(root(context), name);
        //noinspection ResultOfMethodCallIgnored
        dir.mkdirs();
        return new Session(dir);
    }

    /** All recordings, newest first. */
    static List<Session> list(Context context) {
        File[] dirs = root(context).listFiles(File::isDirectory);
        List<Session> out = new ArrayList<>();
        if (dirs == null) return out;
        Arrays.sort(dirs, (a, b) -> b.getName().compareTo(a.getName()));
        for (File d : dirs) out.add(new Session(d));
        return out;
    }

    File videoFile() {
        return new File(dir, name + ".mp4");
    }

    File gpsFile() {
        return new File(dir, name + "_gps.csv");
    }

    File metaFile() {
        return new File(dir, name + ".json");
    }

    boolean isComplete() {
        return metaFile().exists() && videoFile().exists();
    }

    /** Files to share/export, in a stable order. */
    List<File> files() {
        List<File> out = new ArrayList<>();
        for (File f : new File[]{videoFile(), gpsFile(), metaFile()}) {
            if (f.exists()) out.add(f);
        }
        return out;
    }

    /** Writes the GPS CSV and the metadata JSON once the video has been finalised. */
    void writeTrack(List<GeoTrack.Fix> fixes, long videoStartNanos, long durationNanos,
                    String resolution, int clockSamples, String appVersion) throws IOException {
        Files.write(gpsFile().toPath(), GeoTrack.toCsv(fixes, videoStartNanos).getBytes(StandardCharsets.UTF_8));

        long startUtcMillis = System.currentTimeMillis()
                - (SystemClock.elapsedRealtimeNanos() - videoStartNanos) / 1_000_000;
        SimpleDateFormat iso = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        iso.setTimeZone(TimeZone.getTimeZone("UTC"));

        try {
            JSONObject meta = new JSONObject()
                    .put("format", "road-recorder/1")
                    .put("video", videoFile().getName())
                    .put("gps", gpsFile().getName())
                    .put("started_utc", iso.format(new Date(startUtcMillis)))
                    .put("duration_sec", durationNanos / 1e9)
                    .put("resolution", resolution)
                    .put("gps_fixes", fixes.size())
                    .put("distance_m", Math.round(GeoTrack.distanceMeters(fixes)))
                    .put("clock_sync", "camerax recorded-duration vs elapsedRealtimeNanos")
                    .put("clock_sync_samples", clockSamples)
                    .put("device", Build.MANUFACTURER + " " + Build.MODEL)
                    .put("android", Build.VERSION.RELEASE)
                    .put("app_version", appVersion)
                    .put("uploads", new JSONArray());
            double acc = GeoTrack.medianAccuracy(fixes);
            if (!Double.isNaN(acc)) meta.put("median_accuracy_m", Math.round(acc * 10) / 10.0);
            writeMeta(meta);
        } catch (JSONException e) {
            throw new IOException(e);
        }
    }

    JSONObject readMeta() {
        try {
            return new JSONObject(new String(Files.readAllBytes(metaFile().toPath()), StandardCharsets.UTF_8));
        } catch (IOException | JSONException e) {
            return null;
        }
    }

    private void writeMeta(JSONObject meta) throws IOException, JSONException {
        Files.write(metaFile().toPath(), meta.toString(2).getBytes(StandardCharsets.UTF_8));
    }

    /** Records a successful server run in the metadata. */
    void markUploaded(String server, String jobId) {
        JSONObject meta = readMeta();
        if (meta == null) return;
        try {
            JSONArray uploads = meta.optJSONArray("uploads");
            if (uploads == null) uploads = new JSONArray();
            uploads.put(new JSONObject().put("server", server).put("job_id", jobId)
                    .put("time_ms", System.currentTimeMillis()));
            meta.put("uploads", uploads);
            writeMeta(meta);
        } catch (IOException | JSONException ignored) {
            // Upload state is a convenience; the recording itself is unaffected.
        }
    }

    /** Job id of the most recent successful server run, or null. */
    String lastJobId() {
        JSONObject meta = readMeta();
        JSONArray uploads = meta == null ? null : meta.optJSONArray("uploads");
        if (uploads == null || uploads.length() == 0) return null;
        return uploads.optJSONObject(uploads.length() - 1).optString("job_id", null);
    }

    void delete() {
        File[] files = dir.listFiles();
        if (files != null) {
            //noinspection ResultOfMethodCallIgnored
            for (File f : files) f.delete();
        }
        //noinspection ResultOfMethodCallIgnored
        dir.delete();
    }
}
