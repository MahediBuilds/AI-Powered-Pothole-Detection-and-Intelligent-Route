package com.potholes.roadrecorder;

import android.os.Handler;
import android.os.Looper;

import androidx.annotation.NonNull;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import okio.Buffer;
import okio.BufferedSink;
import okio.Okio;
import okio.Source;

/**
 * Talks to app.py: lists models, uploads a recording to /api/process, then polls the
 * job until the pipeline finishes. Network calls run off the main thread; listener
 * callbacks are delivered on the main thread.
 */
final class Uploader {

    interface Listener {
        void onUploadProgress(int percent);

        void onProcessing(int percent, String currentStep);

        /** total/mild/moderate/severe are counts across the whole detections.json. */
        void onDone(String jobId, int total, int mild, int moderate, int severe);

        void onError(String message);
    }

    static final class Model {
        final String label;
        final String path;

        Model(String label, String path) {
            this.label = label;
            this.path = path;
        }
    }

    private static final long POLL_INTERVAL_MS = 1500;
    private static final OkHttpClient CLIENT = new OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .writeTimeout(5, TimeUnit.MINUTES)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();
    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    private Uploader() {}

    /** Blocking; call off the main thread. */
    static List<Model> fetchModels(String baseUrl) throws IOException {
        JSONObject body = getJson(baseUrl + "/api/models");
        List<Model> models = new ArrayList<>();
        JSONArray arr = body.optJSONArray("models");
        if (arr != null) {
            for (int i = 0; i < arr.length(); i++) {
                JSONObject m = arr.optJSONObject(i);
                if (m != null) models.add(new Model(m.optString("label"), m.optString("path")));
            }
        }
        return models;
    }

    static void upload(String baseUrl, Session session, String modelPath, boolean append, Listener listener) {
        new Thread(() -> {
            try {
                String jobId = post(baseUrl, session, modelPath, append, listener);
                poll(baseUrl, jobId, listener);
            } catch (IOException e) {
                String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                MAIN.post(() -> listener.onError(msg));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }, "upload-" + session.name).start();
    }

    private static String post(String baseUrl, Session session, String modelPath, boolean append,
                               Listener listener) throws IOException {
        File video = session.videoFile();
        RequestBody videoBody = new ProgressBody(video, MediaType.get("video/mp4"),
                percent -> MAIN.post(() -> listener.onUploadProgress(percent)));

        MultipartBody.Builder form = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("video", video.getName(), videoBody)
                .addFormDataPart("model_path", modelPath)
                .addFormDataPart("append", append ? "true" : "false");
        File gps = session.gpsFile();
        if (gps.exists()) {
            form.addFormDataPart("gps", gps.getName(), RequestBody.create(gps, MediaType.get("text/csv")));
        }

        Request request = new Request.Builder().url(baseUrl + "/api/process").post(form.build()).build();
        try (Response response = CLIENT.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("server returned HTTP " + response.code());
            return parse(response.body()).getString("job_id");
        } catch (JSONException e) {
            throw new IOException("unexpected response from /api/process", e);
        }
    }

    private static void poll(String baseUrl, String jobId, Listener listener)
            throws IOException, InterruptedException {
        while (true) {
            Thread.sleep(POLL_INTERVAL_MS);
            JSONObject job = getJson(baseUrl + "/api/process/" + jobId + "/status");
            String status = job.optString("status");
            int progress = job.optInt("progress", 0);
            JSONArray log = job.optJSONArray("log");
            String lastLine = log != null && log.length() > 0 ? log.optString(log.length() - 1) : "";
            String step = latestStep(log);

            if ("done".equals(status)) {
                JSONObject summary = job.optJSONObject("summary");
                JSONObject breakdown = summary != null ? summary.optJSONObject("breakdown") : null;
                int total = summary != null ? summary.optInt("total_detections") : 0;
                int mild = breakdown != null ? breakdown.optInt("Mild") : 0;
                int moderate = breakdown != null ? breakdown.optInt("Moderate") : 0;
                int severe = breakdown != null ? breakdown.optInt("Severe") : 0;
                MAIN.post(() -> listener.onDone(jobId, total, mild, moderate, severe));
                return;
            }
            if ("error".equals(status)) {
                String msg = lastLine.isEmpty() ? "processing failed on the server" : lastLine;
                MAIN.post(() -> listener.onError(msg));
                return;
            }
            MAIN.post(() -> listener.onProcessing(progress, step));
        }
    }

    /** Most recent "Step n/4: ..." line app.py logged; its other lines are raw subprocess output. */
    private static String latestStep(JSONArray log) {
        if (log != null) {
            for (int i = log.length() - 1; i >= 0; i--) {
                String line = log.optString(i);
                if (line.startsWith("Step ")) return line;
            }
        }
        return "";
    }

    private static JSONObject getJson(String url) throws IOException {
        Request request = new Request.Builder().url(url).build();
        try (Response response = CLIENT.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("server returned HTTP " + response.code());
            return parse(response.body());
        }
    }

    private static JSONObject parse(ResponseBody body) throws IOException {
        if (body == null) throw new IOException("empty response");
        try {
            return new JSONObject(body.string());
        } catch (JSONException e) {
            throw new IOException("server response wasn't JSON", e);
        }
    }

    /** File request body that reports how much of it has been sent. */
    private static final class ProgressBody extends RequestBody {
        interface OnProgress {
            void onProgress(int percent);
        }

        private final File file;
        private final MediaType type;
        private final OnProgress onProgress;

        ProgressBody(File file, MediaType type, OnProgress onProgress) {
            this.file = file;
            this.type = type;
            this.onProgress = onProgress;
        }

        @Override
        public MediaType contentType() {
            return type;
        }

        @Override
        public long contentLength() {
            return file.length();
        }

        @Override
        public void writeTo(@NonNull BufferedSink sink) throws IOException {
            long total = file.length();
            long sent = 0;
            int lastPercent = -1;
            Buffer buffer = new Buffer();
            try (Source source = Okio.source(file)) {
                long read;
                while ((read = source.read(buffer, 64 * 1024)) != -1) {
                    sink.write(buffer, read);
                    sent += read;
                    int percent = total > 0 ? (int) (sent * 100 / total) : 100;
                    if (percent != lastPercent) {
                        lastPercent = percent;
                        onProgress.onProgress(percent);
                    }
                }
            }
        }
    }
}
