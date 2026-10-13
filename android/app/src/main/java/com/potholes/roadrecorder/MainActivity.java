package com.potholes.roadrecorder;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Location;
import android.media.MediaMetadataRetriever;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.ColorRes;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FallbackStrategy;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.textfield.TextInputEditText;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Camera screen: shows the road, the live GPS status, and the record button.
 *
 * While recording, CameraX progress events feed a {@link VideoClock} so that when the
 * video is finalised every GPS fix can be given an exact timestamp on the video timeline.
 */
public class MainActivity extends AppCompatActivity {

    /** GPS fixes kept from before the first frame / after the last, so the ends interpolate. */
    private static final long PRE_ROLL_NANOS = 5_000_000_000L;
    private static final long POST_ROLL_NANOS = 2_000_000_000L;
    private static final long GPS_STALE_MS = 5000;
    private static final String[] PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
    };

    private View root;
    private PreviewView previewView;
    private TextView gpsChip;
    private TextView recChip;
    private TextView message;
    private ImageButton recordButton;
    private Button settingsButton;
    private Button sessionsButton;

    private Prefs prefs;
    private GpsLogger gps;
    private ProcessCameraProvider cameraProvider;
    private VideoCapture<Recorder> videoCapture;
    private Recording recording;
    private Session session;
    private VideoClock clock;

    private final Handler ticker = new Handler(Looper.getMainLooper());
    private final Runnable gpsStalenessCheck = new Runnable() {
        @Override
        public void run() {
            if (gps.hasPermission() && gps.lastLocation() != null && gps.lastFixAgeMillis() > GPS_STALE_MS) {
                setGpsChip(R.color.gps_bad, getString(R.string.gps_lost));
            }
            ticker.postDelayed(this, 1000);
        }
    };

    private final ActivityResultLauncher<String[]> permissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(), this::onPermissionResult);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        hideSystemBars();

        root = findViewById(R.id.root);
        previewView = findViewById(R.id.preview);
        gpsChip = findViewById(R.id.gpsChip);
        recChip = findViewById(R.id.recChip);
        message = findViewById(R.id.message);
        recordButton = findViewById(R.id.recordButton);
        settingsButton = findViewById(R.id.settingsButton);
        sessionsButton = findViewById(R.id.sessionsButton);

        View overlay = findViewById(R.id.overlay);
        ViewCompat.setOnApplyWindowInsetsListener(overlay, (v, insets) -> {
            Insets i = insets.getInsets(WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout());
            v.setPadding(i.left, i.top, i.right, i.bottom);
            return insets;
        });

        prefs = new Prefs(this);
        gps = new GpsLogger(this, this::onFix);

        recordButton.setEnabled(false);
        recordButton.setOnClickListener(v -> toggleRecording());
        settingsButton.setOnClickListener(v -> showSettings());
        sessionsButton.setOnClickListener(v -> startActivity(new Intent(this, SessionsActivity.class)));

        if (granted(Manifest.permission.CAMERA) && gps.hasPermission()) {
            startCamera();
        } else {
            permissionLauncher.launch(PERMISSIONS);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        startGps();
    }

    @Override
    protected void onStop() {
        super.onStop();
        ticker.removeCallbacks(gpsStalenessCheck);
        // Leaving the screen stops the camera too (CameraX is lifecycle-bound), which
        // finalises any recording, so GPS isn't needed in the background.
        gps.stop();
    }

    private void hideSystemBars() {
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowInsetsControllerCompat controller =
                WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView());
        controller.hide(WindowInsetsCompat.Type.systemBars());
        controller.setSystemBarsBehavior(WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
    }

    private boolean granted(String permission) {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED;
    }

    private void onPermissionResult(Map<String, Boolean> result) {
        if (!granted(Manifest.permission.CAMERA)) {
            showMessage(getString(R.string.need_camera));
            return;
        }
        startCamera();
        if (!startGps()) setGpsChip(R.color.gps_bad, getString(R.string.gps_denied));
    }

    private boolean startGps() {
        if (!gps.start()) return false;
        ticker.removeCallbacks(gpsStalenessCheck);
        ticker.post(gpsStalenessCheck);
        return true;
    }

    // ------------------------------------------------------------------ GPS

    private void onFix(Location loc) {
        int acc = Math.round(loc.getAccuracy());
        @ColorRes int color = acc <= 10 ? R.color.gps_good : acc <= 30 ? R.color.gps_warn : R.color.gps_bad;
        setGpsChip(color, getString(R.string.gps_fix, acc, loc.getLatitude(), loc.getLongitude()));
    }

    private void setGpsChip(@ColorRes int dotColor, String text) {
        SpannableStringBuilder sb = new SpannableStringBuilder("● ").append(text);
        sb.setSpan(new ForegroundColorSpan(ContextCompat.getColor(this, dotColor)), 0, 1,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        gpsChip.setText(sb);
    }

    // --------------------------------------------------------------- Camera

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindCamera();
            } catch (Exception e) {
                showMessage(getString(R.string.camera_failed, e.getMessage()));
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCamera() {
        Quality quality = Prefs.QUALITY_1080.equals(prefs.quality()) ? Quality.FHD : Quality.HD;
        Recorder recorder = new Recorder.Builder()
                .setQualitySelector(QualitySelector.from(quality, FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)))
                .build();
        videoCapture = VideoCapture.withOutput(recorder);

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, videoCapture);
            message.setVisibility(View.GONE);
            recordButton.setEnabled(true);
        } catch (IllegalArgumentException | IllegalStateException e) {
            showMessage(getString(R.string.camera_failed, e.getMessage()));
        }
    }

    private void showMessage(String text) {
        message.setText(text);
        message.setVisibility(View.VISIBLE);
    }

    // ------------------------------------------------------------ Recording

    private void toggleRecording() {
        if (recording != null) {
            recording.stop();
            recordButton.setEnabled(false);  // re-enabled once the file is finalised
            return;
        }
        if (videoCapture == null) return;

        session = Session.create(this);
        clock = new VideoClock();

        FileOutputOptions.Builder options = new FileOutputOptions.Builder(session.videoFile());
        Location last = gps.lastLocation();
        if (last != null) options.setLocation(last);  // also tags the MP4 itself with the start point

        try {
            recording = videoCapture.getOutput()
                    .prepareRecording(this, options.build())
                    .start(ContextCompat.getMainExecutor(this), this::onRecordEvent);
        } catch (RuntimeException e) {
            session.delete();
            session = null;
            Toast.makeText(this, getString(R.string.record_failed, e.getMessage()), Toast.LENGTH_LONG).show();
            return;
        }
        setRecordingUi(true);
    }

    private void onRecordEvent(VideoRecordEvent event) {
        long now = SystemClock.elapsedRealtimeNanos();
        if (event instanceof VideoRecordEvent.Start) {
            clock.onStart(now);
        } else if (event instanceof VideoRecordEvent.Status) {
            long durationNanos = event.getRecordingStats().getRecordedDurationNanos();
            clock.onProgress(now, durationNanos);
            int fixes = gps.track().countSince(clock.videoStartNanos() - PRE_ROLL_NANOS);
            recChip.setText(getString(R.string.rec_status, Formats.duration(durationNanos / 1e9), fixes));
        } else if (event instanceof VideoRecordEvent.Finalize) {
            onFinalized((VideoRecordEvent.Finalize) event);
        }
    }

    private void onFinalized(VideoRecordEvent.Finalize event) {
        recording = null;
        setRecordingUi(false);
        Session s = session;
        session = null;

        // Errors such as the app being backgrounded still leave a playable file, so
        // keep anything that was written; only "no valid data" means nothing to save.
        if (event.getError() == VideoRecordEvent.Finalize.ERROR_NO_VALID_DATA
                || !s.videoFile().exists() || s.videoFile().length() == 0) {
            s.delete();
            Toast.makeText(this, R.string.nothing_recorded, Toast.LENGTH_SHORT).show();
            return;
        }

        long durationNanos = event.getRecordingStats().getRecordedDurationNanos();
        long start = clock.videoStartNanos();
        List<GeoTrack.Fix> fixes = gps.track().window(start - PRE_ROLL_NANOS, start + durationNanos + POST_ROLL_NANOS);
        try {
            s.writeTrack(fixes, start, durationNanos, readResolution(s.videoFile()), clock.samples(), appVersion());
        } catch (IOException e) {
            Toast.makeText(this, getString(R.string.save_failed, e.getMessage()), Toast.LENGTH_LONG).show();
            return;
        }

        String text = fixes.size() < 2
                ? getString(R.string.saved_no_gps, Formats.duration(durationNanos / 1e9))
                : getString(R.string.saved_summary, Formats.duration(durationNanos / 1e9),
                        Formats.distance(GeoTrack.distanceMeters(fixes)), fixes.size());
        Snackbar.make(root, text, Snackbar.LENGTH_LONG)
                .setAction(R.string.view, v -> startActivity(new Intent(this, SessionsActivity.class)))
                .show();
    }

    private void setRecordingUi(boolean recordingNow) {
        recordButton.setImageResource(recordingNow ? R.drawable.rec_active : R.drawable.rec_idle);
        recordButton.setContentDescription(getString(recordingNow ? R.string.stop_recording : R.string.start_recording));
        recordButton.setEnabled(true);
        recChip.setVisibility(recordingNow ? View.VISIBLE : View.GONE);
        recChip.setText(getString(R.string.rec_status, Formats.duration(0), 0));
        settingsButton.setEnabled(!recordingNow);
        sessionsButton.setEnabled(!recordingNow);
    }

    /** "1280x720"-style size of the finished video, or "" if it can't be read. */
    private static String readResolution(File video) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(video.getPath());
            String w = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
            String h = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
            return w != null && h != null ? w + "x" + h : "";
        } catch (RuntimeException e) {
            return "";
        } finally {
            try {
                retriever.release();
            } catch (Exception ignored) {
                // nothing to clean up beyond this
            }
        }
    }

    private String appVersion() {
        try {
            return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (PackageManager.NameNotFoundException e) {
            return "";
        }
    }

    // ------------------------------------------------------------- Settings

    private void showSettings() {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_settings, null);
        TextInputEditText urlInput = view.findViewById(R.id.serverUrl);
        RadioGroup qualityGroup = view.findViewById(R.id.qualityGroup);
        urlInput.setText(prefs.rawServerUrl());
        qualityGroup.check(Prefs.QUALITY_1080.equals(prefs.quality()) ? R.id.quality1080 : R.id.quality720);

        new MaterialAlertDialogBuilder(this)
                .setTitle(R.string.settings)
                .setView(view)
                .setPositiveButton(R.string.save, (d, w) -> saveSettings(urlInput, qualityGroup, false))
                .setNeutralButton(R.string.test_connection, (d, w) -> saveSettings(urlInput, qualityGroup, true))
                .setNegativeButton(R.string.cancel, null)
                .show();
    }

    private void saveSettings(TextInputEditText urlInput, RadioGroup qualityGroup, boolean test) {
        prefs.setServerUrl(String.valueOf(urlInput.getText()));
        String quality = qualityGroup.getCheckedRadioButtonId() == R.id.quality1080 ? Prefs.QUALITY_1080 : Prefs.QUALITY_720;
        if (!quality.equals(prefs.quality())) {
            prefs.setQuality(quality);
            if (cameraProvider != null && recording == null) bindCamera();
        }
        if (test) testConnection();
    }

    private void testConnection() {
        String base = prefs.serverUrl();
        new Thread(() -> {
            try {
                int n = Uploader.fetchModels(base).size();
                runOnUiThread(() -> Toast.makeText(this, getString(R.string.connected, n), Toast.LENGTH_LONG).show());
            } catch (IOException e) {
                runOnUiThread(() -> new MaterialAlertDialogBuilder(this)
                        .setTitle(R.string.unreachable_title)
                        .setMessage(getString(R.string.unreachable_body, base, e.getMessage()))
                        .setPositiveButton(R.string.ok, null)
                        .show());
            }
        }, "test-connection").start();
    }
}
