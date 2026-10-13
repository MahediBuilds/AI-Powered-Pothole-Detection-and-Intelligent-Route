package com.potholes.roadrecorder;

import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Lists saved recordings and sends them to the detection server, shares or deletes them. */
public class SessionsActivity extends AppCompatActivity {

    private Prefs prefs;
    private LinearLayout list;
    private TextView empty;
    private int activeUploads;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sessions);
        prefs = new Prefs(this);
        list = findViewById(R.id.list);
        empty = findViewById(R.id.empty);

        MaterialToolbar toolbar = findViewById(R.id.toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());
        toolbar.inflateMenu(R.menu.sessions);
        toolbar.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == R.id.action_open_map) {
                openMap();
                return true;
            }
            return false;
        });

        render();
    }

    private void render() {
        list.removeAllViews();
        List<Session> sessions = Session.list(this);
        empty.setVisibility(sessions.isEmpty() ? View.VISIBLE : View.GONE);
        LayoutInflater inflater = LayoutInflater.from(this);
        for (Session s : sessions) {
            View item = inflater.inflate(R.layout.item_session, list, false);
            bind(item, s);
            list.addView(item);
        }
    }

    private void bind(View item, Session s) {
        TextView title = item.findViewById(R.id.title);
        TextView stats = item.findViewById(R.id.stats);
        TextView status = item.findViewById(R.id.status);
        Button upload = item.findViewById(R.id.uploadButton);
        Button share = item.findViewById(R.id.shareButton);
        Button delete = item.findViewById(R.id.deleteButton);

        title.setText(s.name);
        JSONObject meta = s.readMeta();
        if (meta == null || !s.isComplete()) {
            stats.setVisibility(View.GONE);
            status.setText(R.string.session_incomplete);
            upload.setEnabled(s.videoFile().exists());
        } else {
            stats.setText(getString(R.string.session_stats,
                    Formats.duration(meta.optDouble("duration_sec", 0)),
                    Formats.distance(meta.optDouble("distance_m", 0)),
                    meta.optInt("gps_fixes", 0),
                    Formats.accuracy(meta.optDouble("median_accuracy_m", Double.NaN))));
            String jobId = s.lastJobId();
            status.setText(jobId == null ? getString(R.string.status_not_uploaded) : getString(R.string.status_uploaded, jobId));
        }

        upload.setOnClickListener(v -> startUpload(item, s));
        share.setOnClickListener(v -> share(s));
        delete.setOnClickListener(v -> confirmDelete(s));
    }

    // --------------------------------------------------------------- Upload

    private void startUpload(View item, Session s) {
        TextView status = item.findViewById(R.id.status);
        Button upload = item.findViewById(R.id.uploadButton);
        String base = prefs.serverUrl();
        upload.setEnabled(false);
        status.setText(getString(R.string.status_connecting, base));

        new Thread(() -> {
            try {
                List<Uploader.Model> models = Uploader.fetchModels(base);
                runOnUiThread(() -> chooseModel(item, s, base, models));
            } catch (IOException e) {
                runOnUiThread(() -> {
                    upload.setEnabled(true);
                    status.setText(getString(R.string.status_failed, e.getMessage()));
                    new MaterialAlertDialogBuilder(this)
                            .setTitle(R.string.unreachable_title)
                            .setMessage(getString(R.string.unreachable_body, base, e.getMessage()))
                            .setPositiveButton(R.string.ok, null)
                            .show();
                });
            }
        }, "fetch-models").start();
    }

    private void chooseModel(View item, Session s, String base, List<Uploader.Model> models) {
        Button upload = item.findViewById(R.id.uploadButton);
        TextView status = item.findViewById(R.id.status);
        if (models.isEmpty()) {
            upload.setEnabled(true);
            status.setText(getString(R.string.status_failed, getString(R.string.no_models)));
            return;
        }

        View view = LayoutInflater.from(this).inflate(R.layout.dialog_upload, null);
        RadioGroup group = view.findViewById(R.id.modelGroup);
        CheckBox append = view.findViewById(R.id.appendBox);
        int checkedId = View.NO_ID;
        for (int i = 0; i < models.size(); i++) {
            RadioButton rb = new RadioButton(this);
            rb.setId(View.generateViewId());
            rb.setText(models.get(i).label);
            rb.setTag(models.get(i).path);
            group.addView(rb, new RadioGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            // Server lists pothole-trained models first, so default to the first unless one was used before.
            if (i == 0 || models.get(i).path.equals(prefs.lastModel())) checkedId = rb.getId();
        }
        group.check(checkedId);

        new MaterialAlertDialogBuilder(this)
                .setTitle(R.string.choose_model)
                .setView(view)
                .setPositiveButton(R.string.upload, (d, w) -> {
                    RadioButton chosen = group.findViewById(group.getCheckedRadioButtonId());
                    String path = (String) chosen.getTag();
                    prefs.setLastModel(path);
                    runUpload(item, s, base, path, append.isChecked());
                })
                .setNegativeButton(R.string.cancel, (d, w) -> {
                    upload.setEnabled(true);
                    bind(item, s);
                })
                .setOnCancelListener(d -> {
                    upload.setEnabled(true);
                    bind(item, s);
                })
                .show();
    }

    private void runUpload(View item, Session s, String base, String modelPath, boolean append) {
        TextView status = item.findViewById(R.id.status);
        Button upload = item.findViewById(R.id.uploadButton);
        Button delete = item.findViewById(R.id.deleteButton);
        LinearProgressIndicator progress = item.findViewById(R.id.progress);

        delete.setEnabled(false);
        progress.setVisibility(View.VISIBLE);
        progress.setProgressCompat(0, false);
        uploadStarted();

        Uploader.upload(base, s, modelPath, append, new Uploader.Listener() {
            @Override
            public void onUploadProgress(int percent) {
                progress.setProgressCompat(percent, true);
                status.setText(getString(R.string.status_uploading, percent));
            }

            @Override
            public void onProcessing(int percent, String currentStep) {
                progress.setProgressCompat(percent, true);
                status.setText(getString(R.string.status_processing, currentStep));
            }

            @Override
            public void onDone(String jobId, int total, int mild, int moderate, int severe) {
                s.markUploaded(base, jobId);
                progress.setVisibility(View.GONE);
                upload.setEnabled(true);
                delete.setEnabled(true);
                status.setText(getString(R.string.status_done, total, mild, moderate, severe));
                uploadFinished();
            }

            @Override
            public void onError(String message) {
                progress.setVisibility(View.GONE);
                upload.setEnabled(true);
                delete.setEnabled(true);
                status.setText(getString(R.string.status_failed, message));
                uploadFinished();
            }
        });
    }

    /** Keep the screen on while anything is uploading, so the phone doesn't sleep mid-transfer. */
    private void uploadStarted() {
        if (activeUploads++ == 0) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void uploadFinished() {
        if (--activeUploads == 0) getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    // ---------------------------------------------------------- Share/delete

    private void share(Session s) {
        ArrayList<Uri> uris = new ArrayList<>();
        for (File f : s.files()) {
            uris.add(FileProvider.getUriForFile(this, getPackageName() + ".files", f));
        }
        Intent intent = new Intent(Intent.ACTION_SEND_MULTIPLE)
                .setType("*/*")
                .putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(intent, getString(R.string.share_title)));
    }

    private void confirmDelete(Session s) {
        new MaterialAlertDialogBuilder(this)
                .setTitle(getString(R.string.delete_title, s.name))
                .setMessage(R.string.delete_body)
                .setPositiveButton(R.string.delete, (d, w) -> {
                    s.delete();
                    render();
                })
                .setNegativeButton(R.string.cancel, null)
                .show();
    }

    private void openMap() {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(prefs.serverUrl() + "/")));
        } catch (ActivityNotFoundException ignored) {
            // No browser installed; nothing useful to do.
        }
    }
}
