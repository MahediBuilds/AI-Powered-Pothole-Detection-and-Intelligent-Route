package com.potholes.roadrecorder;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Location;
import android.os.Looper;
import android.os.SystemClock;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;

/**
 * Streams high-accuracy location fixes (about 1 per second) into a {@link GeoTrack}.
 *
 * Every fix is kept for the life of the app (a few KB per hour), stamped with
 * Location.getElapsedRealtimeNanos(), which is when the fix was actually taken rather
 * than when it was delivered. Recordings later cut their window out of this track.
 */
final class GpsLogger {

    interface Listener {
        void onFix(Location location);
    }

    private static final long INTERVAL_MS = 1000;
    private static final long MIN_INTERVAL_MS = 500;

    private final Context context;
    private final FusedLocationProviderClient client;
    private final Listener listener;
    private final GeoTrack track = new GeoTrack();
    private Location lastLocation;
    private boolean running;

    private final LocationCallback callback = new LocationCallback() {
        @Override
        public void onLocationResult(@NonNull LocationResult result) {
            for (Location loc : result.getLocations()) {
                track.add(toFix(loc));
                lastLocation = loc;
            }
            if (lastLocation != null) listener.onFix(lastLocation);
        }
    };

    GpsLogger(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.client = LocationServices.getFusedLocationProviderClient(this.context);
        this.listener = listener;
    }

    boolean hasPermission() {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED;
    }

    /** Starts location updates; returns false if location permission isn't granted. */
    boolean start() {
        if (running) return true;
        if (!hasPermission()) return false;
        LocationRequest request = new LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, INTERVAL_MS)
                .setMinUpdateIntervalMillis(MIN_INTERVAL_MS)
                .setWaitForAccurateLocation(false)
                .build();
        try {
            client.requestLocationUpdates(request, callback, Looper.getMainLooper());
            running = true;
        } catch (SecurityException e) {
            return false;
        }
        return true;
    }

    void stop() {
        if (!running) return;
        client.removeLocationUpdates(callback);
        running = false;
    }

    GeoTrack track() {
        return track;
    }

    Location lastLocation() {
        return lastLocation;
    }

    /** Milliseconds since the last fix was taken, or Long.MAX_VALUE if there's been none. */
    long lastFixAgeMillis() {
        if (lastLocation == null) return Long.MAX_VALUE;
        return (SystemClock.elapsedRealtimeNanos() - lastLocation.getElapsedRealtimeNanos()) / 1_000_000;
    }

    private static GeoTrack.Fix toFix(Location loc) {
        return new GeoTrack.Fix(
                loc.getElapsedRealtimeNanos(),
                loc.getTime(),
                loc.getLatitude(),
                loc.getLongitude(),
                loc.hasAccuracy() ? loc.getAccuracy() : Double.NaN,
                loc.hasSpeed() ? loc.getSpeed() : Double.NaN,
                loc.hasBearing() ? loc.getBearing() : Double.NaN,
                loc.hasAltitude() ? loc.getAltitude() : Double.NaN);
    }
}
