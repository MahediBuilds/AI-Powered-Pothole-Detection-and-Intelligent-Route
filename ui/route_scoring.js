/* Route scoring helpers for the route planner in index.html.

   Kept in their own file so they can be unit-tested with Node
   (tests/route_scoring.test.js). In the browser they are exposed as
   globals; under Node they are exported via module.exports. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function distancePointToSegmentMeters(p, a, b) {
    // approximate: project in a local flat-earth sense (fine at city scale)
    const toXY = (pt, origin) => [
      (pt[1] - origin[1]) * 111320 * Math.cos(origin[0] * Math.PI / 180),
      (pt[0] - origin[0]) * 110540,
    ];
    const [px, py] = toXY(p, a);
    const [bx, by] = toXY(b, a);
    const t = Math.max(0, Math.min(1, (px * bx + py * by) / (bx * bx + by * by || 1)));
    const projX = t * bx, projY = t * by;
    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
  }

  // True if the detection lies within radiusMeters of any segment of the route.
  function isNearRoute(detection, coords, radiusMeters) {
    const p = [detection.latitude, detection.longitude];
    for (let i = 0; i < coords.length - 1; i++) {
      if (distancePointToSegmentMeters(p, coords[i], coords[i + 1]) <= radiusMeters) return true;
    }
    return false;
  }

  // Counts each pothole at most once, however many route segments pass near it.
  // (Counting per segment inflated the totals, since OSRM's full-resolution
  // geometry puts many short segments within the radius of a single pothole.)
  function scoreRoute(coords, severeDetections, moderateDetections, radiusMeters) {
    return {
      severeCount: severeDetections.filter(d => isNearRoute(d, coords, radiusMeters)).length,
      moderateCount: moderateDetections.filter(d => isNearRoute(d, coords, radiusMeters)).length,
    };
  }

  return { distancePointToSegmentMeters, isNearRoute, scoreRoute };
});
