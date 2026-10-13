// Run with: node --test tests/route_scoring.test.js
const test = require("node:test");
const assert = require("node:assert/strict");

const { distancePointToSegmentMeters, isNearRoute, scoreRoute } = require("../ui/route_scoring.js");

// A straight east-west road through Bengaluru, densely sampled every ~1 m
// (OSRM's overview=full geometry is similarly dense in city streets).
function denseRoute(lat, lonStart, lonEnd, points) {
  const coords = [];
  for (let i = 0; i < points; i++) {
    coords.push([lat, lonStart + (lonEnd - lonStart) * (i / (points - 1))]);
  }
  return coords;
}

const LAT = 12.9716;
const route = denseRoute(LAT, 77.5900, 77.5910, 110); // ~108 m long
const pothole = (latOffsetMeters, lon) => ({ latitude: LAT + latOffsetMeters / 110540, longitude: lon });

test("distance to a segment is the perpendicular distance in metres", () => {
  const d = distancePointToSegmentMeters([LAT + 10 / 110540, 77.5905], [LAT, 77.5900], [LAT, 77.5910]);
  assert.ok(Math.abs(d - 10) < 0.01, `expected ~10 m, got ${d}`);
});

test("distance past a segment's end is measured to the endpoint", () => {
  const d = distancePointToSegmentMeters([LAT, 77.5920], [LAT, 77.5900], [LAT, 77.5910]);
  assert.ok(d > 100, `expected >100 m, got ${d}`);
});

test("a pothole next to a densely sampled route is counted once", () => {
  const { severeCount, moderateCount } = scoreRoute(route, [pothole(5, 77.5905)], [], 20);
  assert.equal(severeCount, 1);
  assert.equal(moderateCount, 0);
});

test("each pothole within the radius counts once, ones outside are ignored", () => {
  const severe = [pothole(3, 77.5902), pothole(-8, 77.5907), pothole(50, 77.5905)];
  const moderate = [pothole(0, 77.5901), pothole(19, 77.5909)];
  assert.deepEqual(scoreRoute(route, severe, moderate, 20), { severeCount: 2, moderateCount: 2 });
});

test("isNearRoute handles a route with fewer than two points", () => {
  assert.equal(isNearRoute(pothole(0, 77.5900), [[LAT, 77.5900]], 20), false);
});
