---
layout: subchapter
title: Additional Exercises
custom_js:
- /assets/js/draw.js
- /assets/js/custom.js
- /assets/js/paper-full.min.js
custom_css:
- /assets/css/draw.css
---

Exercises that span the different possible algorithms

Some ideas:

 - Trilateration in 3D?
 - Increase / decrease the number of sensors for trilateration, do something with that?

### Trilateration example
~~~~
///fold:
var drawPoints = function(canvas, positions, radius, strokeColor){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(next[0], next[1], radius, strokeColor, "rgba(0, 0, 0, 0)");
  drawPoints(canvas, positions.slice(1), radius, strokeColor);
};

var drawPointsMultiRadius = function(canvas, positions, radii, strokeColor){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  var radius = radii[0];
  canvas.circle(next[0], next[1], radius, strokeColor, "rgba(0, 0, 0, 0)");
  drawPointsMultiRadius(canvas, positions.slice(1), radii.slice(1), strokeColor);
};

var distance = function(p1, p2) {
  var xdiff = p1[0] - p2[0];
  var ydiff = p1[1] - p2[1];
  return Math.sqrt(xdiff*xdiff + ydiff*ydiff);
};
///

var noisyObserve = function(loc, stationLoc, targetDist, noise) {
  var observedDist = distance(loc, stationLoc);
  factor(Gaussian({mu: targetDist, sigma: noise}).score(observedDist));
}

var trilaterate = function(stations, distances) {
  // Weak prior over location
  var loc = [gaussian(200, 100), gaussian(200, 100)];
  
  // Condition on noisy measurements
  var noise = 10;
  mapIndexed(function(i, stationLoc) {
    noisyObserve(loc, stationLoc, distances[i], noise);
  }, stations);
  
  return loc;
};

var stations = [
  [100, 150],
  [200, 250],
  [250, 100]
];
var distances = [60, 90, 100];

var posterior = Infer({
//     method: 'MCMC',
//     samples: 1000,
//     burn: 1000,
//     kernel: {
//       HMC : { steps: 10, stepSize: 0.1 }
//     }
    method: 'SMC',
    particles: 5000
  }, function() {
  return trilaterate(stations, distances);
});

var drawStationDists = function(canvas, stations, distances) {
  drawPoints(canvas, stations, 5, 'black');
  drawPointsMultiRadius(canvas, stations, distances, 'red');
}

var canvas = Draw(400, 400, true);
drawStationDists(canvas, stations, distances);
var postSamps = repeat(100, function() { sample(posterior); });
drawPoints(canvas, postSamps, 2, 'blue');
~~~~