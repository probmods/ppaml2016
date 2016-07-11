---
layout: subchapter
title: Rejection Sampling and Particle Filtering
custom_js:
- /assets/js/draw.js
- /assets/js/custom.js
- /assets/js/paper-full.min.js
custom_css:
- /assets/css/draw.css
---

## Rejection Sampling

WebPPL provides a routine for basic [rejection sampling](http://docs.webppl.org/en/master/inference.html#rejection-sampling). Here's a simple program that uses rejection sampling to compute the uniform distribution of 2D points over the unit circle:

~~~~

var uniformCircle = function() {
	var x = uniform(-1, 1)
	var y = uniform(-1, 1)
	condition(x*x + y*y < 1)
	return {x: x, y: y}
};

viz.auto(Infer({method: 'rejection', samples: 1000}, uniformCircle));
~~~~

## Particle Filtering

In cases where the evidence is unlikely (e.g. an observed Gaussian variable taking on a particular value), it is best to use `factor` instead of `condition`, which allows posterior computation through likelihood-weighted samples. Here is an example of inferring the 2D location of a static object given several noisy observations of its position, i.e. from a radar detector:

~~~~
///fold:
var drawPoints = function(canvas, positions, strokeColor){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(next[0], next[1], 5, strokeColor, "white");
  drawPoints(canvas, positions.slice(1), strokeColor);
};
///

var observe = function(pos, obs) {
	factor(Gaussian({mu: pos[0], sigma: 5}).score(obs[0]));
	factor(Gaussian({mu: pos[1], sigma: 5}).score(obs[1]));
};

var radarStaticObject = function(observations) {
	var pos = [gaussian(200, 50), gaussian(200, 50)];
	map(function(obs) { observe(pos, obs); }, observations);
	return pos;
};

var observations = [[120, 75], [110, 100], [125, 70]];
var posterior = Infer({method: 'SMC', particles: 1000}, function() {
	return radarStaticObject(observations);
});
var posEstimate = sample(posterior);

var canvas = Draw(400, 400, true);
drawPoints(canvas, observations, 'red');
drawPoints(canvas, [posEstimate], 'blue');
~~~~

Note that this code uses the `'SMC'` inference method; this corresponds to the [Sequential Monte Carlo](http://docs.webppl.org/en/master/inference.html#smc) family of algorithms, also known as particle filters. These algorithms maintain a collection of samples (particles) that are resampled upon encountering new evidence; hence, they are especially well-suited to programs that interleave random choices with `factor` statements. Below, we extend the the radar detection example to infer the trajectory of a moving object, rather than the position of a static one--the program receives a sequence of noisy observations and must infer the underlying sequence of true object locations. Our program assumes that the object's motion is governed by a momentum term which is a function of its previous two locations; this tends to produce smoother trajectories.

The code below generates synthetic observations from a randomly-sampled underlying trajectory:

~~~~
///fold:
var drawLines = function(canvas, start, positions){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.line(start[0], start[1], next[0], next[1], 4, 0.2);
  drawLines(canvas, next, positions.slice(1));
};

var drawPoints = function(canvas, positions){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(next[0], next[1], 2, "red", "white");
  drawPoints(canvas, positions.slice(1));
};
///

var genObservation = function(pos){
  return map(
    function(x){ return gaussian(x, 5); },
	pos
  );
};

var init = function(){
	var state1 = [gaussian(200, 1), gaussian(200, 1)];
	var state2 = [gaussian(200, 1), gaussian(200, 1)];
	var states = [state1, state2];
  	var observations = map(genObservation, states);
	return {
		states: states,
		observations: observations
	};
};

var transition = function(lastPos, secondLastPos){
  return map2(
    function(lastX, secondLastX){
      var momentum = (lastX - secondLastX) * .7;
      return gaussian(lastX + momentum, 3);
    },
	lastPos,
    secondLastPos
  );
};

var trajectory = function(n) {
  var prevData = (n == 2) ? init() : trajectory(n - 1);
  var prevStates = prevData.states;
  var prevObservations = prevData.observations;
  var newState = transition(last(prevStates), secondLast(prevStates));
  var newObservation = genObservation(newState);
  return {
    states: prevStates.concat([newState]),
    observations: prevObservations.concat([newObservation])
  }
};

var numSteps = 80;
var synthObservations = trajectory(numSteps).observations;
var canvas = Draw(400, 400, true)
drawPoints(canvas, synthObservations)
~~~~

We can then use `'SMC'` inference to estimate the underlying trajectory which generated a synthetic observation sequence:

~~~~
///fold:
var drawLines = function(canvas, start, positions){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.line(start[0], start[1], next[0], next[1], 4, 0.2);
  drawLines(canvas, next, positions.slice(1));
};

var drawPoints = function(canvas, positions){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(next[0], next[1], 2, "red", "white");
  drawPoints(canvas, positions.slice(1));
};

var genObservation = function(pos){
  return map(
    function(x){ return gaussian(x, 5); },
	pos
  );
};

var init = function(){
	var state1 = [gaussian(200, 1), gaussian(200, 1)];
	var state2 = [gaussian(200, 1), gaussian(200, 1)];
	var states = [state1, state2];
  	var observations = map(genObservation, states);
	return {
		states: states,
		observations: observations
	};
};

var transition = function(lastPos, secondLastPos){
  return map2(
    function(lastX, secondLastX){
      var momentum = (lastX - secondLastX) * .7;
      return gaussian(lastX + momentum, 3);
    },
	lastPos,
    secondLastPos
  );
};

var trajectory = function(n) {
  var prevData = (n == 2) ? init() : trajectory(n - 1);
  var prevStates = prevData.states;
  var prevObservations = prevData.observations;
  var newState = transition(last(prevStates), secondLast(prevStates));
  var newObservation = genObservation(newState);
  return {
    states: prevStates.concat([newState]),
    observations: prevObservations.concat([newObservation])
  }
};
///

var observe = function(pos, trueObs){
  return map2(
    function(x, trueObs) {
    	return factor(Gaussian({mu: x, sigma: 5}).score(trueObs));
    },    
	pos,
    trueObs
  );
};

var initWithObs = function(trueObs){
	var state1 = [gaussian(200, 1), gaussian(200, 1)];
	var state2 = [gaussian(200, 1), gaussian(200, 1)];
  	var obs1 = observe(state1, trueObs[0]);
  	var obs2 = observe(state2, trueObs[1]);
	return {
		states: [state1, state2],
		observations: [obs1, obs2]
	}
};

var trajectoryWithObs = function(n, trueObservations) {
  var prevData = (n == 2) ?
  	initWithObs(trueObservations.slice(0, 2)) :
    trajectoryWithObs(n-1, trueObservations.slice(0, n-1));
  var prevStates = prevData.states;
  var prevObservations = prevData.observations;
  var newState = transition(last(prevStates), secondLast(prevStates));
  var newObservation = observe(newState, trueObservations[n-1]);
  return {
    states: prevStates.concat([newState]),
    observations: prevObservations.concat([newObservation])
  }
};

var numSteps = 80;

// Gen synthetic observations
var trueObservations = trajectory(numSteps).observations;

// Infer underlying trajectory using particle filter
var posterior = ParticleFilter(
  function(){
    return trajectoryWithObs(numSteps, trueObservations);
  }, 10) // Try reducing the number of samples to 1!

var inferredTrajectory = sample(posterior).states;

// Draw model output
var canvas = Draw(400, 400, true)
drawPoints(canvas, trueObservations)
drawLines(canvas, inferredTrajectory[0], inferredTrajectory.slice(1))
~~~~

Particle filtering also works well for the program we showed early which matches rendered lines to a target image:

~~~~
var targetImage = Draw(50, 50, false);
loadImage(targetImage, "/ppaml2016/assets/img/box.png")

var drawLines = function(drawObj, lines){
  var line = lines[0];
  drawObj.line(line[0], line[1], line[2], line[3]);
  if (lines.length > 1) {
    drawLines(drawObj, lines.slice(1));
  }
}

var makeLines = function(n, lines, prevScore){
  // Add a random line to the set of lines
  var x1 = randomInteger(50);
  var y1 = randomInteger(50);
  var x2 = randomInteger(50);
  var y2 = randomInteger(50);
  var newLines = lines.concat([[x1, y1, x2, y2]]);
  // Compute image from set of lines
  var generatedImage = Draw(50, 50, false);
  drawLines(generatedImage, newLines);
  // Factor prefers images that are close to target image
  var newScore = -targetImage.distance(generatedImage)/1000;
  factor(newScore - prevScore);
  generatedImage.destroy();
  // Generate remaining lines (unless done)
  return (n==1) ? newLines : makeLines(n-1, newLines, newScore);
}

Infer(
  {method: 'SMC', particles: 100},
  function(){
    var lines = makeLines(4, [], 0);
    var finalGeneratedImage = Draw(50, 50, true);
	drawLines(finalGeneratedImage, lines);
   })
~~~~

Try running this program multiple times. Note that while each run produces different outputs, within a run, all of the output particles look extremely similar. We will return to this issue later on in the next section.

## Exercises

TODO. Some ideas below

- Introduce guides, ask them to try to make a guide for the trajectory program (i.e. using one-step lookahead on the observation data?)

[Next: Markov Chain Monte Carlo]({{ "/chapters/4-2-mcmc.html" | prepend: site.baseurl }})