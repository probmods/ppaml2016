---
layout: chapter
title: Approximate Inference Algorithms
description: "The various approximate inference algorithms WebPPL provides and the classes of programs for which they are each best suited."
custom_js:
- /assets/js/draw.js
- /assets/js/custom.js
- /assets/js/paper-full.min.js
custom_css:
- /assets/css/draw.css
---

<!-- - Content: Algorithms
  - Rejection
  - Enumeration
  - Particle filters
  - Basic MCMC
  - MCMC
    - Incremental
    - HMC
  - Variational inference
  - Challenges: give models without Infer options and ask students to choose algorithms to make inference work. -->

Thus far, we have used exact inference in the form of `'enumerate'` to compute the posterior distributions of probabilistic programs. While enumeration can solve many interesting problems, it struggles when faced with the following scenarios:

### Continuous random choices
Such as `gaussian` and `gamma`. Such choices can take on an infinite number of possible values, so it is not possible to enumerate all of them. Trying to enumerate this program, for example

~~~~
var gaussianModel = function() {
	return gaussian(0, 1)
};
Infer({method: 'enumerate'}, gaussianModel);
~~~~

causes a runtime error.

### Very large state spaces
As a program makes more random choices, and as these choices gain more possible values, the number of possible execution paths through the program grows exponentially. Explicitly enumerating all of these paths can be prohibitively expensive. For instance, consider this program which computes the posterior distribution on rendered 2D lines, conditioned on those lines approximately matching a target image:

~~~~
var targetImage = Draw(50, 50, true);
loadImage(targetImage, "/ppaml2016/assets/img/box.png");
~~~~

~~~~
///fold:
var targetImage = Draw(50, 50, false);
loadImage(targetImage, "/ppaml2016/assets/img/box.png");

var drawLines = function(drawObj, lines){
  var line = lines[0];
  drawObj.line(line[0], line[1], line[2], line[3]);
  if (lines.length > 1) {
    drawLines(drawObj, lines.slice(1));
  }
};
///

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
};

var lineDist = Infer(
  { method: 'enumerate', strategy: 'depthFirst', maxExecutions: 10 },
  function(){
    var lines = makeLines(4, [], 0);
    var finalGeneratedImage = Draw(50, 50, true);
    drawLines(finalGeneratedImage, lines);
    return lines;
  });

viz.table(lineDist);
~~~~

Running this program, we can see that enumeration starts by growing a line from the bottom-right corner of the image, and then proceeds to methodically plot out every possible line length that could be generated. These are all fairly terrible at matching the target image, and there are billions more states like them that enumeration would have to wade through in order to find those few that have high probability.

In these situations, we can instead use one of WebPPL's many approximate, sampling-based inference algorithms.

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

TODO: Exercises. Some ideas below

- Introduce guides, ask them to try to make a guide for the trajectory program (i.e. using one-step lookahead on the observation data?)


## Markov Chain Monte Carlo

Outline:

- Example where all randomness happens before all factors, motivate need ('black box, can use whenever,' etc.)
- Basic MCMC
- Vision example with MCMC - note different behavior from SMC (also note: MCMC good for simulation-based models)
- Custom proposal functions (drift, probably, as example) to get better performance.
- Mode lock (toy) example, introduce HMC
- More practical HMC example?
- Rejuvenation - vision example, or...?
- Exercises

As mentioned above, SMC often works well when random choices are interleaved with evidence. However, there are many useful models that do not conform to this structure. Often, a model will perform all random choices up-front, followed by one or more `factor` statements.  In such settings, [Markov Chain Monte Carlo (MCMC)](http://docs.webppl.org/en/master/inference.html#mcmc) methods typically work better. Whereas SMC evovles a collection of multiple samples approximately distributed according to the posterior, MCMC instead iteratively mutates a single sample such that over time, the sequence of mutated samples (also called a 'chain') is approximately distributed according to the posterior.

As a first example, we'll consider a simple program for modeling data generated by mixture of three 2D Gaussians.
We'll first generate some synthetic observations:

~~~~
///fold:
var drawPoints = function(canvas, positions, strokeColor){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(next[0], next[1], 5, strokeColor, "white");
  drawPoints(canvas, positions.slice(1), strokeColor);
};
///

var genFrom2DGaussMixture = function(mus, sigmas, weights) {
  var i = discrete(weights);
  return [
    gaussian(mus[i][0], sigmas[i][0]),
    gaussian(mus[i][1], sigmas[i][1]),
  ];
};

var mus = [[100, 200], [200, 100], [200, 200]];
var sigmas = [[20, 5], [10, 10], [5, 20]];
var weights = [0.2, 0.5, 0.3];
var nObservations = 50;
var synthData = repeat(nObservations, function() {
  return genFrom2DGaussMixture(mus, sigmas, weights);
});

var canvas = Draw(400, 400, true);
drawPoints(canvas, synthData, 'red');
~~~~

The program below then infers the parameters of the Gaussian mixture which generated this data. Note that it samples all of the model parameters up-front and then iterates over observed data to compute its likelihood. We'll use Markov Chain Monte Carlo (specified by the `'MCMC'` argument to `Infer`) to perform inference. By default, this method uses the [Metropolis-Hastings](http://docs.webppl.org/en/master/inference.html#wingate11) algorithm, mutating samples by changing one random choice at a time. Note that since we're only interested in a *maximum a posteriori* estimate of the model parameters, we provide the `onlyMAP` option, as well. This gives a slight performance boost.

~~~~
///fold:
var drawPoints = function(canvas, positions, strokeColor){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(next[0], next[1], 5, strokeColor, "white");
  drawPoints(canvas, positions.slice(1), strokeColor);
};

var genFrom2DGaussMixture = function(mus, sigmas, weights) {
  var i = discrete(weights);
  return [
    gaussian(mus[i][0], sigmas[i][0]),
    gaussian(mus[i][1], sigmas[i][1]),
  ];
};

var mus = [[100, 200], [200, 100], [200, 200]];
var sigmas = [[20, 5], [10, 10], [5, 20]];
var weights = [0.2, 0.5, 0.3];
var nObservations = 50;
var synthData = repeat(nObservations, function() {
  return genFrom2DGaussMixture(mus, sigmas, weights);
});
///

var logsumexp = function(xs) {
  return Math.log(sum(map(function(x) { return Math.exp(x); }, xs)));
};

var gaussMixtureObs = function(observations) {
  var mus = repeat(3, function() {
    return [gaussian(200, 40), gaussian(200, 40)];
  });
  var sigmas = repeat(3, function() {
    return [gamma(1, 10), gamma(1, 10)];
  });
  var weights = dirichlet(Vector([1, 1, 1])).toFlatArray();
  
  map(function(obs) {
    // Compute likelihood of observation by summing over all three gaussians
    var xscores = mapIndexed(function(i, w) {
      return Gaussian({mu: mus[i][0], sigma: sigmas[i][0]}).score(obs[0])
             + Math.log(w);
    }, weights);
    var yscores = mapIndexed(function(i, w) {
      return Gaussian({mu: mus[i][1], sigma: sigmas[i][1]}).score(obs[1])
             + Math.log(w);
    }, weights);
    factor(logsumexp(xscores));
    factor(logsumexp(yscores));
  }, observations);
  
  return {
    mus: mus,
    sigmas: sigmas,
    weights: weights
  };
};

var post = Infer({method: 'MCMC', samples: 1000, onlyMAP: true}, function() {
  return gaussMixtureObs(synthData);
});
var params = sample(post);
var dataFromLearnedModel = repeat(nObservations, function() {
  genFrom2DGaussMixture(params.mus, params.sigmas, params.weights);
});

var canvas = Draw(400, 400, true);
drawPoints(canvas, synthData, 'red');
drawPoints(canvas, dataFromLearnedModel, 'blue');
params;
~~~~

We can also use MCMC to perform inference on the line drawing program from before. Note the different behavior of MCMC, how the results are a sequence of images, each a slight modification on the last:

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

var makeLines = function(n, lines){
  var x1 = randomInteger(50);
  var y1 = randomInteger(50);
  var x2 = randomInteger(50);
  var y2 = randomInteger(50);
  var newLines = lines.concat([[x1, y1, x2, y2]]);
  return (n==1) ? newLines : makeLines(n-1, newLines);
}

var finalImgSampler = Infer(
  { method: 'MCMC', samples: 500},
  function(){
    var lines = makeLines(4, []);
    var finalGeneratedImage = Draw(50, 50, true);
    drawLines(finalGeneratedImage, lines);
    var newScore = -targetImage.distance(finalGeneratedImage)/1000;
    factor(newScore);
    return lines
   });

var finalImage = Draw(100, 100, false);
var finalLines = sample(finalImgSampler);
drawLines(finalImage, finalLines);
~~~~

This program uses only a single `factor` at the end of `finalImgSampler`, rather than one per each line rendered as in the SMC version. The fact that MCMC supports such a pattern makes it well-suited for programs that invoke complicated, 'black-box' simulations in order to compute likelihoods. It also makes MCMC a good default go-to inference method for most programs.

### Custom MH Proposals(?)

Do we want this to be a thing?

### Hamiltonian Monte Carlo

When the input to a `factor` statement is a function of multiple variables, those variables become correlated in the posterior distribution. If the induced correlation is particularly strong, MCMC can sometimes become 'stuck,' generating many very similar samples which result in a poor approximation of the true posterior. Take this example below, where we use a Gaussian likelihood factor to encourage ten uniform random numbers to sum to the value 5:

~~~~
var bin = function(x) {
  return Math.floor(x * 1000) / 1000;
};

var model = function() {
  var xs = repeat(10, function() {
    return uniform(0, 1);
  });
  var targetSum = xs.length / 2;
  factor(Gaussian({mu: targetSum, sigma: 0.001}).score(sum(xs)));
  return map(bin, xs);
};

var post = Infer({method: 'MCMC', samples: 5000}, model);
repeat(10, function() { return sample(post); });
~~~~

Running this program produces some random samples from the computed posterior distribution over the list of ten numbers---you'll notice that they are all very similiar, despite there being many distinct ways for ten real numbers to sum to 5.


## Variational Inference