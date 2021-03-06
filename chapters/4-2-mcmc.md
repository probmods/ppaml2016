---
layout: subchapter
title: Markov Chain Monte Carlo
custom_js:
- /assets/js/draw.js
- /assets/js/custom.js
- /assets/js/paper-full.min.js
custom_css:
- /assets/css/draw.css
---

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

var mus = [[100, 200], [200, 100], [200, 300]];
var sigmas = [[20, 5], [10, 10], [5, 20]];
var weights = [0.33, 0.33, 0.33];
var nObservations = 100;
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

var mus = [[100, 200], [200, 100], [200, 300]];
var sigmas = [[20, 5], [10, 10], [5, 20]];
var weights = [0.33, 0.33, 0.33];
var nObservations = 100;
var synthData = repeat(nObservations, function() {
  return genFrom2DGaussMixture(mus, sigmas, weights);
});
///

var gaussMixtureObs = function(observations) {
  var mus = repeat(3, function() {
    return [gaussian(200, 40), gaussian(200, 40)];
  });
  var sigmas = repeat(3, function() {
    return [gamma(1, 10), gamma(1, 10)];
  });
  var weights = dirichlet(Vector([1, 1, 1]));
  
  map(function(obs) {
    var i = discrete(weights);
    factor(Gaussian({mu: mus[i][0], sigma: sigmas[i][0]}).score(obs[0]));
    factor(Gaussian({mu: mus[i][1], sigma: sigmas[i][1]}).score(obs[1]));
  }, observations);
  
  return {
    mus: mus,
    sigmas: sigmas,
    weights: weights
  };
};

var post = Infer({method: 'MCMC', samples: 5000, onlyMAP: true}, function() {
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

// 'justSample' retains an array of samples as a property of the object
//    returned from inference
// samples are {value: , score: } objects
var post = Infer(
  { method: 'MCMC', samples: 500, justSample: true},
  function(){
    var lines = makeLines(4, []);
    var finalGeneratedImage = Draw(50, 50, false);
    drawLines(finalGeneratedImage, lines);
    var newScore = -targetImage.distance(finalGeneratedImage)/1000;
    factor(newScore);
    return lines
   });

map(function(samp) {
  var lines = samp.value;
  var img = Draw(50, 50, true);
  drawLines(img, lines);
}, post.samples);
~~~~

This program uses only a single `factor` at the end of `finalImgSampler`, rather than one per each line rendered as in the SMC version. The fact that MCMC supports such a pattern makes it well-suited for programs that invoke complicated, 'black-box' simulations in order to compute likelihoods. It also makes MCMC a good default go-to inference method for most programs. However, note that MCMC has difficulty inferring the position of all four lines, often producing results that have only three correctly positioned. The take-away here: when you can restructure the program to have multiple `factor` statements throughout, as opposed to one at the end, it is often a good idea to do so and to use SMC instead.

### Custom MH Proposals

By default, WebPPL's MH algorithm proposes a change to a random choice by resampling from its prior. While this is always correct, it isn't always the most efficient proposal strategy. In practice, it's often advantageous to make a proposal based on the current value of the random choice: for example, a small perturbation of the current value is more likely to be accepted than an independently resampled value.

WebPPL supports these sorts of custom proposals via an option `driftKernel` that can be provided to any call to `sample`. Below, we show an example proposal for discrete random choices that forces MH to choose a different value than the current one (by setting the probability of the current value to zero):

~~~~
var model = function() {
  var x = sample(Discrete({ps: [0.25, 0.25, 0.25, 0.25]}, {
    driftKernel: function(currVal) {
      var newps = ps.slice(0, currVal).concat([0]).concat(ps.slice(currVal+1));
      return Discrete({ps: newps});
    }
  }));
  return x;
}

Infer({method: 'MCMC', samples: 100}, model);
~~~~

You might try using this strategy with the Gaussian mixture model program above to see how it improves efficiency.

### Hamiltonian Monte Carlo

When the input to a `factor` statement is a function of multiple variables, those variables become correlated in the posterior distribution. If the induced correlation is particularly strong, MCMC can sometimes become 'stuck,' generating many very similar samples which result in a poor approximation of the true posterior. Take this example below, where we use a Gaussian likelihood factor to encourage ten uniform random numbers to sum to the value 5:

~~~~
var bin = function(x) {
  return Math.floor(x * 1000) / 1000;
};

var constrainedSumModel = function() {
  var xs = repeat(10, function() {
    return uniform(0, 1);
  });
  var targetSum = xs.length / 2;
  factor(Gaussian({mu: targetSum, sigma: 0.005}).score(sum(xs)));
  return map(bin, xs);
};

var post = Infer({
	method: 'MCMC',
	samples: 5000,
	callbacks: [MCMC_Callbacks.finalAccept]
}, constrainedSumModel);
var samps = repeat(10, function() { return sample(post); });
reduce(function(x, acc) {
	return acc + 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + x.toString() + '\n';
}, '', samps);
~~~~

Running this program produces some random samples from the computed posterior distribution over the list of ten numbers---you'll notice that they are all very similiar, despite there being many distinct ways for ten real numbers to sum to 5. This program also uses the `callbacks` option to `MCMC` to display the final acceptance ratio (i.e. the percentage of proposed samples that were accepted)--it should be around 1-2%, which is very inefficient.

To deal with situations like this one, WebPPL provides an implementation of [Hamiltonian Monte Carlo](http://docs.webppl.org/en/master/inference.html#kernels), or HMC. HMC automatically computes the gradient of the posterior with respect to the random choices made by the program. It can then use the gradient information to make coordinated proposals to all the random choices, maintaining posterior correlations. Below, we apply HMC to `constrainedSumModel`:

~~~~
///fold:
var bin = function(x) {
  return Math.floor(x * 1000) / 1000;
};

var constrainedSumModel = function() {
  var xs = repeat(10, function() {
    return uniform(0, 1);
  });
  var targetSum = xs.length / 2;
  factor(Gaussian({mu: targetSum, sigma: 0.005}).score(sum(xs)));
  return map(bin, xs);
};
///

var post = Infer({
	method: 'MCMC',
	samples: 100,
	callbacks: [MCMC_Callbacks.finalAccept],
	kernel: {
		HMC : { steps: 50, stepSize: 0.0025 }
	}
}, constrainedSumModel);
var samps = repeat(10, function() { return sample(post); });
reduce(function(x, acc) {
	return acc + 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + x.toString() + '\n';
}, '', samps);
~~~~

The approximate posterior samples produced by this program are more varied, and the final acceptance rate is much higher.

There are a couple of caveats to keep in mind when using HMC:

 - Its parameters can be extremely sensitive. Try increasing the `stepSize` option to `0.004` and seeing how the output samples degenerate. 
 - It is only applicable to continuous random choices, due to its gradient-based nature. You can still use HMC with models that include discrete choices, though: under the hood, this will alternate between HMC for the continuous choices and MH for the discrete choices.

## Particle Rejuvenation

MCMC can also be used to improve the output of SMC. Recall the example from the previous section which tries to match rendered lines to a target image:

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

var numParticles = 100;

var post = Infer(
  {method: 'SMC', particles: numParticles},
  function(){
    return makeLines(4, [], 0);
   });

repeat(numParticles, function() {
  var finalGeneratedImage = Draw(50, 50, true);
  var lines = sample(post);
  drawLines(finalGeneratedImage, lines);
});
~~~~

We observed before that the posterior samples from this program are all nearly identical. One way to combat this problem is to run MCMC for a small number of steps after each particle resampling step; this process is often termed 'particle rejuvenation'. To apply this technique in WebPPL, just provide an integer greater than zero to the `rejuvSteps` option for SMC:


~~~~
///fold:
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
///

var numParticles = 100;

var post = Infer(
  {method: 'SMC', particles: numParticles, rejuvSteps: 10},
  function(){
    return makeLines(4, [], 0);
   });

repeat(numParticles, function() {
  var finalGeneratedImage = Draw(50, 50, true);
  var lines = sample(post);
  drawLines(finalGeneratedImage, lines);
});
~~~~

You can also rejuvenate with HMC, by specifying HMC as the `rejuvKernel` option. See the WebPPL [documentation](http://docs.webppl.org/en/master/inference.html#smc) for more information.

## Exercises

### 1. Sampling Implicit Curves

The code box below uses rejection sampling to sample points along the boundary of an implicit curve defined by the `curve` function:

~~~~
var curve = function(x, y) {
  var x2 = x*x;
  var term1 = y - Math.pow(x2, 1/3);
  return x2 + term1*term1 - 1;
};
var xbounds = [-1, 1];
var ybounds = [-1, 1.6];

var xmu = 0.5 * (xbounds[0] + xbounds[1]);
var ymu = 0.5 * (ybounds[0] + ybounds[1]);
var xsigma = 0.5 * (xbounds[1] - xbounds[0]);
var ysigma = 0.5 * (ybounds[1] - ybounds[0]);

var model = function() {
  var x = gaussian(xmu, xsigma);
  var y = gaussian(ymu, ysigma);
  var c_xy = curve(x, y);
  condition(Math.abs(c_xy) < 0.01);
  return {x: x, y: y};
};

var post = Infer({method: 'rejection', samples: 1000}, model);
viz.auto(post);
~~~~

Try using MCMC instead of rejection sampling--you'll notice that it doesn't fare as well. Why does this happen, and how can you change the model (or the inference algorithm) to make MCMC perform better? There may be multiple ways to approach this problem.

Hints:

 - The documentation on WebPPL's primitive [Distributions](http://docs.webppl.org/en/master/distributions.html) may be helpful. Be aware that some distributions operate on [Vectors](http://docs.webppl.org/en/master/tensors.html), rather than arrays.

<!-- ~~~~
// Solution 1: Using multivariate gaussian
var curve = function(x, y) {
  var x2 = x*x;
  var term1 = y - Math.pow(x2, 1/3);
  return x2 + term1*term1 - 1;
};
var xbounds = [-1, 1];
var ybounds = [-1, 1.6];

var xmu = 0.5 * (xbounds[0] + xbounds[1]);
var ymu = 0.5 * (ybounds[0] + ybounds[1]);
var xsigma = 0.5 * (xbounds[1] - xbounds[0]);
var ysigma = 0.5 * (ybounds[1] - ybounds[0]);

var mu = Vector([xmu, ymu]);
var sigma = Vector([xsigma, ysigma]);

var model = function() {
  var xy = sample(DiagCovGaussian({mu: mu, sigma: sigma}));
  var x = T.get(xy, 0);
  var y = T.get(xy, 1);
  var c_xy = curve(x, y);
  condition(Math.abs(c_xy) < 0.01);
  return {x: x, y: y};
};

var post = Infer({method: 'MCMC', samples: 30000}, model);
viz.auto(post);
~~~~

~~~~
// Solution 2: Using HMC
var curve = function(x, y) {
  var x2 = x*x;
  var term1 = y - Math.pow(x2, 1/3);
  return x2 + term1*term1 - 1;
};
var xbounds = [-1, 1];
var ybounds = [-1, 1.6];

var xmu = 0.5 * (xbounds[0] + xbounds[1]);
var ymu = 0.5 * (ybounds[0] + ybounds[1]);
var xsigma = 0.5 * (xbounds[1] - xbounds[0]);
var ysigma = 0.5 * (ybounds[1] - ybounds[0]);

var model = function() {
  var x = gaussian(xmu, xsigma);
  var y = gaussian(ymu, ysigma);
  var c_xy = curve(x, y);
  condition(Math.abs(c_xy) < 0.01);
  return {x: x, y: y};
};

var post = Infer({
  method: 'MCMC',
  kernel: { HMC: { stepSize: 0.1, steps: 10 } },
  samples: 10000
}, model);
viz.auto(post);
~~~~ -->


### 2. Pathfinding

In this exercise, you'll use MCMC to infer paths in a 2D environment from a start position to a target end position. The paths should avoid any obstacles in the environment. The code below draws the environment, including the start and goal positions:

~~~~
///fold:
var drawCircle = function(canvas, center, radius, color) {
  canvas.circle(T.get(center, 0), T.get(center, 1), radius, color,
                "rgba(0, 0, 0, 0)");
};

var drawObstacles = function(canvas, obstacles, color){
  if (obstacles.length == 0) { return []; }
  drawCircle(canvas, obstacles[0].center, obstacles[0].radius, color);
  drawObstacles(canvas, obstacles.slice(1), color);
};

var circle = function(c, r) { return {center: c, radius: r}; };
///

var start = Vector([200, 390]);
var goal = Vector([375, 50]);
    
var obstacles = [
  circle(Vector([75, 300]), 30),
  circle(Vector([175, 330]), 40),
  circle(Vector([200, 200]), 60),
  circle(Vector([300, 150]), 55),
  circle(Vector([375, 175]), 25),
];

var canvas = Draw(400, 400, true);
drawCircle(canvas, start, 5, 'blue');
drawCircle(canvas, goal, 5, 'green');
drawObstacles(canvas, obstacles, 'red');
~~~~

In this case, a path is an ordered list of points in space. The paths your code infers should:

- Travel from the start position (blue circle) to the goal position (green circle)
- Avoid all obstacles (red circles)
- Be as short as possible
- Stay within the bounds of the drawing window at all times

The code box below sets things up for you, including several useful helper functions.

~~~~
///fold:
var drawCircle = function(canvas, center, radius, color) {
  canvas.circle(T.get(center, 0), T.get(center, 1), radius, color,
                "rgba(0, 0, 0, 0)");
};

var drawObstacles = function(canvas, obstacles, color){
  if (obstacles.length == 0) { return []; }
  drawCircle(canvas, obstacles[0].center, obstacles[0].radius, color);
  drawObstacles(canvas, obstacles.slice(1), color);
};

var drawPath = function(canvas, positions, color){
  if (positions.length <= 1) { return []; }
  var start = positions[0];
  var end = positions[1];
  canvas.line(T.get(start, 0), T.get(start, 1),
              T.get(end, 0), T.get(end, 1), 3, 0.5, color);
  drawPath(canvas, positions.slice(1), color);
};

var dot = function(v1, v2) {
  return T.get(T.dot(T.transpose(v1), v2), 0);
};

var norm = function(v) {
  return Math.sqrt(dot(v, v));
};

var circle = function(c, r) { return {center: c, radius: r}; };

var rayCircleIntersect = function(a, d, circ) {
  var c = circ.center;
  var r = circ.radius;
  var ac = T.sub(a, c);
  var A = dot(d, d);
  var B = 2 * dot(ac, d);
  var C = dot(ac, ac) - r*r;
  var discr = B*B - 4*A*C;
  // Negative discriminant; no intersection
  if (discr < 0) { return [false]; }
  // Zero discriminant (dual root); one intersection
  if (discr === 0) { return [true, -B / (2*A)]; }
  // Positive discriminant; two intersections
  var sqrtDiscr = Math.sqrt(discr);
  var t1 = (-B + sqrtDiscr) / (2*A);
  var t2 = (-B - sqrtDiscr) / (2*A);
  // No intersection if both are negative
  if (t1 < 0 && t2 < 0) { return [false]; }
  // Pick closest one if both are nonnegative
  if (t1 >=0 && t1 >= 0) { return [true, Math.min(t1, t2)]; }
  // Else return the one nonnegative one
  return [true, t1 >= 0 ? t1 : t2];
};

var lineSegIntersectsCircle = function(a, b, circ) {
  var d = T.sub(b, a);
  var result = rayCircleIntersect(a, d, circ);
  if (result[0] === false) { return false; }
  var t = result[1];
  return t <= 1;
};

var pathSegBlocked = function(a, b, obstacles) {
  if (obstacles.length === 0) { return false; }
  if (lineSegIntersectsCircle(a, b, obstacles[0])) {
    return true;
  }
  return pathSegBlocked(a, b, obstacles.slice(1));
};

var start = Vector([200, 390]);
var goal = Vector([375, 50]);
    
var obstacles = [
  circle(Vector([75, 300]), 30),
  circle(Vector([175, 330]), 40),
  circle(Vector([200, 200]), 60),
  circle(Vector([300, 150]), 55),
  circle(Vector([375, 175]), 25),
];
///

// Returns true if the path is blocked by the obstacles
var pathBlocked = function(path, obstacles)
///fold:
{
  if (path.length < 2) { return false; }
  var a = path[0];
  var b = path[1];
  if (pathSegBlocked(a, b, obstacles)) { return true; }
  return pathBlocked(path.slice(1), obstacles);
};
///

// Returns the traversal length of a path
var pathLength = function(path)
///fold:
{
  if (path.length < 2) { return 0; }
  var a = path[0];
  var b = path[1];
  return norm(T.sub(b, a)) + pathLength(path.slice(1));
};
///

// Return true if the path is entirely contained within the provided
//    x, y bounds.
var pathInsideWindow = function(path, bounds)
///fold:
{
  if (path.length === 0) { return true; }
  var xlo = bounds[0];
  var xhi = bounds[1];
  var ylo = bounds[2];
  var yhi = bounds[3];
  var p = path[0];
  var px = T.get(p, 0);
  var py = T.get(p, 1);
  if (px < xlo || px > xhi || py < ylo || py > yhi) { return false; }
  return pathInsideWindow(path.slice(1), bounds);
};
///

var bounds = [0, 400, 0, 400];

// TODO: Replace this with sample from inferred posterior using MCMC.
var path = [start, goal];

var canvas = Draw(400, 400, true);
drawCircle(canvas, start, 5, 'blue');
drawCircle(canvas, goal, 5, 'green');
drawObstacles(canvas, obstacles, 'red');
drawPath(canvas, path, 'black');
~~~~

Some hints:

 - Add constraints one at time, i.e. first make sure you can generate paths from start-to-goal, then make them avoid obstacles, then make them short.
 - How you choose to represent and sample paths from start-to-goal can have a big impact on inference efficiency.

If you can get things working, here are some optional challenges / extentions worth considering:

 - Play with different configurations of obstacles, start position, and goal position. What configurations make inference especially challenging?
 - Try using other inference algorithms (especially SMC). How does their behavior compare with that of MCMC?

<!-- ~~~~
// Solution

var killProb = 0.55;
var baseVariance = 200;
var subdividePath = function(start, end, level) {
  if (flip(killProb)) { return [start, end]; }
  var midpoint = T.mul(T.add(start, end), 0.5);
  var variance = baseVariance / level;
  var newpoint = Vector([
    gaussian(T.get(midpoint, 0), variance),
    gaussian(T.get(midpoint, 1), variance)
  ]);
  return subdividePath(start, newpoint, level + 1).concat(
    subdividePath(newpoint, end, level + 1)
  );
};

var findPath = function(start, goal, obstacles) {
  
  var path = subdividePath(start, goal, 1);
  
  // Be inside window
  condition(pathInsideWindow(path, [0, 400, 0, 400]));
  
  // Be short
  var plen = pathLength(path);
  factor(Gaussian({mu: 0, sigma: 1}).score(plen));
  
  // Avoid obstacles
  condition(!pathBlocked(path, obstacles));
  
  return path;
};

var post = Infer({method: 'MCMC', samples: 1000}, function() {
  return findPath(start, goal, obstacles);
});
var path = sample(post);
~~~~ -->




<!-- ### 2. Model Selection

This didn't work very well.

~~~~
var data = [{"x":0.13473403270505463,"y":-0.2629565322178335},{"x":0.9126369177285534,"y":0.9567046429764398},{"x":0.001420248082787973,"y":-0.5753426558716854},{"x":0.42985188074337366,"y":-0.04441769644042026},{"x":0.45237485704422314,"y":0.14989187006486818},{"x":0.2093600228968304,"y":-0.3165985794148729},{"x":0.45238804885365863,"y":-0.014727917758442143},{"x":0.23318134766221613,"y":-0.19465903856927458},{"x":0.8163403491720688,"y":0.24000279744450384},{"x":0.3377257287983323,"y":-0.09487503848640304},{"x":0.6367977764792434,"y":0.07421055600754164},{"x":0.8792156403767458,"y":-0.04841981411864199},{"x":0.9544359768103792,"y":0.06609901738248972},{"x":0.8521427581035771,"y":0.8593272351775502},{"x":0.9149114940169959,"y":0.2507301693758486},{"x":0.4892677240289921,"y":-0.39585175691073826},{"x":0.43776432749525573,"y":-0.040550651608737096},{"x":0.9367781826068432,"y":1.254289662212853},{"x":0.41616142663105843,"y":-0.36819575589861475},{"x":0.8269278562846453,"y":0.8438779241666086}];

viz.scatter(data);
~~~~

~~~~
///fold:
var data = [{"x":0.3455425789094027,"y":-0.1373162164331799},{"x":0.394310502049926,"y":-0.48369720957062085},{"x":0.9215603031923841,"y":0.5682190176360146},{"x":0.9912941891578686,"y":1.1601241372071083},{"x":0.837913225141722,"y":0.5829580542054823},{"x":0.33661620204789267,"y":-0.35167416354663056},{"x":0.058864384417884025,"y":-0.48841738417370284},{"x":0.9192393957676772,"y":0.9032936686910031},{"x":0.35393452324861285,"y":-0.4342151438066949},{"x":0.6623384686909153,"y":-0.007435940554668334},{"x":0.1892645577627515,"y":-0.3269172295557968},{"x":0.8333821943877054,"y":0.4698744158499081},{"x":0.41466964963696384,"y":-0.1808924698146419},{"x":0.8796573332384006,"y":0.5681766127465091},{"x":0.7135217936531182,"y":0.15003997494648874},{"x":0.33886590102787584,"y":-0.23354408852538033},{"x":0.34720798041752954,"y":-0.3516400712376001},{"x":0.2645384506679998,"y":-0.4168499458988008},{"x":0.6094061966180758,"y":0.3238385201413541},{"x":0.4093943716706563,"y":-0.3708483568007901}];
///

// var linear = function(x, c) {
//   return c[0]*x + c[1];
// };

// var quadratic = function(x, c) {
//   return c[0]*x*x + c[1]*x + c[2];
// };

// var cubic = function(x, c) {
//   return c[0]*x*x*x + c[1]*x*x + c[2]*x + c[3];
// };

var linear = function(x, c) {
  return c[1]*x + c[0];
};

var quadratic = function(x, c) {
  return c[2]*x*x + linear(x, c);
};

var cubic = function(x, c) {
  return c[3]*x*x*x + quadratic(x, c);
};

var models = [linear, quadratic, cubic];

var modelSelection = function() {
  var modelClass = randomInteger(3);
  var model = models[modelClass];
  var degree = 1 + modelClass;
  var ncoeffs = degree + 1;
  var c = repeat(ncoeffs, function() { return gaussian(0, 1); });
  map(function(datum) {
    var y_predicted = model(datum.x, c);
    var y_observed = datum.y;
    factor(Gaussian({mu: y_predicted, sigma: 0.1}).score(y_observed));
  }, data);
  return degree;
};

var post = Infer({method: 'MCMC', samples: 1000}, modelSelection);
viz.auto(post);
~~~~ -->

<!-- ~~~~
// Code used to generate the data
var linear = function(x, coeffs) {
  return coeffs[0] + coeffs[1]*x;
};

var quadratic = function(x, coeffs) {
  return linear(x, coeffs) + coeffs[2]*x*x;
};

var cubic = function(x, coeffs) {
  return quadratic(x, coeffs) + coeffs[3]*x*x*x;
};

var weights = [0.2, 0.5, 0.3];
var n = 20;

var genData = function() {
  var coeffs = repeat(4, function() {
    return gaussian(0, 1);
  });
  return repeat(n, function() {
    var x = uniform(0, 1);
    //var degree = 1 + randomInteger(3);
    // var y = (degree === 1) ? linear(x, coeffs) :
    //         (degree === 2) ? quadratic(x, coeffs) :
    //         (degree === 3) ? cubic(x, coeffs) :
    //         undefined;
    var y = cubic(x, coeffs);
    return {x: x, y: y + gaussian(0, 0.1)};
  });
};

var data = genData();
viz.scatter(data);
data;
~~~~ -->





<!-- 
### 2. Constraint-based Layout

I don't really like this example; it devolves into a really tedious and difficult parameter tuning exercise.

~~~~
///fold:
var drawCircles = function(canvas, positions){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.circle(T.get(next, 0), T.get(next, 1), 5, 'black', 'blue');
  drawCircles(canvas, positions.slice(1));
};

var dot = function(v1, v2) {
  return T.get(T.dot(T.transpose(v1), v2), 0);
}

var norm = function(v) {
  return Math.sqrt(dot(v, v));
};

var normalize = function(v) {
  return T.div(v, norm(v));
};

var distance = function(p1, p2) {
  return norm(T.sub(p2, p1));
};

var collinearity = function(p1, p2, p3) {
  var v1 = normalize(T.sub(p1, p2));
  var v2 = normalize(T.sub(p3, p2));
  var d = dot(v1, v2); // between [-1, 1]
  // perfect collinearity is d === -1
  return 0.5 * (-d + 1);
};
///

var nPoints = 8;
var targetDist = 50;

var model = function() {
  // Sample point positions
  var positions = repeat(nPoints, function() {
    return sample(DiagCovGaussian({mu: Vector([200, 200]),
                                   sigma: Vector([100, 100])}));
  });
  
  // Constraints
  mapIndexed(function(i, p1) {
    // Constraint 1: distance between subsequent points
    var p2 = positions[(i+1) % positions.length];
    var d = distance(p1, p2);
    factor(Gaussian({mu: targetDist, sigma: 2}).score(d));
    
    // Constraint 2: collinearity of consecutive triples
    // 'collinearity' returns a score between 0 and 1
    var p3 = positions[(i+2) % positions.length];
    var cl = collinearity(p1, p2, p3);
    factor(Gaussian({mu: 1, sigma: 0.02}).score(cl));
  }, positions);
  
  return positions;
};

// var post = Infer({method: 'MCMC', samples: 10000}, model);
var post = Infer({
  method: 'MCMC',
  kernel: { HMConly: { stepSize: 0.033, steps: 30 } },
  samples: 1000,
  callbacks: [MCMC_Callbacks.finalAccept]
}, model);

repeat(10, function() {
  var samp = sample(post);
  var canvas = Draw(400, 400, true);
  drawCircles(canvas, samp);
});

// mapIndexed(function(i, p1) {
//   display('-----------------')
//   // Constraint 1: distance between subsequent points
//   var p2 = samp[(i+1) % samp.length];
//   var d = distance(p1, p2);
//   display(d);

//   // Constraint 2: collinearity of consecutive triples
//   // 'collinearity' returns a score between 0 and 1
//   var p3 = samp[(i+2) % samp.length];
//   var cl = collinearity(p1, p2, p3);
//   display(cl);
// }, samp);


~~~~ -->

[Next: Variational Inference]({{ "/chapters/4-3-variational.html" | prepend: site.baseurl }})