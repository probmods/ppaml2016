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
  var pos = [gaussian(200, 100), gaussian(200, 100)];
  map(function(obs) { observe(pos, obs); }, observations);
  return pos;
};

// var observations = [[120, 75], [110, 100], [125, 70]];
var observations = repeat(20, function() {
  return [ gaussian(100, 20), gaussian(100, 20) ];
});
var posterior = Infer({method: 'SMC', particles: 1000}, function() {
  return radarStaticObject(observations);
});
var posEstimate = sample(posterior);

var canvas = Draw(400, 400, true);
drawPoints(canvas, observations, 'red');
drawPoints(canvas, [posEstimate], 'blue');
posEstimate;
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

[Here](http://dritchie.github.io/web-procmod/) is a more complex example of using SMC to generate a 3D model that matches a given volumetric target (Note: this demo uses a much older version of WebPPL, so some of the syntax is different / not compatible with the code we've been working with).

## Exercises

### 1. Robot Localization

Suppose we have a mobile robot moving around a 2D environment enclosed by a collection of walls:

~~~~
///fold:
// Environment is a collection of walls, where walls are just horizontal or
//    vertical line segments

var WallType = { Vertical: 0, Horizontal: 1 };

var makeVerticalWall = function(start, end) {
  return {
    start: start,
    end: end,
    type: WallType.Vertical,
    x: start[0],
    ylo: Math.min(start[1], end[1]),
    yhi: Math.max(start[1], end[1])
  }
};

var makeHorizontalWall = function(start, end) {
  return {
    start: start,
    end: end,
    type: WallType.Horizontal,
    y: start[1],
    xlo: Math.min(start[0], end[0]),
    xhi: Math.max(start[0], end[0])
  };
}; 

var makeWall = function(start, end) {
  if (start[0] === end[0]) {
    return makeVerticalWall(start, end);
  } else if (start[1] === end[1]) {
    return makeHorizontalWall(start, end)
  } else {
    console.error('Wall provided was not horizontal or vertical!');
    factor(NaN);
  }
};

// Connects subsequent points
var makeWalls = function(points) {
  if (points.length <= 1) return [];
  var start = points[0];
  var end = points[1];
  return [makeWall(start, end)].concat(makeWalls(points.slice(1)));
};

var drawWalls = function(canvas, walls) {
  map(function(wall) {
    canvas.line(wall.start[0], wall.start[1], wall.end[0], wall.end[1],
      4, 1, 'black');
  }, walls);
  return null;
};
///

var walls = makeWalls([
  [50, 25],
  [50, 150],
  [150, 150],
  [150, 300],
  [50, 300],
  [50, 375],
  [225, 375],
  [225, 275],
  [350, 275],
  [350, 225],
  [225, 225],
  [225, 100],
  [375, 100],
  [375, 25],
  [175, 25],
  [175, 75],
  [125, 75],
  [125, 25],
  [50, 25]
]);

var canvas = Draw(400, 400, true);
drawWalls(canvas, walls);
~~~~

As it moves, the robot records observations of its environment through an onboard sensor; this sensor records the distance to the nearest wall in several directions. The sensor isn't perfect: its measurements are noisy, and it can't sense walls beyond a certain maximum distance.

First, write a program to generate motion trajectories this robot might follow if it moved randomly. 
The code box below provides several utility functions, including the transition function the robot obeys as it moves from time step to time step.
All you need to do is fill in the `genTrajectory` function. You should use the provided `collisionWithWall` function to ensure that the generated trajectories don't involve the robot walking through walls (hint: note that the function you're filling in is called via rejection sampling).

~~~~
///fold:
// Utilities

var lerp = function(a, b, t) {
  return (1-t)*a + t*b;
};

var polar2rect = function(r, theta) {
  return [r*Math.cos(theta), r*Math.sin(theta)];
};

var range = function(n) {
  return n === 0 ? [] : range(n-1).concat([n-1]);
};

var min = function(nums) {
  return reduce(function(x, accum) {
    return Math.min(x, accum);
  }, Infinity, nums);
};

// ----------------------------------------------------------------------------

// Vector math

var vec_sub = function(v1, v0) {
  return [
    v1[0] - v0[0],
    v1[1] - v0[1]
  ];
};

// ----------------------------------------------------------------------------

// Environment is a collection of walls, where walls are just horizontal or
//    vertical line segments

var WallType = { Vertical: 0, Horizontal: 1 };

var makeVerticalWall = function(start, end) {
  return {
    start: start,
    end: end,
    type: WallType.Vertical,
    x: start[0],
    ylo: Math.min(start[1], end[1]),
    yhi: Math.max(start[1], end[1])
  }
};

var makeHorizontalWall = function(start, end) {
  return {
    start: start,
    end: end,
    type: WallType.Horizontal,
    y: start[1],
    xlo: Math.min(start[0], end[0]),
    xhi: Math.max(start[0], end[0])
  };
}; 

var makeWall = function(start, end) {
  if (start[0] === end[0]) {
    return makeVerticalWall(start, end);
  } else if (start[1] === end[1]) {
    return makeHorizontalWall(start, end)
  } else {
    console.error('Wall provided was not horizontal or vertical!');
    factor(NaN);
  }
};

// Connects subsequent points
var makeWalls = function(points) {
  if (points.length <= 1) return [];
  var start = points[0];
  var end = points[1];
  return [makeWall(start, end)].concat(makeWalls(points.slice(1)));
};

// ----------------------------------------------------------------------------

// Intersection tests

var intersectWall = function(start, dir, wall) {
  if (wall.type === WallType.Vertical) {
    var t = (wall.x - start[0]) / dir[0];
    var y = start[1] + dir[1]*t;
    return (y >= wall.ylo && y <= wall.yhi) ? t : Infinity;
  } else if (wall.type === WallType.Horizontal) {
    var t = (wall.y - start[1]) / dir[1];
    var x = start[0] + dir[0]*t;
    return (x >= wall.xlo && x <= wall.xhi) ? t : Infinity;
  }
};


var intersectWalls = function(start, dir, walls) {
  return min(filter(function(t) {
    return t >= 0;
  }, map(function(wall) {
    return intersectWall(start, dir, wall);
  }, walls)));
};

// ----------------------------------------------------------------------------

// Rendering

var drawWalls = function(canvas, walls) {
  map(function(wall) {
    canvas.line(wall.start[0], wall.start[1], wall.end[0], wall.end[1],
      4, 1, 'black');
  }, walls);
  return null;
};

var drawTrajectory = function(canvas, positions, color){
  if (positions.length <= 1) { return []; }
  var start = positions[0];
  var end = positions[1];
  canvas.line(start[0], start[1], end[0], end[1], 3, 0.5, color);
  drawTrajectory(canvas, positions.slice(1), color);
};

// ----------------------------------------------------------------------------

// Previously defined

var walls = makeWalls([
  [50, 25],
  [50, 150],
  [150, 150],
  [150, 300],
  [50, 300],
  [50, 375],
  [225, 375],
  [225, 275],
  [350, 275],
  [350, 225],
  [225, 225],
  [225, 100],
  [375, 100],
  [375, 25],
  [175, 25],
  [175, 75],
  [125, 75],
  [125, 25],
  [50, 25]
]);

///

// Robot motion prior is a semi-markov random walk
// (or just a random walk, when there's only been one prior timestep)
var transition = function(lastPos, secondLastPos){
  if (!secondLastPos) {
    return map(
      function(lastX) {
        return gaussian(lastX, 10);
      },
      lastPos
    );
  } else {
    return map2(
      function(lastX, secondLastX){
        var momentum = (lastX - secondLastX) * .9;
        return gaussian(lastX + momentum, 4);
      },
      lastPos,
      secondLastPos
    );
  }
};

// Sensor is a set of n raycasters that shoot rays outward at evenly-spaced
//    angular intervals.
// Has a maximum sensor distance (beyond which it just reports max distance)
// Also degrades (e.g. exhibits greater noise) with distance
var makeSensor = function(n, maxDist, minNoise, maxNoise) {
  return {
    rayDirs: map(function(i) {
      var ang = 2 * Math.PI * (i/n);
      return polar2rect(1, ang);
    }, range(n)),
    maxDist: maxDist,
    minNoise: minNoise,
    maxNoise: maxNoise
  };
};

// Generate a sensor observation
var sense = function(sensor, curPos, walls) {
  return map(function(dir) {
    var trueDist = intersectWalls(curPos, dir, walls);
    var cappedDist = Math.min(trueDist, sensor.maxDist);
    var t = Math.min(1, cappedDist/sensor.maxDist);
    var noise = lerp(sensor.minNoise, sensor.maxNoise, t);
    return gaussian(cappedDist, noise);
  }, sensor.rayDirs);
};

// Returns true if the robot collides with a wall by moving from secondLastPos
//    to lastPos
var collisionWithWall = function(lastPos, secondLastPos, walls) {
///fold:
  var dir = vec_sub(lastPos, secondLastPos);
  var helper = function(walls) {
    if (walls.length === 0) return false;
    var wall = walls[0];
    var t = intersectWall(secondLastPos, dir, wall);
    if (t >= 0 && t <= 1) return true;
    return helper(walls.slice(1));
  };
  return helper(walls);
///
};

var genTrajectory = function(n, initPos, sensor, walls) {
  // Fill in
};


var sensor = makeSensor(8, 40, 0.1, 3);
var initPos = [75, 50];
var trajectoryLength = 50;

var post = Infer({method: 'rejection', samples: 1}, function() {
  return genTrajectory(trajectoryLength, initPos, sensor, walls);
});
var trajectory = sample(post);

wpEditor.put('sensor', sensor);
wpEditor.put('initPos', initPos);
wpEditor.put('trajectoryLength', trajectoryLength);
wpEditor.put('trajectory', trajectory);

var canvas = Draw(400, 400, true);
drawWalls(canvas, walls);
drawTrajectory(canvas, trajectory.states, 'blue');

~~~~

<!-- ~~~~
// Solution
var init = function(initPos, sensor, walls) {
  return {
    states: [ initPos ],
    observations: [ sense(sensor, initPos, walls) ]
  };
};

var genTrajectory = function(n, initPos, sensor, walls) {
  var helper = function(n) {
        var prevData = (n == 1) ? init(initPos, sensor, walls) : helper(n-1);
    var prevStates = prevData.states;
    var prevObs = prevData.observations;
    var newState = transition(last(prevStates), secondLast(prevStates));
    var collision = collisionWithWall(newState, last(prevStates), walls);
    factor(collision ? -Infinity : 0);
    var newObs = sense(sensor, newState, walls);
    return {
      states: prevStates.concat([newState]),
      observations: prevObs.concat([newObs])
    };
  };
  return helper(n);
};
~~~~ -->

Note that this program takes quite a while to generate trajectories. Try switching from rejection sampling to SMC and play with the number of particles used. How does the behavior of SMC differ from that of rejection sampling, and why?

Now that you can generate random plausible robot motion trajectories, see if you can infer a trajectory given only the sensor observations a robot received while it was moving. The code box below sets up the scenario for you; you just need to fill in the `inferTrajectory` function.

~~~~
///fold:
// Utilities

var lerp = function(a, b, t) {
  return (1-t)*a + t*b;
};

var polar2rect = function(r, theta) {
  return [r*Math.cos(theta), r*Math.sin(theta)];
};

var range = function(n) {
  return n === 0 ? [] : range(n-1).concat([n-1]);
};

var min = function(nums) {
  return reduce(function(x, accum) {
    return Math.min(x, accum);
  }, Infinity, nums);
};

// ----------------------------------------------------------------------------

// Vector math

var vec_sub = function(v1, v0) {
  return [
    v1[0] - v0[0],
    v1[1] - v0[1]
  ];
};

// ----------------------------------------------------------------------------

// Environment is a collection of walls, where walls are just horizontal or
//    vertical line segments

var WallType = { Vertical: 0, Horizontal: 1 };

var makeVerticalWall = function(start, end) {
  return {
    start: start,
    end: end,
    type: WallType.Vertical,
    x: start[0],
    ylo: Math.min(start[1], end[1]),
    yhi: Math.max(start[1], end[1])
  }
};

var makeHorizontalWall = function(start, end) {
  return {
    start: start,
    end: end,
    type: WallType.Horizontal,
    y: start[1],
    xlo: Math.min(start[0], end[0]),
    xhi: Math.max(start[0], end[0])
  };
}; 

var makeWall = function(start, end) {
  if (start[0] === end[0]) {
    return makeVerticalWall(start, end);
  } else if (start[1] === end[1]) {
    return makeHorizontalWall(start, end)
  } else {
    console.error('Wall provided was not horizontal or vertical!');
    factor(NaN);
  }
};

// Connects subsequent points
var makeWalls = function(points) {
  if (points.length <= 1) return [];
  var start = points[0];
  var end = points[1];
  return [makeWall(start, end)].concat(makeWalls(points.slice(1)));
};

// ----------------------------------------------------------------------------

// Intersection tests

var intersectWall = function(start, dir, wall) {
  if (wall.type === WallType.Vertical) {
    var t = (wall.x - start[0]) / dir[0];
    var y = start[1] + dir[1]*t;
    return (y >= wall.ylo && y <= wall.yhi) ? t : Infinity;
  } else if (wall.type === WallType.Horizontal) {
    var t = (wall.y - start[1]) / dir[1];
    var x = start[0] + dir[0]*t;
    return (x >= wall.xlo && x <= wall.xhi) ? t : Infinity;
  }
};


var intersectWalls = function(start, dir, walls) {
  return min(filter(function(t) {
    return t >= 0;
  }, map(function(wall) {
    return intersectWall(start, dir, wall);
  }, walls)));
};

// ----------------------------------------------------------------------------

// Rendering

var drawWalls = function(canvas, walls) {
  map(function(wall) {
    canvas.line(wall.start[0], wall.start[1], wall.end[0], wall.end[1],
      4, 1, 'black');
  }, walls);
  return null;
};

var drawTrajectory = function(canvas, positions, color){
  if (positions.length <= 1) { return []; }
  var start = positions[0];
  var end = positions[1];
  canvas.line(start[0], start[1], end[0], end[1], 3, 0.5, color);
  drawTrajectory(canvas, positions.slice(1), color);
};

// ----------------------------------------------------------------------------

// Functions and values defined previously

// Returns true if the robot collides with a wall by moving from secondLastPos
//    to lastPos
var collisionWithWall = function(lastPos, secondLastPos, walls) {
  var dir = vec_sub(lastPos, secondLastPos);
  var helper = function(walls) {
    if (walls.length === 0) return false;
    var wall = walls[0];
    var t = intersectWall(secondLastPos, dir, wall);
    if (t >= 0 && t <= 1) return true;
    return helper(walls.slice(1));
  };
  return helper(walls);
};

// Robot motion prior is a semi-markov random walk
// (or just a random walk, when there's only been one prior timestep)
var transition = function(lastPos, secondLastPos){
  if (!secondLastPos) {
    return map(
      function(lastX) {
        return gaussian(lastX, 10);
      },
      lastPos
    );
  } else {
    return map2(
      function(lastX, secondLastX){
        var momentum = (lastX - secondLastX) * .9;
        return gaussian(lastX + momentum, 4);
      },
      lastPos,
      secondLastPos
    );
  }
};

// Sensor is a set of n raycasters that shoot rays outward at evenly-spaced
//    angular intervals.
// Has a maximum sensor distance (beyond which it just reports max distance)
// Also degrades (e.g. exhibits greater noise) with distance
var makeSensor = function(n, maxDist, minNoise, maxNoise) {
  return {
    rayDirs: map(function(i) {
      var ang = 2 * Math.PI * (i/n);
      return polar2rect(1, ang);
    }, range(n)),
    maxDist: maxDist,
    minNoise: minNoise,
    maxNoise: maxNoise
  };
};

var walls = makeWalls([
  [50, 25],
  [50, 150],
  [150, 150],
  [150, 300],
  [50, 300],
  [50, 375],
  [225, 375],
  [225, 275],
  [350, 275],
  [350, 225],
  [225, 225],
  [225, 100],
  [375, 100],
  [375, 25],
  [175, 25],
  [175, 75],
  [125, 75],
  [125, 25],
  [50, 25]
]);

var sensor = wpEditor.get('sensor');
var initPos = wpEditor.get('initPos');
var trajectoryLength = wpEditor.get('trajectoryLength');
///

var inferTrajectory = function(n, initPos, sensor, walls, sensorReadings) {
  // Fill in
};

var trueTrajectory = wpEditor.get('trajectory');
var sensorReadings = trueTrajectory.observations;

var post = Infer({method: 'SMC', particles: 100}, function() {
  return inferTrajectory(trajectoryLength, initPos, sensor, walls,
                         sensorReadings);
});
var inferredTrajectory = sample(post);

var canvas = Draw(400, 400, true);
drawWalls(canvas, walls);
drawTrajectory(canvas, trueTrajectory.states, 'blue');
drawTrajectory(canvas, inferredTrajectory.states, 'red');

~~~~

<!-- ~~~~
// Solution

// Observe a sensor reading
var observe = function(sensor, curPos, walls, sensorReading) {
  mapIndexed(function(i, dir) {
    var trueDist = intersectWalls(curPos, dir, walls);
    var cappedDist = Math.min(trueDist, sensor.maxDist);
    var t = Math.min(1, cappedDist/sensor.maxDist);
    var noise = lerp(sensor.minNoise, sensor.maxNoise, t);
    factor(Gaussian({mu: cappedDist, sigma: noise}).score(sensorReading[i]));
  }, sensor.rayDirs);
  return sensorReading;
};

var init = function(initPos, sensor, walls, initSensorReading) {
  return {
    states: [ initPos ],
    observations: [ observe(sensor, initPos, walls, initSensorReading) ]
  };
};

var inferTrajectory = function(n, initPos, sensor, walls, sensorReadings) {
  var helper = function(n) {
        var prevData = (n == 1) ? init(initPos, sensor, walls, sensorReadings[0]) : helper(n-1);
    var prevStates = prevData.states;
    var prevObs = prevData.observations;
    var newState = transition(last(prevStates), secondLast(prevStates));
    var collision = collisionWithWall(newState, last(prevStates), walls);
    factor(collision ? -Infinity : 0);
    var newObs = observe(sensor, newState, walls, sensorReadings[n-1]);
    return {
      states: prevStates.concat([newState]),
      observations: prevObs.concat([newObs])
    };
  };
  return helper(n);
};
~~~~ -->

Once you've gotten this to work, you might consider changing some of the parameters in this model, such as:

 - The length of the trajectory
 - The robot's initial postition
 - The nature of the sensor, such as how many directions it sees in, it's noise model, and it's maximum sensing distance.

How do different settings of these parameters affect the number of SMC particles required to infer accurate trajectories?

[Next: Markov Chain Monte Carlo]({{ "/chapters/4-2-mcmc.html" | prepend: site.baseurl }})