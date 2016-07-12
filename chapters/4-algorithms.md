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

In these situations, we can instead use one of WebPPL's many approximate inference algorithms.

## Chapter Contents

[Rejection Sampling & Particle Filtering]({{ "/chapters/4-1-rejectionAndPF.html" | prepend: site.baseurl }})

[Markov Chain Monte Carlo]({{ "/chapters/4-2-mcmc.html" | prepend: site.baseurl }})

[Variational Inference]({{ "/chapters/4-3-variational.html" | prepend: site.baseurl }})

[Additional Exercises]({{ "/chapters/4-4-exercises.html" | prepend: site.baseurl }})
