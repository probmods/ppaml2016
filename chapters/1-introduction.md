---
layout: chapter
title: Introduction
description: "An introduction."
---

WebPPL is a probabilistic programming language based on Javascript. WebPPL can be used most easily through [webppl.org](http://webppl.org). It can also be [installed locally](http://webppl.readthedocs.io/en/dev/installation.html) and run from the [command line](http://webppl.readthedocs.io/en/dev/usage.html).

The deterministic part of WebPPL is a [subset of Javascript](http://dippl.org/chapters/02-webppl.html).
The probabilistic aspects come from: [distributions](http://webppl.readthedocs.io/en/dev/distributions.html) and [sampling](http://webppl.readthedocs.io/en/dev/sample.html),
marginal [inference](http://webppl.readthedocs.io/en/dev/inference/index.html),
and [factors](http://webppl.readthedocs.io/en/dev/inference/index.html#factor).




- Resources
  - This is a mashup of the introductory chapters of dippl, probmods, and agentmodels.
    - http://dippl.org/chapters/02-webppl.html
    - https://probmods.org/generative-models.html
    - https://probmods.org/conditioning.html
    - https://probmods.org/patterns-of-inference.html
    - http://agentmodels.org/chapters/02-webppl.html
- Content
    - point to io packages (csv, json, fs)
    - rwebppl
  - The WebPPL language
    - ERPs as reified distributions
    - Visualizing distributions
    - Examples that illustrate the language
      - Coin flipping
      - Geometric distribution as example with structure change
        - Recursion
  - Conditioning and factors
  - Inference operators
  - Graphical models as programs with fixed control flow
  - More interesting (but still simple) models
    - Tug of war

A code box example:

~~~~
// Using the stochastic function `flip` we build a function that
// returns 'H' and 'T' with equal probability:

var coin = function(){
  return flip(.5) ? 'H' : 'T';
};

var flips = [coin(), coin(), coin()];
print("Some coin flips: " + flips);


// We now use `flip` to define a sampler for the geometric distribution:

var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};

var boundedGeometric = Enumerate(
  function(){ return geometric(0.5); },
  {maxExecutions: 20});

print('Histogram of (bounded) Geometric distribution');
viz.auto(boundedGeometric);
~~~~

Here we link to the [next chapter](2-tour.html).
