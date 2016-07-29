---
layout: chapter
title: Introduction
description: "A brief introduction."
---

WebPPL is a probabilistic programming language based on Javascript. WebPPL can be used most easily through [webppl.org](http://webppl.org). It can also be [installed locally](http://webppl.readthedocs.io/en/dev/installation.html) and run from the [command line](http://webppl.readthedocs.io/en/dev/usage.html).

The deterministic part of WebPPL is a [subset of Javascript](http://dippl.org/chapters/02-webppl.html).
Assignment is generally not allowed (except by using the [globalstore](http://webppl.readthedocs.io/en/dev/globalstore.html)).
This also means looping constructs (such as `for`) are not available; we use functional programming instead to operate on [arrays](http://webppl.readthedocs.io/en/dev/functions/arrays.html).
(Note that [tensors](http://webppl.readthedocs.io/en/dev/functions/tensors.html) are not arrays.)

The probabilistic aspects of WebPPL come from: [distributions](http://webppl.readthedocs.io/en/dev/distributions.html) and [sampling](http://webppl.readthedocs.io/en/dev/sample.html),
marginal [inference](http://webppl.readthedocs.io/en/dev/inference/index.html),
and [factors](http://webppl.readthedocs.io/en/dev/inference/index.html#factor).

- Resources
  - [Language intro from DIPPL](http://dippl.org/chapters/02-webppl.html).
  - [Language intro from AgentModels](http://agentmodels.org/chapters/02-webppl.html).
  - [Generative and cognitive modeling ideas (with Church)](https://probmods.org).
  - WebPPL [packages](http://webppl.readthedocs.io/en/dev/packages.html) (e.g. csv, json, fs).
  - If you would rather use WebPPL from R: [RWebPPL](https://github.com/mhtess/rwebppl).

A few examples to illustrate:

~~~~
// Using the stochastic function `flip` we build a function that
// returns 'H' and 'T' with equal probability:

var coin = function(){
  return sample(Bernoulli({p: .5})) ? 'H' : 'T';
};

var flips = [coin(), coin(), coin()];
print("Some coin flips: " + flips);
~~~~

~~~~
// Defining the geometric distribution as a stochastic recursion:

var geometric = function(p) {
  return bernoulli(p) ? 1 + geometric(p) : 1
};

// Favoring values around 10 via factor:
var skewedGeometric = function(){
  var g = geometric(0.5);
  factor(-Math.pow(10-g,2))
  return g
}

// Finding the marginal distribution via Infer:
var m = Infer({method: 'enumerate', maxExecutions: 100}, skewedGeometric)

print('Histogram of skewed Geometric distribution');
viz.auto(m);
~~~~

## HMM

## regression

## marbles?

## LDA
