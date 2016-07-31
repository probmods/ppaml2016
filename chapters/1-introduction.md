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
  - [Language intro from AgentModels](http://agentmodels.org/chapters/2-webppl.html).
  - [Generative and cognitive modeling ideas (with Church)](https://probmods.org).
  - WebPPL [packages](http://webppl.readthedocs.io/en/dev/packages.html) (e.g. csv, json, fs).
  - If you would rather use WebPPL from R: [RWebPPL](https://github.com/mhtess/rwebppl).

## Two simple models

Here is an example using distributions and `sample`.

~~~~
// We build a function that returns 'H' and 'T' with equal probability:

var coin = function(){
  var aDist = Bernoulli({p: .5});
  return sample(aDist) ? 'H' : 'T';
};

var flips = [coin(), coin(), coin()];
print("Some coin flips: " + flips);
~~~~

Here is an example using `factor` and `Infer`, and illustrating stochastic recursion.

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
var m = Infer({method: 'enumerate', maxExecutions: 100},
              skewedGeometric)

print('Histogram of skewed Geometric distribution');
viz.auto(m);
~~~~

Some things to note:

- The short form `bernoulli(p)` which is the same as `sample(Bernoulli({p: p}))`.
- We can access deterministic functions from `Math` (and other js standard libraries) transparently. From the command line additional libraries can be required as [packages](http://webppl.readthedocs.io/en/dev/packages.html).
- The [viz](https://github.com/probmods/webppl-viz) package provides easy visualization methods. (It is pre-loaded in the browser version.)


## Regression

Here is a simple logistic regression model.

~~~~
//some data:
var data = [{feature: -10, label: false},
            {feature: -5, label: false},
            {feature: 2, label: true},
            {feature: 6, label: true},
            {feature: 10, label: true}]

//a helper fn:
var sigmoid = function(x) {
  return 1 / (1 + Math.exp(-x))
}

var model = function() {
  //slope, intercept, and noise for the regression:
  var m = gaussian(0, 1)
  var b = gaussian(0, 1)
  var sigmaSquared = gamma(1, 1)

  //the regression model itself
  var labelDist = function(x) {
    var y = gaussian(m * x + b, sigmaSquared);
    return Bernoulli({p: sigmoid(y)});
  }

  //include observed data
  map(function(d){factor(labelDist(d.feature).score(d.label))},
      data)

  return sample(labelDist(0))
//   return labelDist(0).score(true) //alternate return to explore confidence
}

viz.auto(Infer({method: 'MCMC',
                samples: 10000,
                callbacks: [editor.MCMCProgress()]},
               model))
~~~~


## HMM

~~~~
var transition = function(s) {
  return s ? flip(0.7) : flip(0.3)
}

var observe = function(s) {
  return s ? flip(0.9) : flip(0.1)
}

var hmm = function(n) {
  var prev = (n == 1) ? {states: [true], observations: []} : hmm(n - 1);
  var newState = transition(prev.states[prev.states.length - 1]);
  var newObs = observe(newState);
  return {
    states: prev.states.concat([newState]),
    observations: prev.observations.concat([newObs])
  };
}

var trueObservations = [false, false, false];

var stateDist = Infer({method: 'enumerate'},
  function() {
    var r = hmm(3);
    map2(function(o,d){condition(o===d)}, r.observations, trueObservations)
    return r.states;
  }
)

viz.hist(stateDist)
~~~~


## LDA

~~~~
var nTopics = 2;
var vocabulary = ['zebra', 'wolf', 'html', 'css'];
var docs = {
  'doc1': 'zebra wolf zebra wolf zebra wolf html wolf zebra wolf'.split(' '),
  'doc2': 'html css html css html css html css html css'.split(' '),
  'doc3': 'zebra wolf zebra wolf zebra wolf zebra wolf zebra wolf'.split(' '),
  'doc4': 'html css html css html css html css html css'.split(' '),
  'doc5': 'zebra wolf zebra html zebra wolf zebra wolf zebra wolf'.split(' ')
};

var makeWordDist = function() { dirichlet(ones([vocabulary.length,1])) };
var makeTopicDist = function() { dirichlet(ones([nTopics,1])) };

var model = function() {
  var wordDistForTopic = repeat(nTopics, makeWordDist);

  mapObject(function(docname, words) {
    var topicDist = makeTopicDist();
    map(function(word) {
      var topic = discrete(topicDist);
      var wordDist = wordDistForTopic[topic];
      var wordID = vocabulary.indexOf(word);
      factor(Discrete({ps: wordDist}).score(wordID));
    }, words);
  }, docs);

  return map(function(v) { return _.toArray(v.data); }, wordDistForTopic);
};

var post = Infer({
  method: 'MCMC',
  burn: 10000,
  samples: 1,
  callbacks: [editor.MCMCProgress()]
}, model);

var samp = sample(post);

print("Topic 1:"); viz.bar(vocabulary, samp[0]);
print("Topic 2:"); viz.bar(vocabulary, samp[1]);
~~~~
