---
layout: chapter
title: Data - analysis and prediction
custom_js:
- /assets/js/cityData.js
- /assets/js/bdaHelpers.js
description: "Analyzing data to gain insight into the processes that may have generated it and to make predictions on new data."
---

# Part 1: Bayesian data analysis

Statistical analysis of data is useful for understanding the processes that may have generated that data and to make predictions about new data. Bayesian data analysis is a general purpose data analysis approach for making explicit hypotheses about where the data came from (e.g. the hypothesis that data from 2 experimental conditions came from two different distributions). Inference is then performed to *invert* the model: go from data to inferences about hypotheses. In this chapter, we will walk through some basic Bayesian data analysis models, and gradually build more complex models of data.

Many of the examples and explanations of basic concepts are borrowed from [Lee & Wagenmakers (2013)](https://bayesmodels.com/).

## Basics: Parameters and predictives

Bayes’ rule provides a bridge between the unobserved parameters of models and the observed data. The most useful part of this bridge is that data allow us to update the uncertainty, represented by probability distributions, about parameters. But the bridge can handle two-way traffic, and so there is a richer set of possibilities for relating parameters to data. For a given Bayesian model (together with data), there are four conceptually distinct distributions:

+ First, the *prior distribution over parameters* captures our initial beliefs or state of knowledge about the latent variables those parameters represent.
+ Second, the *prior predictive distribution* tells us what data to expect, given our model and our current state of knowledge. The prior predictive is a distribution over data, and gives the relative probability of different observable outcomes before any data have been seen.
+ Third, the *posterior distribution over parameters* captures what we know about the latent variables having updated the prior information with the evidence provided by data.
+ Fourth, the *posterior predictive distribution* tells us what data to expect, given the same model we started with, but with a current state of knowledge that has been updated by the observed data. Again, the posterior predictive is a distribution over data, and gives the relative probability of different observable outcomes after data have been seen.

Loosely speaking, *predictive* distribuions are in "data space" and *parameter* distributions are in "latent space".

### A simple illustration

Here, we explore the result of an experiment with 15 trials and binary outcomes (e.g., flipping a coin with an uncertain weight, asking people if they'll vote for Hillary Clinton or Donald Trump, ...)

~~~~
// Unpack data
var k = 1 // number of successes
var n = 15  // number of attempts

var model = function() {

   var p = uniform( {a:0, b:1} );

   // Observed k number of successes, assuming a binomial
   observe(Binomial({p : p, n: n}), k)

   // sample from binomial with updated p
   var posteriorPredictive = binomial({p : p, n: n});

   // sample fresh p
   var prior_p = uniform({ a: 0, b: 1});
   // sample from binomial with fresh p
   var priorPredictive = binomial({p : prior_p, n: n})

   return {
       prior: prior_p, priorPredictive : priorPredictive,
       posterior : p, posteriorPredictive : posteriorPredictive
    };
}

var numSamples = 2000;
var inferOpts = {
  method: "rejection",
  samples: numSamples
};

var posterior = Infer(inferOpts, model);

viz.marginals(posterior)
~~~~

### Exercises

1. Make sure you understand the prior, posterior, prior predictive, and posterior predictive distributions, and how they relate to each other. Why are some plots densities and others bar graphs? Understanding these ideas is a key to understanding Bayesian analysis. Check your understanding by trying other data sets, varying both k and n.

2. Try different priors on `theta`, by changing `theta = uniform(0, 1)` to `theta = beta(10,10)`, `beta(1,5)` and `beta(0.1,0.1)`. Use the figures produced to understand the assumptions these priors capture, and how they interact with the same data to produce posterior inferences and predictions.

3. Predictive distributions are not restricted to exactly the same experiment as the observed data, and can be used in the context of any experiment where the inferred model parameters make predictions. In the current simple binomial setting, for example, predictive distributions could be found by an experiment that is different because it has `n' != n` observations. Change the model to implement an example of this.


## Posterior prediction and model checking

One important use of posterior predictive distributions is to examine the descriptive adequacy of a model. The posterior predictive can be viewed as a set of predictions about what data the model expects to see, based on the posterior distribution over parameters. If these predictions do not match the data *already seen*, the model is descriptively inadequate.

Let's say we ran 2 experiments that we believe are conceptually the same (e.g., asking 10 people whether or not they would vote for "Hillary Clinton or Donald Trump" and asking a separate group of 10 people if they would vote for "Donald Trump or Hillary Clinton"). Suppose we observed the following data from those 2 experiments: `k1=0; k2=10`.

~~~~
// Successes in 2 experiments
var k1 = 0;
var k2 = 10;

// Number of trials in 2 experiments
var n1 = 10;
var n2 = 10;

var model = function() {

  // Sample rate from uniform distribution
  var p = uniform( {a:0, b:1} );

  var scr = Binomial({p: p, n: n1}).score(k1) +
            Binomial({p: p, n: n2}).score(k2);

  factor(scr)

  // alternatively, you could write:
  // observe(Binomial({p: p, n: n1}), k1);
  // observe(Binomial({p: p, n: n2}), k2);

  var posteriorPredictive1 = binomial({p : p, n : n1})
  var posteriorPredictive2 = binomial({p : p, n : n2})

  return {posterior : p,
          posteriorPredictive1: posteriorPredictive1,
          posteriorPredictive2: posteriorPredictive2
  };
}

var numSamples = 20000;
var inferOpts = {
  method: "MCMC",
  samples: numSamples,
  burn: numSamples/2,
  callbacks: [editor.MCMCProgress()]
};

var posterior = Infer(inferOpts, model);

viz.marginals(posterior)
~~~~

### Exercises 2

1.  What do you conclude about the descriptive adequacy of the model, based on the relationship between the observed data and the posterior predictive distribution? Recall the observed data is `k1 = 0; n1 = 10` and  `k2 = 10; n2 = 10`.

2. What can you conclude about the parameter `theta`?


## A/B testing

[link]({{ "/chapters/5-ab.html" | prepend: site.baseurl }})

<!--
- Resources
  - This is a shortened version of MH's BDA course.
  - http://forestdb.org/models/bayesian-data-analysis.html
-Content
  - Occam's razor
  - Various models useful for BDA
  - Use BDA to compare two of the models shown earlier on a dataset
    - logistic regression vs Bayesian neural net
    - a rich cognitive model vs regression
    - text analysis models (topic models, hmms, etc)
  - Making predictions from data -->


## "The Full Bayesian Thing" (Bayesian data analysis of Bayesian models)

[link]({{ "/chapters/5-bdaBCM.html" | prepend: site.baseurl }})

<hr style='width: 180%; height: 0; border: 1px solid #99ccff'/>

# Part 2: Making predictions from data

Some times, we're more interested in getting model *predictions* than, say, analyzing learned model *parameters*.

## Uncertain and reversible financial models

Estimating Amazon hosting costs for a fictional video streaming company:

~~~~
var model = function() {
  var resources = [
    {name: 'a',
     size: 1,
     requests: beta(0.01,0.1) * 20000 + 30000
    },
    {name: 'b',
     size: 10,
     requests: beta(0.02,0.1) * 20000 + 30000
    },
    {name: 'c',
     size: 7,
     requests: beta(0.05,0.1) * 20000 + 30000
    }
  ];
  var storage =  sum(map(function(r) { r.size },               resources))
  var requests = sum(map(function(r) { r.requests },           resources))
  var transfer = sum(map(function(r) { r.requests * r.size  }, resources))

  var costs = {
    requests: 0.004 * (requests / 10000),
    storage: 0.03 * storage,
    transfer: transfer *
      (transfer <= 1 ? 0 :
       (transfer <= 10000 ? 0.09 :
        (transfer <= 40000 ? 0.085 : (transfer <= 100000 ? 0.07 : 0.05))))
  };
  // /* B goes viral */ condition(resources[1].requests > 40000)
  return sum(_.values(costs))/100
}

var dist = Infer({method: 'MCMC', samples: 30000}, model);
print("Expected cost: $" + expectation(dist))
viz.density(dist, {bounds: [300,500]})
~~~~

Advantages:

- Gives uncertainties about price
- Can ask what-if questions: what if B goes viral?

Exercise: play with what the request priors and ask, under your priors, what could cause costs to be high?

## Censored weight regression

Now, an example where we first learn some parameters from data and then use them to make predictions about new settings.

First, make some synthetic data and censor it:

~~~~
var genders = repeat(50, function() { uniformDraw(['m','f'])});
var heights = map(function(g) { gaussian(g == 'm' ? 11 : 10, 3) },
                  genders)

var interceptMap = {m: 4, f: 0},
    slopeMap = {m: 10, f: 8};

var weights = map2(function(h,g) {
  var intercept = interceptMap[g];
  var slope = slopeMap[g];
  return intercept + slope * h + gaussian(0, 1.5);
},
                   heights,
                   genders);

var pureData = map(function(i) {
  return {height: heights[i], weight: weights[i], gender: genders[i]}
}, _.range(50))

print('pure data:')
viz.scatter(pureData)

var data = map(function(i) {
  return {height: heights[i], weight: Math.min(110, weights[i]), gender: genders[i]}
}, _.range(50))

print('censored data:')
viz.scatter(data)

editor.put('censored', data)
~~~~

Learn parameters from data and use it to predict Bob's weight:

~~~~
var data = editor.get('censored');

var bob = {
  height: 12,
  gender: 'm'
}

var model = function() {
  var slopeMap = {
    m: uniform(5, 15),
    f: uniform(5, 15)
  };
  var interceptMap = {
    m: uniform(0, 10),
    f: uniform(0, 10)
  };

  var predictWeight = function(person) {
    var intercept = interceptMap[person.gender];
    var slope = slopeMap[person.gender];
    return intercept + slope * person.height;
  }

  map(function(person) {
    var sampledWeight = predictWeight(person);
    observe(Gaussian({mu: Math.min(110, sampledWeight), sigma: 1}), person.weight);
  },
     data)

  predictWeight(bob)
}

var dist = Infer({method: 'MCMC',
                  kernel: {HMC: {stepSize: 0.02, steps: 10}},
                  samples: 1000,
                  callbacks: [editor.MCMCProgress()]},
                 model)

viz.auto(dist)
~~~~

Exercise: add one or two more sources of noise to this example:

- Fluctuation in actual weight. This assumes that people's true weight fluctuates throughout the day.
- Systematic underreporting. If you ask people to self-report their weight, they might report slightly lower numbers than the scale.

## Presidential election

[link]({{ "/chapters/5-election.html" | prepend: site.baseurl }})

## Digit classification

[link]({{ "/chapters/5-vae.html" | prepend: site.baseurl }})




## Exercises

Extend any of these analysis or prediction examples

Other ideas:

- Power of Ten (Planet Money podcast)
