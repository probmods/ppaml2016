---
layout: chapter
title: Data - analysis and prediction
custom_js:
- /assets/js/cityData.js
- /assets/js/bdaHelpers.js
description: "Analyzing data to gain insight into the processes that may have generated it and to make predictions on new data."
---

# Bayesian data analysis

Statistical analysis of data is useful for understanding the processes that may have generated that data and to make predictions about new data. Bayesian data analysis is a general purpose data analysis approach for making explicit hypotheses about where the data came from (e.g. the hypothesis that data from 2 experimental conditions came from two different distributions). Inference is then performed to *invert* the model: go from data to inferences about hypotheses. In this chapter, we will walk through some basic Bayesian data analysis models, and gradually build more complex models of data.

## Basics: Parameters and predictives

Bayes’ rule provides a bridge between the unobserved parameters of models and the observed data. The most useful part of this bridge is that data allow us to update the uncertainty, represented by probability distributions, about parameters. But the bridge can handle two-way traffic, and so there is a richer set of possibilities for relating parameters to data. There are really four distributions available, and they are all important and useful.

+ First, the *prior distribution over parameters* captures our initial beliefs or state of knowledge about the latent variables they represent.
+ Second, the *prior predictive distribution* tells us what data to expect, given our model and our current state of knowledge. The prior predictive is a distribution over data, and gives the relative probability of different observable outcomes before any data have been seen.
+ Third, the *posterior distribution over parameters* captures what we know about the latent variables having updated the prior information with the evidence provided by data.
+ Fourth, the *posterior predictive distribution* tells us what data to expect, given the same model we started with, but with a current state of knowledge that has been updated by the observed data. Again, the posterior predictive is a distribution over data, and gives the relative probability of different observable outcomes after data have been seen.

### A simple illustration

~~~~
// Unpack data
var k = 1 // number of successes
var n = 15  // number of attempts

var model = function() {

   var p = uniform( {a:0, b:1} );

   // Observed k number of successes
   var scr = Binomial( {p : p, n: n }).score(k);
   factor(scr)

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

1. Make sure you understand the prior, posterior, prior predictive, and posterior predictive distributions, and how they relate to each other (why are some plots densities and others bar graphs?). Understanding these ideas is a key to understanding Bayesian analysis. Check your understanding by trying other data sets, varying both k and n.

2. Try different priors on `theta`, by changing `theta = uniform(0, 1)` to `theta = beta(10,10)`, `beta(1,5)` and `beta(0.1,0.1)`. Use the figures produced to understand the assumptions these priors capture, and how they interact with the same data to produce posterior inferences and predictions.

3. Predictive distributions are not restricted to exactly the same experiment as the observed data, and can be used in the context of any experiment where the inferred model parameters make predictions. In the current simple binomial setting, for example, predictive distributions could be found by an experiment that is different because it has `n' != n` observations. Change the model to implement an example of this.


## Posterior prediction and model checking

One important use of posterior predictive distributions is to examine the descriptive adequacy of a model. It can be viewed as a set of predictions about what data the model expects to see, based on the posterior distribution over parameters. If these predictions do not match the data already seen, the model is descriptively inadequate.


The model below has k1 = 0 successes out of n1 = 10 observations, and k2 = 10 successes out of n2 = 10 observations. The code draws the posterior distribution for the rate and the posterior predictive distribution.

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

1.  What do you conclude about the descriptive adequacy of the model, based on the relationship between the observed data and the posterior predictive distribution?

2. What can you conclude about the parameter `theta`?


## A / B testing

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


# Making predictions from data

## Censored regression with multiple noise sources

We give people a scale and ask them to weigh themselves and report the numbers. There are some realistic complications:

1. The scale has a maximum limit (making this a so-called censored model)
1. Peoples' weights fluctuate
1. There's a systematic bias for people to under-report their weights

What we really care about is just estimates of peoples' true weight, don't care that much about interpreting parameters (though #4 is kind of interesting)


~~~~
///fold:
var softEqual = function(expected, reported) {
  factor(Gaussian({mu: expected, sigma: 0.1}).score(reported))
};
///

var predict = function(reported) {
  var model = function() {
    var latent = beta(2,5) * 600;

    // fluctuation throughout the day
    var noiseFluctuation = gaussian({mu: 0, sigma: 3});

    // systematic under-reporting
    var noiseUnderreporting = -3 * beta({a: 5, b: 5});

    // scale has a maximum weight
    var sampled = Math.min(300, latent + noiseFluctuation + noiseUnderreporting);

    softEqual(sampled, reported)

    return latent
  }

  var optsHMC = {method: 'MCMC',
                 kernel: {HMC: {stepSize: 0.00001}},
                 callbacks: [editor.MCMCProgress()],
                 samples: 1e4}

  var optsMH = {method: 'MCMC',
                samples: 5e5,
                callbacks: [editor.MCMCProgress()],
                burn: 1e4}

  var optsPF = {method: 'SMC',
               samples: 1000}

  Infer(optsHMC, model)
}

var dist = predict(100);
viz.table(dist, {top: 5})
 // workaround for hist, density bugs
if (dist.support().length > 1) {
  viz.density(dist)
  viz.hist(dist, {numBins: 10})
}
expectation(dist)
~~~~

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

var dist = MH(model, 30000);
print("Expected cost: $" + expectation(dist))
viz.density(dist, {bounds: [300,500]})
~~~~

Advantages:

- Gives uncertainties about price
- Can ask what-if questions: what if B goes viral? what could cause costs to be high?

## Variational autoencoder

[link]({{ "/chapters/5-vae.html" | prepend: site.baseurl }})

## Presidential election

[link]({{ "/chapters/5-election.html" | prepend: site.baseurl }})



## Exercises

extend any of these analysis or prediction examples

other examples:

- power of ten (planet money)
- mark and recapture
