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


## A - B testing

Designing new products for the market is hard. There are many --- hundreds, thousands, ... --- of design decisions along the way. For many of them, your intuition will fail and you will have to resort to collecting data.

Let's pretend we're building a website --- *GrubWatch* --- that allows users to log the food they've eaten in the past day and will compute health data for them. Your business partner has the intuition that changing the top banner of the webpage from grey to green will increase the attractiveness of the webpage. You are skeptical, however, that the color of the banner would have any effect. Your friend and you decide to allocate the next 200 visitors to this experiment: randomly paint the banner either grey or green. 

This is a what's called an A/B Test (here, a grey/green test). A/B testing is employed in marketing and business development to determine if one version of a product or website is better than another (better, usually according to how many people end up buying the product or using the webpage). Note that A/B testing is a special case of a psychology experiment: It is testing conditions that are thought to manipulate human behavior. 


## Inferring a rate

You collect data on 200 visitors to your website. Your data includes the banner color each visitors was assigned (grey or green), the amount of time they spent on the website, whether or not their visitor "converted" (i.e., they bought your product, or signed up for your service), and misc demographic data (their browswer, location). Your friend thinks that conversion rate is going to be higher for the group that has a green banner than for the group with the grey banner.

Concretely, we are interested in the rate of conversion for the two groups:

~~~~
///fold:
var foreach = function(lst, fn) {
    var foreach_ = function(i) {
        if (i < lst.length) {
            fn(lst[i]);
            foreach_(i + 1);
        }
    };
    foreach_(0);
};
///

var model = function() {

  // unknown rates of conversion
  var conversionRates = {
    grey: uniform(0,1),
    green: uniform(0,1)
  };

  foreach(bannerData, function(personData) {

      // grab appropriate conversionRate by condition
      var acceptanceRate = conversionRates[personData["condition"]];

      // people are assumed to be i.i.d. samples
      var scr = Bernoulli({p:acceptanceRate}).score(personData["converted"]);

      factor(scr)

  });

  return conversionRates;

}

var numSamples = 10000;
var inferOpts = {
  method: "MCMC", 
  samples: numSamples,
  burn: numSamples/2, 
  callbacks: [editor.MCMCProgress()] 
};

var posterior = Infer(inferOpts, model);

viz.auto(posterior)
viz.marginals(posterior)
~~~~

You show this analysis to your friend. She is unconvinced by your analysis. She says that GrubWatch gets a lot of *accidental* traffic, because visitors are often interested in a different site **GrubMatch**, the slightly more popular dating website based on common food preferences. She says that dozens of visitors visit and leave your website within a few seconds, after they realize they're not at GrubMatch. She says these people are contaminating the data.

This inspires you to create a build a new model, trying to account for this contamination.
~~~~
////fold:
var foreach = function(lst, fn) {
    var foreach_ = function(i) {
        if (i < lst.length) {
            fn(lst[i]);
            foreach_(i + 1);
        }
    };
    foreach_(0);
};
////

var personIDs = _.uniq(_.pluck(bannerData, "id"));

var model = function() {

  // average time spent on the website (in log-seconds)
  var logTimes = {
    bonafide: gaussian(3,3), // exp(3) ~ 20s
    accidental: gaussian(0,2), // exp(2) ~ 7s
  }

  // variance of time spent on website (plausibly different for the two groups)
  var sigmas =  {
    bonafide: uniform(0,3),
    accidental: uniform(0,3),
  }

  var conversionRates = {
    red: uniform(0,1),
    blue: uniform(0,1)
  };

  // mixture parameter (i.e., % of bonafide visitors)
  var probBonafide = uniform(0,1);

  var sampleGroup = function(id) { return [id, flip(probBonafide) ? "bonafide" : "accidental"  ] }

  var personAssignments = _.object(map(sampleGroup, personIDs));

  foreach(personIDs, function(person_id) {

      var personData = _.where(bannerData, {id: person_id})[0];

      var group = personAssignments[person_id];

      var scr1 = Gaussian({mu: logTimes[group], sigma: sigmas[group]}).score(personData.time)

      factor(scr1)

      var acceptanceRate = (group == "bonafide") ? hitRates[personData.condition] : 0.00001

      var scr2 = Bernoulli({p:acceptanceRate}).score(personData.converted)

      factor(scr2)

  } )

  return { logTimes_accidental: logTimes.accidental,
            logTimes_bonafide: logTimes.bonafide,
            sigma_accidental: sigmas.accidental,
            sigma_bonafide: sigmas.bonafide,
            green: hitRates.green,
            grey: hitRates.grey,
            percent_bonafide: probBonafide }

}

var posterior = Infer({method: "MCMC", samples: 10000, burn: 5000,
                      callbacks: [editor.MCMCProgress()]}, model)

viz.marginals(posterior)

~~~~


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
  - Making predictions from data


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
