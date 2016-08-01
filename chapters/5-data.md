---
layout: chapter
title: Data - analysis and prediction
custom_js:
- /assets/js/cityData.js
- /assets/js/bdaHelpers.js
description: "Analyzing data to gain insight into the processes that may have generated it and to make predictions on new data."
---

# Bayesian data analysis

Statistical analysis of data is useful for understanding the processes that may have generated that data and to make predictions about new data.
Bayesian data analysis explicitly posits latent constructs and the generative process of the data. Inference is then performed to *invert* the model: go from data to inferences about latents. In this chapter, we will walk through some basic Bayesian data analysis models, and gradually build more complex models of data.


## A - B testing

Designing new products for the market is hard. There are many --- hundreds, thousands, ... --- of design decisions along the way. For many of them, your intuition will fail. 

Let's pretend we're building a website (*GrubWatch*) that allows users to log the food they've eaten in the past day and will compute health data for them. Your partner has the intuition that changing the top banner of the webpage from grey to green will increase the attractiveness of the webpage. You are skeptical, however, that the color of the banner would have any effect. Your friend and you decide to allocate the next 200 visitors to this experiment: randomly paint the banner either grey or green. 

This is a what's called an A/B Test (here, a grey/green test). A/B testing is employed in **??product developent??** to determine if one version of a product or website is better than another (better, usually according to how many people end up buying the product or using the webpage). Note that A/B testing is a special case of a psychology experiment: It is testing conditions that are thought to manipulate human behavior. 


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

      var acceptanceRate = (group == "bonafide") ? hitRates[personData.condition] : 0.001

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
    -In contrast to this page, we might want 2-3 real datasets
-Content
  - *MH to revise content*
  -Input/output of data
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
