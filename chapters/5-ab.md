---
layout: subchapter
title: Data - analysis (A/B Testing)
custom_js:
- /assets/js/cityData.js
- /assets/js/bdaHelpers.js
---

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
    green: uniform(0,1),
    grey: uniform(0,1)
  };

  // mixture parameter (i.e., % of bonafide visitors)
  var probBonafide = uniform(0,1);

  foreach(bannerData, function(personData) {

      var group = flip(probBonafide) ? "bonafide" : "accidental";

      var scr1 = Gaussian({mu: logTimes[group], sigma: sigmas[group]}).score(personData.time)

      factor(scr1)

      var acceptanceRate = (group == "bonafide") ? conversionRates[personData.condition] : 0.001

      var scr2 = Bernoulli({p:acceptanceRate}).score(personData.converted)

      factor(scr2)

  } )

  return { logTimes_accidental: logTimes.accidental,
            logTimes_bonafide: logTimes.bonafide,
            sigma_accidental: sigmas.accidental,
            sigma_bonafide: sigmas.bonafide,
            green: conversionRates.green,
            grey: conversionRates.grey,
            percent_bonafide: probBonafide }

}

var numSamples = 20000;
var posterior = Infer({method: "incrementalMH", 
                       samples: numSamples, 
                       burn: numSamples/2}, 
                      model)

viz.marginals(posterior)
~~~~
