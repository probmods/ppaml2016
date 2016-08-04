---
layout: subchapter
title: Data - analysis (A/B Testing)
custom_js:
- /assets/js/cityData.js
- /assets/js/bdaHelpers.js
---

Designing new products for the market is hard. There are hundreds, thousands, ... of design decisions along the way. For many of them, your intuition will fail and you will have to collect data to learn about possible design choices. 

Let's pretend we're building a website --- *GrubWatch* --- that allows users to log the food they've eaten in the past day and will compute health data for them. Your business partner has the intuition that changing the top banner of the webpage from grey to green will increase the attractiveness of the webpage. You are skeptical, however, that the color of the banner would have any effect. Your friend and you decide to allocate the next 100 visitors to this experiment: randomly paint the banner either grey or green. 

This is a what's called an A/B Test (here, a grey/green test). A/B testing is employed in marketing and business development to determine if one version of a product or website is better than another (better, usually according to how many people end up buying the product or using the webpage). Note that A/B testing is a special case of a psychology experiment: It is testing conditions that are thought to manipulate human behavior. 


## Inferring a rate

You collect data on 100 visitors to your website. Your data includes the banner color each visitors was assigned (grey or green), the amount of time they spent on the website, whether or not their visitor "converted" (i.e., they bought your product, or signed up for your service), and misc demographic data (their browser, location).

Let's take a look at the data.

~~~~
///fold:
var head = function(ar, l){
	var len = l ? l : 6;
	return print(ar.slice(0, len));
}
///
 
// print first few lines of data set
head(bannerData)

// show all the ids
print(_.pluck(bannerData, "id"))

// how many people saw each banner?
viz.table(_.pluck(bannerData, "condition"))
~~~~

Your friend thinks that conversion rate is going to be higher for the group that has a green banner than for the group with the grey banner.Concretely, we are interested in the rate of conversion for the two groups:

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

      // visitors are i.i.d.
      observe(Bernoulli({p:acceptanceRate}), personData["converted"])

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
viz.marginals(posterior, {bounds: [0.3, 0.7]})
~~~~


1. What can you conclude about the difference between using the grey and the green banner?

You show this analysis to your friend. She is not impressed. She reminds you that GrubWatch gets a lot of *accidental* traffic, because visitors are often interested in a different site **GrubMatch**, the slightly more popular dating website based on common food preferences. She says that dozens of visitors visit and leave your website within a few seconds, after they realize they're not at GrubMatch. She says these people are contaminating the data.

Fortunately, you have recorded how much time users spend on your website. Let's visualize that data.

~~~~
viz.hist(_.pluck(bannerData, "time"), {numBins: 10})
~~~~

This look likes canonical wait time data, following a log-normal distribution. To validate this intuition, let's look at the data taking the log.

~~~~
var logTimeData = map(function(t){
	return Math.log(t);
}, _.pluck(bannerData, "time"))

viz.hist(logTimeData, {numBins: 10})
~~~~

Looks pretty normal, but also looks like there's something funny going on. Your friend may be right: some people are spending substantially less time on your website than other people. Presumably, none of these people visiting your site for a just a few seconds are buying your product. 

How can we account for this potential data contamination? We posit 2 groups of visitors: `"bonafide"` visitors and `"accidental"` visitors. They spend different amounts of time on your website, and don't behave according to the true `conversionRates`.

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

      observe(
      	Gaussian({mu: logTimes[group], sigma: sigmas[group]}), 
      	Math.log(personData.time)
      	)

      // accidental visitors have a very low probability of buying your product
      var acceptanceRate = (group == "bonafide") ? 
      	conversionRates[personData.condition] : 
      	0.0000001

      observe(
      	Bernoulli({p:acceptanceRate}), 
      	personData.converted
      	)

  } )

  return { logTimes_accidental: logTimes.accidental,
            logTimes_bonafide: logTimes.bonafide,
            sigma_accidental: sigmas.accidental,
            sigma_bonafide: sigmas.bonafide,
            green: conversionRates.green,
            grey: conversionRates.grey,
            percent_bonafide: probBonafide }

}

var numSamples = 100000;
var posterior = Infer({method: "incrementalMH", 
                       samples: numSamples, burn: numSamples/2,
                   		verbose: true, verboseLag: numSamples/10}, 
                      model)

// run a big model: takes about 1 minute
editor.put("posterior", posterior)
~~~~

#### Examine posterior

Display marginal posterior over the rate of bonafide visitors.

~~~~
///fold:
var marginalize = function(myDist, label){
    Infer({method: "enumerate"}, function(){
        var x = sample(myDist);
        return x[label]
    });
};
///

var jointPosterior = editor.get("posterior");

var marginalBonafide = marginalize(jointPosterior, "percent_bonafide");

viz.hist(marginalBonafide, {numBins: 15});
~~~~
	
So, indeed, almost 30% of the traffic to your site was judged to be "accidental".

~~~~
///fold:
var marginalizeExponentiate = function(myDist, label){
    Infer({method: "enumerate"}, function(){
        var x = sample(myDist);
        return Math.exp(x[label])
    });
};
///

var jointPosterior = editor.get("posterior");

var marginalTime_accidental = marginalizeExponentiate(jointPosterior, "logTimes_accidental");
var marginalTime_bonafide = marginalizeExponentiate(jointPosterior, "logTimes_bonafide");

print("Inferred time spent by accidental visitors (in seconds)")
viz.hist(marginalTime_accidental, {numBins: 10});

print("Inferred time spent by bonafide visitors (in seconds)")
viz.hist(marginalTime_bonafide, {numBins: 10})
~~~~

### Exercises

1. Now, get out the marginal distributions over the rates of the conversion parameters. Does accounting for the accidental visitors change the conclusions you can draw about the efficacy of the green banner?

2. You show these results to your friend, and even she is surprised by them. Why are the results the way they are? In the above model, we assumed the rate of bonafide visitors was independent of which banner they were assigned. Is that true? Modify the model above to test the hypothesis that the rate of bonafide visitors was different for the different banners?

3. In the above model, we assume that accidental visitors are very unlikely to buy your product. How could we relax this assumption, and say that accidental visitors also have some probability of "accidentally" buying your product? Modify the model to express this possibility, run the model, and draw inferences about the rate at which accidental visitors buy your product.
