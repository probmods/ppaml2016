---
layout: chapter
title: Data - analysis and prediction
custom_js:
- /assets/js/cityData.js
- /assets/js/bdaHelpers.js
description: "Analyzing data to gain insight into the processes that may have generated it and to make predictions on new data."
---

# Bayesian data analysis

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

var personIDs = _.uniq(_.pluck(cityData, "id"));

var model = function() {

  var logTimes = {
    bonafide: gaussian(3,3), // exp(3) ~ 20s
    accidental: gaussian(0,2), // exp(2) ~ 7s
  }

  var sigmas =  {
    bonafide: uniform(0,3),
    accidental: uniform(0,3),
  }

  var hitRates = {
    red: uniform(0,1),
    blue: uniform(0,1)
  };

  var phi = uniform(0,1);

  var sampleGroup = function(id) { return [id, flip(phi) ? "bonafide" : "accidental"  ] }

 var personAssignments = _.object(map(sampleGroup, personIDs));

  foreach(personIDs, function(person_id) {
      // subset is a header.js function from mht
      var personData = subset(data, "id", person_id)[0];
      // var group = function(id) { return flip(phi) ? "bonafide" : "accidental" }

      var group = personAssignments[person_id];

      var scr1 = Gaussian({mu: logTimes[group],
                          sigma: sigmas[group]}).score(personData.time)

      factor(scr1)

      var acceptanceRate = (group == "bonafide") ?
            hitRates[personData.condition] : 0.001


      var scr2 = Bernoulli({p:acceptanceRate}).score(personData.converted)
      factor(scr2)

  })

  return { logTimes_accidental: logTimes.accidental,
            logTimes_bonafide: logTimes.bonafide,
            sigma_accidental: sigmas.accidental,
            sigma_bonafide: sigmas.bonafide,
            blue: hitRates.blue,
            red: hitRates.red,
            percent_bonafide: phi }

}

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

## Censored linear regression with multiple noise sources

We give people a scale and ask them to weigh themselves and report the numbers. There are some realistic complications:

1. The scale has a maximum limit (making this a so-called censored model)
1. The scale is noisy
1. Peoples' weights fluctuate
1. There's a systematic bias for people to under-report their weights

What we really care about is just estimates of peoples' true weight, don't care that much about interpreting parameters (though #4 is kind of interesting)

note that this would be hard to do in a non-PPL (censoring is hard, non-conjugacy)

## Inferring regular expressions

(do both selection and replacement)

Point of this section is that getting posterior probs. exactly right might not be so important because you're just gonna show the user a list anyway.

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

todo: figure out some exercises
