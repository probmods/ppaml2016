---
layout: chapter
title: Data - analysis and prediction
description: "Analyzing data to gain insight into the processes that may have generated it and to make predictions on new data."
---

# Bayesian data analysis

~~~~
var sampleGroup = function() { return flip(0.5) ? "bonafide" : "accidental" }

var model = function() {
  var bonafideLogTime = gaussian(3,3);
  var accidentalLogTime = uniform(0, bonafideLogTime);
  var sigma = uniform(0, 5);

  var personAssignments = repeat(n_people, sampleGroup)

  var hitRates = {
    red: uniform(0,1),
    blue: uniform(0,1)
  };

  foreach(_.range(0, n_people), function(person_id) {
      var personData = _.pluck(data, "id", person_id);

      var pAssignment = personAssignments[person_id];
      var meanLogTime = (pAssignment == "bonafide") ? bonafideLogTime : accidentalLogTime;

      factor(Gaussian({mu: meanLogTime, sigma: sigma}).score(Math.log(personData.time)))

      (pAssignment == "bonafide") ?
        factor(Bernoulli({p: hitRates[personData.condition]}).score(personData.converted == 1) ) :
        null
  })

  return {
    bonafideLogTime : bonafideLogTime,
    accidentalLogTime : accidentalLogTime,
    blue_hitRate : hitRates.blue,
    red_hitRate : hitRates.red
  }

}

~~~~

- Resources
  - This is a shortened version of MH's BDA course.
  - http://forestdb.org/models/bayesian-data-analysis.html
    -In contrast to this page, we might want 2-3 real datasets
- Content
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

## Ordinal regression with endpoint aversion

One advantage of PPLs is that they let you express your prior knowledge to make predictions *at all* (this example would be really hard to do in, say, R).

## Inferring regular expressions

(do both selection and replacement)

Point of this section is that getting posterior probs. exactly right might not be so important because you're just gonna show the user a list anyway.

## Uncertain and reversible spreadsheets

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

## Image denoising

## Variational autoencoder 

## Exercises

todo: figure out some exercises