---
layout: chapter
title: Data - analysis and prediction
description: "Analyzing data to gain insight into the processes that may have generated it and to make predictions on new data."
---

# Bayesian data analysis

- Resources
  - This is a shortened version of MH's BDA course.
  - http://forestdb.org/models/bayesian-data-analysis.html
    - In contrast to this page, we might want 2-3 real datasets
- Content
  - *MH to revise content*
  - Input/output of data
  - Occam's razor
  - Various models useful for BDA
  - Use BDA to compare two of the models shown earlier on a dataset
    - logistic regression vs Bayesian neural net
    - a rich cognitive model vs regression
    - text analysis models (topic models, hmms, etc)
  - Making predictions from data


# Making predictions from data

Estimating Amazon hosting costs for a fictional video streaming company:

~~~~
var transferCost = function(gb) {
  var unitCost =
      (gb <= 1 ? 0 :
       (gb <= 10000 ? 0.09 :
        (gb <= 40000 ? 0.085 : (gb <= 100000 ? 0.07 : 0.05))))
  return unitCost * gb;
}

var requestsCost = function(n) { 0.004 * n / 10000}
var storageCost = function(gb) { 0.03 * gb } // todo: flesh out

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
  var cents = storageCost(storage) + requestsCost(requests) + transferCost(transfer);

  // condition(resources[1].requests > 40000) // B goes viral
  return cents / 100;
}

var dist = MH(model, 30000);
print("Expected cost: $" + expectation(dist))
viz.density(dist)
~~~~
