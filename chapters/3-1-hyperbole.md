---
layout: subchapter
title: Hyperbole
description: "Nonliteral Language Understanding for Number Words."
---

A model of hyperbole understanding as pragmatic reasoning:

The speaker chooses an utterance conditioned on the listener inferring information that is correct and relevant to the speaker’s communicative goal (or QUD). The goal can either be to communicate the state of the world, the speaker’s attitude towards the state of the world (affect), or both. The listener chooses an interpretation conditioned on the speaker selecting the given utterance when intending to communicate this meaning. In this example the state of the world is how much an electric kettle cost.


~~~~
var observe = function(dist,val) {
  factor(dist.score(val))
}

// Define list of kettle prices under consideration (possible price states)
var states = [50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001]

// Prior probability of kettle prices (taken from human experiments)
var statePrior = function() {
  return categorical({ps: [0.4205, 0.3865, 0.0533, 0.0538, 0.0223, 0.0211, 0.0112, 0.0111, 0.0083, 0.0120],
                      vs: states})
}

// Probability that given a price state, the speaker thinks it's too
// expensive (taken from human experiments)
var valencePriors = {50: 0.3173,
                     51: 0.3173,
                     500: 0.7920,
                     501: 0.7920,
                     1000: 0.8933,
                     1001: 0.8933,
                     5000: 0.9524,
                     5001: 0.9524,
                     10000: 0.9864,
                     10001: 0.9864}
var valencePrior = function(state) {
  return bernoulli(valencePriors[state]) //todo: convert to 1,0?
}


// Prior over QUDs (de-refernce through qud name since mem doesn't play nice with function values)
var qudPrior = function() {
  return categorical({ps:[0.17, 0.32, 0.17, 0.17, 0.17],
                      vs:['s',  'v',  'sv', 'as', 'asv'] })
}

var qudFns =
    {s: function(state, valence){return state},
     v: function(state, valence){return valence},
     sv: function(state, valence){return [state, valence]},
     as: function(state, valence){return approx(state, 10)},
     asv: function(state, valence){return [approx(state, 10), valence]}}


// Round x to nearest multiple of b (used for approximate interpretation):
var approx = function(x,b) {return b * Math.round(x / b)}

// Define list of possible utterances (same as price states)
var utterances = states

// Sharp numbers are costlier
var utterancePrior = function() {
  return categorical({vs: utterances,
                      ps: [0.18, 0.1, 0.18, 0.1, 0.18, 0.1, 0.18, 0.1, 0.18, 0.1]})
}

// Literal interpretation "meaning" function, just check if uttered number reflects price state
var literalInterpretation = function(utterance, state) {
  return utterance == state
}

// Literal listener, infers the qud value assuming the utterance is true of the state
var litListener = cache(function(utterance, qud) {
  return Infer({method: 'enumerate'},
   function() {
    var state = statePrior()
    var valence = valencePrior(state)
    var qudfn = qudFns[qud]
    condition(literalInterpretation(utterance, state))
    return qudfn(state, valence)
  })
})

// Speaker, chooses an utterance to convey a particular value of the qud
var speaker = cache(function(val, qud){
  return Infer({method: 'enumerate'},
   function(){
    var utterance = utterancePrior()
    observe(litListener(utterance,qud), val)
    return utterance
  })
})

// Pragmatic listener, jointly infers the price state, speaker valence, and QUD
var pragListener = function(utterance) {
  return Infer({method: 'enumerate'},
   function(){
    var state = statePrior()
    var valence = valencePrior(state)
    var qud = qudPrior()
    var qudfn = qudFns[qud]
    observe(speaker(qudfn(state,valence), qud), utterance)
    return {state: state, valence: valence}
  })
}

viz.auto(pragListener(10000))
~~~~

This example is based on "Nonliteral Language Understanding for Number Words." Kao, Justine T and Wu, Jean Y and Bergen, Leon and Goodman, Noah D (2014). Proceedings of the National Academy of Sciences.
