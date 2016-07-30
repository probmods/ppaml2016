---
layout: 
title: A tour through model space
description: "Looking at a few standard models in WebPPL."
---

In this section we collect together the WebPPL code for a few standard probabilistic models.

## Markov models

Markov model, [by Long](http://longouyang.github.io/nextml/sections/markov-models.html):

~~~~
var words = ["^", "complete", "the", "sandwich", "sentence", "$"];

var T = {
  //                 transition probabilities from current word to
  // current word     ^    complete  the   sandwich   sentence    $
  "^":              [ 0,   0.5,      0.3,  0.1,       0.1,        0   ],
  "complete":       [ 0,   0,        0.6,  0.2,       0.1,        0.1 ],
  "the":            [ 0,   0,        0,    0.6,       0.4,        0   ],
  "sandwich":       [ 0,   0,        0,    0,         0,          1   ],
  "sentence":       [ 0,   0,        0,    0,         0,          1   ]
};

var transition = function(word) {
  var transitionProbabilities = T[word];
  var sampledWordIndex = discrete(transitionProbabilities);
  return words[sampledWordIndex];
}

var _sampleSentence = function(wordsSoFar) {
  var prevWord = wordsSoFar[wordsSoFar.length - 1];
  if (prevWord == "$") {
    return wordsSoFar;
  }
  var nextWord = transition(prevWord);
  return _sampleSentence( wordsSoFar.concat(nextWord) );
}

var sampleSentence = function() {
  return _sampleSentence(["^"]).slice(1,-1).join(" ");
}

viz.hist(repeat(100, sampleSentence))
~~~~

## HMMs

HMM, from webppl `examples/`:

~~~~
var transition = function(s) {
  return s ? flip(0.7) : flip(0.3)
}

var observe = function(s) {
  return s ? flip(0.9) : flip(0.1)
}

var hmm = function(n) {
  var prev = (n == 1) ? {states: [true], observations: []} : hmm(n - 1);
  var newState = transition(prev.states[prev.states.length - 1]);
  var newObs = observe(newState);
  return {
    states: prev.states.concat([newState]),
    observations: prev.observations.concat([newObs])
  };
}

var trueObservations = [false, false, false];

var arrayEq = function(a, b) {
  return (a.length == 0) ? true : (a[0] == b[0] && arrayEq(a.slice(1), b.slice(1)))
}

var stateDist = Enumerate(function() {
  var r = hmm(3);
  factor(arrayEq(r.observations, trueObservations) ? 0 : -Infinity);
  return r.states;
})

viz.hist(stateDist)
~~~~

## PCFGs

PCFG, from webppl `examples/`:

~~~~
var pcfgTransition = function(symbol) {
  var rules = {
    'start': {rhs: [['NP', 'V', 'NP'], ['NP', 'V']], probs: [0.4, 0.6]},
    'NP': {rhs: [['A', 'NP'], ['N']], probs: [0.4, 0.6]}
  };
  return rules[symbol].rhs[discrete(rules[symbol].probs)]
}

var preTerminal = function(symbol) {
  return symbol == 'N' || symbol == 'V' || symbol == 'A'
}

var terminal = function(symbol) {
  var rules = {
    'N': {words: ['John', 'soup'], probs: [0.6, 0.4]},
    'V': {words: ['loves', 'hates', 'runs'], probs: [0.3, 0.3, 0.4]},
    'A': {words: ['tall', 'salty'], probs: [0.6, 0.4]} }
  return rules[symbol].words[discrete(rules[symbol].probs)]
}

var pcfg = function(symbol) {
  preTerminal(symbol) ? [terminal(symbol)] : expand(pcfgTransition(symbol))
}

var expand = function(symbols) {
  if (symbols.length == 0) {
    return []
  } else {
    var f = pcfg(symbols[0])
    return f.concat(expand(symbols.slice(1)))
  }
}

var arrayEq = function(a, b) {
  return (a.length == 0) ? true : (a[0] == b[0] && arrayEq(a.slice(1), b.slice(1)))
}

var nextWordDist = Enumerate(function() {
  var y = pcfg('start')
  factor(arrayEq(y.slice(0, 2), ['tall', 'John']) ? 0 : -Infinity) //yield starts with "tall John"
  return { nextWord: y[2] ? y[2] : '' } //distribution on next word?
}, {maxExecutions: 300})

viz.auto(nextWordDist)
~~~~

## Hierarchical models

Bags of marbles

## Mixture models

Basic LDA, adapted from webppl `examples/`:

~~~~
// Parameters

var vocabulary = ['bear', 'wolf', 'python', 'prolog'];

var topics = {
  'topic1': null,
  'topic2': null
};

var docs = {
  'doc1': 'bear wolf bear wolf bear wolf python wolf bear wolf'.split(' '),
  'doc2': 'python prolog python prolog python prolog python prolog python prolog'.split(' '),
  'doc3': 'bear wolf bear wolf bear wolf bear wolf bear wolf'.split(' '),
  'doc4': 'python prolog python prolog python prolog python prolog python prolog'.split(' '),
  'doc5': 'bear wolf bear python bear wolf bear wolf bear wolf'.split(' ')
};


// Model
var makeWordDist = function() { dirichlet(ones([vocabulary.length,1])) };
var makeTopicDist = function() { dirichlet(ones([_.size(topics),1])) };

var discreteFactor = function(vs, ps /* Vector */, v) {
  var i = vs.indexOf(v);
  factor(Math.log(ps.data[i]));
}

var model = function() {
  var wordDistForTopic = mapObject(makeWordDist, topics);
  var topicDistForDoc = mapObject(makeTopicDist, docs);
  var makeTopicForWord = function(docName, word) {
    var i = discrete(topicDistForDoc[docName]);
    return _.keys(topics)[i];
  };
  var makeWordTopics = function(docName, words) {
    return map(function(word) {return makeTopicForWord(docName, word);},
               words);
  };
  var topicsForDoc = mapObject(makeWordTopics, docs);

  mapObject(
    function(docName, words) {
      map2(
        function(topic, word) {
          discreteFactor(vocabulary, wordDistForTopic[topic], word);
        },
        topicsForDoc[docName],
        words);
    },
    docs);

  return mapObject(function(k,v) { return _.toArray(v.data) },
                   wordDistForTopic)
};

var samp = sample(MH(model, 10000));

print("Topic 1:"); viz.bar(vocabulary, samp.topic1);
print("Topic 2:"); viz.bar(vocabulary, samp.topic2);
~~~~

Collapsed LDA, from webppl `examples/`:

~~~~
// Dirichlet-discrete with sample, observe & getCount (could go in webppl header) // NEW

var makeDirichletDiscrete = function(pseudocounts) {
  var addCount = function(a, i, j) {
    var j = j == undefined ? 0 : j;
    if (a.length == 0) {
      return [];
    } else {
      return [a[0] + (i == j)].concat(addCount(a.slice(1), i, j + 1));
    }
  };
  globalStore.DDindex = 1 + (globalStore.DDindex == undefined ? 0 : globalStore.DDindex);
  var ddname = 'DD' + globalStore.DDindex;
  globalStore[ddname] = pseudocounts;
  var ddSample = function() {
    var pc = globalStore[ddname];  // get current sufficient stats
    var val = sample(Discrete({ps: pc}));  // sample from predictive. (doesn't need to be normalized.)
    globalStore[ddname] = addCount(pc, val); // update sufficient stats
    return val;
  };
  var ddObserve = function(val) {
    var pc = globalStore[ddname];  // get current sufficient stats
    factor(Discrete({ps: normalize(pc)}).score(val));
    // score based on predictive distribution (normalize counts)
    globalStore[ddname] = addCount(pc, val); // update sufficient stats
  };
  var ddCounts = function() {
    return globalStore[ddname];
  };
  return {
    'sample': ddSample,
    'observe': ddObserve,
    'getCounts': ddCounts
  };
};

var dirichletDiscreteFactor = function(vs, dd, v) { // NEW
  var i = vs.indexOf(v);
  var observe = dd.observe;
  observe(i);
}


// Parameters

var vocabulary = ['bear', 'wolf', 'python', 'prolog'];

var topics = {
  'topic1': null,
  'topic2': null
};

var docs = {
  'doc1': 'bear wolf bear wolf bear wolf python wolf bear wolf'.split(' '),
  'doc2': 'python prolog python prolog python prolog python prolog python prolog'.split(' '),
  'doc3': 'bear wolf bear wolf bear wolf bear wolf bear wolf'.split(' '),
  'doc4': 'python prolog python prolog python prolog python prolog python prolog'.split(' '),
  'doc5': 'bear wolf bear python bear wolf bear wolf bear wolf'.split(' ')
};


// Constants and helper functions

var ones = function(n) {
  return repeat(n, function() {return 1.0;});
}

// Model

var makeWordDist = function() {
  return makeDirichletDiscrete(ones(vocabulary.length)); // NEW
};

var makeTopicDist = function() {
  return makeDirichletDiscrete(ones(_.size(topics))); // NEW
};

var model = function() {

  var wordDistForTopic = mapObject(makeWordDist, topics);
  var topicDistForDoc = mapObject(makeTopicDist, docs);
  var makeTopicForWord = function(docName, word) {
    var sampleDD = topicDistForDoc[docName].sample; // NEW
    var i = sampleDD(); // NEW
    return _.keys(topics)[i];
  };
  var makeWordTopics = function(docName, words) {
    return map(function(word) {return makeTopicForWord(docName, word);},
               words);
  };
  var topicsForDoc = mapObject(makeWordTopics, docs);

  mapObject(
      function(docName, words) {
        map2(
            function(topic, word) {
              dirichletDiscreteFactor(vocabulary, wordDistForTopic[topic], word); // NEW
            },
            topicsForDoc[docName],
            words);
      },
      docs);

  // Print out pseudecounts of (dirichlet-discrete) word distributions for topics // NEW
  var getCounts1 = wordDistForTopic['topic1'].getCounts;
  var getCounts2 = wordDistForTopic['topic2'].getCounts;
  var counts = [getCounts1(), getCounts2()];
  // console.log(counts);
  return counts;
};

viz.table(MH(model, 5000), {top: 5})
~~~~

## Logistic regression

From webppl `examples/`:

~~~~
var xs = [-10, -5, 2, 6, 10]
var labels = [false, false, true, true, true]

var model = function() {
  var m = gaussian(0, 1)
  var b = gaussian(0, 1)
  var sigmaSquared = gamma(1, 1)

  var y = function(x) {
    return gaussian(m * x + b, sigmaSquared)
  }
  var sigmoid = function(x) {
    return 1 / (1 + Math.exp(-1 * y(x)))
  }

  map2(
      function(x, label) {
        factor(Bernoulli({p: sigmoid(x)}).score(label))
      },
      xs,
      labels)

  return sigmoid(4)
}

viz.auto(MH(model, 10000))
~~~~

## Bayesian neural net
