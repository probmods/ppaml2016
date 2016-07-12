---
layout: subchapter
title: Variational Inference
custom_js:
- /assets/js/draw.js
- /assets/js/custom.js
- /assets/js/paper-full.min.js
custom_css:
- /assets/css/draw.css
---

Outline:

 - Fit single Gaussian
 - Fit mixture of Gaussians (show how much better when you explicitly marginalize out choice of component)
 - Point of what distributions currently work well, look for "using PW estimator," etc.
 - LDA (with enumerating out trick)
 - Custom guides (sum-to-N example where guide is learned linear function of previous numbers)
 - Amortized inference: using Optimize, saving params, and then sampling from guide.

~~~~
var trueMu = 3.5;
var trueSigma = 0.8;

var data = repeat(100, function() { return gaussian(trueMu, trueSigma); });

var gaussianModel = function() {
  var mu = gaussian(0, 1);
  var sigma = Math.exp(gaussian(0, 1));
  var G = Gaussian({mu: mu, sigma: sigma});
  map(function(d) {
    factor(G.score(d));
  }, data);
  return {mu: mu, sigma: sigma};
};

var post = Infer({
  method: 'optimize',
  optMethod: 'adam',
  steps: 500,
  // Try making it work with MCMC; you'll see it takes longer
//   method: 'MCMC',
//   onlyMAP: true,
//   samples: 5000
}, gaussianModel);

sample(post);
~~~~

~~~~
var mus = [-1.2, 0.5, 3.2];
var sigmas = [0.5, 1.2, 0.3];
var weights = [0.4, 0.1, 0.5];

var data = repeat(100, function() {
  var i = discrete(weights);
  return gaussian(mus[i], sigmas[i]);
});

var gaussianMixtureModel = function() {
  var weights = dirichlet(Vector([1, 1, 1]));
  var mus = repeat(3, function() { return gaussian(0, 1); });
  var sigmas = repeat(3, function() { return Math.exp(gaussian(0, 1)); });
  map(function(d) {
    var i = discrete(weights);
    factor(Gaussian({mu: mus[i], sigma: sigmas[i]}).score(d));
  }, data);
  return {mus: mus, sigmas: sigmas, weights: weights};
};

var post = Infer({
  method: 'optimize',
  optMethod: 'adam',
  steps: 1000
}, gaussianMixtureModel);

sample(post);
~~~~

~~~~
///fold:
var mus = [-1.2, 0.5, 3.2];
var sigmas = [0.5, 1.2, 0.3];
var weights = [0.4, 0.1, 0.5];

var data = repeat(100, function() {
  var i = discrete(weights);
  return gaussian(mus[i], sigmas[i]);
});
///

var logsumexp = function(xs) {
  return Math.log(sum(map(function(x) { return Math.exp(x); }, xs)));
};

var gaussianMixtureModel = function() {
  var weights = dirichlet(Vector([1, 1, 1]));
  var mus = repeat(3, function() { return gaussian(0, 1); });
  var sigmas = repeat(3, function() { return Math.exp(gaussian(0, 1)); });
  map(function(d) {
    // Explicitly integrate out choice of mixture component
    var scores = mapIndexed(function(i, mu) {
      var sigma = sigmas[i];
      var w = T.get(weights, i);  // get element i from tensor
      return Gaussian({mu: mu, sigma: sigma}).score(d) + Math.log(w);
    }, mus);
    factor(logsumexp(scores));
  }, data);
  return {mus: mus, sigmas: sigmas, weights: weights};
};

var post = Infer({
  method: 'optimize',
  optMethod: 'adam',
  steps: 1000
}, gaussianMixtureModel);

sample(post);
~~~~

~~~~
var nTopics = 2;

var vocabulary = ['bear', 'wolf', 'python', 'prolog'];

var docs = {
  'doc1': 'bear wolf bear wolf bear wolf python wolf bear wolf'.split(' '),
  'doc2': 'python prolog python prolog python prolog python prolog python prolog'.split(' '),
  'doc3': 'bear wolf bear wolf bear wolf bear wolf bear wolf'.split(' '),
  'doc4': 'python prolog python prolog python prolog python prolog python prolog'.split(' '),
  'doc5': 'bear wolf bear python bear wolf bear wolf bear wolf'.split(' ')
};

var makeWordDist = function() { dirichlet(ones([vocabulary.length,1])) };
var makeTopicDist = function() { dirichlet(ones([nTopics,1])) };

var model = function() {
  var wordDistForTopic = repeat(nTopics, makeWordDist);
  
  mapObject(function(docname, words) {
    var topicDist = makeTopicDist();
    map(function(word) {
      // Explicitly integrate out choice of topic per word
      var wordMarginal = Enumerate(function() {
        var z = discrete(topicDist);
        return discrete(wordDistForTopic[z]);
      });
      var wordID = vocabulary.indexOf(word);
      factor(wordMarginal.score(wordID));
    }, words);
  }, docs);

  return map(function(v) { return _.toArray(v.data); }, wordDistForTopic);
};

var post = Infer({
  method: 'optimize',
  optMethod: 'adam',
  steps: 400
}, model);

var samp = sample(post);

print("Topic 1:"); viz.bar(vocabulary, samp[0]);
print("Topic 2:"); viz.bar(vocabulary, samp[1]);
~~~~

~~~~
var n = 10;
var targetSum = n / 2;

var affine = function(xs) {
  if (xs.length === 0) {
    return scalarParam(0, 1);
  } else {
    return scalarParam(0, 1) * xs[0] +
      affine(xs.slice(1));
  }
};

var numPrior = Gaussian({mu: 0, sigma: 2});
var sampleNumber = function() {
  var guideMu = affine(globalStore.nums);
  var guideSigma = Math.exp(scalarParam(0, 1));
  return sample(numPrior, {
    guide: Gaussian({mu: guideMu, sigma: guideSigma})
  });
};

var constrainedSum = function() {
  globalStore.nums = [];
  repeat(n, function() {
    var num = sampleNumber();
    globalStore.nums = cons(num, globalStore.nums);
  });
  factor(Gaussian({mu: targetSum, sigma: 0.01}).score(sum(globalStore.nums)));
  return globalStore.nums;
};

var post = Infer({
  method: 'optimize',
  optMethod: { adam: {stepSize: 0.25} },
  estimator: { ELBO : {samples: 5} },
  steps: 500,
  samples: 100
}, constrainedSum);

var samps = repeat(10, function() {
  return sample(post);
});
reduce(function(x, acc) {
	return acc + 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + x.toString() + '\n';
}, '', samps);
~~~~

~~~~
///fold:
var n = 10;
var targetSum = n / 2;

var affine = function(xs) {
  if (xs.length === 0) {
    return scalarParam(0, 1);
  } else {
    return scalarParam(0, 1) * xs[0] +
      affine(xs.slice(1));
  }
};

var numPrior = Gaussian({mu: 0, sigma: 2});
var sampleNumber = function() {
  var guideMu = affine(globalStore.nums);
  var guideSigma = Math.exp(scalarParam(0, 1));
  return sample(numPrior, {
    guide: Gaussian({mu: guideMu, sigma: guideSigma})
  });
};

var constrainedSum = function() {
  globalStore.nums = [];
  repeat(n, function() {
    var num = sampleNumber();
    globalStore.nums = cons(num, globalStore.nums);
  });
  factor(Gaussian({mu: targetSum, sigma: 0.01}).score(sum(globalStore.nums)));
  return globalStore.nums;
};
///

var params = Optimize(constrainedSum, {
  optMethod: { adam: {stepSize: 0.25} },
  estimator: { ELBO : {samples: 5} },
  steps: 500,
});
wpEditor.put('constrainedSumParams', params);
~~~~

~~~~
///fold:
var n = 10;
var targetSum = n / 2;

var affine = function(xs) {
  if (xs.length === 0) {
    return scalarParam(0, 1);
  } else {
    return scalarParam(0, 1) * xs[0] +
      affine(xs.slice(1));
  }
};

var numPrior = Gaussian({mu: 0, sigma: 2});
var sampleNumber = function() {
  var guideMu = affine(globalStore.nums);
  var guideSigma = Math.exp(scalarParam(0, 1));
  return sample(numPrior, {
    guide: Gaussian({mu: guideMu, sigma: guideSigma})
  });
};

var constrainedSum = function() {
  globalStore.nums = [];
  repeat(n, function() {
    var num = sampleNumber();
    globalStore.nums = cons(num, globalStore.nums);
  });
  factor(Gaussian({mu: targetSum, sigma: 0.01}).score(sum(globalStore.nums)));
  return globalStore.nums;
};
///

var post = SampleGuide(constrainedSum, {
	samples: 100,
	params: wpEditor.get('constrainedSumParams')
});

var samps = repeat(10, function() {
  return sample(post);
});
reduce(function(x, acc) {
	return acc + 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + x.toString() + '\n';
}, '', samps);
~~~~

## Exercises

TODO

[Next: Additional Exercises]({{ "/chapters/4-4-exercises.html" | prepend: site.baseurl }})