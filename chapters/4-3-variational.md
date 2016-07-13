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

The previous parts of this chapter focused on Monte Carlo methods for approximate inference: algorithms that generate a (large) collection of samples to represent the posterior distribution. This is a *non-parametric* representation of the posterior. On the other side of the same coin, we have *parametric* representations--that is, we can try to design and fit a paramametrized density function to approximate the posterior distribution. 

This is the approach taken by the family of [variational inference](http://docs.webppl.org/en/master/inference.html#optimization) methods, and WebPPL provides a version of these algorithms via the `optimize` inference option (the name 'optimize' comes from the fact that we're optimizing the parameters of a density function such it is as close as possible to the true posterior).
Below, we use `optimize` to fit the hyperparameters of a Gaussian distribution from data:

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
  // Also try using MCMC and seeing how long it takes to converge
  // method: 'MCMC',
  // onlyMAP: true,
  // samples: 5000
}, gaussianModel);

sample(post);
~~~~

Run this code, then try using MCMC to achieve the same result. You'll notice that MCMC takes significantly longer to converge.

How does `optimize` work?
It takes the given arguments of random choices in the program (in this case, the arguments `(0, 1)` and `(0, 1)` to the two `gaussian` random choices used as priors) and replaces with them with free parameters which it then optimizes to bring the resulting distribution as close as possible to the true posterior. This approach is also known as *mean-field variational inference*: approximating the posterior with a product of independent distributions (one for each random choice in the program).

Here's a more complicated example of using mean-field inference for a simple Latent Dirichlet Allocation model:

~~~~
var nTopics = 2;
var vocabulary = ['zebra', 'wolf', 'html', 'css'];
var docs = {
  'doc1': 'zebra wolf zebra wolf zebra wolf html wolf zebra wolf'.split(' '),
  'doc2': 'html css html css html css html css html css'.split(' '),
  'doc3': 'zebra wolf zebra wolf zebra wolf zebra wolf zebra wolf'.split(' '),
  'doc4': 'html css html css html css html css html css'.split(' '),
  'doc5': 'zebra wolf zebra html zebra wolf zebra wolf zebra wolf'.split(' ')
};

var makeWordDist = function() { dirichlet(ones([vocabulary.length,1])) };
var makeTopicDist = function() { dirichlet(ones([nTopics,1])) };

var model = function() {
  var wordDistForTopic = repeat(nTopics, makeWordDist);
  
  mapObject(function(docname, words) {
    var topicDist = makeTopicDist();
    map(function(word) {
      var topic = discrete(topicDist);
      var wordDist = wordDistForTopic[topic];
      var wordID = vocabulary.indexOf(word);
      factor(Discrete({ps: wordDist}).score(wordID));
    }, words);
  }, docs);

  return map(function(v) { return _.toArray(v.data); }, wordDistForTopic);
};

var post = Infer({
  method: 'optimize',
  optMethod: 'adam',
  steps: 1000
}, model);

var samp = sample(post);

print("Topic 1:"); viz.bar(vocabulary, samp[0]);
print("Topic 2:"); viz.bar(vocabulary, samp[1]);
~~~~

Unfortunately, running this program produces poor results--the resulting word distributions per-topic do not do a good job of separating the animal-related words from the programming-related ones. This is because WebPPL's implementation of variational inference (for the time being, anyway) works much better with continuous random choices than discrete ones (notice the `discrete` choice of topic in the program above). In particular, the algorithm works best when the program contains only random choices from the following distributions:

  - `Gaussian`
  - `Dirichlet`

If, when running `Infer` with method `optimize`, the program prints the message `ELBO: Using PW estimator`, then the program satisfies this criterion. If you see message about a different estimator, then expect things not to work as well.

We can make LDA better suited for variational inference by explicitly integrating out the latent choice of topic per word:

~~~~
///fold:
var nTopics = 2;
var vocabulary = ['zebra', 'wolf', 'html', 'css'];
var docs = {
  'doc1': 'zebra wolf zebra wolf zebra wolf html wolf zebra wolf'.split(' '),
  'doc2': 'html css html css html css html css html css'.split(' '),
  'doc3': 'zebra wolf zebra wolf zebra wolf zebra wolf zebra wolf'.split(' '),
  'doc4': 'html css html css html css html css html css'.split(' '),
  'doc5': 'zebra wolf zebra html zebra wolf zebra wolf zebra wolf'.split(' ')
};
var makeWordDist = function() { dirichlet(ones([vocabulary.length,1])) };
var makeTopicDist = function() { dirichlet(ones([nTopics,1])) };
///

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

The computed posterior now exhibits much better separation between topics.

## Beyond Mean Field: Custom Guide Distributions

Sometimes, the basic mean-field approximation strategy isn't quite enough. Below is a slightly-modified version of the `constrainedSum` program from the previous section, this time using mean-field variational inference.

~~~~
var n = 10;
var targetSum = n / 2;

var numPrior = Gaussian({mu: 0, sigma: 2});
var sampleNumber = function() {
  return sample(numPrior);
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
map(function(x) {
  var numsRounded = map(function(xi) { xi.toFixed(2) }, x).join(' ');
  return 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + numsRounded;
}, samps).join('\n');
~~~~

Try running this program. Notice the structure of the output posterior samples--the mean-field algorithm has essentially learned that to achieve a sum of 5 from ten numbers, it can make each number independently take a value as close as possible ot 0.5. This is not a particularly good approximation of the true posterior.

To do better, we need to move away from the independence assumptions of mean-field and try to capture the dependencies between the different random choices that are induced by the sum-to-5 constraint. One reasonable idea is to posit that each random choice should be close to an affine transformation of all the choices that came before it:

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
map(function(x) {
  var numsRounded = map(function(xi) { xi.toFixed(2) }, x).join(' ');
  return 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + numsRounded;
}, samps).join('\n');
~~~~

In the above program, we are introducing a few new features:

 - `guide` optionally specifies how each random choice should be distributed in the approximate posterior.
 - `scalarParam(mu, sigma)` samples a new optimizable parameter value.

To make this more concrete: for a random choice `gaussian(0, 1)`, mean-field, under the hood, actually does something like:

~~~~
gaussian(0, 1, {
	guide: Gaussian({mu: scalarParam(0, 1), sigma: Math.exp(scalarParam(0, 1))})
});
~~~~

With these new features at our disposal, the revised program above does a much better job of capturing the variability of the true posterior distribution.

## Decoupling Optimization from Sampling

One nice feature of a parametric approximation to the posterior is that once we have optimized its parameters, we can generate arbitrarily many samples from it whenever we want. So, if we first optimize the parameters of the `constrainedSum` program and then save the optimized parameters:

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

we can then draw samples using these optimized parameters without having to re-run optimization:

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

var post = Infer({
  method: 'forward',
  samples: 100,
  guide: true,
  params: wpEditor.get('constrainedSumParams')
}, constrainedSum);

var samps = repeat(10, function() {
  return sample(post);
});
map(function(x) {
  var numsRounded = map(function(xi) { xi.toFixed(2) }, x).join(' ');
  return 'sum: ' + sum(x).toFixed(3) + ' | nums: ' + numsRounded;
}, samps).join('\n');
~~~~

One motivation for this division: if your program makes predictions from data (as the next chapter covers), then you can spend time up-front optimizing parameters for an approximate posterior that work well for many possible input data. Then, when presented with new data, all that's required is to quickly generate some samples using the pre-optimized parameters. This paradigm is sometimes called *amortized inference*.

## Exercises

### 1. Improve Gaussian mixture model efficiency

Below, we attempt to use mean-field inference with a Gaussian mixture model. However, due to the `discrete` choice of mixture component per datapoint, an inefficient estimator is used and results are poor:

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

Try modifying this program to integrate out the `discrete` choice and improve performance. Hint: you may find the `logsumexp` function provided below to be helpful.

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
	// Fill in function body
};

var post = Infer({
  method: 'optimize',
  optMethod: 'adam',
  steps: 1000
}, gaussianMixtureModel);

sample(post);
~~~~

<!-- ~~~~
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
~~~~ -->

[Next: Additional Exercises]({{ "/chapters/4-4-exercises.html" | prepend: site.baseurl }})