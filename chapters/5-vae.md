---
layout: subchapter
title: Data - prediction (VAE)
custom_js:
- /assets/js/sketch.js
- /assets/js/knear.js
- /assets/js/mnist-io.js
- /assets/js/msgpack.min.js
custom_css:
- /assets/css/mnist-io.css
---

<!-- nb: for drawing to work properly,  have to declare width with element attributes, not css -->
<div class='mnist'>
<div class='loading'>
Loading:<br />
<table>
<tr><td>Encoder parameters</td><td class="progress-params"></td></tr>
<tr><td>Latents</td><td class="progress-latents"></td></tr>
</table>
</div>
<div class='ready'>
<canvas width='160' height='160'></canvas>
<div class="downsampled"></div>
<div class="result"><b>Classification</b>: <span></span></div>
<button class="classify">Classify</button> <button class="clear">Clear</button>
</div>
</div>


# Modeling binary images

## A simple model

Here's a simple (but incomplete) model of a 3x3 binary image:

~~~~
var f = function(z) {
  // Not yet implemented.
};
var z = sample(DiagCovGaussian({
  mu: zeros([2, 1]),
  sigma: ones([2, 1])
}));
var probs = f(z);
var pixels = sample(MultivariateBernoulli({ps: probs}));
pixels;
~~~~

Where we would like `f` to be some function that maps our latent `z`(in $$\mathbb{R}^2$$) to a vector of probabilities.

`sample(MultivariateBernoulli({ps: probs}))` is roughly `map(flip, probs)`.
We just have 0/1 not true/false, and a tensor rather than an array.

## Using a neural net for `f`

One choice of function for `f` is a neural net.
Since we don't know what the weights of the net are, so we put a prior on those:

~~~~
///fold:
var printPixels = function(t) {
  var ar = map(function(x) { x > 0.5 ? "*" : " "}, t.data);
  print([ar.slice(0,3).join(' '),
   ar.slice(3,6).join(' '),
   ar.slice(6,9).join(' ')
  ].join('\n'))
}
///

var sampleMatrix = function() {
  var mu = zeros([18,1]);
  var sigma = ones([18,1]);
  T.reshape(sample(DiagCovGaussian({mu: mu, sigma: sigma})),
            [9,2])
};

var z = sample(DiagCovGaussian({mu: zeros([2,1]), sigma: ones([2,1])}));

var W = sampleMatrix();
var f = function(z) {
  return T.sigmoid(T.dot(W, z));
};
var probs = f(z);

var pixels = sample(MultivariateBernoulli({ps: probs}));

printPixels(pixels)
~~~~

## Add observations

We can extend this to multiple images, and add observations in the usual way.
Note that the neural net `f` is shared across all data points.

~~~~
///fold:
var printPixels = function(t) {
  var ar = map(function(x) { x > 0.5 ? "*" : " "}, t.data);
  print([ar.slice(0,3).join(' '),
   ar.slice(3,6).join(' '),
   ar.slice(6,9).join(' ')
  ].join('\n'))
}
///

var letterL = Vector([1,0,0,
                      1,0,0,
                      1,1,0]);

var number7 = Vector([1,1,1,
                      0,0,1,
                      0,1,0])

var data = append(repeat(50, function() { letterL }),
                  repeat(50, function() { number7 }))

var observe = function(dist, x) {
  factor(dist.score(x))
}

// note to self: currently, if distribution is parameterized by
// reals or tensors, we have an auto guide
var sampleMatrix = ///fold:
function() {
  var mu = zeros([18,1]);
  var sigma = ones([18,1]);
  var v = sample(DiagCovGaussian({mu: mu, sigma: sigma}));
  return T.reshape(v, [9,2]);
}
///

var model = function() {

  var W = sampleMatrix();
  var f = function(z) { T.sigmoid(T.dot(W, z));  };

  var zs = map(function(x) {
    var z = sample(DiagCovGaussian({mu: zeros([2,1]), sigma: ones([2,1])}));
    var probs = f(z);
    factor(MultivariateBernoulli({ps: probs}).score(x));
    return z;
  }, data);

  return {W: W, zs: zs}
};

util.seedRNG(1)
var dist = Infer({method: 'optimize',
                  optMethod: {adam: {stepSize: 0.1}},
                  steps: 150
                 }, model)

var out = dist.sample();
var W = out.W;
var someZs = repeat(10, function() { uniformDraw(out.zs) })
map(function(z) {
  printPixels(T.sigmoid(T.dot(W, z)))
  print('---------')
}, someZs)
~~~~

Inference in this model will work... but how well?

* This straight-forward application of VI has *lots* of parameters to
  optimize. Means and variances of the neural net weights, means and
  variances of the latent `z` for each data point.
* Learning the uncertainty in the weights of a neural net could be
  difficult.
* If we wanted to model MNIST digits we would have to map over 60K
  data points during every execution.

There are changes we can make to address these problems:

## Improvement 1 - Recognition net

The default guide gives us a fully factorized guide. i.e. each `z` has
an independent guide distribution. (This is called mean-field
variational inference.)

Instead, we can generate the guide parameters using a neural net that maps a single image to the parameters of the guide for that image.

The weights of this net are still variational parameters (because they
are parameters of the guide), but we now have parameters shared across
data points.

This technique is one approach to "amortized inference".

~~~~
///fold:
var printPixels = function(t) {
  var ar = map(function(x) { x > 0.5 ? "*" : " "}, t.data);
  print([ar.slice(0,3).join(' '),
   ar.slice(3,6).join(' '),
   ar.slice(6,9).join(' ')
  ].join('\n'))
}

var letterL = Vector([1,0,0,
                      1,0,0,
                      1,1,0]);

var number7 = Vector([1,1,1,
                      0,0,1,
                      0,1,0])

var data = append(repeat(50, function() { letterL }),
                  repeat(50, function() { number7 }))
///

var sampleMatrix = ///fold:
function() {
  var mu = zeros([18,1]);
  var sigma = ones([18,1]);
  var v = sample(DiagCovGaussian({mu: mu, sigma: sigma}));
  return T.reshape(v, [9,2]);
}
///

var model = function() {

  var W = sampleMatrix();

  var f = function(z) {
    return T.sigmoid(T.dot(W, z));
  };

  // *** START NEW ***
  var Wmu = tensorParam([2, 9], 0, 0.1);
  var Wsigma = tensorParam([2, 9], 0, 0.1);

  var recogNet = function(x) {
    // the net in the vae has an extra hidden layer
    var mu = T.dot(Wmu, x);
    // pass through exp to ensure sigma is +ve
    var sigma = T.exp(T.dot(Wsigma, x));
    // these parameters match what is expected by DiagCovGaussian
    return {mu: mu, sigma: sigma};
  };
  // *** END NEW ***

  var zs = map(function(x) {

    var z = sample(DiagCovGaussian({mu: zeros([2,1]), sigma: ones([2,1])}), {
      // *** START NEW ***
      guide: DiagCovGaussian(recogNet(x))
      // *** END NEW ***
    });

    var probs = f(z);

    observe(MultivariateBernoulli({ps: probs}), x);

    return z;

  }, data);

  // return the things we're interested in
  return {zs: zs, W: W};

};

util.seedRNG(1)
var dist = Infer({method: 'optimize',
                  optMethod: {adam: {stepSize: 0.1}},
                  steps: 150
                 }, model)

var out = dist.sample();
var W = out.W;
var someZs = repeat(10, function() { uniformDraw(out.zs) })
map(function(z) {
  printPixels(T.sigmoid(T.dot(W, z)))
  print('---------')
}, someZs)
~~~~

## Improvement 2 - Use point estimates of the neural net weights

Instead of trying to infer the full posterior over the weights of `f`,
we can use a `Delta` distribution as the guide. When optimizing the
variational objective, this is equivalent to doing maximum likelihood
with regularization for these parameters. (i.e. MAP estimation.)

~~~~
///fold:
var printPixels = function(t) {
  var ar = map(function(x) { x > 0.5 ? "*" : " "}, t.data);
  print([ar.slice(0,3).join(' '),
   ar.slice(3,6).join(' '),
   ar.slice(6,9).join(' ')
  ].join('\n'))
}

var letterL = Vector([1,0,0,
                      1,0,0,
                      1,1,0]);

var number7 = Vector([1,1,1,
                      0,0,1,
                      0,1,0])

var data = append(repeat(50, function() { letterL }),
                  repeat(50, function() { number7 }))
///

var sampleMatrix = ///fold:
function() {
  var mu = zeros([18,1]);
  var sigma = ones([18,1]);
  var v = sample(DiagCovGaussian({mu: mu, sigma: sigma}), {
    // *** START NEW ***
    guide: Delta({v: tensorParam([18, 1], 0, 1)})
    // *** END NEW ***
  });
  return T.reshape(v, [9,2]);
}
///

var model = function() {

  var W = sampleMatrix()

  var f = function(z) {
    return T.sigmoid(T.dot(W, z));
  };

  var Wmu = tensorParam([2, 9], 0, 0.1);
  var Wsigma = tensorParam([2, 9], 0, 0.1);

  var recogNet = function(x) {
    // the net in the vae has an extra hidden layer
    var mu = T.dot(Wmu, x);
    // pass through exp to ensure sigma is +ve
    var sigma = T.exp(T.dot(Wsigma, x));
    // these are parameters match what is expected by DiagCovGaussian
    return {mu: mu, sigma: sigma};
  };

  var zs = map(function(x) {

    var z = sample(DiagCovGaussian({mu: zeros([2,1]), sigma: ones([2,1])}), {
      guide: DiagCovGaussian(recogNet(x))
    });

    var probs = f(z);

    observe(MultivariateBernoulli({ps: probs}), x);

    return z;

  }, data);

  // return the things we're interested in
  return {zs: zs, W: W};

};

util.seedRNG(1)
var dist = Infer({method: 'optimize',
                  optMethod: {adam: {stepSize: 0.1}},
                  steps: 150
                 }, model)

var out = dist.sample();
var W = out.W;
var someZs = repeat(10, function() { uniformDraw(out.zs) })
map(function(z) {
  printPixels(T.sigmoid(T.dot(W, z)))
  print('---------')
}, someZs)
~~~~

## Improvement 3 - Use `mapData`

With VI it's possible to take an optimization step without looking at
all of the data. Instead, we can sub-sample a mini-batch of data, and
use only this subset to compute gradient estimates.

Sub-sampling the data adds extra stochasticity to the already
stochastic gradient estimates, but they are still correct in
expectation.

In code, all we do to use mini-batches is switch from using `map` to
`mapData`.

Note that by using `mapData` we're asserting that the choices that
happen in the function are conditionally independent, given the random
choices the happen before the `mapData`.

~~~~
///fold:
var printPixels = function(t) {
  var ar = map(function(x) { x > 0.5 ? "*" : " "}, t.data);
  print([ar.slice(0,3).join(' '),
   ar.slice(3,6).join(' '),
   ar.slice(6,9).join(' ')
  ].join('\n'))
}

var letterL = Vector([1,0,0,
                      1,0,0,
                      1,1,0]);

var number7 = Vector([1,1,1,
                      0,0,1,
                      0,1,0])

var data = append(repeat(50, function() { letterL }),
                  repeat(50, function() { number7 }))
///

var sampleMatrix = ///fold:
function() {
  var mu = zeros([18,1]);
  var sigma = ones([18,1]);
  var v = sample(DiagCovGaussian({mu: mu, sigma: sigma}), {
    guide: Delta({v: tensorParam([18, 1], 0, 1)})
  });
  return T.reshape(v, [9,2]);
}
///

var model = function() {

  var W = sampleMatrix()

  var f = function(z) {
    return T.sigmoid(T.dot(W, z));
  };

  var Wmu = tensorParam([2, 9], 0, 0.1);
  var Wsigma = tensorParam([2, 9], 0, 0.1);

  var recogNet = function(x) {
    // the net in the vae has an extra hidden layer
    var mu = T.dot(Wmu, x);
    // pass through exp to ensure sigma is +ve
    var sigma = T.exp(T.dot(Wsigma, x));
    // these are parameters match what is expected by DiagCovGaussian
    return {mu: mu, sigma: sigma};
  };

  // *** NEW ***
  var zs = mapData({data: data, batchSize: 5}, function(x) {

    var z = sample(DiagCovGaussian({mu: zeros([2,1]), sigma: ones([2,1])}), {
      guide: DiagCovGaussian(recogNet(x))
    });

    var probs = f(z);

    observe(MultivariateBernoulli({ps: probs}), x);

    return z;

  });

  // return the things we're interested in
  return {zs: zs, W: W};

};

util.seedRNG(1)
var dist = Infer({method: 'optimize',
                  optMethod: {adam: {stepSize: 0.1}},
                  steps: 150
                 }, model)

var out = dist.sample();
var W = out.W;
var someZs = repeat(10, function() { uniformDraw(out.zs) })
map(function(z) {
  printPixels(T.sigmoid(T.dot(W, z)))
  print('---------')
}, someZs)
~~~~

This model and inference strategy is known as the Variational
Auto-encoder in the machine learning literature. See
[vae.wppl](https://github.com/probmods/webppl-daipp/blob/master/examples/vae.wppl)
for a more complete demo.
