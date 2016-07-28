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
<canvas width='224' height='224'></canvas>
<div class="downsampled"></div>
<div class="result"><b>Classification</b>: <span></span></div>
<button class="classify">Classify</button> <button class="clear">Clear</button>
</div>
</div>


# model binary images

## a simple model

here's a simple model of a 3x3 binary image:

~~~~
var z = sample(TensorGaussian({mu: 0, sigma: 1, dims: [2, 1]}));
var probs = f(z);
var pixels = sample(MultivariateBernoulli({ps: probs}));
pixels;
~~~~

`f` is some function that maps our latent `z`(in $$\mathbb{R}^2$$) to a vector of probabilities.

`sample(MultivariateBernoulli({ps: probs}))` is roughly `map(flip, probs)`.
We just have 0/1 not true/false, and a tensor rather than an array.

## use a neural net for f


we can use a neural net for `f`.
we don't know what the weights of the net are, so we put a prior on those:

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

var z = sample(TensorGaussian({mu: 0, sigma: 1, dims: [2, 1]}));

var W = sample(TensorGaussian({mu: 0, sigma: 1, dims: [9, 2]}));
var f = function(z) {
  return T.sigmoid(T.dot(W, z));
};
var probs = f(z);

var pixels = sample(MultivariateBernoulli({ps: probs}));

printPixels(pixels)
~~~~

this program now runs.

## add observations

we can extend this to multiple images, and add observations in the usual way.
note that the neural net `f` is shared across all data points.

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

var data = append(repeat(100, function() { letterL }),
                  repeat(100, function() { number7 }))

var observe = function(dist, x) {
  factor(dist.score(x))
}

var model = function() {
  var W = sample(TensorGaussian({mu: 0, sigma: 1, dims: [9, 2]}));
  var f = function(z) { T.sigmoid(T.dot(W, z));  };

  var zs = map(function(x) {
    var z = sample(TensorGaussian({mu: 0, sigma: 1, dims: [2, 1]}));
    var probs = f(z);
    factor(MultivariateBernoulli({ps: probs}).score(x));
    return z;
  }, data);

  // return a sampler for a new code z
  var newZ = sample(TensorGaussian({mu: 0, sigma: 1, dims: [2, 1]}));
  var newProbs = f(newZ);
  return newProbs
};

util.seedRNG(1)
var dist = Infer({method: 'MCMC',
                  kernel: 'HMC',
                  samples: 500,
                  callbacks: [wpEditor.MCMCProgress()]
                 }, model)

repeat(10, function() {
  printPixels(dist.sample())
  print('--------')
})
~~~~

inference should work... but how well?

* we have 60K data points to map over if modeling mnist digits.
* a straight-forward application of vi has *lots* of parameters to
  optimize. means and variances of the nn weights, means and variances
  of a z for each data point.
* what else?...

(note the vi doesn't work on the model at this point because `TensorGaussian` can't be guided automatically. i can fix that if we'd like to show it?)

we can make some improvements...

## improvement 1 - recognition net

the default guide (if it worked, see above) would give us a fully factorized guide.
i.e., each `z` would have its own independent guide distribution (this is mean-field).

instead, we can generate the guide parameters using a neural net that maps a single image to the parameters of the guide for that image.

the weights of this net are variational parameters (because they are parameters of the guide).
importantly we now have parameters shared across data points.
say "amortized inference" perhaps?

~~~~
var data = [Vector([1, 0, 0, 0, 1, 0, 0, 0, 1])];

var model = function() {

  var W = sample(TensorGaussian({mu: 0, sigma: 1, dims: [9, 2]}));

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

    var z = sample(TensorGaussian({mu: 0, sigma: 1, dims: [2, 1]}), {
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
~~~~

## improvement 2 - use point estimates of the neural net weights

instead of trying to infer the full posterior over the weights of `f`, we can use a `Delta`distribution as the guide. when optimizing the elbo, this is equivalent to doing maximum likelihood with regularization. (i should really double check the math here...)

~~~~
var data = [Vector([1, 0, 0, 0, 1, 0, 0, 0, 1])];

var model = function() {

  var W = sample(TensorGaussian({mu: 0, sigma: 1, dims: [9, 2]}), {
    // *** START NEW ***
    guide: Delta({v: tensorParam([9, 2], 0, 0.1)})
    // *** END NEW ***
 });

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

    var z = sample(TensorGaussian({mu: 0, sigma: 1, dims: [2, 1]}), {
      guide: DiagCovGaussian(recogNet(x))
    });

    var probs = f(z);

    observe(MultivariateBernoulli({ps: probs}), x);

    return z;

  }, data);

  // return the things we're interested in
  return {zs: zs, W: W};

};
~~~~

this should run, since we've now specified the guide by hand, side-stepping the problem guiding `TensorGaussian` automatically.

~~~~
Infer({method: 'optimize', steps: 10, samples: 10}, model);
~~~~

## improvement 3 - use `mapData`

if we switch from `map` to `mapData` we can do mini-batches.

note that by using `mapData` we're asserting that the choices that happen in the function are conditionally independent, given the stuff outside.

~~~~
// TODO: add mapData example
~~~~

## this is the vae

see [vae.wppl](https://github.com/probmods/webppl-daipp/blob/master/examples/vae.wppl) for a more complete demo.