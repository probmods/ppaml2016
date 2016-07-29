---
layout: subchapter
title: Data - presidential election
---

## presidential election model

#### learn a state-wide preference from poll data

~~~~
// infer true state preference given poll
var trueStatePref = function() {
  var pref = beta(1,1);
  var counts = {trump: 304, clinton: 276};
  var total = sum(_.values(counts));
  factor(Binomial({n: total, p: pref}).score(counts.clinton))
  return pref;
}
var dist = Infer({method: 'MCMC', samples: 10000}, trueStatePref);
viz.density(dist,{bounds: [0.4, 0.6]})
editor.put('pref', dist);
~~~~

#### use the learned preference to simulate general election results

~~~~
// simulating general election result
var prefDist = editor.get('pref');

var sampleElection = function() {
  var pref = sample(prefDist);
  var turnout = 2400000;
  var clintonVotes = binomial({p: pref, n: turnout});
  var trumpVotes = turnout - clintonVotes;
  var winner = clintonVotes > trumpVotes ? "clinton" : "trump";
  return {winner: winner};
}
var dist = Infer({method: 'MCMC', samples: 10}, sampleElection)
viz.hist(dist)
~~~~

~~~~
///fold:
// http://picomath.org/javascript/erf.js.html
var erf =
function(_x) {
    // constants
    var a1 =  0.254829592;
    var a2 = -0.284496736;
    var a3 =  1.421413741;
    var a4 = -1.453152027;
    var a5 =  1.061405429;
    var p  =  0.3275911;

    // Save the sign of x
    var sign = (_x < 0) ? -1 : 1;
    var x = Math.abs(_x);

    // A&S formula 7.1.26
    var t = 1.0/(1.0 + p*x);
    var y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);

    return sign*y;
}

var gaussianCDF = function(x, mu, sigma) {
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.sqrt(2))))
}
///

// arizona results, adapted from 
// http://elections.huffingtonpost.com/pollster/2016-arizona-president-trump-vs-clinton
// ppp 276, 304
// oh  498, 455
// gqr 129, 144
// ppp 340, 358
// bhr 236, 197
var p0 = 1+276+498+129+340+236; // 1480
var p1 = 1+304+455+144+358+197; // 1459
var simulateResult = function() {
  var p = beta(p0, p1); // use conjugacy
  
  // gaussian approximation to binomial because binomial with large n is slow
  var n = 2400000; // 2012 turnout
  var np = n * p;
  
  // use cdf to sample a winner rather than explicitly sampling a number of votes
  var clintonWinProb = (1 - gaussianCDF(n/2, np, np * (1-p) * p));
  //var winner = flip(clintonWinProb) ? "clinton" : "trump";
  
  return clintonWinProb;
}

// write 
util.seedRNG(1);
expectation(Infer({method: 'forward', samples: 1e6}, simulateResult))

// util.seedRNG(1469818976713)
// 0.5107734807109324
~~~~

#### how do we extend to undecided voters?

- could be useful to know their demographics, says the [nyt], but that's about senate elections
- the fact that they are undecided suggests that maybe, as a group, their actual decisions will be less peaked than the early-deciders (could do this as a softmax on the decided polls and try to learn alpha)
- integrate with discrete-time drift process (using particle filter)


[nyt]: http://www.nytimes.com/2014/11/05/upshot/the-secret-about-undecided-voters-theyre-predictable.html?_r=0

~~~~

~~~~