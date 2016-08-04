---
layout: subchapter
title: Data - presidential election
---

# Basic model

## Learning a state-wide preference from poll data

~~~~
// infer true state preference given poll
var trueStatePref = function() {
  var pref = beta(1,1);
  var counts = {trump: 304, clinton: 276};
  var total = sum(_.values(counts));
  observe(Binomial({n: total, p: pref}), counts.clinton);
  return pref;
}
var dist = Infer({method: 'MCMC', samples: 10000}, trueStatePref);
viz.density(dist,{bounds: [0.4, 0.6]})
editor.put('pref', dist);
~~~~

## Using the learned preference to simulate election-day results

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
var dist = Infer({method: 'forward', samples: 1000}, sampleElection)
viz.table(dist)
~~~~

Notice that results depend on quality of inference for learned preference above.
(Aside: only the `forward` and `rejection` inference methods work well for this example because the Binomial scorer takes time linear in `n`).
We can optimize this quite a bit, taking advantage of (1) conjugacy, (2) the Gaussian approximation to binomial, and (3) the CDF:

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

var p0 = 1+276;
var p1 = 1+304;
var simulateResult = function() {
  var p = beta(p0, p1); // use conjugacy

  // gaussian approximation to binomial because binomial with large n is slow
  var n = 2400000; // 2012 turnout
  var np = n * p;

  // use cdf to sample a winner rather than explicitly sampling a number of votes
  var clintonWinProb = (1 - gaussianCDF(n/2, np, Math.sqrt(np * (1-p))));
  //var winner = flip(clintonWinProb) ? "clinton" : "trump";

  return clintonWinProb
}

expectation(Infer({method: 'forward', samples: 1e5}, simulateResult))
~~~~

This implementation is nice because it is fast, but it also has disadvantages.
Can you think of any?

# A time-varying model

~~~~
var dist0 = {
  d: 0.40,
  r: 0.40,
  u: 0.20
};

var alpha = 0.02;

var step = function(curDist) {
  var d = curDist.d, r = curDist.r, u = curDist.u;

  var joiners  = uniform(0, 0.2  ) * u;
  var leaversD = uniform(0, 0.001) * d;
  var leaversR = uniform(0, 0.001) * r;

  // assume that the deciders noisily mirror the people who have already decided
  var joinerSides = dirichlet(T.mul(Vector(_.values(_.omit(curDist, 'u'))), alpha)),
      joinersD = T.get(joinerSides, 0) * joiners,
      joinersR = T.get(joinerSides, 1) * joiners;

  return {
    d: d - leaversD + joinersD,
    r: r - leaversR + joinersR,
    u: u - joiners + leaversR + leaversD
  }
}

var evolve = function(dists, steps) {
  if (steps == 0) {
    return dists
  } else {
    var curDist = _.last(dists),
        nextDist = step(curDist);
    return evolve(dists.concat(nextDist), steps - 1)
  }
}

var steps = 5;
var path = evolve([dist0], steps);

var vdist = _.flatten(mapIndexed(function(t, dist) {
  var keys = _.keys(dist), vals = _.values(dist);
  return map2(function(k,v) { return {time: t, fraction: v, group: k} }, keys, vals)
}, path));

print('support for dems, reps, or undecided over time')
viz.line(vdist, {groupBy: 'group'})

print("percent favoring dems to reps over time");
var ds = _.pluck(path, 'd'),
    rs = _.pluck(path, 'r');
viz.line(_.range(steps+1),
         map2(function(d,r) { d / (d + r) }, ds, rs))
~~~~

Later polls are more accurate than earlier polls:

~~~~
///fold:
var dist0 = {
  d: 0.40,
  r: 0.40,
  u: 0.20
};

var alpha = 0.02;

var step = function(curDist) {
  var d = curDist.d, r = curDist.r, u = curDist.u;

  var joiners  = uniform(0, 0.2  ) * u;
  var leaversD = uniform(0, 0.001) * d;
  var leaversR = uniform(0, 0.001) * r;

  // assume that the deciders noisily mirror the people who have already decided
  var joinerSides = dirichlet(T.mul(Vector(_.values(_.omit(curDist, 'u'))), alpha)),
      joinersD = T.get(joinerSides, 0) * joiners,
      joinersR = T.get(joinerSides, 1) * joiners;

  return {
    d: d - leaversD + joinersD,
    r: r - leaversR + joinersR,
    u: u - joiners + leaversR + leaversD
  }
}

var evolve = function(dists, steps) {
  if (steps == 0) {
    return dists
  } else {
    var curDist = _.last(dists),
        nextDist = step(curDist);
    return evolve(dists.concat(nextDist), steps - 1)
  }
}

var steps = 5;
///

var doPoll = function(pref, size) {
  _.object(_.keys(pref),
           multinomial({ps: _.values(pref), n: size}))
}

var KL = function(pProbs, qProbs) {
  return sum(map2(function(p, q) { p === 0 ? 0 : p * (Math.log(p) - Math.log(q)); },
                  pProbs,
                  qProbs))
}

var pollsDist = Infer(
  {method: 'SMC', particles: 400},
  function() {
    var path = evolve([dist0], 5);

    var poll0 = doPoll(path[0], 100);
    var poll3 = doPoll(path[3], 100);
    var goodness0 = -1 * KL(_.values(poll0), _.values(_.last(path)));
    var goodness3 = -1 * KL(_.values(poll3), _.values(_.last(path)));

    return {"relative quality of poll 3": goodness3 - goodness0}
});

viz.auto(pollsDist);
~~~~

A different view of this:

~~~~
///fold:
var dist0 = {
  d: 0.40,
  r: 0.40,
  u: 0.20
};

var alpha = 0.002;

var step = function(curDist) {
  var d = curDist.d, r = curDist.r, u = curDist.u;

  var joiners  = uniform(0, 0.2  ) * u;
  var leaversD = uniform(0, 0.001) * d;
  var leaversR = uniform(0, 0.001) * r;

  // assume that the deciders noisily mirror the people who have already decided
  var joinerSides = dirichlet(T.mul(Vector(_.values(_.omit(curDist, 'u'))), alpha)),
      joinersD = T.get(joinerSides, 0) * joiners,
      joinersR = T.get(joinerSides, 1) * joiners;

  return {
    d: d - leaversD + joinersD,
    r: r - leaversR + joinersR,
    u: u - joiners + leaversR + leaversD
  }
}

var evolve = function(dists, steps) {
  if (steps == 0) {
    return dists
  } else {
    var curDist = _.last(dists),
        nextDist = step(curDist);
    return evolve(dists.concat(nextDist), steps - 1)
  }
}

var steps = 5;
///

var doPoll = function(pref, size) {
  _.object(_.keys(pref),
           multinomial({ps: _.values(pref), n: size}))
}

var pollsDist = Infer(
  {method: 'SMC', particles: 500, rejuvSteps: 4},
  function() {

    var tick = function() {
      if (!globalStore.path) {
        globalStore.path = [dist0];
      } else {
        globalStore.path = globalStore.path.concat(step(_.last(globalStore.path)));
      }
    }

    var observePoll = function(counts) {
      var currentPs = _.values(_.last(globalStore.path))
      factor(Multinomial({n: sum(counts),
                          ps: currentPs
                         }).score(counts))
    }

    tick();
    tick();
    observePoll([2400, 2400, 1200]);
    tick();
    tick();
    tick();
    // observePoll([2400, 2400, 1200]);
    tick();

    // final time step
    return _.last(globalStore.path).u

    // all time steps
    // return _.object(['t0','t1','t2','t3','t4','t5'], _.pluck(globalStore.path, 'u'))
});

// final time step
var sd = function(dist) {
  var mean = expectation(dist);
  Math.sqrt( expectation(dist, function(x) { (x - mean) * (x - mean) }) )
}

viz.density(pollsDist, {bounds: [0, 0.2]});
'm: ' + expectation(pollsDist).toFixed(3) +
  ' sd: ' + sd(pollsDist).toFixed(3)

// // all time steps
// viz.auto(pollsDist, {bounds: {t0: [0, 0.3], t1: [0, 0.3], t2: [0, 0.3],
//                               t3: [0, 0.3], t4: [0, 0.3], t5: [0, 0.3]
//                              }})
~~~~
