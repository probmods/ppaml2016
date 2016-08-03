---
layout: subchapter
title: Data - analysis of cognitive model
---

*To Run*: Copy to codebox on [agentmodels.org](http://agentmodels.org/chapters/4-reasoning-about-agents.html)

~~~~
var actions = ['italian', 'french'];

var transition = function(state, action){
  var nextStates = ['bad', 'good', 'spectacular'];
  var nextProbs = (action === 'italian') ? [0.2, 0.6, 0.2] : [0.05, 0.9, 0.05];
  return categorical(nextProbs, nextStates);
};

var utility = function(state){
  var table = { 
    bad: -10, 
    good: 6, 
    spectacular: 8 
  };
  return table[state];
};

var softMaxAgent = function(state, alpha){
  return Infer({ method: 'enumerate' }, function(){

    var action = uniformDraw(actions);

    var expectedUtility = function(action){
      return expectation(Infer({ method: 'enumerate' }, function(){
        return utility(transition(state, action));
      }));
    };
    factor(alpha * expectedUtility(action));
    return action;
  })
};

var data = ['italian', 'french','french','french','french','french','french',
           'french','french','french','french','french','italian', 'italian',
           'french','french','french','french','french','french',
           'french','french','french','french','french','french',
           'french','french','french','french','french','french',
           'french','french','french','french','french','french',
           'french','french','french','french','french','french']

var dataAnalysisModel = function(){
  var alpha = uniform(0, 5);
  var cognitiveModel = softMaxAgent('initialState', alpha);
  map(function(d){observe(cognitiveModel, d)}, data)
  return {
    alpha: alpha,
    french_prediction: Math.exp(cognitiveModel.score("french"))
  }
}

var numSamples = 5000;

var inferOpts = {
  method: "MCMC",
  samples: numSamples,
  burn: numSamples / 2,
  callbacks: [editor.MCMCProgress()]
}

var posterior = Infer(inferOpts, dataAnalysisModel);

viz.auto(posterior);
~~~~


Check out: 

+ Prior predictive
+ What if your model is wrong? (switch agent's beliefs about utilities)