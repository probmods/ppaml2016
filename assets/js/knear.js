// Copyright (c) 2014 Nathan Epstein

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


//compute the euclidean distance between two vectors
//function assumes vectors are arrays of equal length
var dist = function(v1,v2){
  var sum = 0;
  v1.forEach(function(val,index){
    sum += Math.pow(val - v2[index],2);
  });
  return Math.sqrt(sum);
};

var updateMax = function(val,arr){
    var max = 0;
    arr.forEach(function(obj){
        max = Math.max(max,obj.d);
    });
    return max;
};

function mode(store){
  var frequency = {};  // array of frequency.
  var max = 0;  // holds the max frequency.
  var result;   // holds the max frequency element.
  for(var v in store) {
          frequency[store[v]]=(frequency[store[v]] || 0)+1; // increment frequency.
          if(frequency[store[v]] > max) { // is this frequency > max so far ?
                  max = frequency[store[v]];  // update max.
                  result = store[v];          // update result.
          }
  }
    return result;
}




var kNear = function(k){
  //PRIVATE
  var training = [];


  //PUBLIC

  //add a point to the training set
  this.learn = function(vector, label){
    var obj = {v:vector, lab: label};
    training.push(obj);
  };

  this.classify = function(v){
    var voteBloc = [];
    var maxD = 0;
    training.forEach(function(obj){
      var o = {d:dist(v,obj.v), vote:obj.lab};
      if (voteBloc.length < k){
        voteBloc.push(o);
        maxD = updateMax(maxD,voteBloc);
      }
      else {
        if (o.d < maxD){
          var bool = true;
          var count = 0;
          while (bool){
            if (Number(voteBloc[count].d) === maxD){
              voteBloc.splice(count,1,o);
              maxD = updateMax(maxD,voteBloc);
              bool = false;
            }
            else{
              if(count < voteBloc.length-1){
                count++;
              }
              else{
                bool = false;
              }
            }
          }
        }
      }

    });
    var votes = [];
    voteBloc.forEach(function(el){
      votes.push(el.vote);
    });
    return mode(votes);
  };
};
