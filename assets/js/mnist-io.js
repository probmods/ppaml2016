// - get pixels of the full image
// - binarize
// - downsample to 20 x 20
// - embed center bounding box into 28x28 image
var extractPixels = function(cv) {
  cv = cv || $('.mnist canvas')[0];
  var fw = cv.width, fh = cv.height; //160

  var imageData = cv.getContext('2d').getImageData(0,0,fw,fh).data;
  // imageData is a Uint8ClampedArray array with fw * fh * 4 (rgba) elements

  var binaryPixels = [];
  // just look at the alpha channel, since marker is black
  for(var i = 0, ii = imageData.length/4; i < ii; i++) {
    binaryPixels.push(imageData[(i+1) * 4 - 1])
  }

  var mode = function(xs) {
    var numOn = _.reduce(xs,
                         function(acc, x) {return acc + (x >= 128 ? 1 : 0)},
                         0);
    var numOff = xs.length - numOn;

    return numOn > numOff ? 1 : 0;
  }

  var n = 8; // TODO: don't hard code

  var pixelTable = []

  for(var r = 0; r < 20; r++) {
    var tableRow = [];
    for (var c = 0; c < 20; c++) {
      // pixel (r,c) in the downsampled image
      // corresponds to the pixels in the full image between
      // (r*n, c*n) to ((r+1)*n - 1, (c+1)*n - 1)

      var block = [];
      for(var fr = r*n, frr = (r+1)*n; fr < frr; fr++) {
        for(var fc = c*n, fcc = (c+1)*n; fc < fcc; fc++) {

          // convert fr and fc to flattened indices
          var fi = (fr * fw) + fc;
          block.push(binaryPixels[fi]);
        }
      }
      tableRow.push(mode(block));
    }
    pixelTable.push(tableRow);
  }

  var minR = 999, maxR = -1,
      minC = 999, maxC = -1;
  for(var r = 0; r < 20; r++) {
    for (var c = 0; c < 20; c++) {
      if (pixelTable[r][c]) {
        if (r < minR) {
          minR = r
        }
        if (r > maxR){
          maxR = r
        }
        if (c < minC) {
          minC = c
        }
        if (c > maxC) {
          maxC = c
        }
      }
    }
  }

  // top left corner is [minR, minC]
  // bottom right corner is [maxR, maxC]

  var bbWidth = maxC - minC + 1;
  var bbHeight = maxR - minR + 1;

  // calculate offsets for centering 20x20 image
  var centeringOffsetR = Math.round((20 - bbHeight)/2 - minR);
  var centeringOffsetC = Math.round((20 - bbWidth)/2 - minC);

  // include offset for embedding in 28x28
  var offsetR = centeringOffsetR + 4,
      offsetC = centeringOffsetC + 4;

  var fullPixelTable = [];

  for(var r = 0; r < 28; r++) {
    var row = [];
    for(var c = 0; c < 28; c++) {
      row.push(0);
    }
    fullPixelTable.push(row)
  }

  for(var r = 0; r < 20; r++) {
    var row = [];
    for(var c = 0; c < 20; c++) {
      if (pixelTable[r][c] == 1) {
        var rPrime = Math.min(27, Math.max(0, r + offsetR));
        var cPrime = Math.min(27, Math.max(0, c + offsetC));

        fullPixelTable[rPrime][cPrime] = 1;
      }
    }
  }

  return _.flatten(fullPixelTable);
}

// NB: I'm writing a webppl model here, which I compile later on
// using webppl.compile. I pass arguments to this thing using globalStore
var encoderProgram = function() {
  var zDim = 10;
  var hDecodeDim = 500;
  var hEncodeDim = 500;
  var xDim = 784;

  var x = encoderStore['x'];

  var W0 = encoderStore['W0'],
      W1 = encoderStore['W1'],
      W2 = encoderStore['W2'],
      b0 = encoderStore['b0'],
      b1 = encoderStore['b1'],
      b2 = encoderStore['b2'];

  var h = T.tanh(T.add(T.dot(W0, x), b0));

  var mu = T.add(T.dot(W1, h), b1);
  var sigma = T.exp(T.mul(T.add(T.dot(W2, h), b2),
                          1/2));

  return {mu: mu, sigma: sigma};
}

var encoder = eval.call({},
                        webppl.compile(['(',encoderProgram.toString(),')()'].join('')));


$(function() {

  var codec = msgpack.createCodec();
  codec.addExtUnpacker(0x3F, unpackTensor);

  var Tensor = ad.tensor.__Tensor; /* TODO: this takes advantage of an undocumented hack i found in webppl for daipp-friendliness; submit a PR in webppl to export Tensor constructor */
  function unpackTensor(buffer) {
    var obj = msgpack.decode(buffer);
    var t = new Tensor(obj.dims);
    t.data = obj.data;
    return t;
  }

  var loaded = 0;
  var start = function() {
    if (loaded == 2) {
      setTimeout(function() {
        $(".mnist .loading").hide()
        $(".mnist .ready").show()
      }, 200);
    }
  }

  // load encoder params
  var xhr1 = new XMLHttpRequest();
  xhr1.onreadystatechange = function(){
    if (this.readyState == 4 && this.status == 200){
      if (this.response) {
        var byteArray = new Uint8Array(this.response);
        encoderParams = msgpack.decode(byteArray, {codec: codec});
        encoderStore = _.mapObject(encoderParams, _.first);
        loaded++;
        start();
      }
    }
  }
  var xhrPrefix = location.href.indexOf("127.0.0.1") > 0 ? "../assets/data/"  :
      "http://s3-us-west-2.amazonaws.com/cdn.webppl.org/";
  xhr1.open('GET', xhrPrefix + 'encoder-params.msp');
  xhr1.responseType = "arraybuffer";
  xhr1.addEventListener("progress", function(e) {
    var pct = Math.floor(e.loaded * 100 / e.total);
    $(".progress-params").text(pct + "%");
  });
  xhr1.send();

  // load pre-trained latents
  var xhr2 = new XMLHttpRequest();
  xhr2.onreadystatechange = function(){
    if (this.readyState == 4 && this.status == 200){
      if (this.response) {
        var byteArray = new Uint8Array(this.response);
        latents = msgpack.decode(byteArray, {codec: codec});
        knn = new kNear(30);
        _.each(latents,
               function(latent) {
                 knn.learn(latent.mu.data, latent.label + '')
               });
        loaded++;
        start();
      }
    }
  }
  xhr2.open('GET', xhrPrefix + 'latents.msp');
  xhr2.responseType = "arraybuffer";
  xhr2.addEventListener("progress", function(e) {
    var pct = Math.floor(e.loaded * 100 / e.total);
    $(".progress-latents").text(pct + "%");
  });
  xhr2.send();

  var $canvas = $('.mnist canvas'),
      canvas = $canvas[0];
  $canvas.sketch({defaultSize: 15});
  sketch = $canvas.sketch();

  // wire up Clear button
  var reset = function() {
    $(".mnist .result span").empty()
    $(".mnist .downsampled").empty()
    sketch.set('tool', 'eraser');
    // fake event needs pageX and pageY for eraser to work
    var position = $(sketch.el).position(),
        pageX = position.left + 10,
        pageY = position.top + 10,
        eData = {pageX: pageX, pageY: pageY};
    var e1 = jQuery.Event("mousedown", eData)
    var e2 = jQuery.Event("mouseup", eData)
    $canvas.trigger(e1);
    $canvas.trigger(e2);
    sketch.set('tool', 'marker');
  }
  $('.mnist .clear').click(reset);

  // wire up Classify button
  var classify = function() {
    var pixels = extractPixels();

    // visualize downsampled picture
    // var i = 0;
    // var table = ["<table>"];
    // for(var r = 0; r < 28; r++) {
    //   var row = ["<tr>"];
    //   for(var c = 0; c < 28; c++) {
    //     var pixel = pixels[i],
    //         className = (pixel == 1 ? 'on' : 'off');
    //     row.push('<td class="' + className  +  '">&nbsp</td>');
    //     i++;
    //   }
    //   row.push("</tr>");
    //   table.push(row.join(""))
    // }
    // table.push("</table>");
    // table = table.join("\n");
    // $(".mnist .downsampled").html(table);

    var afterClassify = function(s,x) {
      $(".mnist .result span").text(knn.classify(x.mu.data))
    }

    // pass pixels through encoder and do knn
    var handleError = function() { console.error(arguments) }
    var baseRunner = util.trampolineRunners.web();
    var prepared = webppl.prepare(encoder,
                                  afterClassify,
                                  {errorHandlers: [handleError], debug: true, baseRunner: baseRunner});

    encoderStore.x = new Tensor([pixels.length, 1]);
    encoderStore.x.data = new Float64Array(pixels);

    prepared.run();
  }
  $('.mnist .classify').click(classify);

});
