/* get pixels of the full image, binarize, and then downsample to 28 x 28 */
var extractPixels = function(cv) {
  cv = cv || $('.mnist-draw')[0];
  var fw = cv.width,
      fh = cv.height; //224

  var imageData = cv.getContext('2d').getImageData(0,0,fw,fh).data;
  // imageData is a Uint8ClampedArray array with fw * fh * 4 (rgba) elements

  var binaryPixels = [];
  // just look at the alpha channel, since marker is black
  for(var i = 0, ii = imageData.length/4; i < ii; i++) {
    binaryPixels.push(imageData[(i+1) * 4 - 1])
  }

  var mode = function(xs) {
    var numOn = _.reduce(xs,
                         function(acc, x) {return acc + (x > 128 ? 1 : 0)},
                         0);
    var numOff = xs.length - numOn;

    return numOn > numOff ? 1 : 0;
  }

  var pixels = [];
  var n = 8; // TODO: don't hard code
  for(var r = 0; r < 28; r++) {
    for (var c = 0; c < 28; c++) {
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
      pixels.push(mode(block));
    }
  }
  return pixels;
}

// NB: I'm writing a webppl model here, which I compile later on
// using webppl.compile. I pass this thing arguments using globalStore
var encoderProgram = function() {
  var zDim = 10;
  var hDecodeDim = 500;
  var hEncodeDim = 500;
  var xDim = 784;

  var x = globalStore['x'];

  var W0 = globalStore['W0'],
      W1 = globalStore['W1'],
      W2 = globalStore['W2'],
      b0 = globalStore['b0'],
      b1 = globalStore['b1'],
      b2 = globalStore['b2'];

  var h = T.tanh(T.add(T.dot(W0, x), b0));

  var mu = T.add(T.dot(W1, h), b1);
  var sigma = T.exp(T.mul(T.add(T.dot(W2, h), b2),
                          1/2));

  return {mu: mu, sigma: sigma};
}

// TODO: will have to switch to the compile-prepare-run stratey once #570 is merged
var encoder = eval.call({},
                        webppl.compile(['(',encoderProgram.toString(),')()'].join('')));

$(function() {

  // load encoder params
  var xhr1 = new XMLHttpRequest();
  xhr1.onreadystatechange = function(){
    if (this.readyState == 4 && this.status == 200){
      //this.response is what you're looking for
      //console.log(this.response, typeof this.response);
      if (this.response) {
        var byteArray = new Uint8Array(this.response);
        encoderParams = msgpack.decode(byteArray, {codec: codec});
        encoderStore = _.mapObject(encoderParams, _.first);
        console.log('got encoder params')


      }
    }
  }
  xhr1.open('GET', '../assets/data/encoder-params.msp');
  xhr1.responseType = "arraybuffer";
  xhr1.send();

  // load pre-trained latents
  var xhr2 = new XMLHttpRequest();
  xhr2.onreadystatechange = function(){
    if (this.readyState == 4 && this.status == 200){
      //this.response is what you're looking for
      //console.log(this.response, typeof this.response);
      if (this.response) {
        var byteArray = new Uint8Array(this.response);
        latents = msgpack.decode(byteArray, {codec: codec});
        console.log('got latents');
        knn = new kNear(30);
        _.each(latents,
               function(latent) {
                 knn.learn(latent.mu.data, latent.label + '')
               });
        console.log('learned neighbors');
      }

    }
  }
  xhr2.open('GET', '../assets/data/latents.msp');
  xhr2.responseType = "arraybuffer";
  xhr2.send();

  var $canvas = $('.mnist-draw'),
      canvas = $canvas[0];
  $canvas.sketch({defaultSize: 12});
  sketch = $canvas.sketch();

  $('#mnist-clear').click(function() {
    $("#mnist-result span").empty()
    $("#downsampled").empty()
    sketch.set('tool', 'eraser');
    // fake event needs pageX and pageY for drawing to work
    var position = $(sketch.el).position(),
        pageX = position.left + 10,
        pageY = position.top + 10,
        eData = {pageX: pageX, pageY: pageY};
    var e1 = jQuery.Event("mousedown", eData)
    var e2 = jQuery.Event("mouseup", eData)
    $canvas.trigger(e1);
    $canvas.trigger(e2);
    sketch.set('tool', 'marker');
  });

  $('#mnist-classify').click(function() {
    var pixels = extractPixels();

    // visualize downsampled picture
    var i = 0;
    var table = ["<table>"];
    for(var r = 0; r < 28; r++) {
      var row = ["<tr>"];
      for(var c = 0; c < 28; c++) {
        var pixel = pixels[i],
            className = (pixel == 1 ? 'on' : 'off');
        row.push('<td class="' + className  +  '">&nbsp</td>');
        i++;
      }
      row.push("</tr>");
      table.push(row.join(""))
    }
    table.push("</table>");
    table = table.join("\n");

    $("#downsampled").html(table);

    var runner = util.trampolineRunners.web(function() { console.error(arguments) });
    var f = encoder(runner);

    // pass pixels through encoder
    encoderStore.x = new Tensor([pixels.length, 1]);
    encoderStore.x.data = new Float64Array(pixels);
    var res = f(encoderStore,
                function(s,x) {
                  $("#mnist-result span").text(knn.classify(x.mu.data))
                },
                '');

    // sample codes

  })

  var codec = msgpack.createCodec();
  codec.addExtUnpacker(0x3F, unpackTensor);

  var Tensor = ad.tensor.__Tensor; /* TODO: this takes advantage of an undocumented hack i found in webppl for daipp-friendliness; submit a PR in webppl to export Tensor constructor */
  function unpackTensor(buffer) {
    var obj = msgpack.decode(buffer);
    var t = new Tensor(obj.dims);
    t.data = obj.data;
    return t;
  }






});
