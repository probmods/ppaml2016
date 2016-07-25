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
    //debugger;

    // assume that all pixels are either 0 or 255
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

$(function() {
  var $canvas = $('.mnist-draw'),
      canvas = $canvas[0];
  $canvas.sketch({defaultSize: 40});
  sketch = $canvas.sketch();

  $('#mnist-clear').click(function() {
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
    //debugger;

    // visualize downsampled as a picture
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

  })

});
