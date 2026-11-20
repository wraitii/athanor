export const examples: Record<string, string> = {
  'Gaussian blur': `var WIDTH = 2048;
var HEIGHT = 2048;
var PASSES = 5;

function gaussianBlur(image, kernel, output, width, height) {
  for (var y = 2; y < height - 2; ++y)
    for (var x = 2; x < width - 2; ++x) {
      var sum = 0;
      for (var ky = 0; ky < 5; ++ky)
        for (var kx = 0; kx < 5; ++kx)
          sum += image[(y + ky - 2) * width + x + kx - 2] * kernel[ky * 5 + kx];
      output[y * width + x] = sum;
    }
}

function main() {
  var image = new Float32Array(WIDTH * HEIGHT);
  for (var i = 0; i < image.length; ++i)
    image[i] = (i * 7919) % 256 / 255;
  var row = [1, 4, 6, 4, 1];
  var kernel = new Float32Array(25);
  for (var k = 0; k < 25; ++k)
    kernel[k] = row[Math.floor(k / 5)] * row[k % 5] / 256;
  var output = new Float32Array(WIDTH * HEIGHT);
  for (var pass = 0; pass < PASSES; ++pass)
    gaussianBlur(image, kernel, output, WIDTH, HEIGHT);
  var checksum = 0;
  for (var j = 0; j < output.length; ++j)
    checksum += output[j];
  console.log("checksum", checksum);
}
`,
  'Multiply-add': `function multiplyAdd(a, b) {
  return a * b + 1;
}
`,
  'Object fields': `function dot(points) {
  var total = 0;
  for (var i = 0; i < points.length; i++)
    total += points[i].x * points[i].y;
  return total;
}
`,
};
