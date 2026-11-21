var GaussianBlurInt = new BenchmarkSuite('GaussianBlurInt', 10000,
  [new Benchmark('GaussianBlurInt', runGaussianBlurInt, setupGaussianBlurInt)]);

var intBlurImage;
var intBlurKernel;
var intBlurOutput;
var intBlurWidth = 256;
var intBlurHeight = 256;

function setupGaussianBlurInt()
{
  intBlurImage = new Uint8Array(intBlurWidth * intBlurHeight);
  intBlurOutput = new Uint8Array(intBlurWidth * intBlurHeight);
  intBlurKernel = new Int16Array(25);
  var row = [1, 4, 6, 4, 1];
  for (var k = 0; k < 25; ++k)
    intBlurKernel[k] = row[Math.floor(k / 5)] * row[k % 5];
  for (var i = 0; i < intBlurImage.length; ++i)
    intBlurImage[i] = (i * 7919) % 256;
}

function gaussianBlurInt(image, kernel, output, width, height)
{
  for (var y = 2; y < height - 2; ++y)
    for (var x = 2; x < width - 2; ++x)
    {
      var sum = 0;
      for (var ky = 0; ky < 5; ++ky)
        for (var kx = 0; kx < 5; ++kx)
          sum += image[(y + ky - 2) * width + x + kx - 2] * kernel[ky * 5 + kx];
      output[y * width + x] = sum >> 8;
    }
}

function runGaussianBlurInt()
{
  gaussianBlurInt(intBlurImage, intBlurKernel, intBlurOutput,
    intBlurWidth, intBlurHeight);
}
