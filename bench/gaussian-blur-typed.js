var GaussianBlurTyped = new BenchmarkSuite('GaussianBlurTyped', 10000,
  [new Benchmark('GaussianBlurTyped', runGaussianBlurTyped, setupGaussianBlurTyped)]);

var typedBlurImage;
var typedBlurKernel;
var typedBlurOutput;
var typedBlurWidth = 256;
var typedBlurHeight = 256;

function setupGaussianBlurTyped()
{
  typedBlurImage = new Float32Array(typedBlurWidth * typedBlurHeight);
  typedBlurOutput = new Float32Array(typedBlurWidth * typedBlurHeight);
  typedBlurKernel = new Float32Array(25);
  var row = [1, 4, 6, 4, 1];
  for (var k = 0; k < 25; ++k)
    typedBlurKernel[k] = row[Math.floor(k / 5)] * row[k % 5] / 256;
  for (var i = 0; i < typedBlurImage.length; ++i)
    typedBlurImage[i] = (i * 7919) % 256 / 255;
}

function gaussianBlurTyped(image, kernel, output, width, height)
{
  for (var y = 2; y < height - 2; ++y)
    for (var x = 2; x < width - 2; ++x)
    {
      var sum = 0;
      for (var ky = 0; ky < 5; ++ky)
        for (var kx = 0; kx < 5; ++kx)
          sum += image[(y + ky - 2) * width + x + kx - 2] * kernel[ky * 5 + kx];
      output[y * width + x] = sum;
    }
}

function runGaussianBlurTyped()
{
  gaussianBlurTyped(typedBlurImage, typedBlurKernel, typedBlurOutput,
    typedBlurWidth, typedBlurHeight);
}
