var GaussianBlur = new BenchmarkSuite('GaussianBlur', 1000,
  [new Benchmark('GaussianBlur', runGaussianBlur, setupGaussianBlur)]);

var blurImage;
var blurKernel;
var blurOutput;
var blurWidth = 128;
var blurHeight = 128;

function setupGaussianBlur()
{
  blurImage = [];
  blurOutput = [];
  blurKernel = [1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16];
  for (var i = 0; i < blurWidth * blurHeight; ++i)
  {
    blurImage[i] = (i * 17 + (i % blurWidth) * 13) % 256;
    blurOutput[i] = 0;
  }
}

function gaussianBlur(image, kernel, output, width, height)
{
  for (var y = 1; y < height - 1; ++y)
    for (var x = 1; x < width - 1; ++x)
    {
      var sum = 0;
      for (var ky = 0; ky < 3; ++ky)
        for (var kx = 0; kx < 3; ++kx)
          sum += image[(y + ky - 1) * width + x + kx - 1] * kernel[ky * 3 + kx];
      output[y * width + x] = sum;
    }
}

function runGaussianBlur()
{
  gaussianBlur(blurImage, blurKernel, blurOutput, blurWidth, blurHeight);
}
