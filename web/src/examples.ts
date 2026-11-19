export const examples: Record<string, string> = {
  'Gaussian blur': `function gaussianBlur(image, kernel, output, width, height) {
  for (var y = 1; y < height - 1; ++y)
    for (var x = 1; x < width - 1; ++x) {
      var sum = 0;
      for (var ky = 0; ky < 3; ++ky)
        for (var kx = 0; kx < 3; ++kx)
          sum += image[(y + ky - 1) * width + x + kx - 1] * kernel[ky * 3 + kx];
      output[y * width + x] = sum;
    }
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
