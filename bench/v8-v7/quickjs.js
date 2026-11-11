function run(script)
{
  for (var file of ["bench/v8-v7/base.js", script, "bench/v8-v7/driver.js"])
    hostEvaluateSource(hostReadFile(file), file);
  return "";
}
