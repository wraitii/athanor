BenchmarkSuite.RunSuites({
  NotifyResult: function (name, result) { print(name + ": " + result + "\n"); },
  NotifyError: function (name, error) { print(name + ": ERROR " + error + "\n"); },
  NotifyScore: function (score) { print("Score: " + score + "\n"); },
});
