var fs = require("fs");
var vm = require("vm");
var dir = __dirname;
globalThis.print = function (text) { process.stdout.write(text); };
for (var file of ["base.js"].concat(process.argv.slice(2).map(function (name) { return name + ".js"; }), ["driver.js"]))
  vm.runInThisContext(fs.readFileSync(dir + "/" + file, "utf8"), { filename: file });
