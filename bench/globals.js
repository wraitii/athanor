function emit(line) {
  if (typeof console !== "undefined")
    console.log(line);
  else
    print(line);
}

var sink = 0;

function bench(name, fn, opsPerCall) {
  var warmStart = Date.now();
  var warmRuns = 0;
  while (warmRuns < 20 || Date.now() - warmStart < 50) {
    sink += fn();
    warmRuns += 1;
  }

  var batch = 1;
  while (true) {
    var t0 = Date.now();
    for (var i = 0; i < batch; ++i)
      sink += fn();
    var elapsed = Date.now() - t0;
    if (elapsed >= 50)
      break;
    batch = batch * 2;
  }

  var runs = 0;
  var start = Date.now();
  var total = 0;
  while (total < 250) {
    for (var j = 0; j < batch; ++j)
      sink += fn();
    runs += batch;
    total = Date.now() - start;
  }

  var ns = (total * 1000000) / (runs * opsPerCall);
  var rounded = Math.round(ns * 100) / 100;
  emit(name + ": " + rounded + " ns/op");
}

var gVal = 3;
var gA = 1;
var gB = 2;
var gC = 3;
var gD = 4;
var gSink = 0;
var gCounter = 0;
var gObj = { x: 5 };

function makeRing(len) {
  var nodes = [];
  for (var i = 0; i < len; ++i)
    nodes[i] = { val: i & 7, next: null };
  for (var j = 0; j < len; ++j)
    nodes[j].next = nodes[(j + 1) % len];
  return nodes[0];
}

var gCur = makeRing(64);

function gAdd(a, b) {
  return a + b;
}

var N = 50000;

function globalRead(n) {
  var sum = 0;
  for (var i = 0; i < n; ++i)
    sum += gVal;
  return sum;
}

function globalRead4(n) {
  var sum = 0;
  for (var i = 0; i < n; ++i)
    sum += gA + gB + gC + gD;
  return sum;
}

function globalWrite(n) {
  for (var i = 0; i < n; ++i)
    gSink = i;
  return gSink;
}

function globalRmw(n) {
  for (var i = 0; i < n; ++i)
    gCounter = gCounter + 1;
  return gCounter;
}

function globalWriteRead(n) {
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    gSink = i;
    sum += gA;
  }
  return sum;
}

function globalObjRead(n) {
  var sum = 0;
  for (var i = 0; i < n; ++i)
    sum += gObj.x;
  return sum;
}

function globalObjRing(n) {
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += gCur.val;
    gCur = gCur.next;
  }
  return sum;
}

function globalFnCall(n) {
  var sum = 0;
  for (var i = 0; i < n; ++i)
    sum = gAdd(sum, 1);
  return sum;
}

bench("global-read", function () { return globalRead(N); }, N);
bench("global-read-4", function () { return globalRead4(N); }, N);
bench("global-write", function () { return globalWrite(N); }, N);
bench("global-rmw", function () { return globalRmw(N); }, N);
bench("global-write-read", function () { return globalWriteRead(N); }, N);
bench("global-obj-read", function () { return globalObjRead(N); }, N);
bench("global-obj-ring", function () { return globalObjRing(N); }, N);
bench("global-fn-call", function () { return globalFnCall(N); }, N);

function main() {}
