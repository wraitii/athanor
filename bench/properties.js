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

function makeMonoRing(len) {
  var nodes = [];
  for (var i = 0; i < len; ++i)
    nodes[i] = { val: i & 7, next: null };
  for (var j = 0; j < len; ++j)
    nodes[j].next = nodes[(j + 1) % len];
  return nodes[0];
}

function makeMultiRing(len) {
  var nodes = [];
  for (var i = 0; i < len; ++i)
    nodes[i] = { a: i & 7, b: (i + 1) & 7, c: (i + 2) & 7, d: (i + 3) & 7, next: null };
  for (var j = 0; j < len; ++j)
    nodes[j].next = nodes[(j + 1) % len];
  return nodes[0];
}

function makeShapeNode(kind, val) {
  if (kind === 0) return { val: val, next: null };
  if (kind === 1) return { p0: 1, val: val, next: null };
  if (kind === 2) return { p0: 1, p1: 2, val: val, next: null };
  if (kind === 3) return { p0: 1, p1: 2, p2: 3, val: val, next: null };
  if (kind === 4) return { q0: 1, val: val, next: null };
  if (kind === 5) return { q0: 1, q1: 2, val: val, next: null };
  if (kind === 6) return { q0: 1, q1: 2, q2: 3, val: val, next: null };
  if (kind === 7) return { r0: 1, val: val, next: null };
  if (kind === 8) return { r0: 1, r1: 2, val: val, next: null };
  if (kind === 9) return { r0: 1, r1: 2, r2: 3, val: val, next: null };
  if (kind === 10) return { s0: 1, val: val, next: null };
  return { s0: 1, s1: 2, val: val, next: null };
}

function makePolyRing(len, shapes) {
  var nodes = [];
  for (var i = 0; i < len; ++i)
    nodes[i] = makeShapeNode(i % shapes, i & 7);
  for (var j = 0; j < len; ++j)
    nodes[j].next = nodes[(j + 1) % len];
  return nodes[0];
}

function ProtoNode(next) {
  this.next = next;
}
ProtoNode.prototype.val = 3;

function makeProtoRing(len) {
  var nodes = [];
  for (var i = 0; i < len; ++i)
    nodes[i] = new ProtoNode(null);
  for (var j = 0; j < len; ++j)
    nodes[j].next = nodes[(j + 1) % len];
  return nodes[0];
}

var monoRing = makeMonoRing(64);
var multiRing = makeMultiRing(64);
var poly2Ring = makePolyRing(64, 2);
var poly4Ring = makePolyRing(64, 4);
var megaRing = makePolyRing(64, 12);
var protoRing = makeProtoRing(64);
var singleObj = { val: 0 };

var N = 50000;

function monoReadRing(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += cur.val;
    cur = cur.next;
  }
  return sum;
}

function monoReadMulti(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += cur.a + cur.b + cur.c + cur.d;
    cur = cur.next;
  }
  return sum;
}

function monoWriteRing(head, n) {
  var cur = head;
  for (var i = 0; i < n; ++i) {
    cur.val = i;
    cur = cur.next;
  }
  return cur.val;
}

function monoRmwRing(head, n) {
  var cur = head;
  for (var i = 0; i < n; ++i) {
    cur.val = cur.val + 1;
    cur = cur.next;
  }
  return cur.val;
}

function monoWriteSame(obj, n) {
  for (var i = 0; i < n; ++i)
    obj.val = i;
  return obj.val;
}

function polyReadRing(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += cur.val;
    cur = cur.next;
  }
  return sum;
}

function poly4ReadRing(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += cur.val;
    cur = cur.next;
  }
  return sum;
}

function megaReadRing(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += cur.val;
    cur = cur.next;
  }
  return sum;
}

function protoReadRing(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    sum += cur.val;
    cur = cur.next;
  }
  return sum;
}

function missingReadRing(head, n) {
  var cur = head;
  var sum = 0;
  for (var i = 0; i < n; ++i) {
    if (cur.absent === undefined)
      sum += 1;
    cur = cur.next;
  }
  return sum;
}

bench("mono-read-ring", function () { return monoReadRing(monoRing, N); }, N);
bench("mono-read-multi", function () { return monoReadMulti(multiRing, N); }, N);
bench("mono-write-ring", function () { return monoWriteRing(monoRing, N); }, N);
bench("mono-rmw-ring", function () { return monoRmwRing(monoRing, N); }, N);
bench("mono-write-same", function () { return monoWriteSame(singleObj, N); }, N);
bench("poly2-read-ring", function () { return polyReadRing(poly2Ring, N); }, N);
bench("poly4-read-ring", function () { return poly4ReadRing(poly4Ring, N); }, N);
bench("mega-read-ring", function () { return megaReadRing(megaRing, N); }, N);
bench("proto-read-ring", function () { return protoReadRing(protoRing, N); }, N);
bench("missing-read-ring", function () { return missingReadRing(monoRing, N); }, N);

function main() {}
