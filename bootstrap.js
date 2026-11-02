var modRoot = "";

globalThis.TERRAIN_TILE_SIZE = 4;
globalThis.MAP_BORDER_WIDTH = 3;

function joinPath(...parts)
{
	return parts
		.filter(part => part !== "")
		.map((part, index) => index === 0 ? part.replace(/\/+$/, "") : part.replace(/^\/+|\/+$/g, ""))
		.join("/");
}

function readText(path)
{
	return hostReadFile(path);
}

function listPaths(path, filter, recursive)
{
	return hostListFiles(path, filter, recursive)
		.split("\n")
		.filter(path => path.length > 0);
}

function relativeToModRoot(path)
{
	const prefix = modRoot.replace(/\/+$/, "") + "/";
	return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}

function log(...values)
{
}

function warn(...values)
{
	print("WARN: " + values.join(" ") + "\n");
}

function error(...values)
{
	throw new Error(values.join(" "));
}

function deepfreeze(value)
{
	if (value === null || typeof value !== "object" || Object.isFrozen(value))
		return value;

	for (const key of Object.getOwnPropertyNames(value))
		deepfreeze(value[key]);

	return Object.freeze(value);
}

function fail(message)
{
	throw new Error(message);
}

globalThis.TS_FAIL = function(message)
{
	fail("Assertion failed: " + message);
};

globalThis.TS_ASSERT = function(value)
{
	if (!value)
		fail("Assertion failed");
};

globalThis.TS_ASSERT_EQUALS = function(actual, expected)
{
	if (actual !== expected)
		fail("Expected " + String(expected) + ", got " + String(actual));
};

globalThis.TS_ASSERT_DIFFER = function(actual, expected)
{
	if (actual === expected)
		fail("Expected values to differ, got " + String(actual));
};

globalThis.TS_ASSERT_NUMBER = function(value)
{
	if (typeof value !== "number" || !isFinite(value))
		fail("Expected a finite number");
};

globalThis.TS_ASSERT_LESS = function(left, right)
{
	if (!(left < right))
		fail("Expected " + String(left) + " < " + String(right));
};

globalThis.TS_ASSERT_GREATER = function(left, right)
{
	if (!(left > right))
		fail("Expected " + String(left) + " > " + String(right));
};

globalThis.TS_ASSERT_LESS_EQUAL = function(left, right)
{
	if (!(left <= right))
		fail("Expected " + String(left) + " <= " + String(right));
};

globalThis.TS_ASSERT_GREATER_EQUAL = function(left, right)
{
	if (!(left >= right))
		fail("Expected " + String(left) + " >= " + String(right));
};

globalThis.TS_ASSERT_EQUALS_APPROX = function(actual, expected, difference)
{
	if (Math.abs(actual - expected) > difference)
		fail("Expected " + String(actual) + " to be within " + String(difference) + " of " + String(expected));
};

globalThis.TS_ASSERT_EXCEPTION = function(callback)
{
	try
	{
		callback();
	}
	catch (exception)
	{
		return;
	}
	fail("Expected an exception");
};

globalThis.TS_ASSERT_UNEVAL_EQUALS = function(actual, expected)
{
	if (JSON.stringify(actual) !== JSON.stringify(expected))
		fail("Expected " + JSON.stringify(expected) + ", got " + JSON.stringify(actual));
};

globalThis.global = globalThis;

var Engine = {
	LoadLibrary: function(name)
	{
		if (Engine.loadedLibraries.has(name))
			return true;

		Engine.loadedLibraries.add(name);
		const directory = joinPath(modRoot, "maps/random", name);
		for (const path of listPaths(directory, "*.js", false))
			Engine.evaluateFile(path);
		return true;
	},
	ReadJSONFile: function(path)
	{
		return JSON.parse(readText(joinPath(modRoot, path)));
	},
	ListDirectoryFiles: function(path, filter, recursive)
	{
		return listPaths(joinPath(modRoot, path), filter, recursive).map(relativeToModRoot);
	},
	GetTemplate: function(path)
	{
		return {
			Identity: {
				GenericName: null,
				Icon: null,
				History: null,
				Civ: path.split("/")[1] || null
			},
			Obstruction: {
				Active: "true",
				BlockMovement: "true",
				BlockPathfinding: "true",
				BlockFoundation: "true",
				BlockConstruction: "true",
				Static: {
					"@width": "4",
					"@depth": "4"
				}
			},
			ResourceSupply: {
				Max: "100"
			},
			Cost: {
				Resources: {
					food: "50",
					wood: "50",
					stone: "50",
					metal: "50"
				}
			},
			Treasure: {
				Resources: {
					food: "100",
					wood: "100",
					stone: "100",
					metal: "100"
				}
			},
			WallSet: {
				Templates: {
					Tower: "structures/wall/tower",
					Gate: "structures/wall/gate",
					Fort: "structures/wall/fort",
					WallLong: "structures/wall/long",
					WallMedium: "structures/wall/medium",
					WallShort: "structures/wall/short"
				},
				MaxTowerOverlap: "0",
				MinTowerOverlap: "0"
			},
			WallPiece: {
				Length: "8",
				Orientation: "0",
				Indent: "0",
				Bend: "0"
			}
		};
	},
	TemplateExists: function()
	{
		return true;
	},
	GetMicroseconds: function()
	{
		return Date.now() * 1000;
	},
	ProfileStart: function()
	{
	},
	ProfileStop: function()
	{
	},
	TS_FAIL: function(message)
	{
		fail("Assertion failed: " + message);
	},
	evaluateFile: function(path)
	{
		try
		{
			hostEvaluateSource(readText(path), path);
		}
		catch (exception)
		{
			throw new Error("Failed loading " + path + ": " + exception);
		}
	},
	loadedLibraries: new Set()
};

function loadGlobalScripts()
{
	for (const path of listPaths(joinPath(modRoot, "globalscripts"), "*.js", false))
		Engine.evaluateFile(path);
}

function loadMapScript(path)
{
	let source = readText(path);
	source = source.replace(/\bexport\s+function\s*\*\s*generateMap\b/, "function* generateMap");
	hostEvaluateSource(source, path);
	if (typeof generateMap !== "function")
		throw new Error("Map script did not define generateMap");
}

function runGenerator(generator)
{
	let step = generator.next();
	while (!step.done)
		step = generator.next();

	return step.value;
}

function fnv1a(values)
{
	let hash = 0x811c9dc5;
	for (let i = 0; i < values.length; ++i)
	{
		hash ^= values[i] & 0xff;
		hash = Math.imul(hash, 0x01000193);
		hash ^= (values[i] >>> 8) & 0xff;
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16).padStart(8, "0");
}

function summarizeMap(map)
{
	if (map === undefined)
		return null;

	const result = map.MakeExportable();
	return JSON.stringify({
		size: result.size,
		seaLevel: result.seaLevel,
		textureNames: result.textureNames,
		heightLength: result.height.length,
		heightHash: fnv1a(result.height),
		tileIndexLength: result.tileData.index.length,
		tileIndexHash: fnv1a(result.tileData.index),
		tilePriorityHash: fnv1a(result.tileData.priority),
		entityCount: result.entities.length
	});
}

function run(scriptPath, dataRoot, settingsJson)
{
	modRoot = dataRoot.replace(/\/+$/, "");
	Engine.loadedLibraries.clear();

	const settings = JSON.parse(settingsJson);
	globalThis.g_MapSettings = settings;

	loadGlobalScripts();
	loadMapScript(scriptPath);

	const started = Date.now();
	const map = runGenerator(generateMap(settings));
	const elapsed = Date.now() - started;

	const summary = summarizeMap(map);
	return JSON.stringify({
		elapsedMilliseconds: elapsed,
		map: summary === null ? null : JSON.parse(summary)
	});
}
