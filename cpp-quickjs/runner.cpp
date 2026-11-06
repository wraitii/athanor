#include "quickjs/quickjs.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr const char* DEFAULT_SCRIPT{"../../binaries/data/mods/public/maps/random/tests/test_Generator.js"};
constexpr const char* DEFAULT_BOOTSTRAP{"../bootstrap.js"};
constexpr const char* DEFAULT_DATA{"../../binaries/data/mods/public"};
constexpr const char* DEFAULT_SETTINGS{"{\"Size\":128,\"Seed\":0,\"PlayerData\":[null,{\"Civ\":\"athen\"},{\"Civ\":\"brit\"}]}"};

struct HostContext
{
	std::string dataRoot;
	std::uint64_t randomState = 0x330e;
};

HostContext* GetHost(JSContext* cx)
{
	return static_cast<HostContext*>(JS_GetContextOpaque(cx));
}

JSValue HostSeedRandom(JSContext* cx, JSValueConst, int argc, JSValueConst* argv)
{
	std::uint32_t seed = 0;
	if (argc >= 1 && JS_ToUint32(cx, &seed, argv[0]))
		return JS_EXCEPTION;

	GetHost(cx)->randomState = (std::uint64_t{seed} << 16) | 0x330e;
	return JS_UNDEFINED;
}

JSValue MathRandom(JSContext* cx, JSValueConst, int, JSValueConst*)
{
	std::uint64_t& state = GetHost(cx)->randomState;
	state = (state * 0x5deece66dULL + 11) & ((1ULL << 48) - 1);
	return JS_NewFloat64(cx, static_cast<double>(state >> 17) / 2147483648.0);
}

bool ToUTF8(JSContext* cx, JSValueConst value, std::string& out)
{
	const char* text = JS_ToCString(cx, value);
	if (!text)
		return false;

	out.assign(text);
	JS_FreeCString(cx, text);
	return true;
}

JSValue HostReadFile(JSContext* cx, JSValueConst, int argc, JSValueConst* argv)
{
	if (argc < 1)
		return JS_ThrowTypeError(cx, "hostReadFile expects a path");

	std::string path;
	if (!ToUTF8(cx, argv[0], path))
		return JS_EXCEPTION;

	std::ifstream file(path, std::ios::binary);
	if (!file)
		return JS_ThrowTypeError(cx, "Unable to open %s", path.c_str());

	std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	return JS_NewStringLen(cx, content.data(), content.size());
}

bool MatchesFilter(const std::string& filename, const std::string& filter)
{
	if (filter.empty() || filter == "*")
		return true;

	if (filter.front() == '*')
	{
		const std::string suffix = filter.substr(1);
		return filename.size() >= suffix.size() &&
			filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
	}

	return filename == filter;
}

JSValue HostListFiles(JSContext* cx, JSValueConst, int argc, JSValueConst* argv)
{
	if (argc < 1)
		return JS_ThrowTypeError(cx, "hostListFiles expects a path");

	std::string path;
	std::string filter = "*";
	if (!ToUTF8(cx, argv[0], path))
		return JS_EXCEPTION;
	if (argc >= 2 && !ToUTF8(cx, argv[1], filter))
		return JS_EXCEPTION;

	bool recursive = false;
	if (argc >= 3)
	{
		const int value = JS_ToBool(cx, argv[2]);
		if (value < 0)
			return JS_EXCEPTION;
		recursive = value;
	}

	std::vector<std::string> files;
	std::error_code ec;

	if (recursive)
	{
		for (std::filesystem::recursive_directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec))
			if (it->is_regular_file(ec) && MatchesFilter(it->path().filename().string(), filter))
				files.push_back(it->path().lexically_normal().generic_string());
	}
	else
	{
		for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec))
			if (it->is_regular_file(ec) && MatchesFilter(it->path().filename().string(), filter))
				files.push_back(it->path().lexically_normal().generic_string());
	}

	if (ec)
		return JS_ThrowTypeError(cx, "Unable to list %s", path.c_str());

	std::sort(files.begin(), files.end());

	std::string result;
	for (const std::string& file : files)
	{
		result += file;
		result += '\n';
	}

	return JS_NewStringLen(cx, result.data(), result.size());
}

JSValue EvaluateSource(JSContext* cx, const std::string& source, const std::string& filename)
{
	return JS_Eval(cx, source.data(), source.size(), filename.c_str(), JS_EVAL_TYPE_GLOBAL);
}

JSValue HostEvaluateSource(JSContext* cx, JSValueConst, int argc, JSValueConst* argv)
{
	if (argc < 2)
		return JS_ThrowTypeError(cx, "hostEvaluateSource expects source and filename");

	std::string source;
	std::string filename;
	if (!ToUTF8(cx, argv[0], source) || !ToUTF8(cx, argv[1], filename))
		return JS_EXCEPTION;

	return EvaluateSource(cx, source, filename);
}

JSValue HostPrint(JSContext* cx, JSValueConst, int argc, JSValueConst* argv)
{
	for (int i = 0; i < argc; ++i)
	{
		std::string value;
		if (!ToUTF8(cx, argv[i], value))
			return JS_EXCEPTION;
		std::cout << value;
	}
	std::cout.flush();
	return JS_UNDEFINED;
}

void PrintException(JSContext* cx)
{
	JSValue exception = JS_GetException(cx);
	std::string message;
	if (ToUTF8(cx, exception, message))
		std::cerr << "Uncaught exception: " << message << '\n';
	else
		std::cerr << "Uncaught exception\n";
	JS_FreeValue(cx, exception);
}

std::string FindArgument(int argc, char** argv, const std::string& name, const char* fallback)
{
	for (int i = 1; i + 1 < argc; ++i)
		if (argv[i] == name)
			return argv[i + 1];
	return fallback;
}

std::string FindPositional(int argc, char** argv, int position)
{
	int current = 0;
	for (int i = 1; i < argc; ++i)
	{
		if (argv[i][0] == '-')
		{
			++i;
			continue;
		}

		if (current++ == position)
			return argv[i];
	}
	return {};
}

JSValue MakeString(JSContext* cx, const std::string& value)
{
	return JS_NewStringLen(cx, value.data(), value.size());
}

}

int main(int argc, char** argv)
{
	const std::string dataRoot = std::filesystem::absolute(
		FindArgument(argc, argv, "--data", DEFAULT_DATA)).lexically_normal().string();
	const std::string bootstrapPath = std::filesystem::absolute(
		FindArgument(argc, argv, "--bootstrap", DEFAULT_BOOTSTRAP)).lexically_normal().string();
	const std::string settings = FindArgument(argc, argv, "--settings", DEFAULT_SETTINGS);
	const std::string scriptArgument = FindPositional(argc, argv, 0);
	const std::string scriptPath = std::filesystem::absolute(
		scriptArgument.empty() ? DEFAULT_SCRIPT : scriptArgument).lexically_normal().string();

	HostContext host{dataRoot};

	JSRuntime* rt = JS_NewRuntime();
	if (!rt)
	{
		std::cerr << "Failed to initialize QuickJS\n";
		return 1;
	}

	JS_SetMaxStackSize(rt, 4 * 1024 * 1024);

	JSContext* cx = JS_NewContext(rt);
	if (!cx)
	{
		std::cerr << "Failed to create a JS context\n";
		JS_FreeRuntime(rt);
		return 1;
	}

	JS_SetContextOpaque(cx, &host);

	JSValue global = JS_GetGlobalObject(cx);
	JSValue math = JS_GetPropertyStr(cx, global, "Math");
	if (JS_IsException(math))
	{
		PrintException(cx);
		JS_FreeValue(cx, global);
		JS_FreeContext(cx);
		JS_FreeRuntime(rt);
		return 1;
	}

	JS_SetPropertyStr(cx, math, "random", JS_NewCFunction(cx, MathRandom, "random", 0));
	JS_FreeValue(cx, math);

	JS_SetPropertyStr(cx, global, "hostSeedRandom",
		JS_NewCFunction(cx, HostSeedRandom, "hostSeedRandom", 1));
	JS_SetPropertyStr(cx, global, "print",
		JS_NewCFunction(cx, HostPrint, "print", 1));
	JS_SetPropertyStr(cx, global, "hostReadFile",
		JS_NewCFunction(cx, HostReadFile, "hostReadFile", 1));
	JS_SetPropertyStr(cx, global, "hostListFiles",
		JS_NewCFunction(cx, HostListFiles, "hostListFiles", 3));
	JS_SetPropertyStr(cx, global, "hostEvaluateSource",
		JS_NewCFunction(cx, HostEvaluateSource, "hostEvaluateSource", 2));

	std::ifstream bootstrapFile(bootstrapPath, std::ios::binary);
	if (!bootstrapFile)
	{
		std::cerr << "Unable to open bootstrap script " << bootstrapPath << '\n';
		JS_FreeValue(cx, global);
		JS_FreeContext(cx);
		JS_FreeRuntime(rt);
		return 1;
	}

	const std::string bootstrapSource(
		(std::istreambuf_iterator<char>(bootstrapFile)), std::istreambuf_iterator<char>());

	JSValue ignored = EvaluateSource(cx, bootstrapSource, bootstrapPath);
	if (JS_IsException(ignored))
	{
		PrintException(cx);
		JS_FreeValue(cx, global);
		JS_FreeContext(cx);
		JS_FreeRuntime(rt);
		return 1;
	}
	JS_FreeValue(cx, ignored);

	JSValue run = JS_GetPropertyStr(cx, global, "run");
	if (!JS_IsFunction(cx, run))
	{
		std::cerr << "Bootstrap did not define run()\n";
		JS_FreeValue(cx, run);
		JS_FreeValue(cx, global);
		JS_FreeContext(cx);
		JS_FreeRuntime(rt);
		return 1;
	}

	JSValue arguments[3] = {
		MakeString(cx, scriptPath),
		MakeString(cx, dataRoot),
		MakeString(cx, settings)
	};

	JSValue result = JS_Call(cx, run, global, 3, arguments);
	JS_FreeValue(cx, arguments[0]);
	JS_FreeValue(cx, arguments[1]);
	JS_FreeValue(cx, arguments[2]);
	JS_FreeValue(cx, run);

	if (JS_IsException(result))
	{
		PrintException(cx);
		JS_FreeValue(cx, global);
		JS_FreeContext(cx);
		JS_FreeRuntime(rt);
		return 1;
	}

	std::string output;
	if (!ToUTF8(cx, result, output))
	{
		PrintException(cx);
		JS_FreeValue(cx, result);
		JS_FreeValue(cx, global);
		JS_FreeContext(cx);
		JS_FreeRuntime(rt);
		return 1;
	}

	std::cout << output << '\n';

	JS_FreeValue(cx, result);
	JS_FreeValue(cx, global);
	JS_FreeContext(cx);
	JS_FreeRuntime(rt);
	return 0;
}
