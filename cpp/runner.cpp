#include <js/CallArgs.h>
#include <js/CallAndConstruct.h>
#include <js/CharacterEncoding.h>
#include <js/CompilationAndEvaluation.h>
#include <js/CompileOptions.h>
#include <js/Conversions.h>
#include <js/Exception.h>
#include <js/GlobalObject.h>
#include <js/Initialization.h>
#include <js/Realm.h>
#include <js/SourceText.h>
#include <js/ValueArray.h>
#include <js/PropertyAndElement.h>
#include <js/String.h>
#include <jsapi.h>

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
	return static_cast<HostContext*>(JS_GetContextPrivate(cx));
}

bool HostSeedRandom(JSContext* cx, unsigned argc, JS::Value* vp)
{
	JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
	std::uint32_t seed;
	if (!JS::ToUint32(cx, args.get(0), &seed))
		return false;

	GetHost(cx)->randomState = (std::uint64_t{seed} << 16) | 0x330e;
	args.rval().setUndefined();
	return true;
}

bool MathRandom(JSContext* cx, unsigned argc, JS::Value* vp)
{
	JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
	std::uint64_t& state = GetHost(cx)->randomState;
	state = (state * 0x5deece66dULL + 11) & ((1ULL << 48) - 1);
	args.rval().setDouble(static_cast<double>(state >> 17) / 2147483648.0);
	return true;
}

bool ToUTF8(JSContext* cx, JS::HandleValue value, std::string& out)
{
	JS::RootedString str(cx, JS::ToString(cx, value));
	if (!str)
		return false;

	JS::UniqueChars utf8 = JS_EncodeStringToUTF8(cx, str);
	if (!utf8)
		return false;

	out.assign(utf8.get());
	return true;
}

bool HostReadFile(JSContext* cx, unsigned argc, JS::Value* vp)
{
	JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
	if (args.length() < 1)
	{
		JS_ReportErrorASCII(cx, "hostReadFile expects a path");
		return false;
	}

	std::string path;
	if (!ToUTF8(cx, args[0], path))
		return false;

	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		JS_ReportErrorASCII(cx, "Unable to open %s", path.c_str());
		return false;
	}

	std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	JS::RootedString result(cx, JS_NewStringCopyUTF8N(cx, JS::UTF8Chars(content.data(), content.size())));
	if (!result)
		return false;

	args.rval().setString(result);
	return true;
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

bool HostListFiles(JSContext* cx, unsigned argc, JS::Value* vp)
{
	JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
	if (args.length() < 1)
	{
		JS_ReportErrorASCII(cx, "hostListFiles expects a path");
		return false;
	}

	std::string path;
	std::string filter = "*";
	if (!ToUTF8(cx, args[0], path))
		return false;
	if (args.length() >= 2 && !ToUTF8(cx, args[1], filter))
		return false;

	const bool recursive = args.length() >= 3 && args[2].toBoolean();
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
	{
		JS_ReportErrorASCII(cx, "Unable to list %s", path.c_str());
		return false;
	}

	std::sort(files.begin(), files.end());

	std::string result;
	for (const std::string& file : files)
	{
		result += file;
		result += '\n';
	}

	JS::RootedString value(cx, JS_NewStringCopyUTF8N(cx, JS::UTF8Chars(result.data(), result.size())));
	if (!value)
		return false;

	args.rval().setString(value);
	return true;
}

bool EvaluateSource(JSContext* cx, const std::string& source, const std::string& filename, JS::MutableHandleValue result);

bool HostEvaluateSource(JSContext* cx, unsigned argc, JS::Value* vp)
{
	JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
	if (args.length() < 2)
	{
		JS_ReportErrorASCII(cx, "hostEvaluateSource expects source and filename");
		return false;
	}

	std::string source;
	std::string filename;
	if (!ToUTF8(cx, args[0], source) || !ToUTF8(cx, args[1], filename))
		return false;

	JS::RootedValue result(cx);
	if (!EvaluateSource(cx, source, filename, &result))
		return false;

	args.rval().set(result);
	return true;
}

bool HostPrint(JSContext* cx, unsigned argc, JS::Value* vp)
{
	JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
	for (unsigned i = 0; i < args.length(); ++i)
	{
		std::string value;
		if (!ToUTF8(cx, args[i], value))
			return false;
		std::cout << value;
	}
	std::cout.flush();
	args.rval().setUndefined();
	return true;
}

void PrintException(JSContext* cx)
{
	JS::RootedValue exception(cx);
	if (!JS_GetPendingException(cx, &exception))
		return;

	JS_ClearPendingException(cx);

	std::string message;
	if (ToUTF8(cx, exception, message))
		std::cerr << "Uncaught exception: " << message << '\n';
	else
		std::cerr << "Uncaught exception\n";
}

bool EvaluateSource(JSContext* cx, const std::string& source, const std::string& filename,
	JS::MutableHandleValue result)
{
	JS::SourceText<mozilla::Utf8Unit> text;
	if (!text.init(cx, source.data(), source.size(), JS::SourceOwnership::Borrowed))
		return false;

	JS::CompileOptions options(cx);
	options.setFileAndLine(filename.c_str(), 1);
	return JS::Evaluate(cx, options, text, result);
}

JS::RootedString MakeString(JSContext* cx, const std::string& value)
{
	return JS::RootedString(cx, JS_NewStringCopyUTF8N(cx, JS::UTF8Chars(value.data(), value.size())));
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

	if (!JS_Init())
	{
		std::cerr << "Failed to initialize SpiderMonkey\n";
		return 1;
	}

	JSContext* cx = JS_NewContext(64 * 1024 * 1024);
	if (!cx)
	{
		std::cerr << "Failed to create a JS context\n";
		JS_ShutDown();
		return 1;
	}

	JS_SetContextPrivate(cx, &host);

	if (!JS::InitSelfHostedCode(cx))
	{
		std::cerr << "Failed to initialize self-hosted JS code\n";
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	static const JSClassOps globalOps = {
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		JS_GlobalObjectTraceHook
	};
	static const JSClass globalClass = {"global", JSCLASS_GLOBAL_FLAGS, &globalOps};

	JS::RealmOptions realmOptions;
	JS::RootedObject global(cx, JS_NewGlobalObject(cx, &globalClass, nullptr,
		JS::DontFireOnNewGlobalHook, realmOptions));
	if (!global)
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	JSAutoRealm realm(cx, global);

	if (!JS::InitRealmStandardClasses(cx))
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	JS::RootedValue mathValue(cx);
	if (!JS_GetProperty(cx, global, "Math", &mathValue) || !mathValue.isObject())
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}
	JS::RootedObject math(cx, &mathValue.toObject());
	if (!JS_DefineFunction(cx, math, "random", MathRandom, 0, 0) ||
		!JS_DefineFunction(cx, global, "hostSeedRandom", HostSeedRandom, 1, 0) ||
		!JS_DefineFunction(cx, global, "print", HostPrint, 1, 0) ||
		!JS_DefineFunction(cx, global, "hostReadFile", HostReadFile, 1, 0) ||
		!JS_DefineFunction(cx, global, "hostListFiles", HostListFiles, 3, 0) ||
		!JS_DefineFunction(cx, global, "hostEvaluateSource", HostEvaluateSource, 2, 0))
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	std::ifstream bootstrapFile(bootstrapPath, std::ios::binary);
	if (!bootstrapFile)
	{
		std::cerr << "Unable to open bootstrap script " << bootstrapPath << '\n';
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	const std::string bootstrapSource(
		(std::istreambuf_iterator<char>(bootstrapFile)), std::istreambuf_iterator<char>());

	JS::RootedValue ignored(cx);
	if (!EvaluateSource(cx, bootstrapSource, bootstrapPath, &ignored))
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	JS::RootedValue run(cx);
	if (!JS_GetProperty(cx, global, "run", &run) || !run.isObject())
	{
		std::cerr << "Bootstrap did not define run()\n";
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	JS::RootedValueArray<3> arguments(cx);
	arguments[0].setString(MakeString(cx, scriptPath));
	arguments[1].setString(MakeString(cx, dataRoot));
	arguments[2].setString(MakeString(cx, settings));

	JS::RootedValue result(cx);
	if (!JS_CallFunctionValue(cx, global, run, arguments, &result))
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	std::string output;
	if (!ToUTF8(cx, result, output))
	{
		PrintException(cx);
		JS_DestroyContext(cx);
		JS_ShutDown();
		return 1;
	}

	std::cout << output << '\n';

	JS_DestroyContext(cx);
	JS_ShutDown();
	return 0;
}
