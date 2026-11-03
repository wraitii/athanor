mod host;

use std::fs;
use std::io::{self, BufRead, Write};

use js_engine::{Engine, Error, Value};

const DEFAULT_SCRIPT: &str = "binaries/data/mods/public/maps/random/new_rms_test.js";
const DEFAULT_DATA: &str = "binaries/data/mods/public";
const DEFAULT_SETTINGS: &str =
    "{\"Size\":128,\"Seed\":0,\"PlayerData\":[null,{\"Civ\":\"athen\"},{\"Civ\":\"brit\"}]}";
const DEFAULT_BOOTSTRAP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bootstrap.js");

fn main() {
    if let Err(error) = run() {
        eprintln!("js-runner: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut script = None;
    let mut preloads = Vec::new();
    let mut eval = None;
    let mut function = String::from("main");
    let mut values = Vec::new();
    let mut repl = false;
    let mut map = false;
    let mut compile = false;
    let mut compile_ir = true;
    let mut compile_ast = false;
    let mut output_option = false;
    let mut compile_asm = false;
    let mut data = None;
    let mut settings = None;
    let mut bootstrap = None;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--eval" | "-e" => {
                eval = Some(arguments.next().ok_or("--eval expects source")?);
            }
            "--function" | "-f" => {
                function = arguments.next().ok_or("--function expects a name")?;
            }
            "--arg" | "-a" => {
                let value = arguments.next().ok_or("--arg expects a value")?;
                values.push(parse_value(&value)?);
            }
            "--load" => preloads.push(arguments.next().ok_or("--load expects a path")?),
            "--map" => map = true,
            "--compile" => compile = true,
            "--no-ir" => {
                compile_ir = false;
                output_option = true;
            }
            "--ir" => {
                compile_ir = true;
                output_option = true;
            }
            "--ast" => {
                compile_ast = true;
                output_option = true;
            }
            "--asm" => {
                compile_asm = true;
                output_option = true;
            }
            "--data" => data = Some(arguments.next().ok_or("--data expects a path")?),
            "--settings" => settings = Some(arguments.next().ok_or("--settings expects JSON")?),
            "--bootstrap" => {
                bootstrap = Some(arguments.next().ok_or("--bootstrap expects a path")?)
            }
            "--repl" => repl = true,
            "--help" | "-h" => {
                usage();
                return Ok(());
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option {other}"));
            }
            other => script = Some(other.to_string()),
        }
    }

    if output_option && !compile {
        return Err("--ir, --no-ir, --ast and --asm require --compile".into());
    }

    if map {
        if compile
            || repl
            || eval.is_some()
            || function != "main"
            || !values.is_empty()
            || !preloads.is_empty()
        {
            return Err("--map cannot be combined with script options".to_string());
        }
        return run_map(script, data, settings, bootstrap);
    }

    if repl {
        if compile
            || eval.is_some()
            || script.is_some()
            || function != "main"
            || !values.is_empty()
            || !preloads.is_empty()
            || data.is_some()
            || settings.is_some()
            || bootstrap.is_some()
        {
            return Err("--repl cannot be combined with script options".to_string());
        }
        return run_repl();
    }

    if data.is_some() || settings.is_some() || bootstrap.is_some() {
        return Err("--data, --settings, and --bootstrap require --map".to_string());
    }

    let (source, filename) = match (eval, script) {
        (Some(source), None) => (source, String::from("<eval>")),
        (None, Some(path)) => (
            fs::read_to_string(&path).map_err(|error| format!("{path}: {error}"))?,
            path,
        ),
        (Some(_), Some(_)) => {
            return Err("pass either --eval or a script path".to_string());
        }
        (None, None) => {
            usage();
            return Err("no source or script path".to_string());
        }
    };

    if compile && (!preloads.is_empty() || function != "main" || !values.is_empty()) {
        return Err("--compile cannot be combined with execution options".into());
    }
    let options = js_engine::EngineOptions {
        compile_dump: compile,
        compile_ir,
        compile_ast,
        compile_asm,
        ..Default::default()
    };
    let mut engine = Engine::with_options(options).map_err(display)?;
    if compile {
        return engine.compile_only(&source, &filename).map_err(display);
    }
    engine.install_host(host::FileHost).map_err(display)?;
    for path in preloads {
        let preload = fs::read_to_string(&path).map_err(|error| format!("{path}: {error}"))?;
        engine.compile(&preload, &path).map_err(display)?;
    }
    engine.compile(&source, &filename).map_err(display)?;
    let result = engine.call(&function, &values).map_err(display)?;
    print_value(&result);
    Ok(())
}

fn print_value(value: &js_engine::OwnedValue) {
    match value.text() {
        Some(text) => println!("{text}"),
        None => println!("{}", value.value()),
    }
}

fn run_map(
    script: Option<String>,
    data: Option<String>,
    settings: Option<String>,
    bootstrap: Option<String>,
) -> Result<(), String> {
    let script = script.unwrap_or_else(|| DEFAULT_SCRIPT.to_string());
    let data = data.unwrap_or_else(|| DEFAULT_DATA.to_string());
    let settings = settings.unwrap_or_else(|| DEFAULT_SETTINGS.to_string());
    let bootstrap = bootstrap.unwrap_or_else(|| DEFAULT_BOOTSTRAP.to_string());

    let mut engine = Engine::new().map_err(display)?;
    engine.install_host(host::FileHost).map_err(display)?;
    let source = fs::read_to_string(&bootstrap).map_err(|error| format!("{bootstrap}: {error}"))?;
    engine.compile(&source, &bootstrap).map_err(display)?;

    let script = engine.string(&script).map_err(display)?;
    let data = engine.string(&data).map_err(display)?;
    let settings = engine.string(&settings).map_err(display)?;
    let arguments = [script.value(), data.value(), settings.value()];
    let result = engine.call("run", &arguments).map_err(display)?;
    print_value(&result);
    Ok(())
}

fn run_repl() -> Result<(), String> {
    let mut engine = Engine::new().map_err(display)?;
    engine.install_host(host::FileHost).map_err(display)?;
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("> ");
        io::stdout().flush().map_err(|error| error.to_string())?;
        let Some(line) = lines.next() else {
            break;
        };
        let line = line.map_err(|error| error.to_string())?;
        let input = line.trim();
        if input == "exit" || input == ".exit" {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let expression = input.trim_end_matches(';').trim_end();
        let result = if input
            .strip_prefix("function")
            .is_some_and(|rest| rest.starts_with(char::is_whitespace))
        {
            engine.compile(input, "<repl>")
        } else {
            let source = format!("var __repl_result = ({expression});");
            match engine.compile(&source, "<repl>") {
                Ok(()) => {
                    if let Some(value) = engine.global().get("__repl_result") {
                        print_value(&value);
                    }
                    Ok(())
                }
                Err(Error::Parse(_)) => engine.compile(input, "<repl>"),
                Err(error) => Err(error),
            }
        };
        if let Err(error) = result {
            eprintln!("{error}");
        }
    }
    Ok(())
}

fn parse_value(value: &str) -> Result<Value, String> {
    match value {
        "undefined" => Ok(Value::UNDEFINED),
        "null" => Ok(Value::NULL),
        "true" => Ok(Value::TRUE),
        "false" => Ok(Value::FALSE),
        _ => value
            .parse::<f64>()
            .map(Value::number)
            .map_err(|_| format!("invalid argument value {value}")),
    }
}

fn display(error: Error) -> String {
    error.to_string()
}

fn usage() {
    eprintln!(
        "usage: js-runner [--load SCRIPT]... [--function NAME] [--arg VALUE]... <script>\n       js-runner [--function NAME] [--arg VALUE]... --eval SOURCE\n       js-runner --map [--data ROOT] [--settings JSON] [--bootstrap SCRIPT] [SCRIPT]\n       js-runner --compile [--no-ir] [--ast] [--asm] (--eval SOURCE | SCRIPT)
       js-runner --repl"
    );
}
