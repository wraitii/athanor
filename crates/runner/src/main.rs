use std::fs;

use js_engine::{Engine, Error, Value};

fn main() {
    if let Err(error) = run() {
        eprintln!("js-runner: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut script = None;
    let mut eval = None;
    let mut function = String::from("main");
    let mut values = Vec::new();

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

    let mut engine = Engine::new().map_err(display)?;
    engine.compile(&source, &filename).map_err(display)?;
    let result = engine.call(&function, &values).map_err(display)?;
    let result = result.value();
    println!("{result}");
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
        "usage: js-runner [--function NAME] [--arg VALUE]... <script>\n       js-runner [--function NAME] [--arg VALUE]... --eval SOURCE"
    );
}
