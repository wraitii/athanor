use std::sync::Arc;

use js_engine::{BoundSpec, CompileOptions, Engine, EngineOptions, GpuKernel, ReportSink, Scale};
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn compile(
    source: &str,
    filter: Option<String>,
    inlining: bool,
    versioning: bool,
    continuations: bool,
) -> Result<String, String> {
    console_error_panic_hook::set_once();
    let reports = ReportSink::default();
    let engine = Engine::with_options(EngineOptions {
        compile: Some(Arc::new(CompileOptions {
            filter,
            after: Some("all".into()),
            reports: Some(reports.clone()),
            ..Default::default()
        })),
        background_compile: false,
        inlining,
        versioning,
        continuations,
        ..Default::default()
    })
    .map_err(|error| error.to_string())?;
    engine
        .compile_only(source, "input.js")
        .map_err(|error| error.to_string())?;
    let reports: Vec<_> = reports
        .take()
        .into_iter()
        .map(|report| {
            json!({
                "name": report.name,
                "label": report.label,
                "direct": report.direct,
                "semantic": report.semantic,
                "passes": report
                    .passes
                    .into_iter()
                    .map(|(pass, text)| json!({ "pass": pass, "text": text }))
                    .collect::<Vec<_>>(),
                "gpu": report.gpu.as_ref().map(gpu_json),
            })
        })
        .collect();
    Ok(Value::Array(reports).to_string())
}

fn bound_json(bound: &BoundSpec) -> Value {
    match *bound {
        BoundSpec::Const(value) => json!({ "arg": null, "add": value }),
        BoundSpec::Arg(arg) => json!({ "arg": arg, "add": 0 }),
        BoundSpec::ArgAdd(arg, add) => json!({ "arg": arg, "add": add }),
    }
}

fn scale_json(scale: Scale) -> Value {
    match scale {
        Scale::Const(value) => json!({ "arg": null, "factor": value }),
        Scale::Arg(arg, factor) => json!({ "arg": arg, "factor": factor }),
    }
}

fn gpu_json(kernel: &Result<GpuKernel, String>) -> Value {
    let kernel = match kernel {
        Ok(kernel) => kernel,
        Err(error) => return json!({ "error": error }),
    };
    let axes: Vec<_> = std::iter::once(&kernel.axis_x)
        .chain(&kernel.axis_y)
        .map(|axis| json!({ "start": bound_json(&axis.start), "end": bound_json(&axis.end) }))
        .collect();
    json!({
        "wgsl": kernel.wgsl,
        "buffers": kernel
            .buffers
            .iter()
            .map(|buffer| json!({ "arg": buffer.arg_index, "writable": buffer.writable }))
            .collect::<Vec<_>>(),
        "uniforms": kernel.uniforms,
        "axes": axes,
        "strides": kernel
            .strides
            .iter()
            .map(|check| json!({ "stride": scale_json(check.stride), "inner": check.inner, "axis": check.inner_axis }))
            .collect::<Vec<_>>(),
        "workgroup": kernel.workgroup_size,
    })
}
