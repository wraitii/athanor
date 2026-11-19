use std::sync::Arc;

use js_engine::{CompileOptions, Engine, EngineOptions, ReportSink};
use serde_json::json;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn compile(source: &str, filter: Option<String>) -> Result<String, String> {
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
            })
        })
        .collect();
    Ok(serde_json::Value::Array(reports).to_string())
}
