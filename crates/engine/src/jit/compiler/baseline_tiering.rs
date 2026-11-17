use super::{Compiler, RecompileJob, RecompileRequest, VariantId};
use crate::error::Error;
use crate::jit;
use std::sync::Arc;

impl Compiler {
    pub(crate) fn baseline_osr_compilation(
        &self,
        site: &jit::baseline_tiering::Site,
    ) -> Result<Option<RecompileJob>, Error> {
        let Some(code) = self
            .canonical
            .code(site.source_name.0, site.graph.span.start)
        else {
            return Ok(None);
        };
        if !self
            .emissions
            .contains_key(&VariantId::ordinary(u64::from(code.as_u32())))
            || !self.function_names.contains_key(&code)
        {
            return Ok(None);
        }
        let (graph, inputs) = jit::semantic::osr::baseline_resume(&site.graph, site.header)
            .map_err(Error::Unsupported)?;
        let Some(mut job) = self.recompile_job(
            u64::from(code.as_u32()),
            RecompileRequest::Observing(jit::baseline_tiering::Request {
                site: site as *const _ as usize,
                graph: Arc::new(graph),
            }),
        ) else {
            return Ok(None);
        };
        *site.inputs.borrow_mut() = Some(inputs);
        job.label += &format!(".observing.{}", site.header);
        Ok(Some(job))
    }
}
