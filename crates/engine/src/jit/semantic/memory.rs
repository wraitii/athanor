use super::super::effects::Locations;
use super::js::memory as js;
use super::{Cfg, Facts, Graph, Inst, Op, Value};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Access {
    pub(super) location: Locations,
    pub(super) js: Option<js::Access>,
}

impl Access {
    pub(super) fn broad(location: Locations) -> Self {
        Self { location, js: None }
    }

    pub(super) fn js(location: Locations, base: Option<Value>, field: js::Field) -> Self {
        Self {
            location,
            js: Some(js::Access { base, field }),
        }
    }

    pub(super) fn overlaps(&self, read: &Self) -> bool {
        // Global bindings and heap properties can refer to the same JS state.
        let data = Locations::HEAP.union(Locations::GLOBAL);
        let code = Locations::CODE;
        (self.location.intersects(data) && read.location.intersects(data))
            || (self.location.intersects(code) && read.location.intersects(code))
    }

    pub(super) fn read(graph: &Graph, op: &Op, facts: &Facts) -> Option<Self> {
        let (location, access) = js::Access::read(graph, op, facts)?;
        Some(Self {
            location,
            js: Some(access),
        })
    }

    pub(super) fn written(graph: &Graph, op: &Op, facts: &Facts, location: Locations) -> Self {
        Self {
            location,
            js: js::Access::written(graph, op, facts),
        }
    }

    pub(super) fn is_unknown(&self) -> bool {
        self.js.as_ref().is_none_or(js::Access::is_unknown)
    }
}

pub(super) struct Analysis {
    js: js::Fresh,
}

impl Analysis {
    pub(super) fn compute(graph: &Graph, cfg: &Cfg) -> Self {
        Self {
            js: js::Fresh::compute(graph, cfg),
        }
    }

    pub(super) fn may_clobber(&self, inst: Inst, write: &Access, read: &Access) -> bool {
        if !write.overlaps(read) {
            return false;
        }
        self.js
            .may_clobber(inst, write.js.as_ref(), read.js.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::Signature;
    use cranelift_codegen::isa::CallConv;
    use oxc_span::Span;

    #[test]
    fn broad_accesses_preserve_unknown_js_conflicts() {
        let graph = Graph::new_low_level(
            "memory".into(),
            Span::new(0, 0),
            Signature::new(CallConv::SystemV),
        );
        let analysis = Analysis::compute(&graph, &Cfg::new(&graph));
        let inst = Inst::from_u32(0);
        let fields = [
            js::Field::Any,
            js::Field::Named("x".into()),
            js::Field::Cells,
            js::Field::Packs,
            js::Field::Length,
            js::Field::Elements,
            js::Field::Iteration,
        ];
        for location in [Locations::HEAP, Locations::GLOBAL, Locations::CODE] {
            let broad = Access::broad(location);
            let unknown = Access::js(location, None, js::Field::Any);
            for other in [Locations::HEAP, Locations::GLOBAL, Locations::CODE] {
                for field in &fields {
                    let access = Access::js(other, None, field.clone());
                    assert_eq!(
                        analysis.may_clobber(inst, &broad, &access),
                        analysis.may_clobber(inst, &unknown, &access)
                    );
                    assert_eq!(
                        analysis.may_clobber(inst, &access, &broad),
                        analysis.may_clobber(inst, &access, &unknown)
                    );
                }
            }
        }
    }
}
