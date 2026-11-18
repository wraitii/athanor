use crate::FxHashMap;
use crate::jit::layout::ReadLayout;
use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};
use crate::jit::semantic::JsOp;

use super::super::{Cfg, Graph, Inst, Op};

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct Plan {
    source: usize,
    accesses: Vec<(Site, PropertyAccess, ReadLayout)>,
    polymorphic: Vec<(Site, Vec<ReadLayout>)>,
    shared_reads: FxHashMap<Site, Vec<ReadLayout>>,
    megamorphic: Vec<Site>,
}

impl Plan {
    pub(in crate::jit) fn polymorphic_reads(&self) -> &[(Site, Vec<ReadLayout>)] {
        &self.polymorphic
    }

    pub(in crate::jit::semantic) fn alternatives(
        &self,
        graph: &Graph,
        inst: Inst,
    ) -> Option<&[ReadLayout]> {
        let site = site(graph, inst, self.source);
        self.polymorphic
            .iter()
            .find_map(|(origin, entries)| (*origin == site).then_some(entries.as_slice()))
    }

    pub(super) fn shared_reads(&self, graph: &Graph, inst: Inst) -> &[ReadLayout] {
        self.shared_reads
            .get(&site(graph, inst, self.source))
            .map_or(&[], Vec::as_slice)
    }

    pub(in crate::jit) fn megamorphic_sites(&self) -> &[Site] {
        &self.megamorphic
    }

    pub(in crate::jit::semantic) fn megamorphic(&self, graph: &Graph, inst: Inst) -> bool {
        self.megamorphic.contains(&site(graph, inst, self.source))
    }

    pub(in crate::jit::semantic) fn accesses(&self) -> &[(Site, PropertyAccess, ReadLayout)] {
        &self.accesses
    }

    pub(in crate::jit::semantic) fn read(&self, graph: &Graph, inst: Inst) -> Option<ReadLayout> {
        self.access(graph, inst, PropertyAccess::Get)
    }

    pub(in crate::jit::semantic) fn write(&self, graph: &Graph, inst: Inst) -> Option<ReadLayout> {
        self.access(graph, inst, PropertyAccess::Set)
    }

    fn access(&self, graph: &Graph, inst: Inst, kind: PropertyAccess) -> Option<ReadLayout> {
        let site = site(graph, inst, self.source);
        self.accesses.iter().find_map(|&(origin, access, layout)| {
            (origin == site && access == kind).then_some(layout)
        })
    }
}

fn property_access(op: &Op) -> Option<PropertyAccess> {
    match op {
        Op::Javascript(JsOp::Get(_, name) | JsOp::ProbeProperty(_, name)) if name != "length" => {
            Some(PropertyAccess::Get)
        }
        Op::Javascript(JsOp::Set(..)) => Some(PropertyAccess::Set),
        _ => None,
    }
}

fn site(graph: &Graph, inst: Inst, source: usize) -> Site {
    let data = &graph.insts[inst];
    Site {
        inline_context: data.inline_context,
        source: data.source_name.map_or(source, |name| name.0),
        start: data.span.start,
        end: data.span.end,
    }
}

pub(in crate::jit::semantic) fn plan(
    graph: &Graph,
    bodies: &super::calls::Repository,
    source: usize,
) -> Plan {
    let callees = graph.callees();
    let assumed: FxHashMap<_, _> = graph.blocks[graph.entry]
        .insts
        .iter()
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::EntryAssume(value, super::super::guards::Fact::Shape(shape)) => {
                Some((value, shape))
            }
            _ => None,
        })
        .collect();
    let mut accesses = Vec::new();
    let mut polymorphic = Vec::new();
    let mut shared_reads = FxHashMap::default();
    let mut megamorphic = Vec::new();
    for (inst, data) in graph.insts.iter() {
        if property_access(&data.op) != Some(PropertyAccess::Get) {
            continue;
        }
        let site = site(graph, inst, source);
        if bodies.shape_feedback(site, PropertyAccess::Get) == Some(ShapeAccess::Megamorphic)
            && !megamorphic.contains(&site)
        {
            megamorphic.push(site);
        }
    }
    for block in Cfg::new(graph).rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let Some(property) = property_access(&data.op) else {
                continue;
            };
            let site = site(graph, inst, source);
            let observed = bodies.shape_feedback(site, property);
            if observed.is_none()
                && site.inline_context != 0
                && matches!(data.op, Op::Javascript(JsOp::Get(..)))
                && let Some(ShapeAccess::PolymorphicRead { entries, len }) = bodies.shape_feedback(
                    Site {
                        inline_context: 0,
                        ..site
                    },
                    property,
                )
            {
                shared_reads
                    .entry(site)
                    .or_insert_with(|| entries[..usize::from(len)].to_vec());
            }
            let callee = data.result().is_some_and(|value| callees.contains(&value));
            let object = match data.op {
                Op::Javascript(
                    JsOp::Get(object, _) | JsOp::ProbeProperty(object, _) | JsOp::Set(object, ..),
                ) => object,
                _ => continue,
            };
            if let Some(ShapeAccess::PolymorphicRead { entries, len }) = observed
                && matches!(data.op, Op::Javascript(JsOp::Get(..)))
                && (callee
                    || !assumed.get(&object).is_some_and(|shape| {
                        entries[..usize::from(len)]
                            .iter()
                            .any(|entry| entry.shape == *shape)
                    }))
            {
                if !polymorphic.iter().any(|(origin, _)| *origin == site) {
                    polymorphic.push((site, entries[..usize::from(len)].to_vec()));
                }
                continue;
            }
            if callee
                && !observed
                    .and_then(|access| access.read(None))
                    .is_some_and(|layout| layout.storage.is_some())
            {
                continue;
            }
            let access = match (property, observed) {
                (PropertyAccess::Get, Some(access)) => {
                    let Some(layout) = access.read(assumed.get(&object).copied()) else {
                        continue;
                    };
                    (site, property, layout)
                }
                (PropertyAccess::Set, Some(ShapeAccess::Overwrite { shape, index })) => {
                    (site, property, ReadLayout::own(shape, index))
                }
                _ => continue,
            };
            if !accesses.contains(&access) {
                accesses.push(access);
            }
        }
    }
    Plan {
        source,
        accesses,
        polymorphic,
        shared_reads,
        megamorphic,
    }
}
