use crate::FxHashMap;
use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};

use super::super::{Cfg, Graph, Inst, Op};

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct Plan {
    source: usize,
    accesses: Vec<(Site, PropertyAccess, u32, u32)>,
    polymorphic: Vec<(Site, Vec<(u32, u32)>)>,
    megamorphic: Vec<Site>,
}

impl Plan {
    pub(in crate::jit) fn polymorphic_reads(&self) -> &[(Site, Vec<(u32, u32)>)] {
        &self.polymorphic
    }

    pub(in crate::jit::semantic) fn alternatives(
        &self,
        graph: &Graph,
        inst: Inst,
    ) -> Option<&[(u32, u32)]> {
        let site = site(graph, inst, self.source);
        self.polymorphic
            .iter()
            .find_map(|(origin, entries)| (*origin == site).then_some(entries.as_slice()))
    }

    pub(in crate::jit) fn megamorphic_sites(&self) -> &[Site] {
        &self.megamorphic
    }

    pub(in crate::jit::semantic) fn megamorphic(&self, graph: &Graph, inst: Inst) -> bool {
        self.megamorphic.contains(&site(graph, inst, self.source))
    }

    pub(in crate::jit::semantic) fn accesses(&self) -> &[(Site, PropertyAccess, u32, u32)] {
        &self.accesses
    }

    pub(in crate::jit::semantic) fn read(&self, graph: &Graph, inst: Inst) -> Option<(u32, u32)> {
        self.access(graph, inst, PropertyAccess::Get)
    }

    pub(in crate::jit::semantic) fn write(&self, graph: &Graph, inst: Inst) -> Option<(u32, u32)> {
        self.access(graph, inst, PropertyAccess::Set)
    }

    fn access(&self, graph: &Graph, inst: Inst, kind: PropertyAccess) -> Option<(u32, u32)> {
        let site = site(graph, inst, self.source);
        self.accesses
            .iter()
            .find_map(|&(origin, access, shape, index)| {
                (origin == site && access == kind).then_some((shape, index))
            })
    }
}

fn property_access(op: &Op) -> Option<PropertyAccess> {
    match op {
        Op::Get(_, name) | Op::ProbeProperty(_, name) if name != "length" => {
            Some(PropertyAccess::Get)
        }
        Op::Set(..) => Some(PropertyAccess::Set),
        _ => None,
    }
}

fn site(graph: &Graph, inst: Inst, source: usize) -> Site {
    let data = &graph.insts[inst];
    Site {
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
    let mut megamorphic = Vec::new();
    for (inst, data) in graph.insts.iter() {
        if property_access(&data.op) != Some(PropertyAccess::Get) {
            continue;
        }
        let site = site(graph, inst, source);
        if bodies.shape_feedback.get(&(site, PropertyAccess::Get))
            == Some(&ShapeAccess::Megamorphic)
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
            let callee = data.result().is_some_and(|value| callees.contains(&value));
            let object = match data.op {
                Op::Get(object, _) | Op::ProbeProperty(object, _) | Op::Set(object, ..) => object,
                _ => continue,
            };
            if let Some(&ShapeAccess::PolymorphicRead { entries, len }) =
                bodies.shape_feedback.get(&(site, property))
                && matches!(data.op, Op::Get(..))
                && (callee
                    || !assumed.get(&object).is_some_and(|shape| {
                        entries[..usize::from(len)]
                            .iter()
                            .any(|entry| entry.0 == *shape)
                    }))
            {
                if !polymorphic.iter().any(|(origin, _)| *origin == site) {
                    polymorphic.push((site, entries[..usize::from(len)].to_vec()));
                }
                continue;
            }
            if callee {
                continue;
            }
            let access = match (property, bodies.shape_feedback.get(&(site, property))) {
                (PropertyAccess::Get, Some(&access)) => {
                    let Some((shape, index)) = access.read(assumed.get(&object).copied()) else {
                        continue;
                    };
                    (site, property, shape, index)
                }
                (PropertyAccess::Set, Some(&ShapeAccess::Overwrite { shape, index })) => {
                    (site, property, shape, index)
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
        megamorphic,
    }
}
