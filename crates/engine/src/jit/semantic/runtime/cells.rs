use super::super::cursor::{Cursor, Insert};
use super::super::dispatch;
use super::super::js::memory::Field;
use super::super::{
    Binding, BindingData, BindingKind, Block, BlockData, Constant, FunctionState, Graph, Inst, Op,
    PropertyWrite, Scope, ScopeKind, Target, TerminatorKind, Value,
};
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit::semantic) fn lower_accesses(function: &mut FunctionState) {
    use cranelift_codegen::ir::{MemFlagsData, types};

    function.ensure_cfg();
    let graph = &mut function.graph;
    for &block in &function.facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            match graph.insts[inst].op {
                Op::CellGet(cell) => {
                    let result = graph.insts[inst].result().unwrap();
                    let mut at = Cursor::before(graph, inst);
                    let view = at.add_unassigned(Op::View(cell));
                    let address = at.ins().load(
                        types::I64,
                        MemFlagsData::trusted(),
                        view,
                        crate::heap::DATA_OFFSET,
                    );
                    graph.set_op(inst, Op::PropertyLoad(cell, address, Field::Cells));
                    Cursor::after(graph, inst).add_effect(Op::CheckInitialized(result));
                }
                Op::CellSet(cell, value) => {
                    let index = Cursor::before(graph, inst)
                        .add_unassigned(Op::Constant(Constant::Number(0.0)));
                    graph.set_op(
                        inst,
                        Op::PropertyStore(cell, index, value, Field::Cells, PropertyWrite::Set),
                    );
                }
                _ => {}
            }
        }
    }
    function.invalidate_cfg();
}

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) -> bool {
    let lazy = companion_cells(&mut function.graph);
    let lowered = lowered_bindings(&function.graph);
    let closures = function
        .graph
        .insts
        .values()
        .any(|data| matches!(data.op, Op::Closure(..)));
    if lowered.is_empty() && lazy.is_empty() && !closures {
        return false;
    }
    let graph = &mut function.graph;
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        for inst in graph.blocks[block].insts.clone() {
            lower_inst(graph, &lowered, &lazy, inst);
        }
    }
    for &binding in lowered.iter().chain(lazy.keys()) {
        graph.bindings[binding].captured = false;
    }
    graph.lazy_captures.clear();
    function.invalidate_cfg();
    true
}

fn lowered_bindings(graph: &Graph) -> FxHashSet<Binding> {
    graph
        .bindings
        .iter()
        .filter(|(binding, data)| data.captured && !graph.lazy_captures.contains(binding))
        .map(|(binding, _)| binding)
        .collect()
}

fn companion_cells(graph: &mut Graph) -> FxHashMap<Binding, Binding> {
    let lazy: Vec<_> = graph
        .bindings
        .keys()
        .filter(|binding| graph.lazy_captures.contains(binding))
        .collect();
    let mut companions = FxHashMap::default();
    for binding in lazy {
        let scope = graph.bindings[binding].scope;
        assert!(graph.scopes[scope].kind != ScopeKind::Captures);
        let name = format!("{}.cell", graph.bindings[binding].name);
        let cell = graph.bindings.push(BindingData {
            name,
            kind: BindingKind::Let,
            scope,
            captured: false,
            persistent: false,
        });
        graph.scopes[scope].bindings.push(cell);
        companions.insert(binding, cell);
    }
    companions
}

fn lower_inst(
    graph: &mut Graph,
    lowered: &FxHashSet<Binding>,
    lazy: &FxHashMap<Binding, Binding>,
    inst: Inst,
) {
    let op = graph.insts[inst].op.clone();
    let lowered_op = match op {
        Op::EnterScope(scope) if graph.scopes[scope].kind == ScopeKind::Captures => {
            let mut at = Cursor::after(graph, inst);
            let closure = at.add_unassigned(Op::Callee);
            capture_cells(&mut at, lowered, scope, closure);
            None
        }
        Op::EnterScope(scope) => {
            let fresh = scope_bindings(graph, lowered, scope);
            let cells = lazy_scope_bindings(graph, lazy, scope);
            let mut at = Cursor::after(graph, inst);
            for binding in fresh {
                initialize_fresh(&mut at, binding);
            }
            for (_, cell) in cells {
                let undefined = at.add_unassigned(Op::Constant(Constant::Undefined));
                at.add_effect(Op::Initialize(cell, undefined));
            }
            None
        }
        Op::EnterCaptures(scope, closure) => {
            let mut at = Cursor::after(graph, inst);
            capture_cells(&mut at, lowered, scope, closure);
            Some(Op::EnterScope(scope))
        }
        Op::NextIteration(scope) => {
            for (binding, cell) in lazy_scope_bindings(graph, lazy, scope) {
                let split = split_on_cell(graph, inst, cell);
                let mut shared = Cursor::at_end(graph, split.shared, inst);
                let value = shared.add_unassigned(Op::CellGet(split.cell));
                shared.add_effect(Op::Initialize(binding, value));
                let undefined = shared.add_unassigned(Op::Constant(Constant::Undefined));
                shared.add_effect(Op::Initialize(cell, undefined));
                shared.terminate_current(jump(split.fallback, Vec::new()));
            }
            let copied = scope_bindings(graph, lowered, scope);
            let mut at = Cursor::after(graph, inst);
            for binding in copied {
                let current = at.add_unassigned(Op::BindingLoad(binding));
                let value = at.add_unassigned(Op::CellGet(current));
                let cell = at.add_unassigned(Op::CellNew(value));
                at.add_effect(Op::Initialize(binding, cell));
            }
            None
        }
        Op::FreshIteration(scope) => {
            let fresh = scope_bindings(graph, lowered, scope);
            let cells = lazy_scope_bindings(graph, lazy, scope);
            let mut at = Cursor::after(graph, inst);
            for binding in fresh {
                initialize_fresh(&mut at, binding);
            }
            for (binding, cell) in cells {
                let undefined = at.add_unassigned(Op::Constant(Constant::Undefined));
                at.add_effect(Op::Initialize(cell, undefined));
                let uninitialized = at.add_unassigned(Op::Constant(Constant::Uninitialized));
                at.add_effect(Op::Initialize(binding, uninitialized));
            }
            None
        }
        Op::Read(binding) if lowered.contains(&binding) => {
            let cell = Cursor::before(graph, inst).add_unassigned(Op::BindingLoad(binding));
            Some(Op::CellGet(cell))
        }
        Op::Read(binding) if lazy.contains_key(&binding) => {
            let split = split_on_cell(graph, inst, lazy[&binding]);
            let mut shared = Cursor::at_end(graph, split.shared, inst);
            let value = shared.add_unassigned(Op::CellGet(split.cell));
            shared.terminate_current(jump(split.continuation, vec![value]));
            None
        }
        Op::Write(binding, value) if lowered.contains(&binding) => {
            let mut at = Cursor::before(graph, inst);
            let cell = at.add_unassigned(Op::BindingLoad(binding));
            at.add_unassigned(Op::CellGet(cell));
            Some(Op::CellSet(cell, value))
        }
        Op::Write(binding, value) if lazy.contains_key(&binding) => {
            let split = split_on_cell(graph, inst, lazy[&binding]);
            let mut shared = Cursor::at_end(graph, split.shared, inst);
            shared.add_unassigned(Op::CellGet(split.cell));
            shared.add_effect(Op::CellSet(split.cell, value));
            shared.terminate_current(jump(split.continuation, Vec::new()));
            None
        }
        Op::Initialize(binding, value) if lowered.contains(&binding) => {
            let cell = Cursor::before(graph, inst).add_unassigned(Op::BindingLoad(binding));
            Some(Op::CellSet(cell, value))
        }
        Op::Initialize(binding, value) if lazy.contains_key(&binding) => {
            let split = split_on_cell(graph, inst, lazy[&binding]);
            let mut shared = Cursor::at_end(graph, split.shared, inst);
            shared.add_effect(Op::CellSet(split.cell, value));
            shared.terminate_current(jump(split.continuation, Vec::new()));
            None
        }
        Op::Closure(span, id, arrow, generator, lexical_this, captures, receiver) => {
            let cells = if captures.iter().any(|binding| lazy.contains_key(binding)) {
                materialize_cells(graph, inst, lazy, &captures)
            } else {
                FxHashMap::default()
            };
            let mut at = Cursor::before(graph, inst);
            let mut values = Vec::with_capacity(captures.len());
            for binding in captures {
                let value = match cells.get(&binding) {
                    Some(&cell) => cell,
                    None => at.add_unassigned(Op::BindingLoad(binding)),
                };
                values.push(value);
            }
            Some(Op::ClosureCells(
                span,
                id,
                arrow,
                generator,
                lexical_this,
                values,
                receiver,
            ))
        }
        _ => None,
    };
    if let Some(op) = lowered_op {
        graph.set_op(inst, op);
    }
}

fn capture_cells(at: &mut Cursor, lowered: &FxHashSet<Binding>, scope: Scope, closure: Value) {
    let bindings = at.graph().scopes[scope].bindings.clone();
    for (index, binding) in bindings.into_iter().enumerate() {
        if lowered.contains(&binding) {
            let cell = at.add_unassigned(Op::CaptureGet(closure, index));
            at.add_effect(Op::Initialize(binding, cell));
        }
    }
}

fn initialize_fresh(at: &mut Cursor, binding: Binding) {
    let uninitialized = at.add_unassigned(Op::Constant(Constant::Uninitialized));
    let cell = at.add_unassigned(Op::CellNew(uninitialized));
    at.add_effect(Op::Initialize(binding, cell));
}

fn scope_bindings(graph: &Graph, lowered: &FxHashSet<Binding>, scope: Scope) -> Vec<Binding> {
    graph.scopes[scope]
        .bindings
        .iter()
        .copied()
        .filter(|binding| lowered.contains(binding))
        .collect()
}

fn lazy_scope_bindings(
    graph: &Graph,
    lazy: &FxHashMap<Binding, Binding>,
    scope: Scope,
) -> Vec<(Binding, Binding)> {
    graph.scopes[scope]
        .bindings
        .iter()
        .filter_map(|&binding| lazy.get(&binding).map(|&cell| (binding, cell)))
        .collect()
}

struct LazySplit {
    continuation: Block,
    fallback: Block,
    shared: Block,
    cell: Value,
}

fn split_on_cell(graph: &mut Graph, inst: Inst, cell: Binding) -> LazySplit {
    let origin = graph.insts[inst].block;
    let cold = graph.cold.contains(&origin);
    let first = graph.blocks.len();
    let (block, continuation, fallback) = dispatch::split(graph, inst);
    if !cold {
        graph.cold.remove(&fallback);
    }
    let shared = graph.blocks.push(BlockData::default());
    let mut at = Cursor::at_end(graph, block, inst);
    let cell_value = at.add_unassigned(Op::BindingLoad(cell));
    let is_undefined = at.add_unassigned(Op::IsUndefined(cell_value));
    at.terminate_current(TerminatorKind::Branch(
        is_undefined,
        Target::new(fallback),
        Target::new(shared),
    ));
    dispatch::extend_regions(graph, origin, first);
    LazySplit {
        continuation,
        fallback,
        shared,
        cell: cell_value,
    }
}

fn materialize_cells(
    graph: &mut Graph,
    inst: Inst,
    lazy: &FxHashMap<Binding, Binding>,
    captures: &[Binding],
) -> FxHashMap<Binding, Value> {
    let origin = graph.insts[inst].block;
    let cold = graph.cold.contains(&origin);
    let first = graph.blocks.len();
    let (mut current, _, fallback) = dispatch::split(graph, inst);
    if !cold {
        graph.cold.remove(&fallback);
    }
    let mut cells = FxHashMap::default();
    for &binding in captures {
        let Some(&cell) = lazy.get(&binding) else {
            continue;
        };
        let create = graph.blocks.push(BlockData::default());
        let join = graph.blocks.push(BlockData::default());
        let joined = graph.add_param(join);
        let mut at = Cursor::at_end(graph, current, inst);
        let existing = at.add_unassigned(Op::BindingLoad(cell));
        let is_undefined = at.add_unassigned(Op::IsUndefined(existing));
        at.terminate_current(TerminatorKind::Branch(
            is_undefined,
            Target::new(create),
            Target {
                block: join,
                args: vec![existing],
            },
        ));
        let mut at = Cursor::at_end(graph, create, inst);
        let raw = at.add_unassigned(Op::BindingLoad(binding));
        let created = at.add_unassigned(Op::CellNew(raw));
        at.add_effect(Op::Initialize(cell, created));
        let undefined = at.add_unassigned(Op::Constant(Constant::Undefined));
        at.add_effect(Op::Initialize(binding, undefined));
        at.terminate_current(jump(join, vec![created]));
        cells.insert(binding, joined);
        current = join;
    }
    Cursor::at_end(graph, current, inst).terminate_current(jump(fallback, Vec::new()));
    dispatch::extend_regions(graph, origin, first);
    cells
}

fn jump(block: Block, args: Vec<Value>) -> TerminatorKind {
    TerminatorKind::Jump(Target { block, args })
}
