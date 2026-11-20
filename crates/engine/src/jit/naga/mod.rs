mod affine;
mod kernel;
#[cfg(feature = "gpu")]
pub(super) mod runtime;

use naga::back::wgsl::{self, WriterFlags};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, BinaryOperator as NagaBinOp, Binding as NagaBinding, Block as NagaBlock,
    BuiltIn, EntryPoint, Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal,
    LocalVariable, Module, ResourceBinding, Scalar, ScalarKind, ShaderStage, Span as NagaSpan,
    Statement, StorageAccess, StructMember, Type, TypeInner, UnaryOperator as NagaUnaryOp,
    VectorSize,
};
use oxc_syntax::operator::BinaryOperator;

use crate::{FxHashMap, FxHashSet};

use super::semantic::low_level;
use super::semantic::{
    Block, Cfg, Constant, Graph, JsOp, Op, Target, TerminatorKind, UnaryOp, Value, natural_loops,
};
use affine::Affine;
use kernel::{AxisSpec, BoundSpec, GpuKernel, KernelBuffer, StrideCheck};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarTy {
    I32,
    F32,
    Bool,
}

struct CountedLoop {
    header: Block,
    latch: Block,
    body_entry: Block,
    exit: Block,
    iv: Value,
    start: Value,
    end: Value,
    next: Value,
    cond: Value,
    body: FxHashSet<Block>,
}

struct BufferInfo {
    arg: Value,
    arg_index: usize,
    writable: bool,
}

pub(super) fn compile_wgsl(graph: &Graph) -> Result<String, String> {
    compile_kernel(graph).map(|kernel| kernel.to_string())
}

#[cfg(feature = "gpu")]
pub(in crate::jit) fn try_inject_gpu(graph: &mut Graph) -> bool {
    use super::effects::{Effects, HEAP_READ, HEAP_WRITE, INTERNAL};
    use super::helpers::Helper;
    use super::semantic::{BlockData, Repr, ValueType};

    if std::env::var("JS_JIT_GPU").as_deref() != Ok("1") {
        return false;
    }
    let Ok(kernel) = compile_kernel(graph) else {
        return false;
    };
    let arity = kernel
        .buffers
        .iter()
        .map(|buffer| buffer.arg_index)
        .chain(kernel.uniforms.iter().copied())
        .max()
        .map_or(0, |max| max + 1);
    let kernel_id = runtime::register_kernel(kernel);
    let old_entry = graph.entry;
    let span = graph.span;
    let gpu_entry = graph.blocks.push(BlockData::default());
    let gpu_done = graph.blocks.push(BlockData::default());
    let mut arg_values = Vec::with_capacity(arity);
    for i in 0..arity {
        arg_values.push(graph.append_value(gpu_entry, span, Op::Argument(i)));
    }
    let args_array = graph.append_value(gpu_entry, span, Op::Javascript(JsOp::NewArray(arg_values)));
    let id_val = graph.append_value(
        gpu_entry,
        span,
        Op::LowLevel(low_level::Op::Integer {
            ty: cranelift_codegen::ir::types::I64,
            bits: kernel_id as u64,
        }),
    );
    let dispatched = graph.append_value(
        gpu_entry,
        span,
        Op::HelperCall {
            helper: Helper::GpuDispatch,
            args: vec![id_val, args_array],
            uses: vec![Repr::I64, Repr::Boxed],
            produces: Some(Repr::I64),
            result_type: ValueType::INTEGER,
            effects: Effects::of(HEAP_READ | HEAP_WRITE | INTERNAL),
        },
    );
    let zero = graph.append_value(
        gpu_entry,
        span,
        Op::LowLevel(low_level::Op::Integer {
            ty: cranelift_codegen::ir::types::I64,
            bits: 0,
        }),
    );
    let ok = graph.append_value(
        gpu_entry,
        span,
        Op::LowLevel(low_level::Op::IntCompare {
            condition: cranelift_codegen::ir::condcodes::IntCC::NotEqual,
            left: dispatched,
            right: zero,
        }),
    );
    graph.terminate(
        gpu_entry,
        span,
        TerminatorKind::Branch(ok, Target::new(gpu_done), Target::new(old_entry)),
    );
    let undef = graph.append_value(
        gpu_done,
        span,
        Op::Javascript(JsOp::Constant(Constant::Undefined)),
    );
    graph.terminate(gpu_done, span, TerminatorKind::Return(undef));
    graph.entry = gpu_entry;
    true
}

fn extract_bound(graph: &Graph, value: Value) -> Option<BoundSpec> {
    match graph.defining_op(value)? {
        Op::Javascript(JsOp::Constant(Constant::Number(n))) if n.fract() == 0.0 => {
            Some(BoundSpec::Const(*n as i32))
        }
        Op::Argument(i) => Some(BoundSpec::Arg(*i)),
        Op::Javascript(JsOp::Binary(BinaryOperator::Subtraction, l, r)) => {
            let Op::Argument(i) = graph.defining_op(*l)? else {
                return None;
            };
            let Op::Javascript(JsOp::Constant(Constant::Number(n))) = graph.defining_op(*r)? else {
                return None;
            };
            Some(BoundSpec::ArgAdd(*i, -(*n as i32)))
        }
        Op::Javascript(JsOp::Binary(BinaryOperator::Addition, l, r)) => {
            if let (Some(Op::Argument(i)), Some(Op::Javascript(JsOp::Constant(Constant::Number(n))))) =
                (graph.defining_op(*l), graph.defining_op(*r))
            {
                return Some(BoundSpec::ArgAdd(*i, *n as i32));
            }
            if let (Some(Op::Javascript(JsOp::Constant(Constant::Number(n)))), Some(Op::Argument(i))) =
                (graph.defining_op(*l), graph.defining_op(*r))
            {
                return Some(BoundSpec::ArgAdd(*i, *n as i32));
            }
            None
        }
        Op::LowLevel(low_level::Op::Binary {
            op: low_level::Binary::Subtract,
            left,
            right,
        }) => {
            let Op::Argument(i) = graph.defining_op(*left)? else {
                return None;
            };
            let Op::Javascript(JsOp::Constant(Constant::Number(n))) = graph.defining_op(*right)?
            else {
                return None;
            };
            Some(BoundSpec::ArgAdd(*i, -(*n as i32)))
        }
        _ => None,
    }
}

fn compile_kernel(graph: &Graph) -> Result<GpuKernel, String> {
    if graph.generator {
        return Err("generators are not supported".into());
    }
    if graph.has_handlers {
        return Err("exception handlers are not supported".into());
    }
    let cfg = Cfg::new(graph);
    let loops = natural_loops(graph, &cfg);
    let (outer_y, outer_x) = find_grid_loops(graph, &cfg, &loops)?;
    let (kernel_header, kernel_entry, kernel_latch, kernel_blocks) = match &outer_x {
        Some(inner) => (inner.header, inner.body_entry, inner.latch, &inner.body),
        None => (outer_y.header, outer_y.body_entry, outer_y.latch, &outer_y.body),
    };
    let grid: Vec<&CountedLoop> = std::iter::once(&outer_y).chain(outer_x.as_ref()).collect();
    check_isolated(graph, &grid, kernel_blocks, kernel_header)?;
    let mut inner_loops: FxHashMap<Block, FxHashSet<Block>> = FxHashMap::default();
    for (header, body) in &loops {
        if *header == outer_y.header {
            continue;
        }
        if outer_x.as_ref().is_some_and(|x| *header == x.header) {
            continue;
        }
        if kernel_blocks.contains(header) {
            inner_loops.insert(*header, body.clone());
        }
    }
    let types = infer_types(graph, &outer_y, outer_x.as_ref());
    let (buffers, uniforms) = collect_inputs(graph, kernel_blocks, &outer_y, outer_x.as_ref());
    if buffers.is_empty() {
        return Err("no storage buffers found".into());
    }
    let axes: Vec<Value> = match &outer_x {
        Some(x_loop) => vec![x_loop.iv, outer_y.iv],
        None => vec![outer_y.iv],
    };
    let strides = check_independent(graph, kernel_blocks, kernel_header, &axes, &buffers)?;
    let (axis_x, axis_y, workgroup_size) = if let Some(ref x_loop) = outer_x {
        let ax = AxisSpec {
            start: extract_bound(graph, x_loop.start)
                .ok_or_else(|| "non-affine x start".to_string())?,
            end: extract_bound(graph, x_loop.end)
                .ok_or_else(|| "non-affine x end".to_string())?,
        };
        let ay = AxisSpec {
            start: extract_bound(graph, outer_y.start)
                .ok_or_else(|| "non-affine y start".to_string())?,
            end: extract_bound(graph, outer_y.end)
                .ok_or_else(|| "non-affine y end".to_string())?,
        };
        (ax, Some(ay), [16, 16, 1])
    } else {
        let ax = AxisSpec {
            start: extract_bound(graph, outer_y.start)
                .ok_or_else(|| "non-affine start".to_string())?,
            end: extract_bound(graph, outer_y.end)
                .ok_or_else(|| "non-affine end".to_string())?,
        };
        (ax, None, [64, 1, 1])
    };
    let mut emitter = Emitter::new(graph, &cfg, &types, kernel_blocks, &inner_loops);
    emitter.declare_inputs(&buffers, &uniforms);
    emitter.emit_entry_point(&outer_y, outer_x.as_ref(), kernel_entry, kernel_latch)?;
    let module = emitter.finish();
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|error| format!("naga validation failed: {error:?}"))?;
    let wgsl = wgsl::write_string(&module, &info, WriterFlags::empty())
        .map_err(|error| format!("wgsl emit failed: {error}"))?;
    Ok(GpuKernel {
        wgsl,
        #[cfg(feature = "gpu")]
        module,
        buffers: buffers
            .iter()
            .map(|b| KernelBuffer {
                arg_index: b.arg_index,
                writable: b.writable,
            })
            .collect(),
        uniforms: uniforms.iter().map(|&(_, index)| index).collect(),
        axis_x,
        axis_y,
        strides,
        workgroup_size,
    })
}

fn find_grid_loops(
    graph: &Graph,
    cfg: &Cfg,
    loops: &[(Block, FxHashSet<Block>)],
) -> Result<(CountedLoop, Option<CountedLoop>), String> {
    let mut candidates = Vec::new();
    for (header, body) in loops {
        if let Some(counted) = match_counted_loop(graph, cfg, *header, body) {
            candidates.push(counted);
        }
    }
    let outer = candidates
        .iter()
        .enumerate()
        .max_by_key(|(_, l)| l.body.len())
        .map(|(i, _)| i)
        .ok_or_else(|| "no counted outer loop found".to_string())?;
    let outer = candidates.remove(outer);
    let inner = candidates
        .into_iter()
        .filter(|l| outer.body.contains(&l.header) && l.header != outer.header)
        .max_by_key(|l| l.body.len())
        .filter(|inner| {
            follows_empty_chain(graph, outer.body_entry, inner.header)
                && follows_empty_chain(graph, inner.exit, outer.latch)
        });
    Ok((outer, inner))
}

fn follows_empty_chain(graph: &Graph, mut cur: Block, target: Block) -> bool {
    let mut visited = FxHashSet::default();
    while cur != target {
        if !visited.insert(cur) {
            return false;
        }
        if !graph.blocks[cur].insts.is_empty() {
            return false;
        }
        match graph.blocks[cur].terminator.as_ref().map(|t| &t.kind) {
            Some(TerminatorKind::Jump(next)) => cur = next.block,
            _ => return false,
        }
    }
    true
}

fn match_counted_loop(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &FxHashSet<Block>,
) -> Option<CountedLoop> {
    if graph.blocks[header].params.len() != 1 {
        return None;
    }
    let iv = graph.blocks[header].params[0];
    let outside: Vec<Block> = cfg.preds[header]
        .iter()
        .copied()
        .filter(|p| !body.contains(p))
        .collect();
    let [preheader] = outside[..] else {
        return None;
    };
    let inside: Vec<Block> = cfg.preds[header]
        .iter()
        .copied()
        .filter(|p| body.contains(p))
        .collect();
    let [latch] = inside[..] else {
        return None;
    };
    let start = jump_arg(graph, preheader, header, 0)?;
    let next = jump_arg(graph, latch, header, 0)?;
    if !is_unit_increment(graph, next, iv) {
        return None;
    }
    let TerminatorKind::Branch(cond, then_t, else_t) =
        &graph.blocks[header].terminator.as_ref()?.kind
    else {
        return None;
    };
    let Op::Javascript(JsOp::Binary(BinaryOperator::LessThan, left, right)) =
        graph.defining_op(*cond)?
    else {
        return None;
    };
    if *left != iv || !body.contains(&then_t.block) || body.contains(&else_t.block) {
        return None;
    }
    Some(CountedLoop {
        header,
        latch,
        body_entry: then_t.block,
        exit: else_t.block,
        iv,
        start,
        end: *right,
        next,
        cond: *cond,
        body: body.clone(),
    })
}

fn jump_arg(graph: &Graph, from: Block, to: Block, index: usize) -> Option<Value> {
    let target = match &graph.blocks[from].terminator.as_ref()?.kind {
        TerminatorKind::Jump(target) if target.block == to => target,
        TerminatorKind::Branch(_, t, _) if t.block == to => t,
        TerminatorKind::Branch(_, _, e) if e.block == to => e,
        _ => return None,
    };
    target.args.get(index).copied()
}

fn is_unit_increment(graph: &Graph, next: Value, iv: Value) -> bool {
    match graph.defining_op(next) {
        Some(Op::Javascript(JsOp::Unary(UnaryOp::Increment, v))) => *v == iv,
        Some(Op::Javascript(JsOp::Binary(BinaryOperator::Addition, l, r))) => {
            (*l == iv && is_const_one(graph, *r)) || (*r == iv && is_const_one(graph, *l))
        }
        Some(Op::LowLevel(low_level::Op::Binary {
            op: low_level::Binary::Add,
            left,
            right,
        })) => (*left == iv && is_const_one(graph, *right)) || (*right == iv && is_const_one(graph, *left)),
        _ => false,
    }
}

fn is_const_one(graph: &Graph, value: Value) -> bool {
    matches!(
        graph.defining_op(value),
        Some(Op::Javascript(JsOp::Constant(Constant::Number(n)))) if *n == 1.0
    )
}

fn check_isolated(
    graph: &Graph,
    grid: &[&CountedLoop],
    kernel_blocks: &FxHashSet<Block>,
    kernel_header: Block,
) -> Result<(), String> {
    let allowed: FxHashSet<Value> = grid.iter().flat_map(|l| [l.end, l.next, l.cond]).collect();
    let mut seen = FxHashSet::default();
    let mut stack = vec![graph.entry];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        stack.extend(graph.successors(block));
        if block != kernel_header && kernel_blocks.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let pure = matches!(
                data.op,
                Op::Argument(_)
                    | Op::Javascript(JsOp::Constant(_))
                    | Op::Observe(..)
                    | Op::ObserveMethod(..)
                    | Op::ReadHint(..)
            );
            if !pure && !data.result().is_some_and(|value| allowed.contains(&value)) {
                return Err(format!("code outside the loop nest in {block}"));
            }
        }
        match graph.blocks[block].terminator.as_ref().map(|t| &t.kind) {
            Some(TerminatorKind::Jump(_)) => {}
            Some(TerminatorKind::Branch(..)) if grid.iter().any(|l| l.header == block) => {}
            Some(TerminatorKind::Return(value))
                if matches!(
                    graph.defining_op(*value),
                    Some(Op::Javascript(JsOp::Constant(Constant::Undefined)))
                ) => {}
            _ => return Err(format!("unsupported control flow outside the loop nest in {block}")),
        }
    }
    Ok(())
}

fn check_independent(
    graph: &Graph,
    kernel_blocks: &FxHashSet<Block>,
    kernel_header: Block,
    axes: &[Value],
    buffers: &[BufferInfo],
) -> Result<Vec<StrideCheck>, String> {
    let mut stores: Vec<(Value, Affine)> = Vec::new();
    let mut reads = Vec::new();
    for &block in kernel_blocks {
        if block == kernel_header {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            match &graph.insts[inst].op {
                Op::Javascript(JsOp::SetIndex(recv, index, _)) => {
                    let index = Affine::of(graph, *index, axes)
                        .ok_or("store index is not affine in the grid")?;
                    match stores.iter().find(|(buffer, _)| buffer == recv) {
                        Some((_, existing)) if *existing != index => {
                            return Err("buffer stored at several indices".into());
                        }
                        Some(_) => {}
                        None => stores.push((*recv, index)),
                    }
                }
                Op::Javascript(JsOp::GetIndex(recv, index) | JsOp::TypedIndex(recv, index, _))
                    if buffers.iter().any(|b| b.writable && b.arg == *recv) =>
                {
                    reads.push((*recv, *index));
                }
                _ => {}
            }
        }
    }
    for (recv, index) in reads {
        let stored = stores.iter().find(|(buffer, _)| *buffer == recv).map(|(_, index)| index);
        if Affine::of(graph, index, axes).as_ref() != stored {
            return Err("written buffer is read at another iteration's index".into());
        }
    }
    let mut checks = Vec::new();
    for (_, index) in &stores {
        checks.extend(index.injectivity(axes.len())?);
    }
    Ok(checks)
}

fn infer_types(
    graph: &Graph,
    outer_y: &CountedLoop,
    outer_x: Option<&CountedLoop>,
) -> FxHashMap<Value, ScalarTy> {
    let mut types: FxHashMap<Value, ScalarTy> = FxHashMap::default();
    types.insert(outer_y.iv, ScalarTy::I32);
    if let Some(x) = outer_x {
        types.insert(x.iv, ScalarTy::I32);
    }
    for (_, data) in graph.insts.iter() {
        let Some(result) = data.result() else {
            continue;
        };
        let ty = match &data.op {
            Op::Javascript(JsOp::GetIndex(_, _) | JsOp::TypedIndex(_, _, _)) => Some(ScalarTy::F32),
            Op::Javascript(JsOp::Constant(Constant::Number(n))) => {
                if n.fract() == 0.0 && *n >= i32::MIN as f64 && *n <= i32::MAX as f64 {
                    Some(ScalarTy::I32)
                } else {
                    Some(ScalarTy::F32)
                }
            }
            Op::Javascript(JsOp::Constant(Constant::Boolean(_))) => Some(ScalarTy::Bool),
            Op::Javascript(JsOp::Unary(UnaryOp::Not, _)) => Some(ScalarTy::Bool),
            Op::Javascript(JsOp::Binary(
                BinaryOperator::LessThan
                | BinaryOperator::LessEqualThan
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterEqualThan
                | BinaryOperator::StrictEquality
                | BinaryOperator::StrictInequality
                | BinaryOperator::Equality
                | BinaryOperator::Inequality,
                _,
                _,
            )) => Some(ScalarTy::Bool),
            Op::LowLevel(low_level::Op::Binary { op, .. }) => match op {
                low_level::Binary::FloatAdd
                | low_level::Binary::FloatSubtract
                | low_level::Binary::FloatMultiply
                | low_level::Binary::FloatDivide => Some(ScalarTy::F32),
                _ => Some(ScalarTy::I32),
            },
            Op::Javascript(JsOp::NumericFloat(_)) => Some(ScalarTy::F32),
            Op::Argument(_) => Some(ScalarTy::I32),
            _ => None,
        };
        if let Some(ty) = ty {
            types.insert(result, ty);
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (_, data) in graph.insts.iter() {
            let Some(result) = data.result() else {
                continue;
            };
            let next = match &data.op {
                Op::Javascript(JsOp::Binary(
                    BinaryOperator::Addition
                    | BinaryOperator::Subtraction
                    | BinaryOperator::Multiplication
                    | BinaryOperator::Division
                    | BinaryOperator::Remainder,
                    l,
                    r,
                )) => {
                    let lt = types.get(l).copied().unwrap_or(ScalarTy::I32);
                    let rt = types.get(r).copied().unwrap_or(ScalarTy::I32);
                    if lt == ScalarTy::F32 || rt == ScalarTy::F32 {
                        Some(ScalarTy::F32)
                    } else {
                        Some(ScalarTy::I32)
                    }
                }
                Op::Javascript(JsOp::Unary(
                    UnaryOp::Increment | UnaryOp::Decrement | UnaryOp::Negate | UnaryOp::Plus,
                    v,
                )) => Some(types.get(v).copied().unwrap_or(ScalarTy::I32)),
                Op::Javascript(JsOp::AsInteger(_)) => Some(ScalarTy::I32),
                Op::Javascript(JsOp::AsNumber(_)) => Some(ScalarTy::F32),
                _ => None,
            };
            if let Some(ty) = next
                && types.get(&result).copied() != Some(ty)
            {
                types.insert(result, ty);
                changed = true;
            }
        }
        for (_, block) in graph.blocks.iter() {
            let Some(term) = &block.terminator else {
                continue;
            };
            let targets: Vec<&Target> = match &term.kind {
                TerminatorKind::Jump(t) => vec![t],
                TerminatorKind::Branch(_, t, e) => vec![t, e],
                _ => Vec::new(),
            };
            for target in targets {
                for (&param, &arg) in graph.blocks[target.block]
                    .params
                    .iter()
                    .zip(target.args.iter())
                {
                    if param == outer_y.iv || outer_x.is_some_and(|x| param == x.iv) {
                        continue;
                    }
                    let arg_ty = types.get(&arg).copied().unwrap_or(ScalarTy::I32);
                    let cur_ty = types.get(&param).copied();
                    let merged = match (cur_ty, arg_ty) {
                        (Some(ScalarTy::F32), _) | (_, ScalarTy::F32) => ScalarTy::F32,
                        (Some(ScalarTy::Bool), _) | (_, ScalarTy::Bool) => ScalarTy::Bool,
                        _ => ScalarTy::I32,
                    };
                    if cur_ty != Some(merged) {
                        types.insert(param, merged);
                        changed = true;
                    }
                }
            }
        }
    }
    types
}

fn collect_inputs(
    graph: &Graph,
    kernel_blocks: &FxHashSet<Block>,
    outer_y: &CountedLoop,
    outer_x: Option<&CountedLoop>,
) -> (Vec<BufferInfo>, Vec<(Value, usize)>) {
    let mut buf_map: FxHashMap<Value, (usize, bool)> = FxHashMap::default();
    let mut scalar_args: FxHashMap<Value, usize> = FxHashMap::default();
    for &block in kernel_blocks {
        for &inst in &graph.blocks[block].insts {
            match &graph.insts[inst].op {
                Op::Javascript(JsOp::GetIndex(recv, _) | JsOp::TypedIndex(recv, _, _)) => {
                    if let Some(Op::Argument(index)) = graph.defining_op(*recv) {
                        buf_map.entry(*recv).or_insert((*index, false));
                    }
                }
                Op::Javascript(JsOp::SetIndex(recv, _, _)) => {
                    if let Some(Op::Argument(index)) = graph.defining_op(*recv) {
                        let entry = buf_map.entry(*recv).or_insert((*index, true));
                        entry.1 = true;
                    }
                }
                _ => {}
            }
        }
    }
    let mut queue: Vec<Value> = vec![outer_y.start, outer_y.end];
    if let Some(x) = outer_x {
        queue.push(x.start);
        queue.push(x.end);
    }
    for &block in kernel_blocks {
        for &inst in &graph.blocks[block].insts {
            queue.extend(graph.insts[inst].op.operands());
        }
    }
    let mut seen = FxHashSet::default();
    while let Some(v) = queue.pop() {
        if !seen.insert(v) || buf_map.contains_key(&v) {
            continue;
        }
        if let Some(op) = graph.defining_op(v) {
            if let Op::Argument(index) = op {
                scalar_args.insert(v, *index);
            } else {
                queue.extend(op.operands());
            }
        }
    }
    let mut buffers: Vec<BufferInfo> = buf_map
        .into_iter()
        .map(|(arg, (arg_index, writable))| BufferInfo {
            arg,
            arg_index,
            writable,
        })
        .collect();
    buffers.sort_by_key(|b| b.arg_index);
    let mut uniforms: Vec<(Value, usize)> = scalar_args.into_iter().collect();
    uniforms.sort_by_key(|&(_, idx)| idx);
    (buffers, uniforms)
}

struct Emitter<'a> {
    graph: &'a Graph,
    cfg: &'a Cfg,
    types: &'a FxHashMap<Value, ScalarTy>,
    kernel_blocks: &'a FxHashSet<Block>,
    inner_loops: &'a FxHashMap<Block, FxHashSet<Block>>,
    module: Module,
    function: Function,
    ty_i32: Handle<Type>,
    ty_f32: Handle<Type>,
    ty_bool: Handle<Type>,
    ty_vec3_u32: Handle<Type>,
    ty_arr_f32: Handle<Type>,
    values: FxHashMap<Value, (Handle<Expression>, ScalarTy)>,
    buffers: FxHashMap<Value, Handle<GlobalVariable>>,
    uniform_global: Option<Handle<GlobalVariable>>,
    uniform_fields: FxHashMap<Value, (u32, ScalarTy)>,
    param_locals: FxHashMap<(Block, usize), (Handle<LocalVariable>, ScalarTy)>,
}

impl<'a> Emitter<'a> {
    fn new(
        graph: &'a Graph,
        cfg: &'a Cfg,
        types: &'a FxHashMap<Value, ScalarTy>,
        kernel_blocks: &'a FxHashSet<Block>,
        inner_loops: &'a FxHashMap<Block, FxHashSet<Block>>,
    ) -> Self {
        let mut module = Module::default();
        let ty_i32 = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::I32),
            },
            NagaSpan::UNDEFINED,
        );
        let ty_f32 = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            NagaSpan::UNDEFINED,
        );
        let ty_bool = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::BOOL),
            },
            NagaSpan::UNDEFINED,
        );
        let ty_vec3_u32 = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            NagaSpan::UNDEFINED,
        );
        let ty_arr_f32 = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: ty_f32,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            NagaSpan::UNDEFINED,
        );
        Self {
            graph,
            cfg,
            types,
            kernel_blocks,
            inner_loops,
            module,
            function: Function::default(),
            ty_i32,
            ty_f32,
            ty_bool,
            ty_vec3_u32,
            ty_arr_f32,
            values: FxHashMap::default(),
            buffers: FxHashMap::default(),
            uniform_global: None,
            uniform_fields: FxHashMap::default(),
            param_locals: FxHashMap::default(),
        }
    }

    fn naga_ty(&self, ty: ScalarTy) -> Handle<Type> {
        match ty {
            ScalarTy::I32 => self.ty_i32,
            ScalarTy::F32 => self.ty_f32,
            ScalarTy::Bool => self.ty_bool,
        }
    }

    fn declare_inputs(&mut self, buffers: &[BufferInfo], uniforms: &[(Value, usize)]) {
        let mut binding = 0u32;
        for buf in buffers {
            let access = if buf.writable {
                StorageAccess::LOAD | StorageAccess::STORE
            } else {
                StorageAccess::LOAD
            };
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("arg{}", buf.arg_index)),
                    space: AddressSpace::Storage { access },
                    binding: Some(ResourceBinding { group: 0, binding }),
                    ty: self.ty_arr_f32,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                NagaSpan::UNDEFINED,
            );
            self.buffers.insert(buf.arg, global);
            binding += 1;
        }
        if !uniforms.is_empty() {
            let mut members = Vec::new();
            for (field_idx, &(val, arg_index)) in uniforms.iter().enumerate() {
                let ty = self.types.get(&val).copied().unwrap_or(ScalarTy::I32);
                members.push(StructMember {
                    name: Some(format!("arg{arg_index}")),
                    ty: self.naga_ty(ty),
                    binding: None,
                    offset: (field_idx as u32) * 4,
                });
                self.uniform_fields.insert(val, (field_idx as u32, ty));
            }
            let span = (uniforms.len() as u32 * 4).div_ceil(16) * 16;
            let struct_ty = self.module.types.insert(
                Type {
                    name: Some("Params".into()),
                    inner: TypeInner::Struct { members, span },
                },
                NagaSpan::UNDEFINED,
            );
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some("params".into()),
                    space: AddressSpace::Uniform,
                    binding: Some(ResourceBinding { group: 0, binding }),
                    ty: struct_ty,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                NagaSpan::UNDEFINED,
            );
            self.uniform_global = Some(global);
        }
    }

    fn emit_pre(&mut self, expr: Expression) -> Handle<Expression> {
        self.function.expressions.append(expr, NagaSpan::UNDEFINED)
    }

    fn emit(&mut self, block: &mut NagaBlock, expr: Expression) -> Handle<Expression> {
        let start = self.function.expressions.len();
        let handle = self.function.expressions.append(expr, NagaSpan::UNDEFINED);
        let range = self.function.expressions.range_from(start);
        block.push(Statement::Emit(range), NagaSpan::UNDEFINED);
        handle
    }

    fn coerce(
        &mut self,
        block: &mut NagaBlock,
        (expr, ty): (Handle<Expression>, ScalarTy),
        target: ScalarTy,
    ) -> Handle<Expression> {
        if ty == target {
            return expr;
        }
        let kind = match target {
            ScalarTy::I32 => ScalarKind::Sint,
            ScalarTy::F32 => ScalarKind::Float,
            ScalarTy::Bool => ScalarKind::Bool,
        };
        let convert = if target == ScalarTy::Bool {
            None
        } else {
            Some(4)
        };
        self.emit(block, Expression::As { expr, kind, convert })
    }

    fn get_or_emit_value(
        &mut self,
        block: &mut NagaBlock,
        value: Value,
    ) -> Result<(Handle<Expression>, ScalarTy), String> {
        if let Some(&existing) = self.values.get(&value) {
            return Ok(existing);
        }
        if let Some(&(field, ty)) = self.uniform_fields.get(&value) {
            let global = self.uniform_global.unwrap();
            let base = self.emit_pre(Expression::GlobalVariable(global));
            let ptr = self.emit(block, Expression::AccessIndex { base, index: field });
            let loaded = self.emit(block, Expression::Load { pointer: ptr });
            self.values.insert(value, (loaded, ty));
            return Ok((loaded, ty));
        }
        let op = self
            .graph
            .defining_op(value)
            .ok_or_else(|| format!("undefined value {value}"))?;
        let pair = self.emit_op(block, value, op)?;
        self.values.insert(value, pair);
        Ok(pair)
    }

    fn emit_op(
        &mut self,
        block: &mut NagaBlock,
        value: Value,
        op: &Op,
    ) -> Result<(Handle<Expression>, ScalarTy), String> {
        match op {
            Op::Javascript(JsOp::Constant(Constant::Number(n))) => {
                let ty = self.types.get(&value).copied().unwrap_or(ScalarTy::F32);
                let lit = match ty {
                    ScalarTy::I32 => Literal::I32(*n as i32),
                    _ => Literal::F32(*n as f32),
                };
                Ok((self.emit_pre(Expression::Literal(lit)), ty))
            }
            Op::Javascript(JsOp::Constant(Constant::Boolean(b))) => {
                Ok((self.emit_pre(Expression::Literal(Literal::Bool(*b))), ScalarTy::Bool))
            }
            Op::Javascript(JsOp::Unary(UnaryOp::Increment, v)) => {
                let (expr, ty) = self.get_or_emit_value(block, *v)?;
                let one = match ty {
                    ScalarTy::I32 => self.emit_pre(Expression::Literal(Literal::I32(1))),
                    _ => self.emit_pre(Expression::Literal(Literal::F32(1.0))),
                };
                let res = self.emit(
                    block,
                    Expression::Binary {
                        op: NagaBinOp::Add,
                        left: expr,
                        right: one,
                    },
                );
                Ok((res, ty))
            }
            Op::Javascript(JsOp::Unary(UnaryOp::Decrement, v)) => {
                let (expr, ty) = self.get_or_emit_value(block, *v)?;
                let one = match ty {
                    ScalarTy::I32 => self.emit_pre(Expression::Literal(Literal::I32(1))),
                    _ => self.emit_pre(Expression::Literal(Literal::F32(1.0))),
                };
                let res = self.emit(
                    block,
                    Expression::Binary {
                        op: NagaBinOp::Subtract,
                        left: expr,
                        right: one,
                    },
                );
                Ok((res, ty))
            }
            Op::Javascript(JsOp::Unary(UnaryOp::Negate, v)) => {
                let (expr, ty) = self.get_or_emit_value(block, *v)?;
                let res = self.emit(
                    block,
                    Expression::Unary {
                        op: NagaUnaryOp::Negate,
                        expr,
                    },
                );
                Ok((res, ty))
            }
            Op::Javascript(JsOp::Unary(UnaryOp::Not, v)) => {
                let pair = self.get_or_emit_value(block, *v)?;
                let expr = self.coerce(block, pair, ScalarTy::Bool);
                let res = self.emit(
                    block,
                    Expression::Unary {
                        op: NagaUnaryOp::LogicalNot,
                        expr,
                    },
                );
                Ok((res, ScalarTy::Bool))
            }
            Op::Javascript(JsOp::Binary(bin, l, r)) => {
                let lp = self.get_or_emit_value(block, *l)?;
                let rp = self.get_or_emit_value(block, *r)?;
                let common = if lp.1 == ScalarTy::F32 || rp.1 == ScalarTy::F32 {
                    ScalarTy::F32
                } else {
                    ScalarTy::I32
                };
                let left = self.coerce(block, lp, common);
                let right = self.coerce(block, rp, common);
                let (naga_op, out_ty) = match bin {
                    BinaryOperator::Addition => (NagaBinOp::Add, common),
                    BinaryOperator::Subtraction => (NagaBinOp::Subtract, common),
                    BinaryOperator::Multiplication => (NagaBinOp::Multiply, common),
                    BinaryOperator::Division => (NagaBinOp::Divide, common),
                    BinaryOperator::Remainder => (NagaBinOp::Modulo, common),
                    BinaryOperator::LessThan => (NagaBinOp::Less, ScalarTy::Bool),
                    BinaryOperator::LessEqualThan => (NagaBinOp::LessEqual, ScalarTy::Bool),
                    BinaryOperator::GreaterThan => (NagaBinOp::Greater, ScalarTy::Bool),
                    BinaryOperator::GreaterEqualThan => (NagaBinOp::GreaterEqual, ScalarTy::Bool),
                    BinaryOperator::StrictEquality | BinaryOperator::Equality => {
                        (NagaBinOp::Equal, ScalarTy::Bool)
                    }
                    BinaryOperator::StrictInequality | BinaryOperator::Inequality => {
                        (NagaBinOp::NotEqual, ScalarTy::Bool)
                    }
                    other => return Err(format!("unsupported binary op {other:?}")),
                };
                let res = self.emit(
                    block,
                    Expression::Binary {
                        op: naga_op,
                        left,
                        right,
                    },
                );
                Ok((res, out_ty))
            }
            Op::LowLevel(low_level::Op::Binary { op, left, right }) => {
                let lp = self.get_or_emit_value(block, *left)?;
                let rp = self.get_or_emit_value(block, *right)?;
                let (naga_op, ty) = match op {
                    low_level::Binary::Add => (NagaBinOp::Add, ScalarTy::I32),
                    low_level::Binary::Subtract => (NagaBinOp::Subtract, ScalarTy::I32),
                    low_level::Binary::Multiply => (NagaBinOp::Multiply, ScalarTy::I32),
                    low_level::Binary::FloatAdd => (NagaBinOp::Add, ScalarTy::F32),
                    low_level::Binary::FloatSubtract => (NagaBinOp::Subtract, ScalarTy::F32),
                    low_level::Binary::FloatMultiply => (NagaBinOp::Multiply, ScalarTy::F32),
                    low_level::Binary::FloatDivide => (NagaBinOp::Divide, ScalarTy::F32),
                    other => return Err(format!("unsupported low-level binary op {other:?}")),
                };
                let l = self.coerce(block, lp, ty);
                let r = self.coerce(block, rp, ty);
                let res = self.emit(
                    block,
                    Expression::Binary {
                        op: naga_op,
                        left: l,
                        right: r,
                    },
                );
                Ok((res, ty))
            }
            Op::Javascript(JsOp::GetIndex(recv, idx) | JsOp::TypedIndex(recv, idx, _)) => {
                let &global = self
                    .buffers
                    .get(recv)
                    .ok_or_else(|| format!("non-buffer receiver {recv}"))?;
                let idx_pair = self.get_or_emit_value(block, *idx)?;
                let index = self.coerce(block, idx_pair, ScalarTy::I32);
                let base = self.emit_pre(Expression::GlobalVariable(global));
                let ptr = self.emit(block, Expression::Access { base, index });
                let loaded = self.emit(block, Expression::Load { pointer: ptr });
                Ok((loaded, ScalarTy::F32))
            }
            Op::Javascript(JsOp::NumericFloat(v) | JsOp::AsNumber(v)) => {
                let pair = self.get_or_emit_value(block, *v)?;
                let expr = self.coerce(block, pair, ScalarTy::F32);
                Ok((expr, ScalarTy::F32))
            }
            Op::Javascript(JsOp::AsInteger(v)) => {
                let pair = self.get_or_emit_value(block, *v)?;
                let expr = self.coerce(block, pair, ScalarTy::I32);
                Ok((expr, ScalarTy::I32))
            }
            other => Err(format!("unsupported op {other:?}")),
        }
    }

    fn param_local(&mut self, block: Block, index: usize, value: Value) -> (Handle<LocalVariable>, ScalarTy) {
        if let Some(&pair) = self.param_locals.get(&(block, index)) {
            return pair;
        }
        let ty = self.types.get(&value).copied().unwrap_or(ScalarTy::I32);
        let local = self.function.local_variables.append(
            LocalVariable {
                name: Some(format!("p_{}_{}", block, index)),
                ty: self.naga_ty(ty),
                init: None,
            },
            NagaSpan::UNDEFINED,
        );
        self.param_locals.insert((block, index), (local, ty));
        (local, ty)
    }

    fn store_target_args(&mut self, block: &mut NagaBlock, target: &Target) -> Result<(), String> {
        let params = &self.graph.blocks[target.block].params;
        for (index, (&param, &arg)) in params.iter().zip(target.args.iter()).enumerate() {
            let (local, ty) = self.param_local(target.block, index, param);
            let pair = self.get_or_emit_value(block, arg)?;
            let value = self.coerce(block, pair, ty);
            let pointer = self.emit_pre(Expression::LocalVariable(local));
            block.push(Statement::Store { pointer, value }, NagaSpan::UNDEFINED);
        }
        Ok(())
    }

    fn load_block_params(&mut self, block: &mut NagaBlock, bb: Block) {
        let params = self.graph.blocks[bb].params.clone();
        for (index, param) in params.into_iter().enumerate() {
            let (local, ty) = self.param_local(bb, index, param);
            let pointer = self.emit_pre(Expression::LocalVariable(local));
            let loaded = self.emit(block, Expression::Load { pointer });
            self.values.insert(param, (loaded, ty));
        }
    }

    fn emit_block_insts(&mut self, block: &mut NagaBlock, bb: Block) -> Result<(), String> {
        for &inst in &self.graph.blocks[bb].insts {
            let data = &self.graph.insts[inst];
            match &data.op {
                Op::Observe(_, _, _) | Op::ObserveMethod(_, _, _) | Op::ReadHint(_, _, _, _) => {}
                Op::Javascript(JsOp::SetIndex(recv, idx, val)) => {
                    let &global = self
                        .buffers
                        .get(recv)
                        .ok_or_else(|| format!("non-buffer store receiver {recv}"))?;
                    let idx_pair = self.get_or_emit_value(block, *idx)?;
                    let index = self.coerce(block, idx_pair, ScalarTy::I32);
                    let val_pair = self.get_or_emit_value(block, *val)?;
                    let value = self.coerce(block, val_pair, ScalarTy::F32);
                    let base = self.emit_pre(Expression::GlobalVariable(global));
                    let unsigned = self.emit(
                        block,
                        Expression::As {
                            expr: index,
                            kind: ScalarKind::Uint,
                            convert: Some(4),
                        },
                    );
                    let length = self.emit(block, Expression::ArrayLength(base));
                    let condition = self.emit(
                        block,
                        Expression::Binary {
                            op: NagaBinOp::Less,
                            left: unsigned,
                            right: length,
                        },
                    );
                    let mut accept = NagaBlock::new();
                    let pointer = self.emit(&mut accept, Expression::Access { base, index });
                    accept.push(Statement::Store { pointer, value }, NagaSpan::UNDEFINED);
                    block.push(
                        Statement::If {
                            condition,
                            accept,
                            reject: NagaBlock::new(),
                        },
                        NagaSpan::UNDEFINED,
                    );
                }
                op => {
                    if let Some(result) = data.result() {
                        let pair = self.emit_op(block, result, op)?;
                        self.values.insert(result, pair);
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_region(
        &mut self,
        start: Block,
        stop: Block,
        active_loop: Option<Block>,
        stmts: &mut NagaBlock,
    ) -> Result<(), String> {
        let mut cur = start;
        while cur != stop {
            if !self.kernel_blocks.contains(&cur) {
                return Err(format!("control flow leaves the kernel at {cur}"));
            }
            if Some(cur) != active_loop
                && let Some(loop_body) = self.inner_loops.get(&cur)
            {
                let mut loop_stmts = NagaBlock::new();
                self.load_block_params(&mut loop_stmts, cur);
                self.emit_block_insts(&mut loop_stmts, cur)?;
                let TerminatorKind::Branch(cond, then_t, else_t) = self.graph.blocks[cur]
                    .terminator
                    .as_ref()
                    .ok_or_else(|| format!("missing terminator in {cur}"))?
                    .kind
                    .clone()
                else {
                    return Err(format!("inner loop header {cur} must branch"));
                };
                let cond_pair = self.get_or_emit_value(&mut loop_stmts, cond)?;
                let cond_expr = self.coerce(&mut loop_stmts, cond_pair, ScalarTy::Bool);
                let (inside_target, exit_target, break_on_true) =
                    if loop_body.contains(&then_t.block) && !loop_body.contains(&else_t.block) {
                        (then_t, else_t, false)
                    } else if loop_body.contains(&else_t.block) && !loop_body.contains(&then_t.block)
                    {
                        (else_t, then_t, true)
                    } else {
                        return Err(format!("unsupported inner loop exit shape at {cur}"));
                    };
                let break_cond = if break_on_true {
                    cond_expr
                } else {
                    self.emit(
                        &mut loop_stmts,
                        Expression::Unary {
                            op: NagaUnaryOp::LogicalNot,
                            expr: cond_expr,
                        },
                    )
                };
                let mut break_block = NagaBlock::new();
                self.store_target_args(&mut break_block, &exit_target)?;
                break_block.push(Statement::Break, NagaSpan::UNDEFINED);
                loop_stmts.push(
                    Statement::If {
                        condition: break_cond,
                        accept: break_block,
                        reject: NagaBlock::new(),
                    },
                    NagaSpan::UNDEFINED,
                );
                self.store_target_args(&mut loop_stmts, &inside_target)?;
                if !self.graph.blocks[inside_target.block].params.is_empty() {
                    self.load_block_params(&mut loop_stmts, inside_target.block);
                }
                self.emit_region(inside_target.block, cur, Some(cur), &mut loop_stmts)?;
                stmts.push(
                    Statement::Loop {
                        body: loop_stmts,
                        continuing: NagaBlock::new(),
                        break_if: None,
                    },
                    NagaSpan::UNDEFINED,
                );
                self.load_block_params(stmts, cur);
                if !self.graph.blocks[exit_target.block].params.is_empty() {
                    self.load_block_params(stmts, exit_target.block);
                }
                cur = exit_target.block;
                continue;
            }
            self.emit_block_insts(stmts, cur)?;
            let term = self.graph.blocks[cur]
                .terminator
                .as_ref()
                .ok_or_else(|| format!("missing terminator in {cur}"))?
                .kind
                .clone();
            match term {
                TerminatorKind::Jump(target) => {
                    self.store_target_args(stmts, &target)?;
                    if target.block == stop {
                        break;
                    }
                    if !self.graph.blocks[target.block].params.is_empty()
                        && !self.inner_loops.contains_key(&target.block)
                    {
                        self.load_block_params(stmts, target.block);
                    }
                    cur = target.block;
                }
                TerminatorKind::Branch(cond, then_t, else_t) => {
                    let merge = find_merge(self.graph, self.cfg, then_t.block, else_t.block, stop)
                        .ok_or_else(|| format!("cannot structurize branch at {cur}"))?;
                    let cond_pair = self.get_or_emit_value(stmts, cond)?;
                    let cond_expr = self.coerce(stmts, cond_pair, ScalarTy::Bool);
                    let mut accept = NagaBlock::new();
                    self.store_target_args(&mut accept, &then_t)?;
                    if then_t.block != merge {
                        self.emit_region(then_t.block, merge, active_loop, &mut accept)?;
                    }
                    let mut reject = NagaBlock::new();
                    self.store_target_args(&mut reject, &else_t)?;
                    if else_t.block != merge {
                        self.emit_region(else_t.block, merge, active_loop, &mut reject)?;
                    }
                    stmts.push(
                        Statement::If {
                            condition: cond_expr,
                            accept,
                            reject,
                        },
                        NagaSpan::UNDEFINED,
                    );
                    if merge == stop {
                        break;
                    }
                    if !self.graph.blocks[merge].params.is_empty() {
                        self.load_block_params(stmts, merge);
                    }
                    cur = merge;
                }
                _ => return Err(format!("unsupported terminator in {cur}")),
            }
        }
        Ok(())
    }

    fn emit_entry_point(
        &mut self,
        outer_y: &CountedLoop,
        outer_x: Option<&CountedLoop>,
        kernel_entry: Block,
        kernel_latch: Block,
    ) -> Result<(), String> {
        self.function.name = Some("main".into());
        self.function.arguments.push(FunctionArgument {
            name: Some("gid".into()),
            ty: self.ty_vec3_u32,
            binding: Some(NagaBinding::BuiltIn(BuiltIn::GlobalInvocationId)),
        });
        let mut body = NagaBlock::new();
        let uniform_values: Vec<Value> = self.uniform_fields.keys().copied().collect();
        for value in uniform_values {
            self.get_or_emit_value(&mut body, value)?;
        }
        let gid = self.emit_pre(Expression::FunctionArgument(0));
        let (workgroup_size, guard_cond) = if let Some(x_loop) = outer_x {
            let gid_x = self.emit(&mut body, Expression::AccessIndex { base: gid, index: 0 });
            let gid_y = self.emit(&mut body, Expression::AccessIndex { base: gid, index: 1 });
            let gid_x_i32 = self.emit(
                &mut body,
                Expression::As {
                    expr: gid_x,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                },
            );
            let gid_y_i32 = self.emit(
                &mut body,
                Expression::As {
                    expr: gid_y,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                },
            );
            let sx = self.get_or_emit_value(&mut body, x_loop.start)?;
            let sx = self.coerce(&mut body, sx, ScalarTy::I32);
            let sy = self.get_or_emit_value(&mut body, outer_y.start)?;
            let sy = self.coerce(&mut body, sy, ScalarTy::I32);
            let x_val = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::Add,
                    left: gid_x_i32,
                    right: sx,
                },
            );
            let y_val = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::Add,
                    left: gid_y_i32,
                    right: sy,
                },
            );
            self.values.insert(x_loop.iv, (x_val, ScalarTy::I32));
            self.values.insert(outer_y.iv, (y_val, ScalarTy::I32));
            let ex = self.get_or_emit_value(&mut body, x_loop.end)?;
            let ex = self.coerce(&mut body, ex, ScalarTy::I32);
            let ey = self.get_or_emit_value(&mut body, outer_y.end)?;
            let ey = self.coerce(&mut body, ey, ScalarTy::I32);
            let oob_x = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::GreaterEqual,
                    left: x_val,
                    right: ex,
                },
            );
            let oob_y = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::GreaterEqual,
                    left: y_val,
                    right: ey,
                },
            );
            let oob = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::LogicalOr,
                    left: oob_x,
                    right: oob_y,
                },
            );
            ([16, 16, 1], oob)
        } else {
            let gid_x = self.emit(&mut body, Expression::AccessIndex { base: gid, index: 0 });
            let gid_x_i32 = self.emit(
                &mut body,
                Expression::As {
                    expr: gid_x,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                },
            );
            let sy = self.get_or_emit_value(&mut body, outer_y.start)?;
            let sy = self.coerce(&mut body, sy, ScalarTy::I32);
            let y_val = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::Add,
                    left: gid_x_i32,
                    right: sy,
                },
            );
            self.values.insert(outer_y.iv, (y_val, ScalarTy::I32));
            let ey = self.get_or_emit_value(&mut body, outer_y.end)?;
            let ey = self.coerce(&mut body, ey, ScalarTy::I32);
            let oob = self.emit(
                &mut body,
                Expression::Binary {
                    op: NagaBinOp::GreaterEqual,
                    left: y_val,
                    right: ey,
                },
            );
            ([64, 1, 1], oob)
        };
        let mut ret_block = NagaBlock::new();
        ret_block.push(Statement::Return { value: None }, NagaSpan::UNDEFINED);
        body.push(
            Statement::If {
                condition: guard_cond,
                accept: ret_block,
                reject: NagaBlock::new(),
            },
            NagaSpan::UNDEFINED,
        );
        self.emit_region(kernel_entry, kernel_latch, None, &mut body)?;
        self.emit_block_insts(&mut body, kernel_latch)?;
        self.function.body = body;
        let function = std::mem::take(&mut self.function);
        self.module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        Ok(())
    }

    fn finish(self) -> Module {
        self.module
    }
}

fn find_merge(
    graph: &Graph,
    cfg: &Cfg,
    a: Block,
    b: Block,
    stop: Block,
) -> Option<Block> {
    let mut reach_a = FxHashSet::default();
    let mut stack = vec![a];
    while let Some(bb) = stack.pop() {
        if !reach_a.insert(bb) || bb == stop {
            continue;
        }
        for succ in graph.successors(bb) {
            if !cfg.dominates(succ, bb) {
                stack.push(succ);
            }
        }
    }
    let mut visited = FxHashSet::default();
    let mut queue = std::collections::VecDeque::from([b]);
    while let Some(bb) = queue.pop_front() {
        if reach_a.contains(&bb) {
            return Some(bb);
        }
        if !visited.insert(bb) || bb == stop {
            continue;
        }
        for succ in graph.successors(bb) {
            if !cfg.dominates(succ, bb) {
                queue.push_back(succ);
            }
        }
    }
    None
}
