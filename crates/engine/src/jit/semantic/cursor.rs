use oxc_span::Span;

use super::low_level::Instructions;
use super::{Block, Graph, Inst, Op as GraphOp, Repr, TerminatorKind, Value};

pub(in crate::jit) trait Insert {
    fn graph(&self) -> &Graph;
    fn graph_mut(&mut self) -> &mut Graph;
    fn place(&mut self, op: GraphOp, has_result: bool) -> (Inst, Option<Value>);
    fn terminate_current(&mut self, kind: TerminatorKind);

    fn add(&mut self, op: GraphOp, repr: Repr) -> Value {
        let value = self.place(op, true).1.unwrap();
        self.graph_mut().reprs[value] = Some(repr);
        value
    }

    fn add_unassigned(&mut self, op: GraphOp) -> Value {
        self.place(op, true).1.unwrap()
    }

    fn add_effect(&mut self, op: GraphOp) -> Inst {
        self.place(op, false).0
    }

    fn as_word(&mut self, value: Value) -> Value {
        self.add(GraphOp::AsWord(value), Repr::Word)
    }

    fn as_boxed(&mut self, word: Value) -> Value {
        self.add(GraphOp::AsBoxed(word), Repr::Boxed)
    }

    fn ins(&mut self) -> Instructions<'_, Self>
    where
        Self: Sized,
    {
        Instructions { inserter: self }
    }
}

pub(in crate::jit) struct Cursor<'g> {
    graph: &'g mut Graph,
    block: Block,
    position: Option<usize>,
    span: Span,
    source_name: Option<(usize, usize)>,
    location: Option<u32>,
    inline_context: u64,
}

impl<'g> Cursor<'g> {
    pub(in crate::jit) fn at_end(graph: &'g mut Graph, block: Block, source: Inst) -> Self {
        let data = &graph.insts[source];
        let (span, source_name, location) = (data.span, data.source_name, data.emission_location);
        let inline_context = data.inline_context;
        Self {
            inline_context,
            graph,
            block,
            position: None,
            span,
            source_name,
            location,
        }
    }

    pub(in crate::jit) fn after(graph: &'g mut Graph, inst: Inst) -> Self {
        let mut cursor = Self::before(graph, inst);
        cursor.position = cursor.position.map(|position| position + 1);
        cursor
    }

    pub(in crate::jit) fn before(graph: &'g mut Graph, inst: Inst) -> Self {
        let data = &graph.insts[inst];
        let (block, span, source_name, location) = (
            data.block,
            data.span,
            data.source_name,
            data.emission_location,
        );
        let position = graph.blocks[block]
            .insts
            .iter()
            .position(|&i| i == inst)
            .unwrap();
        let inline_context = data.inline_context;
        Self {
            inline_context,
            graph,
            block,
            position: Some(position),
            span,
            source_name,
            location,
        }
    }
}

impl Insert for Cursor<'_> {
    fn graph(&self) -> &Graph {
        self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        self.graph
    }

    fn place(&mut self, op: GraphOp, has_result: bool) -> (Inst, Option<Value>) {
        let (inst, value) = match self.position {
            Some(position) => {
                self.position = Some(position + 1);
                self.graph
                    .insert(self.block, position, self.span, op, has_result)
            }
            None => self.graph.append(self.block, self.span, op, has_result),
        };
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        self.graph.insts[inst].inline_context = self.inline_context;
        (inst, value)
    }

    fn terminate_current(&mut self, kind: TerminatorKind) {
        self.graph.terminate(self.block, self.span, kind);
        let terminator = self.graph.blocks[self.block].terminator.as_mut().unwrap();
        terminator.emission_location = self.location;
        terminator.source_name = self.source_name;
    }
}
