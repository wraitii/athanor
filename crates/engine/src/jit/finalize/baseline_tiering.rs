use super::{FunctionCompiler, Helper, IntCC, MemFlagsData, types};

impl FunctionCompiler<'_, '_> {
    pub(super) fn baseline_warmup(&mut self, owner: u64, warmup: u64) {
        let counter = self.data.pin(std::cell::Cell::new(0u64));
        let address = self.b.ins().iconst(self.pointer, counter as i64);
        let flags = MemFlagsData::trusted();
        let count = self.b.ins().load(types::I64, flags, address, 0);
        let cold = self
            .b
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThan, count, warmup as i64);
        let next = self.b.ins().iadd_imm_u(count, 1);
        let count = self.b.ins().select(cold, next, count);
        self.b.ins().store(flags, count, address, 0);
        let warm = self.b.ins().icmp_imm_u(IntCC::Equal, next, warmup as i64);
        let poll = self.b.create_block();
        let done = self.b.create_block();
        self.b.ins().brif(warm, poll, &[], done, &[]);
        self.b.switch_to_block(poll);
        let owner = self.b.ins().iconst(types::I64, owner as i64);
        self.call_helper(Helper::BaselineWarm, &[owner]);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }
}
