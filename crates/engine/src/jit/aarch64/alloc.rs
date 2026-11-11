use crate::jit::alloc::{Registers, Target};

pub(super) const TARGET: Target = Target {
    int: Registers {
        caller_saved: &[12, 13, 14, 15, 8, 0, 1, 7, 6, 5, 4, 3, 2],
        callee_saved: &[19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    },
    float: Registers {
        caller_saved: &[
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 7, 6, 5, 4, 3, 2, 1, 0,
        ],
        callee_saved: &[8, 9, 10, 11, 12, 13, 14, 15],
    },
};
