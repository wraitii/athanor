pub(in crate::jit::semantic) mod conversions;
pub(in crate::jit::semantic) mod cse;
mod dce;
pub(in crate::jit::semantic) mod licm;
pub(in crate::jit::semantic) mod loops;
mod parameters;
mod phis;
mod promote;
mod repair;
pub(in crate::jit::semantic) mod reuse;
mod simplify;
mod trim;
pub(in crate::jit::semantic) mod unroll;

pub(super) use promote::{promote, remove_trivial_params};
pub(super) use repair::repair;
pub(super) use simplify::{Change, Fold, Folder, simplify};
pub(super) use trim::trim;

pub(in crate::jit::semantic) use parameters::simplify_parameters;
