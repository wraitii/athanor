pub(crate) const ELEMENT_FLOAT: u32 = 0b0100;
pub(crate) const ELEMENT_SIGNED: u32 = 0b1000;

pub(crate) const TYPED: u32 = 1 << 8;
pub(crate) const ARRAY: u32 = 1 << 9;
pub(crate) const HOLEY: u32 = 1 << 10;
pub(crate) const OBJECT: u32 = 1 << 11;
pub(crate) const CLOSURE: u32 = 1 << 12;
pub(crate) const STRING: u32 = 1 << 13;
pub(crate) const CELL: u32 = 1 << 14;
pub(crate) const NATIVE: u32 = 1 << 15;
pub(crate) const GENERATOR: u32 = 1 << 16;
pub(crate) const REGEXP: u32 = 1 << 17;
pub(crate) const ENUMERATOR: u32 = 1 << 18;
pub(crate) const SET: u32 = 1 << 19;
pub(crate) const INLINE: u32 = 1 << 20;
pub(crate) const CONSTRUCTOR: u32 = 1 << 22;

pub(crate) const PROTOTYPE_INDEXED: u32 = 1 << 21;
