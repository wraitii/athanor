pub(super) const SP: u8 = 31;
pub(super) const ZR: u8 = 31;
pub(super) const FP: u8 = 29;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Cond {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
}

impl Cond {
    pub(super) fn invert(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Hs => Self::Lo,
            Self::Lo => Self::Hs,
            Self::Mi => Self::Pl,
            Self::Pl => Self::Mi,
            Self::Vs => Self::Vc,
            Self::Vc => Self::Vs,
            Self::Hi => Self::Ls,
            Self::Ls => Self::Hi,
            Self::Ge => Self::Lt,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Width {
    W,
    X,
}

impl Width {
    fn sf(self) -> u32 {
        match self {
            Self::W => 0,
            Self::X => 1 << 31,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Float {
    S,
    D,
}

impl Float {
    fn ftype(self) -> u32 {
        match self {
            Self::S => 0,
            Self::D => 1 << 22,
        }
    }
}

fn rd_rn(rd: u8, rn: u8) -> u32 {
    u32::from(rd) | (u32::from(rn) << 5)
}

fn rd_rn_rm(rd: u8, rn: u8, rm: u8) -> u32 {
    rd_rn(rd, rn) | (u32::from(rm) << 16)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Alu {
    Add,
    Sub,
    Subs,
    And,
    Ands,
    Orr,
    Orn,
    Eor,
}

pub(super) fn alu(op: Alu, width: Width, rd: u8, rn: u8, rm: u8) -> u32 {
    let base = match op {
        Alu::Add => 0x0B00_0000,
        Alu::Sub => 0x4B00_0000,
        Alu::Subs => 0x6B00_0000,
        Alu::And => 0x0A00_0000,
        Alu::Ands => 0x6A00_0000,
        Alu::Orr => 0x2A00_0000,
        Alu::Orn => 0x2A20_0000,
        Alu::Eor => 0x4A00_0000,
    };
    base | width.sf() | rd_rn_rm(rd, rn, rm)
}

pub(super) fn add_imm(width: Width, rd: u8, rn: u8, imm12: u32, shift12: bool) -> u32 {
    debug_assert!(imm12 < 4096);
    0x1100_0000 | width.sf() | (u32::from(shift12) << 22) | (imm12 << 10) | rd_rn(rd, rn)
}

pub(super) fn sub_imm(width: Width, rd: u8, rn: u8, imm12: u32, shift12: bool) -> u32 {
    debug_assert!(imm12 < 4096);
    0x5100_0000 | width.sf() | (u32::from(shift12) << 22) | (imm12 << 10) | rd_rn(rd, rn)
}

pub(super) fn subs_imm(width: Width, rd: u8, rn: u8, imm12: u32, shift12: bool) -> u32 {
    sub_imm(width, rd, rn, imm12, shift12) | (1 << 29)
}

pub(super) fn adds_imm(width: Width, rd: u8, rn: u8, imm12: u32, shift12: bool) -> u32 {
    add_imm(width, rd, rn, imm12, shift12) | (1 << 29)
}

pub(super) fn arith_imm(value: i64) -> Option<(u32, bool)> {
    if (0..4096).contains(&value) {
        return Some((value as u32, false));
    }
    if value & 0xFFF == 0 && (0..4096).contains(&(value >> 12)) {
        return Some(((value >> 12) as u32, true));
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Logical {
    And,
    Ands,
    Orr,
    Eor,
}

fn shifted_mask(value: u64) -> bool {
    value != 0 && {
        let filled = (value - 1) | value;
        filled.wrapping_add(1) & filled == 0
    }
}

fn bitmask(width: Width, value: u64) -> Option<u32> {
    let bits = if width == Width::X { 64 } else { 32 };
    let full = u64::MAX >> (64 - bits);
    if value & !full != 0 || value == 0 || value == full {
        return None;
    }
    let mut size = bits;
    loop {
        size /= 2;
        let mask = (1u64 << size) - 1;
        if value & mask != (value >> size) & mask {
            size *= 2;
            break;
        }
        if size <= 2 {
            break;
        }
    }
    let mask = u64::MAX >> (64 - size);
    let mut element = value & mask;
    let (rotation, ones) = if shifted_mask(element) {
        let rotation = element.trailing_zeros();
        (rotation, (element >> rotation).trailing_ones())
    } else {
        element |= !mask;
        if !shifted_mask(!element) {
            return None;
        }
        let leading = element.leading_ones();
        (
            64 - leading,
            leading + element.trailing_ones() - (64 - size),
        )
    };
    let immr = (size - rotation) & (size - 1);
    let nimms = ((!(u64::from(size) - 1)) << 1) | u64::from(ones - 1);
    let n = ((nimms >> 6) & 1) ^ 1;
    Some(((n as u32) << 22) | (immr << 16) | (((nimms & 0x3F) as u32) << 10))
}

pub(super) fn logical_imm(op: Logical, width: Width, rd: u8, rn: u8, value: u64) -> Option<u32> {
    let base = match op {
        Logical::And => 0x1200_0000,
        Logical::Ands => 0x7200_0000,
        Logical::Orr => 0x3200_0000,
        Logical::Eor => 0x5200_0000,
    };
    Some(base | width.sf() | bitmask(width, value)? | rd_rn(rd, rn))
}

fn bitfield(base: u32, width: Width, rd: u8, rn: u8, immr: u32, imms: u32) -> u32 {
    let n = if width == Width::X { 1 << 22 } else { 0 };
    base | width.sf() | n | (immr << 16) | (imms << 10) | rd_rn(rd, rn)
}

pub(super) fn extract_bits(width: Width, rd: u8, rn: u8, shift: u8, bits: u8) -> u32 {
    bitfield(
        0x5300_0000,
        width,
        rd,
        rn,
        u32::from(shift),
        u32::from(shift) + u32::from(bits) - 1,
    )
}

pub(super) fn shift_imm(op: Shift, width: Width, rd: u8, rn: u8, amount: u32) -> u32 {
    let bits = if width == Width::X { 64 } else { 32 };
    let amount = amount & (bits - 1);
    match op {
        Shift::Lsl => bitfield(
            0x5300_0000,
            width,
            rd,
            rn,
            (bits - amount) & (bits - 1),
            bits - 1 - amount,
        ),
        Shift::Lsr => bitfield(0x5300_0000, width, rd, rn, amount, bits - 1),
        Shift::Asr => bitfield(0x1300_0000, width, rd, rn, amount, bits - 1),
    }
}

pub(super) fn compare_branch(nonzero: bool, width: Width, rt: u8) -> u32 {
    let base = if nonzero { 0x3500_0000 } else { 0x3400_0000 };
    base | width.sf() | u32::from(rt)
}

pub(super) fn mov(width: Width, rd: u8, rm: u8) -> u32 {
    alu(Alu::Orr, width, rd, ZR, rm)
}

pub(super) fn mov_sp(rd: u8, rn: u8) -> u32 {
    add_imm(Width::X, rd, rn, 0, false)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Wide {
    Movz,
    Movn,
    Movk,
}

pub(super) fn mov_wide(op: Wide, width: Width, rd: u8, imm16: u16, hw: u8) -> u32 {
    let base = match op {
        Wide::Movn => 0x1280_0000,
        Wide::Movz => 0x5280_0000,
        Wide::Movk => 0x7280_0000,
    };
    base | width.sf() | (u32::from(hw) << 21) | (u32::from(imm16) << 5) | u32::from(rd)
}

pub(super) fn mov_imm(width: Width, rd: u8, value: u64, mut emit: impl FnMut(u32)) {
    let count = if width == Width::X { 4 } else { 2 };
    let all: [u16; 4] = std::array::from_fn(|i| (value >> (16 * i)) as u16);
    let chunks = &all[..count];
    let ones = chunks.iter().filter(|&&chunk| chunk == 0xFFFF).count();
    let zeros = chunks.iter().filter(|&&chunk| chunk == 0).count();
    let (skip, first) = if ones > zeros {
        (0xFFFF, Wide::Movn)
    } else {
        (0, Wide::Movz)
    };
    let mut started = false;
    for (hw, &chunk) in chunks.iter().enumerate() {
        if chunk == skip {
            continue;
        }
        let word = if started {
            mov_wide(Wide::Movk, width, rd, chunk, hw as u8)
        } else if first == Wide::Movn {
            mov_wide(first, width, rd, !chunk, hw as u8)
        } else {
            mov_wide(first, width, rd, chunk, hw as u8)
        };
        emit(word);
        started = true;
    }
    if !started {
        emit(mov_wide(first, width, rd, 0, 0));
    }
}

pub(super) fn load_imm(rd: u8, value: u64, emit: impl FnMut(u32)) {
    let width = if value >> 32 == 0 { Width::W } else { Width::X };
    mov_imm(width, rd, value, emit);
}

pub(super) fn mul(width: Width, rd: u8, rn: u8, rm: u8) -> u32 {
    0x1B00_7C00 | width.sf() | rd_rn_rm(rd, rn, rm)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Shift {
    Lsl,
    Lsr,
    Asr,
}

pub(super) fn shift(op: Shift, width: Width, rd: u8, rn: u8, rm: u8) -> u32 {
    let base = match op {
        Shift::Lsl => 0x1AC0_2000,
        Shift::Lsr => 0x1AC0_2400,
        Shift::Asr => 0x1AC0_2800,
    };
    base | width.sf() | rd_rn_rm(rd, rn, rm)
}

pub(super) fn csel(width: Width, rd: u8, rn: u8, rm: u8, cond: Cond) -> u32 {
    0x1A80_0000 | width.sf() | rd_rn_rm(rd, rn, rm) | ((cond as u32) << 12)
}

pub(super) fn cset(rd: u8, cond: Cond) -> u32 {
    0x1A80_0400 | rd_rn_rm(rd, ZR, ZR) | ((cond.invert() as u32) << 12)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Sxtb,
    Sxth,
    Sxtw,
}

pub(super) fn extend(op: Extend, width: Width, rd: u8, rn: u8) -> u32 {
    let base = match (op, width) {
        (Extend::Uxtb, _) => 0x5300_1C00,
        (Extend::Uxth, _) => 0x5300_3C00,
        (Extend::Uxtw, _) => return mov(Width::W, rd, rn),
        (Extend::Sxtb, Width::W) => 0x1300_1C00,
        (Extend::Sxth, Width::W) => 0x1300_3C00,
        (Extend::Sxtb, Width::X) => 0x9340_1C00,
        (Extend::Sxth, Width::X) => 0x9340_3C00,
        (Extend::Sxtw, _) => 0x9340_7C00,
    };
    base | rd_rn(rd, rn)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Mem {
    Ldrb,
    Strb,
    Ldrh,
    Strh,
    LdrW,
    StrW,
    LdrX,
    StrX,
    LdrS,
    StrS,
    LdrD,
    StrD,
}

impl Mem {
    pub(super) fn size(self) -> u32 {
        match self {
            Self::Ldrb | Self::Strb => 1,
            Self::Ldrh | Self::Strh => 2,
            Self::LdrW | Self::StrW | Self::LdrS | Self::StrS => 4,
            Self::LdrX | Self::StrX | Self::LdrD | Self::StrD => 8,
        }
    }

    fn scaled(self) -> u32 {
        match self {
            Self::Ldrb => 0x3940_0000,
            Self::Strb => 0x3900_0000,
            Self::Ldrh => 0x7940_0000,
            Self::Strh => 0x7900_0000,
            Self::LdrW => 0xB940_0000,
            Self::StrW => 0xB900_0000,
            Self::LdrX => 0xF940_0000,
            Self::StrX => 0xF900_0000,
            Self::LdrS => 0xBD40_0000,
            Self::StrS => 0xBD00_0000,
            Self::LdrD => 0xFD40_0000,
            Self::StrD => 0xFD00_0000,
        }
    }
}

pub(super) fn mem_scaled(op: Mem, rt: u8, rn: u8, offset: u32) -> Option<u32> {
    let size = op.size();
    if offset % size != 0 || offset / size >= 4096 {
        return None;
    }
    Some(op.scaled() | ((offset / size) << 10) | rd_rn(rt, rn))
}

pub(super) fn mem_unscaled(op: Mem, rt: u8, rn: u8, offset: i32) -> Option<u32> {
    if !(-256..256).contains(&offset) {
        return None;
    }
    let base = op.scaled() & !0x0100_0000;
    Some(base | (((offset as u32) & 0x1FF) << 12) | rd_rn(rt, rn))
}

pub(super) fn mem_register(op: Mem, rt: u8, rn: u8, rm: u8) -> u32 {
    mem_indexed(op, rt, rn, rm, Index::Lsl, false)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Index {
    Lsl,
    Uxtw,
    Sxtw,
}

impl Index {
    fn option(self) -> u32 {
        match self {
            Self::Lsl => 0b011,
            Self::Uxtw => 0b010,
            Self::Sxtw => 0b110,
        }
    }
}

pub(super) fn mem_indexed(op: Mem, rt: u8, rn: u8, rm: u8, index: Index, scaled: bool) -> u32 {
    let base = op.scaled() & !0x0100_0000;
    base | 0x0020_0800 | (index.option() << 13) | (u32::from(scaled) << 12) | rd_rn_rm(rt, rn, rm)
}

pub(super) fn alu_shifted(sub: bool, width: Width, rd: u8, rn: u8, rm: u8, amount: u32) -> u32 {
    let op = if sub { Alu::Sub } else { Alu::Add };
    alu(op, width, rd, rn, rm) | (amount << 10)
}

pub(super) fn alu_extended(sub: bool, rd: u8, rn: u8, rm: u8, index: Index, amount: u32) -> u32 {
    let base = if sub { 0xCB20_0000 } else { 0x8B20_0000 };
    base | (index.option() << 13) | (amount << 10) | rd_rn_rm(rd, rn, rm)
}

pub(super) fn load_acquire(size: u32, rt: u8, rn: u8) -> u32 {
    let base = match size {
        1 => 0x08DF_FC00,
        2 => 0x48DF_FC00,
        4 => 0x88DF_FC00,
        _ => 0xC8DF_FC00,
    };
    base | rd_rn(rt, rn)
}

pub(super) fn pair(load: bool, float: bool, rt: u8, rt2: u8, rn: u8, offset: u32) -> Option<u32> {
    if offset % 8 != 0 || offset / 8 > 63 {
        return None;
    }
    let base = match (float, load) {
        (false, false) => 0xA900_0000,
        (false, true) => 0xA940_0000,
        (true, false) => 0x6D00_0000,
        (true, true) => 0x6D40_0000,
    };
    Some(
        base | ((offset / 8) << 15) | (u32::from(rt2) << 10) | (u32::from(rn) << 5) | u32::from(rt),
    )
}

pub(super) const STP_FP_LR_PRE: u32 = 0xA9BF_7BFD;
pub(super) const LDP_FP_LR_POST: u32 = 0xA8C1_7BFD;
pub(super) const RET: u32 = 0xD65F_03C0;

pub(super) fn blr(rn: u8) -> u32 {
    0xD63F_0000 | (u32::from(rn) << 5)
}

pub(super) fn br(rn: u8) -> u32 {
    0xD61F_0000 | (u32::from(rn) << 5)
}

pub(super) const B: u32 = 0x1400_0000;
pub(super) const BL: u32 = 0x9400_0000;

pub(super) fn b_cond(cond: Cond) -> u32 {
    0x5400_0000 | cond as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

pub(super) fn float_binary(op: FloatOp, ty: Float, rd: u8, rn: u8, rm: u8) -> u32 {
    let base = match op {
        FloatOp::Mul => 0x1E20_0800,
        FloatOp::Div => 0x1E20_1800,
        FloatOp::Add => 0x1E20_2800,
        FloatOp::Sub => 0x1E20_3800,
        FloatOp::Max => 0x1E20_4800,
        FloatOp::Min => 0x1E20_5800,
    };
    base | ty.ftype() | rd_rn_rm(rd, rn, rm)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum FloatUnary {
    Mov,
    Abs,
    Neg,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
}

pub(super) fn float_unary(op: FloatUnary, ty: Float, rd: u8, rn: u8) -> u32 {
    let base = match op {
        FloatUnary::Mov => 0x1E20_4000,
        FloatUnary::Abs => 0x1E20_C000,
        FloatUnary::Neg => 0x1E21_4000,
        FloatUnary::Sqrt => 0x1E21_C000,
        FloatUnary::Ceil => 0x1E24_C000,
        FloatUnary::Floor => 0x1E25_4000,
        FloatUnary::Trunc => 0x1E25_C000,
    };
    base | ty.ftype() | rd_rn(rd, rn)
}

pub(super) fn fcsel(ty: Float, rd: u8, rn: u8, rm: u8, cond: Cond) -> u32 {
    0x1E20_0C00 | ty.ftype() | rd_rn_rm(rd, rn, rm) | ((cond as u32) << 12)
}

pub(super) fn move_to_float(width: Width, rd: u8, rn: u8) -> u32 {
    match width {
        Width::W => 0x1E27_0000 | rd_rn(rd, rn),
        Width::X => 0x9E67_0000 | rd_rn(rd, rn),
    }
}

pub(super) fn move_to_int(width: Width, rd: u8, rn: u8) -> u32 {
    match width {
        Width::W => 0x1E26_0000 | rd_rn(rd, rn),
        Width::X => 0x9E66_0000 | rd_rn(rd, rn),
    }
}

pub(super) fn fcmp(ty: Float, rn: u8, rm: u8) -> u32 {
    0x1E20_2000 | ty.ftype() | rd_rn_rm(0, rn, rm)
}

pub(super) fn int_to_float(signed: bool, width: Width, ty: Float, rd: u8, rn: u8) -> u32 {
    let base = if signed { 0x1E22_0000 } else { 0x1E23_0000 };
    base | width.sf() | ty.ftype() | rd_rn(rd, rn)
}

pub(super) fn float_to_int(signed: bool, width: Width, ty: Float, rd: u8, rn: u8) -> u32 {
    let base = if signed { 0x1E38_0000 } else { 0x1E39_0000 };
    base | width.sf() | ty.ftype() | rd_rn(rd, rn)
}

pub(super) fn demote(rd: u8, rn: u8) -> u32 {
    0x1E62_4000 | rd_rn(rd, rn)
}

pub(super) fn promote(rd: u8, rn: u8) -> u32 {
    0x1E22_C000 | rd_rn(rd, rn)
}

pub(super) fn float_imm8(bits: u64) -> Option<u32> {
    if bits & 0xFFFF_FFFF_FFFF != 0 {
        return None;
    }
    let replicated = match (bits >> 54) & 0xFF {
        0 => 0,
        0xFF => 1,
        _ => return None,
    };
    if (bits >> 62) & 1 == replicated {
        return None;
    }
    let sign = bits >> 63;
    Some(((sign << 7) | (replicated << 6) | ((bits >> 48) & 0x3F)) as u32)
}

pub(super) fn fmov_imm(rd: u8, imm8: u32) -> u32 {
    0x1E60_1000 | (imm8 << 13) | u32::from(rd)
}
