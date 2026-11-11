pub(super) const SP: u8 = 31;
pub(super) const ZR: u8 = 31;
pub(super) const FP: u8 = 29;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Cond {
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
pub(super) enum Width {
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
pub(super) enum Float {
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
pub(super) enum Alu {
    Add,
    Sub,
    Subs,
    And,
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
        Alu::Orr => 0x2A00_0000,
        Alu::Orn => 0x2A20_0000,
        Alu::Eor => 0x4A00_0000,
    };
    base | width.sf() | rd_rn_rm(rd, rn, rm)
}

pub(super) fn add_lsl(width: Width, rd: u8, rn: u8, rm: u8, amount: u32) -> u32 {
    debug_assert!(amount < 64);
    alu(Alu::Add, width, rd, rn, rm) | (amount << 10)
}

pub(super) fn add_imm(width: Width, rd: u8, rn: u8, imm12: u32, shift12: bool) -> u32 {
    debug_assert!(imm12 < 4096);
    0x1100_0000 | width.sf() | (u32::from(shift12) << 22) | (imm12 << 10) | rd_rn(rd, rn)
}

pub(super) fn sub_imm(width: Width, rd: u8, rn: u8, imm12: u32, shift12: bool) -> u32 {
    debug_assert!(imm12 < 4096);
    0x5100_0000 | width.sf() | (u32::from(shift12) << 22) | (imm12 << 10) | rd_rn(rd, rn)
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

pub(super) fn mul(width: Width, rd: u8, rn: u8, rm: u8) -> u32 {
    0x1B00_7C00 | width.sf() | rd_rn_rm(rd, rn, rm)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Shift {
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

pub(super) fn tst_low_byte(rn: u8) -> u32 {
    0x7200_1C1F | (u32::from(rn) << 5)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Extend {
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
pub(super) enum Mem {
    Ldrb,
    Ldrsb,
    Strb,
    Ldrh,
    Ldrsh,
    Strh,
    LdrW,
    Ldrsw,
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
            Self::Ldrb | Self::Ldrsb | Self::Strb => 1,
            Self::Ldrh | Self::Ldrsh | Self::Strh => 2,
            Self::LdrW | Self::Ldrsw | Self::StrW | Self::LdrS | Self::StrS => 4,
            Self::LdrX | Self::StrX | Self::LdrD | Self::StrD => 8,
        }
    }

    fn scaled(self) -> u32 {
        match self {
            Self::Ldrb => 0x3940_0000,
            Self::Ldrsb => 0x3980_0000,
            Self::Strb => 0x3900_0000,
            Self::Ldrh => 0x7940_0000,
            Self::Ldrsh => 0x7980_0000,
            Self::Strh => 0x7900_0000,
            Self::LdrW => 0xB940_0000,
            Self::Ldrsw => 0xB980_0000,
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
    let base = op.scaled() & !0x0100_0000;
    base | 0x0020_6800 | rd_rn_rm(rt, rn, rm)
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

pub(super) const STP_FP_LR_PRE: u32 = 0xA9BF_7BFD;
pub(super) const LDP_FP_LR_POST: u32 = 0xA8C1_7BFD;
pub(super) const RET: u32 = 0xD65F_03C0;

pub(super) fn blr(rn: u8) -> u32 {
    0xD63F_0000 | (u32::from(rn) << 5)
}

pub(super) fn br(rn: u8) -> u32 {
    0xD61F_0000 | (u32::from(rn) << 5)
}

pub(super) fn adr(rd: u8) -> u32 {
    0x1000_0000 | u32::from(rd)
}

pub(super) const B: u32 = 0x1400_0000;
pub(super) const BL: u32 = 0x9400_0000;

pub(super) fn b_cond(cond: Cond) -> u32 {
    0x5400_0000 | cond as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FloatOp {
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
pub(super) enum FloatUnary {
    Abs,
    Neg,
    Sqrt,
    Nearest,
    Ceil,
    Floor,
    Trunc,
}

pub(super) fn float_unary(op: FloatUnary, ty: Float, rd: u8, rn: u8) -> u32 {
    let base = match op {
        FloatUnary::Abs => 0x1E20_C000,
        FloatUnary::Neg => 0x1E21_4000,
        FloatUnary::Sqrt => 0x1E21_C000,
        FloatUnary::Nearest => 0x1E24_4000,
        FloatUnary::Ceil => 0x1E24_C000,
        FloatUnary::Floor => 0x1E25_4000,
        FloatUnary::Trunc => 0x1E25_C000,
    };
    base | ty.ftype() | rd_rn(rd, rn)
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
