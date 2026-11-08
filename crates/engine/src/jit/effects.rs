use super::helpers::Helper;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Effects {
    pub(super) reentry: Reentry,
    pub(super) flags: u16,
    pub(super) assumptions: &'static [Assumption],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Assumption {
    NoAccessors,
    NoCoercionCallbacks,
    DirectIteration,
    StableIndexStorage,
    FixedTypedStorage,
    GuardedMathIdentity,
}

impl std::fmt::Display for Assumption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NoAccessors => "property access does not invoke accessors",
            Self::NoCoercionCallbacks => "coercion does not invoke JS callbacks",
            Self::DirectIteration => "iteration and enumeration use direct engine state",
            Self::StableIndexStorage => "direct numeric array reads have stable storage",
            Self::FixedTypedStorage => "typed array storage never moves or resizes",
            Self::GuardedMathIdentity => "native Math identity is checked by the entry guard",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reentry {
    Closed,
    Arbitrary,
    Unknown,
}

pub(super) const HEAP_READ: u16 = 1;
pub(super) const HEAP_WRITE: u16 = 2;
pub(super) const GLOBAL_READ: u16 = 4;
pub(super) const GLOBAL_WRITE: u16 = 8;
pub(super) const INTERNAL: u16 = 16;
pub(super) const ALLOCATION: u16 = 32;
pub(super) const ERROR: u16 = 64;
pub(super) const UNKNOWN: u16 =
    HEAP_READ | HEAP_WRITE | GLOBAL_READ | GLOBAL_WRITE | INTERNAL | ALLOCATION | ERROR;

pub(super) fn effects(helper: Helper) -> Effects {
    use Helper::*;
    let (reentry, flags, assumptions): (_, _, &'static [Assumption]) = match helper {
        Call
        | CallOwned
        | CallPack
        | Construct
        | ConstructPack
        | ConstructResolved
        | ConstructorCall
        | ConstructorCallPack
        | ConstructPackResolved => (Reentry::Arbitrary, UNKNOWN, &[]),
        FrameOverflow | FrameRelease | IndexUnroot | LocalSet | LocalReplace | Release
        | LocalGet | TempClear | TempRetain | TempOwn | ErrorOrigin | Catch | Throw => {
            (Reentry::Closed, INTERNAL | ERROR, &[])
        }
        ObserveCall | ObserveGet | ObserveIndex | ObserveNumber => {
            (Reentry::Closed, HEAP_READ | INTERNAL, &[])
        }
        CompilePoll | CompileLeave | CallProfileFlush => (Reentry::Closed, INTERNAL, &[]),
        SemanticGlobalProbe => (Reentry::Closed, GLOBAL_READ, &[]),
        ConstructorPrototype
        | SemanticPropertyProbeCached
        | SemanticMathNative
        | ClosureThis
        | LexicalReceiver
        | ArrayIterationGuard
        | IsStringSplit => (Reentry::Closed, HEAP_READ, &[]),
        StringSplit
        | StringSplitOwned
        | StringSplitParts
        | StringSplitPartsOwned
        | Spread
        | NewPack => (
            Reentry::Closed,
            HEAP_READ | ALLOCATION | INTERNAL | ERROR,
            &[],
        ),
        PackInit | PackRelease => (Reentry::Closed, HEAP_WRITE | INTERNAL, &[]),
        PackTake | PackTakeOwned => (Reentry::Closed, HEAP_READ | HEAP_WRITE | INTERNAL, &[]),
        PackAppend | PackExtend => (
            Reentry::Closed,
            HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL,
            &[],
        ),
        IndexRoot => (
            Reentry::Closed,
            HEAP_READ | INTERNAL,
            &[Assumption::StableIndexStorage],
        ),
        MathDirect => (
            Reentry::Closed,
            ALLOCATION | INTERNAL | ERROR,
            &[
                Assumption::GuardedMathIdentity,
                Assumption::NoCoercionCallbacks,
            ],
        ),
        Truthy => (Reentry::Closed, HEAP_READ, &[]),
        TypedIndexGet => (Reentry::Closed, HEAP_READ, &[Assumption::FixedTypedStorage]),
        ArrayMapElement => (Reentry::Closed, HEAP_READ | INTERNAL, &[]),
        ArrayNew | ArrayNewOwned | Rest | ObjectNew | ObjectNewOwned | NewInstance
        | NewInstanceOwned | StringNew | ArrayMapNew | ArrayMapNewOwned => {
            (Reentry::Closed, ALLOCATION | INTERNAL | ERROR, &[])
        }
        GlobalGet | GlobalGetOptional | GlobalGetBorrowed | GlobalGetOptionalBorrowed => {
            (Reentry::Closed, GLOBAL_READ | INTERNAL | ERROR, &[])
        }
        GlobalSet | GlobalInitialize => (Reentry::Closed, GLOBAL_WRITE | INTERNAL | ERROR, &[]),
        PropertyGet | PropertyGetCached | IndexGet | IndexGetNumeric => (
            Reentry::Closed,
            HEAP_READ | GLOBAL_READ | ALLOCATION | INTERNAL | ERROR,
            &[Assumption::NoAccessors, Assumption::NoCoercionCallbacks],
        ),
        IterateNew | IterateNext | EnumerateNew | EnumerateNext => (
            Reentry::Closed,
            HEAP_READ | INTERNAL | ALLOCATION | ERROR,
            &[Assumption::DirectIteration],
        ),
        BitwiseAnd | BitwiseOr | BitwiseXor | ShiftLeft | ShiftRight | ShiftRightUnsigned | Add
        | Sub | Mul | Div | Rem | Pow | LooseEqual | LooseNotEqual | StrictEqual
        | StrictNotEqual | Less | LessEqual | Greater | GreaterEqual | UnaryPlus
        | UnaryNegation | LogicalNot | BitwiseNot => (
            Reentry::Closed,
            HEAP_READ | ALLOCATION | INTERNAL | ERROR,
            &[Assumption::NoCoercionCallbacks],
        ),
        _ => (Reentry::Unknown, UNKNOWN, &[]),
    };
    Effects {
        reentry,
        flags,
        assumptions,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TemporaryEffect {
    None,
    Clear,
    MayRegister,
}

pub(super) fn temporary_effect(helper: Helper) -> TemporaryEffect {
    use Helper::*;
    use TemporaryEffect::*;
    match helper {
        TempClear => Clear,
        TempRetain | TempOwn | Add => MayRegister,
        SemanticPropertyProbeCached
        | TypedIndexGet
        | GlobalGetBorrowed
        | GlobalGetOptionalBorrowed
        | NewInstanceOwned
        | StringSplitOwned
        | StringSplitPartsOwned
        | PackTakeOwned
        | PackInit
        | PackRelease
        | PackAppend
        | PackExtend
        | ArrayNewOwned
        | ObjectNewOwned
        | ArrayMapNewOwned => None,
        FrameOverflow | FrameRelease | IndexUnroot | LocalSet | LocalReplace | Release
        | ErrorOrigin | Throw | MathDirect | ObserveCall | ObserveGet | ObserveIndex
        | ObserveNumber | CompilePoll | CompileLeave | CallProfileFlush | ClosureThis
        | LexicalReceiver | ArrayIterationGuard | IsStringSplit | IndexRoot | Truthy
        | BitwiseAnd | BitwiseOr | BitwiseXor | ShiftLeft | ShiftRight | ShiftRightUnsigned
        | Sub | Mul | Div | Rem | Pow | LooseEqual | LooseNotEqual | StrictEqual
        | StrictNotEqual | Less | LessEqual | Greater | GreaterEqual | UnaryPlus
        | UnaryNegation | LogicalNot | BitwiseNot => None,
        _ => MayRegister,
    }
}
