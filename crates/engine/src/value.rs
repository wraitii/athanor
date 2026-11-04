use std::fmt;

pub(crate) const TAG_SHIFT: u32 = 48;
pub(crate) const TAG_FIRST: u64 = 0x7ff9;
pub(crate) const TAG_LAST: u64 = 0x7fff;
pub(crate) const TAG_SPAN: u64 = TAG_LAST - TAG_FIRST;
const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000;

pub(crate) const TAG_UNDEFINED: u64 = 0x7ff9_0000_0000_0000;
pub(crate) const TAG_NULL: u64 = 0x7ffa_0000_0000_0000;
pub(crate) const TAG_FALSE: u64 = 0x7ffb_0000_0000_0000;
pub(crate) const TAG_TRUE: u64 = 0x7ffc_0000_0000_0000;
pub(crate) const TAG_OBJECT: u64 = 0x7ffd_0000_0000_0000;
pub(crate) const TAG_STRING: u64 = 0x7ffe_0000_0000_0000;
pub(crate) const TAG_UNINITIALIZED: u64 = 0x7fff_0000_0000_0000;

pub(crate) const fn tag_word(bits: u64) -> u64 {
    bits >> TAG_SHIFT
}

#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Value(u64);

impl Value {
    pub(crate) const GLOBAL_OBJECT: Self = Self(TAG_OBJECT);
    pub const UNDEFINED: Self = Self(TAG_UNDEFINED);
    pub const NULL: Self = Self(TAG_NULL);
    pub const FALSE: Self = Self(TAG_FALSE);
    pub const TRUE: Self = Self(TAG_TRUE);
    pub const UNINITIALIZED: Self = Self(TAG_UNINITIALIZED);

    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub fn number(value: f64) -> Self {
        if value.is_nan() {
            Self(CANONICAL_NAN)
        } else {
            Self(value.to_bits())
        }
    }

    pub fn boolean(value: bool) -> Self {
        if value { Self::TRUE } else { Self::FALSE }
    }

    pub fn is_number(self) -> bool {
        !(TAG_FIRST..=TAG_LAST).contains(&tag_word(self.0))
    }

    pub fn as_number(self) -> Option<f64> {
        self.is_number().then(|| f64::from_bits(self.0))
    }

    pub fn is_undefined(self) -> bool {
        self.0 == TAG_UNDEFINED
    }

    pub fn is_null(self) -> bool {
        self.0 == TAG_NULL
    }

    pub fn is_nullish(self) -> bool {
        self.is_undefined() || self.is_null()
    }

    pub fn is_string(self) -> bool {
        (self.0 & 0xffff_0000_0000_0000) == TAG_STRING
    }

    pub fn is_object(self) -> bool {
        (self.0 & 0xffff_0000_0000_0000) == TAG_OBJECT
    }

    pub fn to_number(self) -> f64 {
        if let Some(value) = self.as_number() {
            return value;
        }
        if self.0 == TAG_NULL || self.0 == TAG_FALSE {
            return 0.0;
        }
        if self.0 == TAG_TRUE {
            return 1.0;
        }
        f64::NAN
    }

    pub fn to_boolean(self) -> bool {
        if let Some(value) = self.as_number() {
            return value != 0.0 && !value.is_nan();
        }
        !matches!(self.0, TAG_UNDEFINED | TAG_NULL | TAG_FALSE)
    }

    pub fn type_name(self) -> &'static str {
        if self.is_number() {
            "number"
        } else if self.is_undefined() {
            "undefined"
        } else if self.is_null() {
            "object"
        } else if self.0 == TAG_TRUE || self.0 == TAG_FALSE {
            "boolean"
        } else if self.is_string() {
            "string"
        } else if self.is_object() {
            "object"
        } else {
            "unknown"
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_number() {
            Some(value) => write!(f, "number({value})"),
            None => write!(f, "{}", self.type_name()),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(value) = self.as_number() {
            return f.write_str(&number_text(value));
        }
        if self.is_undefined() {
            return f.write_str("undefined");
        }
        if self.is_null() {
            return f.write_str("null");
        }
        if self.0 == TAG_TRUE {
            return f.write_str("true");
        }
        if self.0 == TAG_FALSE {
            return f.write_str("false");
        }
        f.write_str(self.type_name())
    }
}

fn number_text(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value == 0.0 {
        return "0".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if value.fract() == 0.0 && value.abs() < 1e21 {
        return format!("{value}");
    }
    if value < 0.0 {
        return format!("-{}", number_text(-value));
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exponent.parse::<i32>().unwrap() + 1;
    if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat(-n as usize))
    } else {
        let sign = if n - 1 < 0 { '-' } else { '+' };
        let power = (n - 1).abs();
        match digits.split_at(1) {
            (first, "") => format!("{first}e{sign}{power}"),
            (first, rest) => format!("{first}.{rest}e{sign}{power}"),
        }
    }
}
