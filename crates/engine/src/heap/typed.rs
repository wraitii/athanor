use super::*;

pub(crate) const MAX_TYPED_ARRAY_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TypedKind {
    Uint8,
    Uint16,
    Uint32,
    Int8,
    Int16,
    Int32,
    Float32,
    Float64,
}

impl TypedKind {
    pub(crate) fn element(self) -> u32 {
        match self {
            Self::Uint8 => 0,
            Self::Uint16 => 1,
            Self::Uint32 => 2,
            Self::Int8 => kind::ELEMENT_SIGNED,
            Self::Int16 => kind::ELEMENT_SIGNED | 1,
            Self::Int32 => kind::ELEMENT_SIGNED | 2,
            Self::Float32 => kind::ELEMENT_FLOAT | 2,
            Self::Float64 => kind::ELEMENT_FLOAT | 3,
        }
    }
}

pub(super) struct TypedArray {
    pub(super) kind: TypedKind,
    pub(super) storage: TypedStorage,
}

pub(super) enum TypedStorage {
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl TypedStorage {
    pub(super) fn new(kind: TypedKind, length: usize) -> Option<Self> {
        macro_rules! allocate {
            ($variant:ident, $ty:ty) => {{
                if length.checked_mul(std::mem::size_of::<$ty>())? > MAX_TYPED_ARRAY_BYTES {
                    return None;
                }
                let mut values = Vec::<$ty>::new();
                values.try_reserve_exact(length).ok()?;
                values.resize(length, 0 as $ty);
                Some(Self::$variant(values))
            }};
        }
        match kind {
            TypedKind::Uint8 => allocate!(Uint8, u8),
            TypedKind::Uint16 => allocate!(Uint16, u16),
            TypedKind::Uint32 => allocate!(Uint32, u32),
            TypedKind::Int8 => allocate!(Int8, i8),
            TypedKind::Int16 => allocate!(Int16, i16),
            TypedKind::Int32 => allocate!(Int32, i32),
            TypedKind::Float32 => allocate!(Float32, f32),
            TypedKind::Float64 => allocate!(Float64, f64),
        }
    }
    pub(super) fn raw(&self) -> (*const u8, usize) {
        match self {
            Self::Uint8(values) => (values.as_ptr().cast(), values.len()),
            Self::Uint16(values) => (values.as_ptr().cast(), values.len()),
            Self::Uint32(values) => (values.as_ptr().cast(), values.len()),
            Self::Int8(values) => (values.as_ptr().cast(), values.len()),
            Self::Int16(values) => (values.as_ptr().cast(), values.len()),
            Self::Int32(values) => (values.as_ptr().cast(), values.len()),
            Self::Float32(values) => (values.as_ptr().cast(), values.len()),
            Self::Float64(values) => (values.as_ptr().cast(), values.len()),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Uint8(values) => values.len(),
            Self::Uint16(values) => values.len(),
            Self::Uint32(values) => values.len(),
            Self::Int8(values) => values.len(),
            Self::Int16(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Float32(values) => values.len(),
            Self::Float64(values) => values.len(),
        }
    }
    pub(super) fn get(&self, index: usize) -> Option<f64> {
        match self {
            Self::Uint8(values) => values.get(index).map(|value| *value as f64),
            Self::Uint16(values) => values.get(index).map(|value| *value as f64),
            Self::Uint32(values) => values.get(index).map(|value| *value as f64),
            Self::Int8(values) => values.get(index).map(|value| *value as f64),
            Self::Int16(values) => values.get(index).map(|value| *value as f64),
            Self::Int32(values) => values.get(index).map(|value| *value as f64),
            Self::Float32(values) => values.get(index).map(|value| *value as f64),
            Self::Float64(values) => values.get(index).copied(),
        }
    }
    pub(super) fn integer(number: f64) -> u32 {
        if number.is_finite() && number != 0.0 {
            number.trunc().rem_euclid(4294967296.0) as u32
        } else {
            0
        }
    }
    pub(super) fn set(&mut self, index: usize, number: f64) {
        match self {
            Self::Uint8(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as u8;
                }
            }
            Self::Uint16(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as u16;
                }
            }
            Self::Uint32(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number);
                }
            }
            Self::Int8(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as i8;
                }
            }
            Self::Int16(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as i16;
                }
            }
            Self::Int32(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as i32;
                }
            }
            Self::Float32(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = number as f32;
                }
            }
            Self::Float64(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = number;
                }
            }
        }
    }
    pub(super) fn fill(&mut self, start: usize, end: usize, number: f64) {
        if start >= end {
            return;
        }
        match self {
            Self::Uint8(values) => values[start..end].fill(Self::integer(number) as u8),
            Self::Uint16(values) => values[start..end].fill(Self::integer(number) as u16),
            Self::Uint32(values) => values[start..end].fill(Self::integer(number)),
            Self::Int8(values) => values[start..end].fill(Self::integer(number) as i8),
            Self::Int16(values) => values[start..end].fill(Self::integer(number) as i16),
            Self::Int32(values) => values[start..end].fill(Self::integer(number) as i32),
            Self::Float32(values) => values[start..end].fill(number as f32),
            Self::Float64(values) => values[start..end].fill(number),
        }
    }
}

impl Heap {
    pub(crate) fn typed_new(&self, kind: TypedKind, length: usize) -> Option<Value> {
        let storage = TypedStorage::new(kind, length)?;
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.store(id, Payload::TypedArray(TypedArray { kind, storage }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn typed_kind(&self, value: Value) -> Option<TypedKind> {
        let data = self.0.borrow();
        let array = data.typed_array(heap_id(value)?)?;
        Some(array.kind)
    }

    pub(crate) fn typed_set(&self, value: Value, index: usize, number: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data.typed_array_mut(id) else {
            return false;
        };
        array.storage.set(index, number);
        true
    }

    pub(crate) fn typed_fill(&self, value: Value, start: usize, end: usize, number: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data.typed_array_mut(id) else {
            return false;
        };
        array.storage.fill(start, end, number);
        true
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn float32_copy(&self, value: Value) -> Option<Vec<f32>> {
        let id = heap_id(value)?;
        let data = self.0.borrow();
        let array = data.typed_array(id)?;
        match &array.storage {
            TypedStorage::Float32(values) => Some(values.clone()),
            _ => None,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn float32_write(&self, value: Value, src: &[f32]) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data.typed_array_mut(id) else {
            return false;
        };
        match &mut array.storage {
            TypedStorage::Float32(values) if values.len() == src.len() => {
                values.copy_from_slice(src);
                true
            }
            _ => false,
        }
    }
}
