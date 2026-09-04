//! Minimal GGUF v3 reader — just enough to locate a tensor and hand back its
//! raw bytes. Not a general-purpose parser: metadata values are skipped, not
//! decoded (we only need `general.alignment`).
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use crate::{Error, Result};
use std::collections::HashMap;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

/// The ggml tensor dtypes kiln cares about. Values match the ggml enum.
/// Variant names follow ggml, not Rust casing, so they line up with the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            14 => Self::Q6_K,
            other => return Err(Error::Gguf(format!("unsupported ggml type {other}"))),
        })
    }

    /// (block size in elements, bytes per block).
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_K => (256, 144),
            Self::Q6_K => (256, 210),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset from the start of the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    pub fn n_bytes(&self) -> usize {
        let (blk_elems, blk_bytes) = self.ggml_type.block();
        self.n_elements().div_ceil(blk_elems) * blk_bytes
    }
}

pub struct Gguf {
    bytes: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl Gguf {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::Gguf(e.to_string()))?;
        Self::parse(bytes)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Raw bytes of a tensor, exactly `info.n_bytes()` long.
    pub fn raw(&self, name: &str) -> Result<&[u8]> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Gguf(format!("no tensor named {name}")))?;
        let start = self.data_start + t.offset as usize;
        let end = start + t.n_bytes();
        self.bytes
            .get(start..end)
            .ok_or_else(|| Error::Gguf(format!("tensor {name} runs past end of file")))
    }

    fn parse(bytes: Vec<u8>) -> Result<Self> {
        let mut r = Cursor::new(&bytes);

        if r.u32()? != MAGIC {
            return Err(Error::Gguf("bad magic (not a GGUF file)".into()));
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(Error::Gguf(format!("unsupported GGUF version {version}")));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut alignment: u64 = 32;
        for _ in 0..kv_count {
            let key = r.gguf_string()?;
            let value_type = r.u32()?;
            if key == "general.alignment" {
                alignment = r.read_metadata_value_as_u64(value_type)?;
            } else {
                r.skip_metadata_value(value_type)?;
            }
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.gguf_string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = GgmlType::from_u32(r.u32()?)?;
            let offset = r.u64()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ggml_type,
                    offset,
                },
            );
        }

        // Tensor data starts at the next `alignment` boundary after the header.
        let pos = r.pos();
        let data_start = pos.div_ceil(alignment as usize) * alignment as usize;

        Ok(Self {
            bytes,
            data_start,
            tensors,
        })
    }
}

/// Byte cursor over the header. Everything is little-endian.
struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn pos(&self) -> usize {
        self.p
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self
            .b
            .get(self.p..self.p + n)
            .ok_or_else(|| Error::Gguf("header truncated".into()))?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn gguf_string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let s = self.take(len)?;
        String::from_utf8(s.to_vec()).map_err(|e| Error::Gguf(e.to_string()))
    }

    /// GGUF metadata value type sizes for the scalar types.
    fn scalar_size(t: u32) -> Option<usize> {
        Some(match t {
            0 | 1 | 7 => 1, // u8 / i8 / bool
            2..=3 => 2,     // u16 / i16
            4..=6 => 4,     // u32 / i32 / f32
            10..=12 => 8,   // u64 / i64 / f64
            _ => return None,
        })
    }

    fn skip_metadata_value(&mut self, t: u32) -> Result<()> {
        match t {
            8 => {
                // string
                let len = self.u64()? as usize;
                self.take(len)?;
            }
            9 => {
                // array: elem_type, count, elems
                let elem_t = self.u32()?;
                let count = self.u64()? as usize;
                if elem_t == 8 {
                    for _ in 0..count {
                        let len = self.u64()? as usize;
                        self.take(len)?;
                    }
                } else if elem_t == 9 {
                    return Err(Error::Gguf("nested arrays unsupported".into()));
                } else {
                    let sz = Self::scalar_size(elem_t)
                        .ok_or_else(|| Error::Gguf(format!("bad array elem type {elem_t}")))?;
                    self.take(sz * count)?;
                }
            }
            other => {
                let sz = Self::scalar_size(other)
                    .ok_or_else(|| Error::Gguf(format!("bad metadata value type {other}")))?;
                self.take(sz)?;
            }
        }
        Ok(())
    }

    fn read_metadata_value_as_u64(&mut self, t: u32) -> Result<u64> {
        Ok(match t {
            4 => self.u32()? as u64,
            5 => self.u32()? as i32 as u64,
            10 => self.u64()?,
            11 => self.u64()?,
            _ => return Err(Error::Gguf(format!("alignment has non-integer type {t}"))),
        })
    }
}

/// IEEE-754 half → f32. Used by the dequant kernels' CPU references.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            // subnormal: normalize
            let mut e: i32 = -14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}
