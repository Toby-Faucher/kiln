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

/// A decoded GGUF metadata value.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Any integer variant, widened. `None` for non-integers.
    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            Self::U8(v) => *v as i64,
            Self::I8(v) => *v as i64,
            Self::U16(v) => *v as i64,
            Self::I16(v) => *v as i64,
            Self::U32(v) => *v as i64,
            Self::I32(v) => *v as i64,
            Self::U64(v) => *v as i64,
            Self::I64(v) => *v,
            Self::Bool(v) => *v as i64,
            _ => return None,
        })
    }

    /// Any float or integer variant, as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => self.as_i64().map(|v| v as f32),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// Model hyperparameters pulled from GGUF metadata. Keys are prefixed with the
/// value of `general.architecture` (e.g. `qwen3.block_count`).
#[derive(Debug, Clone)]
pub struct Config {
    pub architecture: String,
    pub n_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub context_length: usize,
    pub vocab_size: usize,
}

pub struct Gguf {
    bytes: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
    metadata: HashMap<String, MetaValue>,
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

    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn meta_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    /// Resolve model hyperparameters. Errors if a required key is missing.
    pub fn config(&self) -> Result<Config> {
        let arch = self
            .meta("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| Error::Gguf("missing general.architecture".into()))?
            .to_string();

        let key = |suffix: &str| format!("{arch}.{suffix}");
        let need_usize = |suffix: &str| -> Result<usize> {
            self.meta(&key(suffix))
                .and_then(MetaValue::as_i64)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Gguf(format!("missing {}.{suffix}", arch)))
        };
        let need_f32 = |suffix: &str| -> Result<f32> {
            self.meta(&key(suffix))
                .and_then(MetaValue::as_f32)
                .ok_or_else(|| Error::Gguf(format!("missing {}.{suffix}", arch)))
        };

        let d_model = need_usize("embedding_length")?;
        let n_heads = need_usize("attention.head_count")?;
        let head_dim = self
            .meta(&key("attention.key_length"))
            .and_then(MetaValue::as_i64)
            .map(|v| v as usize)
            .unwrap_or(d_model / n_heads);

        // vocab: explicit key, else token-list length, else token_embd rows.
        let vocab_size = self
            .meta(&key("vocab_size"))
            .and_then(MetaValue::as_i64)
            .map(|v| v as usize)
            .or_else(|| {
                self.meta("tokenizer.ggml.tokens")
                    .and_then(MetaValue::as_array)
                    .map(<[_]>::len)
            })
            .or_else(|| self.tensor("token_embd.weight").map(|t| t.dims[1] as usize))
            .ok_or_else(|| Error::Gguf("cannot determine vocab_size".into()))?;

        Ok(Config {
            n_layers: need_usize("block_count")?,
            d_model,
            d_ff: need_usize("feed_forward_length")?,
            n_heads,
            n_kv_heads: self
                .meta(&key("attention.head_count_kv"))
                .and_then(MetaValue::as_i64)
                .map(|v| v as usize)
                .unwrap_or(n_heads),
            head_dim,
            rms_eps: need_f32("attention.layer_norm_rms_epsilon")?,
            rope_theta: self
                .meta(&key("rope.freq_base"))
                .and_then(MetaValue::as_f32)
                .unwrap_or(10_000.0),
            context_length: need_usize("context_length")?,
            vocab_size,
            architecture: arch,
        })
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

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.gguf_string()?;
            let value_type = r.u32()?;
            let value = r.read_value(value_type)?;
            metadata.insert(key, value);
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_i64)
            .map(|v| v as u64)
            .unwrap_or(32);

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
            metadata,
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

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Decode one GGUF metadata value of type tag `t`.
    fn read_value(&mut self, t: u32) -> Result<MetaValue> {
        Ok(match t {
            0 => MetaValue::U8(self.take(1)?[0]),
            1 => MetaValue::I8(self.take(1)?[0] as i8),
            2 => MetaValue::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => MetaValue::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(self.f32()?),
            7 => MetaValue::Bool(self.take(1)?[0] != 0),
            8 => MetaValue::String(self.gguf_string()?),
            9 => {
                let elem_t = self.u32()?;
                if elem_t == 9 {
                    return Err(Error::Gguf("nested arrays unsupported".into()));
                }
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_value(elem_t)?);
                }
                MetaValue::Array(items)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(self.f64()?),
            other => return Err(Error::Gguf(format!("bad metadata value type {other}"))),
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
