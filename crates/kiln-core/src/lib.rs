//! kiln engine core.
//!
//! Layering (built bottom-up as the spikes clear — see `CLAUDE.md`):
//!
//! - [`gpu`] — wgpu device/queue acquisition, adapter limits, feature probe.
//! - [`compute`] — pipeline + bind-group + dispatch helpers, one `CommandEncoder` per token.
//! - [`gguf`] — GGUF v3 reader: metadata, `Config`, tensor bytes.
//! - [`dequant`] — CPU dequant refs + WGSL kernels + diff harness.
//! - [`ops`] — scalar f32 CPU ops (the kernel oracle).
//! - [`model`] — CPU reference forward pass (Qwen3). WGSL port diffed against it.
//! - [`backend`] — the trait that lets WebGPU / (later) WebNN / ORT swap in.

pub mod backend;
pub mod compute;
pub mod dequant;
pub mod error;
pub mod gguf;
pub mod gpu;
pub mod model;
pub mod ops;

pub use error::{Error, Result};

/// WGSL source embedded at compile time. Kernels live in `crates/kiln-core/shaders/`.
pub mod shaders {
    /// Spike #1: minimal `x * 2.0` compute pass. Mirror this in the browser to
    /// confirm WebGPU works in Chrome and in Zen on Linux.
    pub const HELLO_COMPUTE: &str = include_str!("../shaders/hello_compute.wgsl");

    /// Spike #2: Q4_K super-block dequant, one invocation per block.
    pub const DEQUANT_Q4K: &str = include_str!("../shaders/dequant_q4k.wgsl");
}
