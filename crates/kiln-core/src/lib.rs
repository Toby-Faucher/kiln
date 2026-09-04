//! kiln engine core.
//!
//! Layering (built bottom-up as the spikes clear — see `CLAUDE.md`):
//!
//! - [`gpu`] — wgpu device/queue acquisition, adapter limits, feature probe.
//! - [`compute`] — pipeline + bind-group + dispatch helpers, one `CommandEncoder` per token.
//! - `gguf` — GGUF v3 parser, Q4_K_M + Q8_0 block types first (todo).
//! - `dequant` — WGSL dequant kernels, each with a CPU reference + diff test (todo).
//! - `model` — architecture graph (Qwen3 dense first), KV cache (todo).
//! - [`backend`] — the trait that lets WebGPU / (later) WebNN / ORT swap in.

pub mod backend;
pub mod compute;
pub mod dequant;
pub mod error;
pub mod gguf;
pub mod gpu;

pub use error::{Error, Result};

/// WGSL source embedded at compile time. Kernels live in `crates/kiln-core/shaders/`.
pub mod shaders {
    /// Spike #1: minimal `x * 2.0` compute pass. Mirror this in the browser to
    /// confirm WebGPU works in Chrome and in Zen on Linux.
    pub const HELLO_COMPUTE: &str = include_str!("../shaders/hello_compute.wgsl");

    /// Spike #2: Q4_K super-block dequant, one invocation per block.
    pub const DEQUANT_Q4K: &str = include_str!("../shaders/dequant_q4k.wgsl");
}
