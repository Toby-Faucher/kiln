//! The seam that keeps WebGPU from being load-bearing everywhere.
//!
//! v1 has exactly one impl (WebGPU via wgpu). The trait exists so a WebNN or
//! ONNX-Runtime backend can be added in 12–24 months without touching the model
//! graph. Keep it *thin* — resist widening it until a second impl actually lands.

use crate::Result;

/// A backend executes a model's forward pass for one decode step and returns
/// next-token logits. Prefill is the same call with a longer input.
pub trait Backend {
    /// Human-readable identity for logs / UI ("webgpu:vulkan:NVIDIA ...").
    fn describe(&self) -> String;

    /// Feed `tokens` (prompt on the first call, one token thereafter) and get
    /// logits over the vocab for the next position.
    fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Drop KV-cache state and start a fresh sequence.
    fn reset(&mut self);
}
