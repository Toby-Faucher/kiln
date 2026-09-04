//! Browser-facing bindings. Instantiated inside a dedicated Web Worker; the main
//! thread only sees `postMessage` traffic (prompt in, tokens out).
//!
//! v1 surface is deliberately tiny: probe the device, run the spike-#1 kernel.
//! The chat API lands once the engine exists.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Spike #1 in the browser: acquire WebGPU, return the adapter report as a JSON
/// string, and assert `hello_compute` produces the doubled vector.
///
/// Call this from the Worker and print the result in Chrome and in Zen.
#[wasm_bindgen]
pub async fn probe() -> Result<String, JsError> {
    let ctx = kiln_core::gpu::acquire(false)
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;

    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let got = kiln_core::compute::hello_compute(&ctx, &input).await;
    let ok = got == [2.0_f32, 4.0, 6.0, 8.0];

    let r = &ctx.report;
    Ok(format!(
        "{{\"adapter\":\"{}\",\"backend\":\"{:?}\",\"shader_f16\":{},\"subgroups\":{},\
          \"max_storage_buffer_binding_size_mib\":{},\"hello_compute_ok\":{}}}",
        r.name,
        r.backend,
        r.shader_f16,
        r.subgroups,
        r.max_storage_buffer_binding_size / (1 << 20),
        ok,
    ))
}
