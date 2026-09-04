// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w
//
// One workgroup normalizes one row of length `n`. Scalar reduction in workgroup
// shared memory — no subgroup ops (not exposed by wgpu-on-wasm; see docs/spikes.md).
//
// Mirrors kiln_core::ops::rmsnorm exactly.

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

struct Params {
    n: u32,       // row length
    n_rows: u32,  // number of rows (one workgroup each)
    eps: f32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> p: Params;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;
var<workgroup> inv_rms: f32;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.n_rows) {
        return;
    }
    let base = row * p.n;
    let tid = lid.x;

    // Each thread sums squares of its strided slice.
    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= p.n) { break; }
        let v = x[base + i];
        acc = acc + v * v;
        i = i + WG;
    }
    partial[tid] = acc;
    workgroupBarrier();

    // Tree reduction in shared memory.
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        let mean_sq = partial[0] / f32(p.n);
        inv_rms = inverseSqrt(mean_sq + p.eps);
    }
    workgroupBarrier();

    let scale = inv_rms;
    var j = tid;
    loop {
        if (j >= p.n) { break; }
        y[base + j] = x[base + j] * scale * w[j];
        j = j + WG;
    }
}
