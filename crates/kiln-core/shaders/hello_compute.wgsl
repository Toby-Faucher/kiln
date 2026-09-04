// Spike #1: the smallest possible compute pass.
// If this runs and returns [2, 4, 6, ...] in a browser, that browser has usable
// WebGPU compute. Run it in Chrome AND in Zen on Linux.

@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&data)) {
        return;
    }
    data[i] = data[i] * 2.0;
}
