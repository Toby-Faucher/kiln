// Q4_K dequantization. One invocation per 256-element super-block.
//
// Block layout (144 bytes = 36 u32 words):
//   word 0        : d (f16, low), dmin (f16, high)   -> unpack2x16float
//   words 1..=3   : scales[0..12]  (8x 6-bit scale + 6-bit min, bit-packed)
//   words 4..=35  : qs[0..128]     (256 4-bit quants)
//
// Mirrors the CPU reference in dequant.rs (ggml dequantize_row_q4_K).

@group(0) @binding(0) var<storage, read>       inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform>             n_blocks: u32;

// byte `k` (0..12) of the packed `scales` array for block at word `base`.
fn sbyte(base: u32, k: u32) -> u32 {
    let w = inp[base + 1u + (k >> 2u)];
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}

// byte `n` (0..128) of `qs` for block at word `base`.
fn qbyte(base: u32, n: u32) -> u32 {
    let w = inp[base + 4u + (n >> 2u)];
    return (w >> ((n & 3u) * 8u)) & 0xFFu;
}

// ggml get_scale_min_k4: (6-bit scale, 6-bit min) for sub-block `j` (0..8).
fn scale_min_k4(base: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(sbyte(base, j) & 63u, sbyte(base, j + 4u) & 63u);
    }
    let d = (sbyte(base, j + 4u) & 0x0Fu) | ((sbyte(base, j - 4u) >> 6u) << 4u);
    let m = (sbyte(base, j + 4u) >> 4u)  | ((sbyte(base, j) >> 6u) << 4u);
    return vec2<u32>(d, m);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if (blk >= n_blocks) {
        return;
    }
    let base = blk * 36u;
    let out_base = blk * 256u;

    let sb = unpack2x16float(inp[base]);
    let d = sb.x;
    let dmin = sb.y;

    var is: u32 = 0u;
    var q_off: u32 = 0u;
    loop {
        if (is >= 8u) {
            break;
        }
        let sm1 = scale_min_k4(base, is);
        let sm2 = scale_min_k4(base, is + 1u);
        let d1 = d * f32(sm1.x);
        let m1 = dmin * f32(sm1.y);
        let d2 = d * f32(sm2.x);
        let m2 = dmin * f32(sm2.y);

        let seg = (is >> 1u) * 64u;
        for (var l: u32 = 0u; l < 32u; l = l + 1u) {
            let byte = qbyte(base, q_off + l);
            outp[out_base + seg + l]        = d1 * f32(byte & 0x0Fu) - m1;
            outp[out_base + seg + 32u + l]  = d2 * f32(byte >> 4u) - m2;
        }
        is = is + 2u;
        q_off = q_off + 32u;
    }
}
