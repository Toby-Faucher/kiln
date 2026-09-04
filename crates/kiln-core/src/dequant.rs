//! Dequantization: CPU references + WGSL kernels + a diff harness.
//!
//! Every CPU path is a line-by-line port of the matching ggml
//! `dequantize_row_*`. The CPU path is the oracle each WGSL kernel is diffed
//! against; `scripts/oracle_dequant.py` cross-checks the CPU path itself against
//! the independent `gguf` Python package.
//!
//! Q4_K super-block (`QK_K = 256`, 144 bytes): `d`/`dmin` f16, `scales[12]`
//! (8×6-bit scale + 6-bit min, packed), `qs[128]` (256 4-bit quants).
//! Q6_K super-block (256, 210 bytes): `ql[128]` + `qh[64]` (6-bit quants),
//! `scales[16]` i8, `d` f16.
//! Q8_0 block (32, 34 bytes): `d` f16, `qs[32]` i8.

use crate::gguf::{f16_to_f32, GgmlType};
use crate::gpu::GpuContext;
use crate::{Error, Result};
use wgpu::util::DeviceExt;

pub const QK_K: usize = 256;
pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q6K_BLOCK_BYTES: usize = 210;
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantize any supported tensor dtype to f32 on the CPU.
pub fn dequant_cpu(t: GgmlType, raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    match t {
        GgmlType::F32 => Ok(bytemuck::cast_slice::<u8, f32>(&raw[..n_elements * 4]).to_vec()),
        GgmlType::F16 => Ok(raw[..n_elements * 2]
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        GgmlType::Q8_0 => q8_0_cpu(raw, n_elements),
        GgmlType::Q4_K => q4k_cpu(raw, n_elements),
        GgmlType::Q6_K => q6k_cpu(raw, n_elements),
    }
}

/// ggml `dequantize_row_q8_0`: `value = d * q[i]`.
pub fn q8_0_cpu(raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if !n_elements.is_multiple_of(Q8_0_BLOCK_ELEMS) {
        return Err(Error::Shape(format!("{n_elements} not a multiple of 32")));
    }
    let n_blocks = n_elements / Q8_0_BLOCK_ELEMS;
    if raw.len() < n_blocks * Q8_0_BLOCK_BYTES {
        return Err(Error::Shape("raw buffer too short for Q8_0".into()));
    }
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let blk = &raw[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push(d * (q as i8) as f32);
        }
    }
    Ok(out)
}

/// ggml `dequantize_row_q6_K`.
pub fn q6k_cpu(raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if !n_elements.is_multiple_of(QK_K) {
        return Err(Error::Shape(format!(
            "{n_elements} not a multiple of {QK_K}"
        )));
    }
    let n_blocks = n_elements / QK_K;
    if raw.len() < n_blocks * Q6K_BLOCK_BYTES {
        return Err(Error::Shape("raw buffer too short for Q6_K".into()));
    }
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let blk = &raw[b * Q6K_BLOCK_BYTES..(b + 1) * Q6K_BLOCK_BYTES];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208]; // i8
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let y = &mut out[b * QK_K..(b + 1) * QK_K];

        for half in 0..2 {
            let ql = &ql[half * 64..];
            let qh = &qh[half * 32..];
            let sc = &sc[half * 8..];
            let y = &mut y[half * 128..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// ggml `get_scale_min_k4`: unpack the 6-bit scale and min for sub-block `j`
/// (0..8) from the 12-byte packed `scales` array.
#[inline]
fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        let d = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        let m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize a Q4_K tensor. `raw` must be exactly `n_elements / 256 * 144` bytes.
pub fn q4k_cpu(raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if !n_elements.is_multiple_of(QK_K) {
        return Err(Error::Shape(format!(
            "{n_elements} not a multiple of {QK_K}"
        )));
    }
    let n_blocks = n_elements / QK_K;
    if raw.len() < n_blocks * Q4K_BLOCK_BYTES {
        return Err(Error::Shape("raw buffer too short for Q4_K".into()));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let blk = &raw[b * Q4K_BLOCK_BYTES..(b + 1) * Q4K_BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let mut is = 0usize;
        let mut q_off = 0usize;
        while is < 8 {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            let q = &qs[q_off..q_off + 32];
            for &byte in q {
                out.push(d1 * (byte & 0x0F) as f32 - min1);
            }
            for &byte in q {
                out.push(d2 * (byte >> 4) as f32 - min2);
            }
            is += 2;
            q_off += 32;
        }
    }
    Ok(out)
}

/// Run `dequant_q4k.wgsl` over `raw` and read back the result.
pub async fn q4k_gpu(ctx: &GpuContext, raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if !n_elements.is_multiple_of(QK_K) {
        return Err(Error::Shape(format!(
            "{n_elements} not a multiple of {QK_K}"
        )));
    }
    let n_blocks = (n_elements / QK_K) as u32;
    let device = &ctx.device;

    // Pad `raw` up to a u32 boundary so the storage buffer cast is clean.
    let mut src = raw[..n_blocks as usize * Q4K_BLOCK_BYTES].to_vec();
    while !src.len().is_multiple_of(4) {
        src.push(0);
    }

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q4k.in"),
        contents: &src,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_bytes = (n_elements * 4) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("q4k.out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("q4k.readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Uniform bindings want a 16-byte-aligned size on some backends; pad.
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q4k.params"),
        contents: bytemuck::cast_slice(&[n_blocks, 0u32, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dequant_q4k.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dequant_q4k.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("q4k.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q4k.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q4k") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q4k.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_blocks.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_bytes);
    ctx.queue.submit([encoder.finish()]);

    let bytes = crate::compute::read_buffer(ctx, &readback).await;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

/// Max absolute and max relative error between two equal-length slices.
pub fn max_error(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut abs = 0.0f32;
    let mut rel = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        abs = abs.max(e);
        let denom = x.abs().max(y.abs()).max(1e-6);
        rel = rel.max(e / denom);
    }
    (abs, rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One hand-built block with d = dmin = 1.0 (f16 0x3C00), all scale bytes 0
    /// => every sub-block scale/min is 0 => every output must be exactly 0.
    #[test]
    fn zero_scales_give_zero() {
        let mut blk = vec![0u8; Q4K_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        blk[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0
        for (i, b) in blk[16..144].iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37); // arbitrary quants
        }
        let out = q4k_cpu(&blk, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "expected all zeros, got {:?}",
            &out[..8]
        );
    }

    /// scale sub-block 0 = 1, min sub-block 0 = 0, d = 1.0: first 32 outputs are
    /// just the low nibbles of qs[0..32].
    #[test]
    fn identity_low_nibbles() {
        let mut blk = vec![0u8; Q4K_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        blk[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0
        blk[4] = 1; // scales[0]: 6-bit scale for sub-block 0 = 1, min = 0
        for (i, b) in blk[16..48].iter_mut().enumerate() {
            *b = (i as u8) & 0x0F; // low nibble = i mod 16, high nibble 0
        }
        let out = q4k_cpu(&blk, QK_K).unwrap();
        for (i, &v) in out[..32].iter().enumerate() {
            assert_eq!(v, (i as f32) % 16.0, "mismatch at {i}");
        }
    }

    /// Q8_0: d = 0.5, qs = [-4, -3, .., 27] => value = 0.5 * q.
    #[test]
    fn q8_0_scaled_identity() {
        let mut blk = vec![0u8; Q8_0_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        for (i, b) in blk[2..34].iter_mut().enumerate() {
            *b = (i as i32 - 4) as i8 as u8;
        }
        let out = q8_0_cpu(&blk, Q8_0_BLOCK_ELEMS).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.5 * (i as f32 - 4.0), "mismatch at {i}");
        }
    }

    /// Q6_K: d = 1.0, all scales = 1, all 6-bit quants = 32 => value = 0.
    #[test]
    fn q6k_centered_zero() {
        let mut blk = vec![0u8; Q6K_BLOCK_BYTES];
        for b in blk[0..128].iter_mut() {
            *b = 0x00; // low nibble 0
        }
        for b in blk[128..192].iter_mut() {
            *b = 0b10_10_10_10; // every 2-bit high group = 2 => quant = 0|32 = 32
        }
        for b in blk[192..208].iter_mut() {
            *b = 1; // scales = 1
        }
        blk[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        let out = q6k_cpu(&blk, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "expected all zeros (q - 32 == 0), got {:?}",
            &out[..8]
        );
    }
}
