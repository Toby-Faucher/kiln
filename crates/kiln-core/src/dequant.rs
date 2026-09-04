//! Q4_K dequantization: CPU reference + WGSL kernel + a diff harness.
//!
//! Q4_K super-block (`QK_K = 256` elements, 144 bytes):
//!   - `d`       f16  super-block scale for the 6-bit sub-block scales
//!   - `dmin`    f16  super-block scale for the 6-bit sub-block mins
//!   - `scales`  [u8; 12]  8×(6-bit scale, 6-bit min), bit-packed
//!   - `qs`      [u8; 128]  256 4-bit quants, low nibble then high nibble
//!
//! Value = `d * scale[sub] * q  -  dmin * min[sub]`.
//!
//! The CPU path is a line-by-line port of ggml's `dequantize_row_q4_K` /
//! `get_scale_min_k4`. It is the oracle the WGSL kernel is diffed against.

use crate::gguf::f16_to_f32;
use crate::gpu::GpuContext;
use crate::{Error, Result};
use wgpu::util::DeviceExt;

pub const QK_K: usize = 256;
pub const Q4K_BLOCK_BYTES: usize = 144;

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

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();
    let view = slice.get_mapped_range().expect("map q4k readback");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    readback.unmap();
    Ok(out)
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
}
