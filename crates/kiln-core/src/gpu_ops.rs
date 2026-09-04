//! WGSL kernel ports of [`crate::ops`], each diffed against the scalar CPU
//! reference. Pipelines are rebuilt per call — fine for the diff harness; the
//! real engine caches them and keeps activations resident on the GPU.

use crate::compute::read_buffer;
use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

fn storage_init(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// RMSNorm over rows of length `w.len()`. `x.len()` must be a multiple of it.
/// Mirrors [`crate::ops::rmsnorm`].
pub async fn rmsnorm(ctx: &GpuContext, x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = w.len();
    assert!(
        n > 0 && x.len().is_multiple_of(n),
        "x.len() must be a multiple of w.len()"
    );
    let n_rows = (x.len() / n) as u32;
    let device = &ctx.device;

    let x_buf = storage_init(device, "rmsnorm.x", x);
    let w_buf = storage_init(device, "rmsnorm.w", w);
    let out_bytes = (x.len() * 4) as u64;
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.y"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut pbytes = [0u8; 16];
    pbytes[0..4].copy_from_slice(&(n as u32).to_le_bytes());
    pbytes[4..8].copy_from_slice(&n_rows.to_le_bytes());
    pbytes[8..12].copy_from_slice(&eps.to_le_bytes());
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmsnorm.params"),
        contents: &pbytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rmsnorm.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rmsnorm.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rmsnorm.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rmsnorm.bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rmsnorm"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_rows, 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &readback, 0, out_bytes);
    ctx.queue.submit([enc.finish()]);

    let bytes = read_buffer(ctx, &readback).await;
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    // Deterministic PRNG so the test data is stable without a dep.
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// Needs a GPU — run with `cargo test -p kiln-core -- --ignored`.
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn rmsnorm_matches_cpu() {
        let n = 1024usize;
        let rows = 3usize;
        let mut s = 0x1234_5678u64;
        let x: Vec<f32> = (0..n * rows).map(|_| lcg(&mut s) * 4.0).collect();
        let w: Vec<f32> = (0..n).map(|_| 0.5 + lcg(&mut s).abs()).collect();
        let eps = 1e-6;

        let ctx = pollster::block_on(crate::gpu::acquire(false)).expect("gpu");
        let got = pollster::block_on(rmsnorm(&ctx, &x, &w, eps));

        let mut want = vec![0.0f32; n * rows];
        for r in 0..rows {
            ops::rmsnorm(
                &x[r * n..(r + 1) * n],
                &w,
                eps,
                &mut want[r * n..(r + 1) * n],
            );
        }
        let (abs, rel) = crate::dequant::max_error(&got, &want);
        assert!(
            rel < 1e-5,
            "rmsnorm GPU vs CPU: abs={abs:.3e} rel={rel:.3e}"
        );
    }
}
