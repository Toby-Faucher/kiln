//! Minimal compute-dispatch helpers.
//!
//! The architectural rule (Maczan, arXiv 2608.08730): minimize dispatch *count*.
//! For real inference every kernel in a token step shares ONE `CommandEncoder`.
//! This module only carries the spike #1 path for now.

use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

/// Map a `MAP_READ` buffer and return its bytes.
///
/// Cross-platform: on native, `device.poll(Wait)` drives the callback to
/// completion before the `await`; on WebGPU, `poll` is a no-op and the browser
/// event loop fires the callback while the `await` is suspended. Using a
/// blocking `mpsc::recv()` here instead deadlocks the single wasm thread —
/// the event loop can never run the map callback.
pub(crate) async fn read_buffer(ctx: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.await
        .expect("map_async callback dropped")
        .expect("buffer map failed");
    let data = slice
        .get_mapped_range()
        .expect("get_mapped_range after successful map")
        .to_vec();
    buffer.unmap();
    data
}

/// Run `hello_compute.wgsl` over `input` and read the result back.
///
/// This round-trips CPU -> GPU -> CPU (slow, map-read every call) on purpose:
/// it is a wiring test, not a template for the hot path.
pub async fn hello_compute(ctx: &GpuContext, input: &[f32]) -> Vec<f32> {
    let device = &ctx.device;
    let byte_len = std::mem::size_of_val(input) as u64;

    let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hello.storage"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hello.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hello_compute.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::shaders::HELLO_COMPUTE.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hello.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hello.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hello"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hello.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = input.len().div_ceil(64) as u32;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, byte_len);
    ctx.queue.submit([encoder.finish()]);

    let bytes = read_buffer(ctx, &readback).await;
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}
