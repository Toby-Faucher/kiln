//! wgpu device acquisition and capability probing.
//!
//! Works on native (Vulkan/Metal/DX12) and on `wasm32` (browser WebGPU). The
//! same `AdapterReport` fields are what spike #1 captures in Chrome vs Zen.
//!
//! NOTE: written against the wgpu 30.0.1 API. If `cargo build` complains about
//! `Instance::new` / `request_adapter` / `request_device` signatures, reconcile
//! with <https://docs.rs/wgpu/30.0.1> — this is expected churn, not a redesign.

use crate::{Error, Result};

/// A live GPU context: everything downstream needs a `&Device` and `&Queue`.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub report: AdapterReport,
}

/// The capability surface that decides which kernel paths are legal.
/// Log this verbatim from every target during spike #1.
#[derive(Debug, Clone)]
pub struct AdapterReport {
    pub backend: wgpu::Backend,
    pub name: String,
    pub device_type: wgpu::DeviceType,
    pub shader_f16: bool,
    pub subgroups: bool,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
}

impl AdapterReport {
    fn probe(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let feats = adapter.features();
        let limits = adapter.limits();
        Self {
            backend: info.backend,
            name: info.name,
            device_type: info.device_type,
            shader_f16: feats.contains(wgpu::Features::SHADER_F16),
            subgroups: feats.contains(wgpu::Features::SUBGROUP),
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        }
    }
}

/// Acquire a compute-capable device.
///
/// `want_f16` requests `SHADER_F16` if the adapter offers it (kiln needs it for
/// the real kernels; the hello-compute spike does not).
pub async fn acquire(want_f16: bool) -> Result<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            // Set unconditionally in trusted code; browsers get bucketed limits.
            apply_limit_buckets: false,
        })
        .await
        .map_err(|_| Error::NoAdapter)?;

    let report = AdapterReport::probe(&adapter);

    let mut required_features = wgpu::Features::empty();
    if want_f16 {
        if !report.shader_f16 {
            return Err(Error::MissingFeature("shader-f16"));
        }
        required_features |= wgpu::Features::SHADER_F16;
    }

    // Spike: request exactly what this adapter offers. The real engine will pin
    // browser-baseline limits and carry an explicit downlevel path.
    let required_limits = adapter.limits();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("kiln"),
            required_features,
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await?;

    Ok(GpuContext {
        device,
        queue,
        report,
    })
}
