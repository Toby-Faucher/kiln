use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no WebGPU adapter available")]
    NoAdapter,

    #[error("failed to acquire wgpu device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    #[error("required feature unavailable: {0}")]
    MissingFeature(&'static str),

    #[error("GGUF parse error: {0}")]
    Gguf(String),

    #[error("shape mismatch: {0}")]
    Shape(String),
}
