//! Native spike harness. Subcommands map 1:1 to the plan in `CLAUDE.md`.
//!
//!   kiln probe                    — spike #1: adapter report + hello_compute.
//!   kiln tensors <model.gguf>     — list Q4_K/Q6_K/Q8_0/F32/F16 tensors.
//!   kiln dequant <model.gguf> [tensor]
//!                                 — spike #2: dequant one Q4_K tensor on GPU and
//!                                   CPU, report max abs/rel error.

use anyhow::{bail, Context, Result};
use kiln_core::gguf::{GgmlType, Gguf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("probe") => pollster::block_on(probe()),
        Some("tensors") => tensors(args.get(1).context("usage: kiln tensors <model.gguf>")?),
        Some("dequant") => pollster::block_on(dequant(
            args.get(1)
                .context("usage: kiln dequant <model.gguf> [tensor]")?,
            args.get(2).map(String::as_str),
        )),
        _ => {
            eprintln!("usage: kiln <probe | tensors <gguf> | dequant <gguf> [tensor]>");
            std::process::exit(2);
        }
    }
}

async fn probe() -> Result<()> {
    let ctx = kiln_core::gpu::acquire(false)
        .await
        .context("no compute-capable GPU adapter")?;

    let r = &ctx.report;
    println!(
        "adapter        : {} ({:?}, {:?})",
        r.name, r.backend, r.device_type
    );
    println!("shader-f16     : {}", r.shader_f16);
    println!("subgroups      : {}", r.subgroups);
    println!("max_buffer_size: {} MiB", r.max_buffer_size / (1 << 20));
    println!(
        "max_storage_buffer_binding_size: {} MiB",
        r.max_storage_buffer_binding_size / (1 << 20)
    );
    println!(
        "max_compute_workgroup_storage_size: {} B",
        r.max_compute_workgroup_storage_size
    );
    println!(
        "max_compute_invocations_per_workgroup: {}",
        r.max_compute_invocations_per_workgroup
    );

    let input: Vec<f32> = (1..=8).map(|n| n as f32).collect();
    let got = kiln_core::compute::hello_compute(&ctx, &input).await;
    let want: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
    println!("\nhello_compute  : {got:?}");
    if got != want {
        bail!("hello_compute mismatch: wanted {want:?}");
    }
    println!("hello_compute  : OK");
    Ok(())
}

fn tensors(path: &str) -> Result<()> {
    let g = Gguf::open(path).context("open gguf")?;
    let mut names: Vec<&str> = g.tensor_names().collect();
    names.sort_unstable();
    for name in names {
        let t = g.tensor(name).unwrap();
        println!(
            "{:<40} {:?}  dims={:?}  {} elems  {} bytes",
            name,
            t.ggml_type,
            t.dims,
            t.n_elements(),
            t.n_bytes()
        );
    }
    Ok(())
}

async fn dequant(path: &str, tensor: Option<&str>) -> Result<()> {
    let g = Gguf::open(path).context("open gguf")?;

    // Default to the first Q4_K tensor found.
    let name = match tensor {
        Some(t) => t.to_string(),
        None => g
            .tensor_names()
            .filter(|n| g.tensor(n).unwrap().ggml_type == GgmlType::Q4_K)
            .min()
            .map(str::to_string)
            .context("no Q4_K tensor in this file")?,
    };

    let info = g.tensor(&name).context("tensor not found")?.clone();
    if info.ggml_type != GgmlType::Q4_K {
        bail!("{name} is {:?}, not Q4_K", info.ggml_type);
    }
    let n = info.n_elements();
    let raw = g.raw(&name)?;
    println!("tensor : {name}");
    println!(
        "dims   : {:?}  ({n} elements, {} Q4_K blocks)",
        info.dims,
        n / 256
    );

    let cpu = kiln_core::dequant::q4k_cpu(raw, n).context("cpu dequant")?;

    let ctx = kiln_core::gpu::acquire(false)
        .await
        .context("acquire gpu")?;
    let gpu = kiln_core::dequant::q4k_gpu(&ctx, raw, n)
        .await
        .context("gpu dequant")?;

    let (abs, rel) = kiln_core::dequant::max_error(&cpu, &gpu);
    println!("cpu[..6]: {:?}", &cpu[..6.min(cpu.len())]);
    println!("gpu[..6]: {:?}", &gpu[..6.min(gpu.len())]);
    println!("max abs error: {abs:.3e}");
    println!("max rel error: {rel:.3e}");

    // Optional: dump the CPU result as raw LE f32 for an external oracle check.
    if let Ok(path) = std::env::var("KILN_DUMP") {
        let mut bytes = Vec::with_capacity(cpu.len() * 4);
        for v in &cpu {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, bytes).context("write KILN_DUMP")?;
        println!("wrote cpu dequant -> {path}");
    }

    if rel > 1e-3 {
        bail!("FAIL: relative error {rel:.3e} exceeds 1e-3");
    }
    println!("PASS (GPU matches CPU reference within 1e-3)");
    Ok(())
}
