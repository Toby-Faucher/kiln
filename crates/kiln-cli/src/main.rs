//! Native spike harness. Subcommands map 1:1 to the plan in `CLAUDE.md`.
//!
//!   kiln probe     — spike #1: acquire an adapter, print its capability report,
//!                    run hello_compute.wgsl, verify the result.
//!   kiln dequant   — spike #2: Q4_K dequant in WGSL vs a CPU reference on one
//!                    real weight tensor from a GGUF file. (stub)

use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "probe" => pollster::block_on(probe()),
        "dequant" => dequant(),
        _ => {
            eprintln!("usage: kiln <probe|dequant>");
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

fn dequant() -> Result<()> {
    bail!(
        "spike #2 not implemented yet.\n\
         plan: parse one Q4_K tensor from a Qwen3-0.6B GGUF, dequant it in WGSL,\n\
         dequant the same bytes with a CPU reference, assert max relative error\n\
         < 1e-3. This is the failure mode that produces coherent-looking garbage."
    )
}
