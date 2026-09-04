//! Native spike / dev harness. Subcommands map to the plan in `CLAUDE.md`.
//!
//!   kiln probe                       — adapter report + hello_compute (spike #1)
//!   kiln config  <model.gguf>        — resolved model hyperparameters
//!   kiln tensors <model.gguf>        — list every tensor with dtype/dims
//!   kiln dequant <model.gguf> [tensor]
//!                                    — dequant one Q4_K tensor GPU vs CPU (spike #2)
//!   kiln dequant --all <model.gguf>  — CPU-dequant every tensor, sanity check

use anyhow::{bail, Context, Result};
use kiln_core::gguf::{GgmlType, Gguf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("probe") => pollster::block_on(probe()),
        Some("config") => config(args.get(1).context("usage: kiln config <model.gguf>")?),
        Some("tensors") => tensors(args.get(1).context("usage: kiln tensors <model.gguf>")?),
        Some("dequant") if args.get(1).map(String::as_str) == Some("--all") => dequant_all(
            args.get(2)
                .context("usage: kiln dequant --all <model.gguf>")?,
        ),
        Some("dequant") => pollster::block_on(dequant(
            args.get(1)
                .context("usage: kiln dequant <model.gguf> [tensor]")?,
            args.get(2).map(String::as_str),
        )),
        Some("forward") => forward(&args[1..]),
        _ => {
            eprintln!(
                "usage: kiln <probe | config <gguf> | tensors <gguf> | \
                 dequant <gguf> [tensor] | dequant --all <gguf> | \
                 forward <gguf> --tokens a,b,c [--gen N]>"
            );
            std::process::exit(2);
        }
    }
}

fn forward(args: &[String]) -> Result<()> {
    let path = args
        .first()
        .context("usage: kiln forward <gguf> --tokens a,b,c")?;
    let mut tokens: Vec<u32> = Vec::new();
    let mut gen = 0usize;
    let mut it = args[1..].iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--tokens" => {
                tokens = it
                    .next()
                    .context("--tokens needs a comma list")?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<std::result::Result<_, _>>()
                    .context("parse token id")?;
            }
            "--gen" => {
                gen = it
                    .next()
                    .context("--gen needs N")?
                    .parse()
                    .context("parse N")?
            }
            other => bail!("unknown flag {other}"),
        }
    }
    if tokens.is_empty() {
        bail!("need at least one token via --tokens");
    }

    let g = Gguf::open(path).context("open gguf")?;
    eprintln!("loading + dequantizing weights…");
    let model = kiln_core::model::Model::load(&g).context("load model")?;
    eprintln!("running forward pass on {} tokens…", tokens.len());

    let logits = model.forward(&tokens);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    println!("top-5 next-token logits:");
    for &i in &idx[..5] {
        println!("  {i:>7}  {:.4}", logits[i]);
    }
    println!("argmax: {}", idx[0]);

    if gen > 0 {
        let out = model.generate(&tokens, gen);
        println!("greedy continuation ({gen}): {out:?}");
    }
    Ok(())
}

fn config(path: &str) -> Result<()> {
    let g = Gguf::open(path).context("open gguf")?;
    let c = g.config().context("resolve config")?;
    println!("{c:#?}");
    Ok(())
}

fn dequant_all(path: &str) -> Result<()> {
    let g = Gguf::open(path).context("open gguf")?;
    let mut names: Vec<&str> = g.tensor_names().collect();
    names.sort_unstable();

    let mut bad = 0usize;
    for name in &names {
        let t = g.tensor(name).unwrap();
        let n = t.n_elements();
        let raw = g.raw(name)?;
        let out = kiln_core::dequant::dequant_cpu(t.ggml_type, raw, n)
            .with_context(|| format!("dequant {name}"))?;

        let nan = out.iter().filter(|v| v.is_nan()).count();
        let inf = out.iter().filter(|v| v.is_infinite()).count();
        let absmax = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let status = if out.len() != n || nan > 0 || inf > 0 {
            bad += 1;
            "BAD "
        } else {
            "ok  "
        };
        println!(
            "{status}{name:<34} {:?}  n={n}  |max|={absmax:.4}  nan={nan} inf={inf}",
            t.ggml_type
        );
    }
    println!("\n{} tensors, {bad} bad", names.len());
    if bad > 0 {
        bail!("{bad} tensor(s) failed to dequant cleanly");
    }
    Ok(())
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
    let n = info.n_elements();
    let raw = g.raw(&name)?;
    println!("tensor : {name}");
    println!(
        "dims   : {:?}  ({n} elements, {:?})",
        info.dims, info.ggml_type
    );

    let cpu = kiln_core::dequant::dequant_cpu(info.ggml_type, raw, n).context("cpu dequant")?;
    println!("cpu[..6]: {:?}", &cpu[..6.min(cpu.len())]);

    // Dump the CPU result as raw LE f32 for the external oracle check.
    if let Ok(dump) = std::env::var("KILN_DUMP") {
        let bytes: Vec<u8> = cpu.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&dump, bytes).context("write KILN_DUMP")?;
        println!("wrote cpu dequant -> {dump}");
    }

    // GPU comparison only where a WGSL kernel exists (Q4_K so far).
    if info.ggml_type == GgmlType::Q4_K {
        let ctx = kiln_core::gpu::acquire(false)
            .await
            .context("acquire gpu")?;
        let gpu = kiln_core::dequant::q4k_gpu(&ctx, raw, n)
            .await
            .context("gpu dequant")?;
        let (abs, rel) = kiln_core::dequant::max_error(&cpu, &gpu);
        println!("gpu[..6]: {:?}", &gpu[..6.min(gpu.len())]);
        println!("max abs error: {abs:.3e}");
        println!("max rel error: {rel:.3e}");
        if rel > 1e-3 {
            bail!("FAIL: GPU vs CPU relative error {rel:.3e} exceeds 1e-3");
        }
        println!("PASS (GPU matches CPU reference within 1e-3)");
    } else {
        println!("(no WGSL kernel for {:?} yet — CPU only)", info.ggml_type);
    }
    Ok(())
}
