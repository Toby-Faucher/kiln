# kiln

**Fire a language model in your browser.**

An in-browser LLM chat runtime written in Rust → WASM + WebGPU. Hand-written WGSL
compute kernels, GGUF Q4_K_M weights, no framework. The Rust answer to WebLLM /
wllama that doesn't currently exist (HuggingFace's `ratchet` was it; abandoned
Nov 2024).

This is a **learning project first**: the point is to build the GPU/systems code
by hand. If it just needs to *work*, forking `wllama` or building on
`transformers.js` v4 is faster — that is a deliberate rejected path, not an
oversight.

## Status

Pre-code. Doing the four de-risking spikes (see below) before committing to the
build.

## The decided stack (do not re-litigate without reason)

| Layer | Choice | Notes |
|---|---|---|
| GPU | `wgpu = "=30.0.1"` directly | Not raw `web-sys::Gpu*`. Not `cubecl` (alpha, no browser-LLM precedent, no tensor cores on WebGPU). |
| Target | `wasm32-unknown-unknown` | Not emscripten. **No** Memory64, **no** JSPI, **no** wasm64. `simd128` on. |
| Toolchain | `wasm-bindgen =0.2.127`, `web-sys =0.3.104`, `wasm-pack 0.15.0`, `wasm-opt`/Binaryen 132 | `wasm-pack`, **not** trunk (this is an engine crate, not a frontend app). |
| Build flags | `target-feature=+simd128,+bulk-memory,+mutable-globals,+reference-types,+sign-ext,+nontrapping-fptoint`; `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`, `strip=true` | Named `Gpu*` web-sys features only, never `["full"]`. `--cfg=web_sys_unstable_apis`. |
| Quant | GGUF **Q4_K_M** for the transformer body, **Q8_0** for `embed_tokens` / `lm_head` | Skip MXFP4 / NVFP4 / ONNX-MatMulNBits / IQ-quants — server formats, no browser story. |
| Attention | **Split-K first** (3 dispatches/layer: score / softmax / value) | Do **NOT** port `flash_attn.wgsl` for v1 — it needs `chromium_experimental_subgroup_matrix`, an unsafe flag that runs on neither stable Chrome nor Firefox. |
| Kernels | Fork `Beledarian/wgpu-llm`'s 12 WGSL shaders as the scaffold | Borrow *only* llama.cpp's Q4_K dequant template (`mul_mat_vec_q_acc.tmpl` Q4_K branch + `mul_mat_decls.tmpl`). Those templates are a reference to *read*, not port wholesale (C++-side wgpu-native, templated preprocessor). |
| Tokenizer | JS: `@huggingface/tokenizers` 0.1.3 + `@huggingface/jinja` 0.5.9 | Behind a trait so it can move to Rust (`tokenizers` `unstable_wasm` + `minijinja 2.24`) later if the JS↔WASM boundary hurts. |
| Loading | Sharded GGUF (`llama-gguf-split --split-max-size 512M`) → HTTP Range → OPFS `FileSystemSyncAccessHandle` in the Worker → 4×1 MB transfer ring → WebGPU storage buffers | **Never load the model into the WASM heap.** Avoid Cache API for the model (Chrome 2 GB single-entry bug). Plan a re-download path for Safari's 7-day ITP wipe. Copy `wllama/src/cache-manager.ts` for the SHA/ETag sidecar. |
| Runtime shape | Dedicated Web Worker hosts the WASM module; main thread is UI + token stream | Single-threaded worker for v1. |
| Perf lever | **Minimize dispatch count, not kernel speed.** Single `CommandEncoder` per token. Fuse RoPE into attention, fuse RMSNorm multiply. | Per-dispatch cost ~24–71 µs is the batch-1 floor. Fusion is worth more than faster kernels. |
| Sampling | CPU-side for non-greedy; GPU `argmax` for greedy | |
| WebNN | Ignore for v1 (CR Draft, nothing shipped) | Thin `Backend` trait so it / ORT can slot in later. |

## Model ladder

1. **Qwen3-0.6B Q4_K_M** — bring-up + numerical validation. Pure attention, tiny,
   fast debug loop.
2. **Qwen3-1.7B Q4_K_M (~1.1 GB)** — v1 engine target. Still pure attention, so
   you prove the engine without also debugging a linear-attention recurrence.
   `Phi-4-mini-instruct` (MIT, 2.49 GB) is the interchangeable alternative.
3. **Qwen3.5-2B** — the showcase / "current 2026 stack" target. Only after the
   engine runs **and** you've ported `gated_delta_net.wgsl` (it uses hybrid
   Gated-DeltaNet attention; it's also multimodal — text path only for now).
   llama.cpp *does* have a `gated_delta_net.wgsl` reference as of Sept 2026.

Defer SmolLM3 / Gemma 4 until the engine works. Any 2026 model: verify it exists,
ships a GGUF, and check whether it's pure attention or hybrid before adopting —
hybrid is a WGSL research subproject, not a config swap.

## The four spikes (do these BEFORE committing weeks)

Progress lives in `docs/spikes.md`. Status: **#1 native PASS**, **#2 PASS**
(Q4_K dequant bit-exact GPU==CPU==gguf-package), **#4 PASS** (no proxy needed).
Open: #1 browser half, #3, and Q6_K dequant.

1. **WebGPU reality check** (~1 day). Run `webgpureport.org` + a 20-line `wgpu`
   compute shader in Chrome **and** in Zen, on this Arch box. Capture
   `adapter.limits`, `shader-f16`, `subgroups`. Firefox/Zen WebGPU on Linux is
   Nightly-only as of Sept 2026 — if it's absent, "Firefox-first" is dead and
   this is Chrome-only.
2. **Q4_K dequant correctness** (~3–5 days). Implement *only* Q4_K dequant. Run it
   on one real weight tensor from a Qwen3-0.6B GGUF. Compare element-wise against
   `llama.cpp` CPU dequant of the same tensor. **This is the failure mode that
   silently produces coherent-looking garbage instead of crashing.** Get the
   super-block scale/min bit layout right in isolation.
3. **Approach validation** (~1–2 weeks). Fork `Beledarian/wgpu-llm`. Run it as a
   native binary first (confirm ~25–66 tok/s per its README). Then get *that*
   running in-browser Chrome with a single `CommandEncoder` per token. Measure.
4. **HF Hub CORS + Range** (~2 hrs). `curl -sIL -H "Origin: https://example.com"
   -H "Range: bytes=0-1023"` a real GGUF shard, follow the Xet redirect to
   `cas-bridge.xethub.hf.co`, confirm `206` + `Access-Control-Allow-Origin` +
   `Access-Control-Expose-Headers: Content-Range`. If it fails, the loader needs
   a proxy or a full-shard-download fallback.

## Numerical validation strategy (both source reports missed this)

The CPU oracle is `llama.cpp` run on the **exact same GGUF file**. Dump per-layer
activation tensors (ggml has debug hooks / `llama-eval-callback`), compare
element-wise. Use **relative tolerance ~1e-2 on late layers**, not bit-identity —
f16 accumulation drift is expected. Bit-identity only applies to fp32 matmul in
isolation.

## Realistic expectations

- **Chrome**, laptop dGPU / M2-class, ~1B q4: **40–100 tok/s** decode, 50–150 ms
  TTFT.
- **Firefox / Zen**, same hardware: **1–10 tok/s** (10–50× slower — dispatch-heavy
  workloads are rate-limited, GPU util 65–70% vs Chromium 90%, subgroups not
  exposed). Ship Chrome-first with an explicit "use Chrome for full speed"
  fallback message.
- **Cold TTFT** is dominated by shader compilation (1–5 s), not model load.
- **Time budget**: ~2–3 months of evenings to a correct-but-slow single-model
  Chrome-only demo. 6–12 months to a genuinely usable multi-model chat app.
  (LlamaWeb was a research *team*; their paper reports "bugs in every WebGPU
  implementation.")

## Key references

- `Beledarian/wgpu-llm` — the scaffold. Rust + wgpu + WGSL, 12 clean shaders.
- `github.com/ggml-org/llama.cpp` — `ggml/src/ggml-webgpu/wgsl-shaders/` — dequant
  templates, `gated_delta_net.wgsl`, the k-quant refactor in PR #24225 (Jun 2026,
  1.34–3.27× faster).
- `ngxson/wllama` — cache-manager / split-file loader pattern. Perf ceiling
  reference (pin `wllama 3.6.1`).
- arXiv **2605.20706** (Levine et al., "Llamas on the Web", May 2026) — WebGPU LLM
  SOTA, portability data. Abstract: +45–69% decode / 29–33% less memory vs
  WebLLM/transformers.js.
- arXiv **2608.08730** (Maczan, Aug 2026) — dispatch overhead: dispatch *count*,
  not kernel quality, is the batch-1 bottleneck; naive timing overstates ~20×.
- `github.com/gpuweb/gpuweb/wiki/Implementation-Status` — browser support truth.

Full research: `~/research/in-browser-llm-rust-webgpu-2026-09.md` and
`~/research/in-browser-llm-stack-report-2026-09-04.md` (plus the reconciliation
notes). Both are accurate on facts; where they disagreed on the target model,
the newer-models call (Qwen3.5 exists, is Apache-2.0) won.

## Repo layout

```
crates/kiln-core/   engine: gpu acquisition, compute helpers, backend trait,
                    (todo) gguf parser, dequant kernels, model graph, KV cache
        shaders/    WGSL — one CPU reference impl + diff test per kernel
crates/kiln-wasm/   cdylib — the API the Web Worker calls (probe(); later: chat)
crates/kiln-cli/    native `kiln` bin — spike harness (`kiln probe`, `kiln dequant`)
web/                spike #1 browser harness (index.html + worker.js)
.cargo/config.toml  wasm32 build flags (native builds untouched)
```

Native probe: `cargo run -p kiln-cli -- probe`
Browser probe: `wasm-pack build crates/kiln-wasm --target web --out-dir ../../web/pkg`
then serve `web/` over http and open in Chrome + Zen.

The wgpu 30 calls in `gpu.rs` / `compute.rs` are written from the docs, not
compile-verified against 30.0.1 — reconcile any signature drift with
<https://docs.rs/wgpu/30.0.1> as the first task.

## Branch flow / how GitHub works here

`PR → dev → alpha → prod`

- **`prod`** — release branch. Only ever fast-forwarded from `alpha`. Tagged on release.
- **`alpha`** — integration / pre-release. `dev` merges up when a milestone is
  stable enough to dogfood.
- **`dev`** — default branch, where day-to-day work lands.
- **feature branches → PR into `dev`.** Every change goes through a PR, even
  solo. Squash-merge. CI (fmt + clippy + test + `wasm-pack build`) must pass.

Promotions (`dev → alpha`, `alpha → prod`) are also PRs, merge-commit (not
squash) so history is preserved. Never commit straight to `dev`, `alpha`, or
`prod`.

## Conventions

- Match surrounding code style. Comment density follows the file you're in.
- WGSL shaders live next to the crate that owns them, in `shaders/`. Every kernel
  gets a CPU reference implementation and a test that diffs them (rel. err
  < 1e-3 for dequant, < 1e-2 for late-layer activations).
- Don't add dependencies without a reason that goes in the commit message.
- Conventional-ish commit subjects (`core:`, `wasm:`, `cli:`, `docs:`, `ci:`).
