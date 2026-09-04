# Spike log

Running record of the de-risking spikes from `CLAUDE.md`. Each entry: what was
tested, the result, and what it means for the build.

---

## Spike 1 — WebGPU reality check

**Native half: PASS** (2026-09-03)

`cargo run -p kiln-cli -- probe` on the dev machine (Arch Linux):

```
adapter        : AMD Radeon RX 9070 XT (RADV GFX1201) (Vulkan, DiscreteGpu)
shader-f16     : true
subgroups      : true
max_buffer_size: 2047 MiB
max_storage_buffer_binding_size: 2047 MiB
max_compute_workgroup_storage_size: 65536 B
max_compute_invocations_per_workgroup: 1024
hello_compute  : [2, 4, 6, 8, ...]  OK
```

Native limits are generous (RADV exposes 2 GiB bindings, 64 KiB workgroup
storage). These are **not** what the browser will report — WebGPU caps
`maxStorageBufferBindingSize` at 128 MiB and workgroup storage at 16 KiB. The
engine must be written to the browser numbers, not these.

**Browser half — Zen: PASS** (2026-09-04)

`http://localhost:8000` (kiln-wasm `probe()`) in Zen on Arch:

```json
{"adapter":"","backend":"BrowserWebGpu","shader_f16":true,
 "subgroups":false,"max_storage_buffer_binding_size_mib":1024,
 "hello_compute_ok":true}
```

- **Zen HAS working WebGPU compute on Linux.** `requestAdapter()` returns a real
  `GPUAdapter`; the `x*2` compute shader ran and returned correct output. The
  research's "Firefox Linux WebGPU is Nightly-only" is bypassed here — the dev
  box has `gfx.webgpu.force-enabled = true` set (from an earlier unrelated
  session), which skips the GPU blocklist.
- `shader_f16: true` — f16 is available.
- **`subgroups: false`** — not exposed in Zen (Firefox stable doesn't ship the
  feature). Confirms the plan: split-K attention, scalar reductions, no
  `subgroupAdd`. Feature-detect and provide the scalar path unconditionally.
- `max_storage_buffer_binding_size: 1024 MiB` — Zen is *more* generous than the
  128 MiB WebGPU baseline. Still design to 128 MiB (Chrome/Safari default) for
  portability; Zen won't be the binding-size constraint.
- `adapter: ""` — Firefox does not populate the adapter name (fingerprinting
  defense). Don't rely on GPU name/vendor strings on this browser.
- Prereqs installed for this: replaced Arch `rust` with `rustup`,
  `rustup target add wasm32-unknown-unknown`, `cargo install wasm-pack`.
- Needed the async-readback fix first (`std::sync::mpsc::recv()` deadlocked the
  wasm thread — the probe hung on "running…" until that landed).

**Browser half — Chromium: PASS** (2026-09-04)

Chromium on Arch, launched `--enable-unsafe-webgpu --enable-features=Vulkan`:

```json
{"adapter":"","backend":"BrowserWebGpu","shader_f16":true,
 "subgroups":false,"max_storage_buffer_binding_size_mib":4095,
 "hello_compute_ok":true}
```

- Compute works. f16 available.
- **`subgroups: false` — same as Zen, and this is the notable result.** Chrome
  has supported WebGPU subgroups at the browser level since Chrome 134. Reading
  `false` here means the **Rust `wgpu` 30 WebGPU-on-wasm backend does not surface
  the subgroups feature** through `adapter.features()`. The RUST research report
  flagged this exact uncertainty ("whether the Rust wgpu crate exposes subgroup
  features on the WebGPU flag — no public confirmation"). Now confirmed negative
  for plain `subgroups`.
  - **Consequence for kiln:** scalar reduction path everywhere — split-K
    attention, scalar RMSNorm / matvec. No `subgroupAdd`, no subgroup-matrix,
    on *any* browser via the Rust+wasm route, until wgpu closes this gap. This
    is already the CLAUDE.md plan; the finding just removes the "maybe we get a
    fast path on Chrome" option.
- `max_storage_buffer_binding_size: 4095 MiB` is inflated by
  `--enable-unsafe-webgpu` (removes limit bucketing). A normal Chrome user gets
  **128 MiB** — design to that.

### Spike 1 summary

WebGPU compute works on both target browsers. Design constraints locked:
`shader-f16` yes, subgroups **no** (wgpu-wasm limitation, not browser),
`maxStorageBufferBindingSize` **128 MiB** (portable baseline), no adapter
name/vendor strings on Firefox.

---

## Spike 4 — HF Hub CORS + Range through the Xet redirect

**PASS — no proxy needed** (2026-09-03)

Tested: `bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf`
with `Origin: https://example.com` + `Range: bytes=0-1023`.

- `resolve/main/...` → **302** to `https://us.aws.cdn.hf.co/xet-bridge-us/...`
  (an AWS CloudFront edge — **not** `cas-bridge.xethub.hf.co` as the research
  reports assumed; the Xet backend moved).
- The redirect response itself carries CORS:
  `access-control-allow-origin: https://example.com`,
  `access-control-expose-headers: ... Accept-Ranges, Content-Range, ETag ...`
- Final CDN response: **HTTP 206** with
  `content-range: bytes 0-1023/484220320`, `accept-ranges: bytes`,
  `access-control-allow-origin: *`, `access-control-expose-headers: *`.
- 1024 bytes actually delivered. Stable `etag`
  `0552b0957f683d7cb507b4ce82b13e749c24954cff625a0a6c6bdedff8db9f75`
  (the Xet content hash — good cache key).

**Implications for the loader:**
- Plain `fetch()` with a `Range` header from the browser works cross-origin. No
  proxy, no server component.
- `Content-Range` is readable from JS (wildcard `expose-headers`), so the loader
  can learn total size from the first ranged request.
- Follow the 302 manually or let `fetch` do it — CORS survives the hop.
- Cache validation: use the `etag` (Xet hash). It's immutable per content, so a
  changed etag == changed file.
- The Jan 2026 CORS-on-OPTIONS bug flagged as "unverified" in the research is
  **resolved** on this path.

---

## Spike 2 — Q4_K dequant correctness

**PASS — all three legs bit-exact** (2026-09-03)

Built: `gguf.rs` (minimal GGUF v3 reader), `dequant.rs` (CPU reference, ported
line-by-line from ggml `dequantize_row_q4_K` / `get_scale_min_k4`),
`shaders/dequant_q4k.wgsl` (one invocation per 256-elem super-block),
`kiln tensors` + `kiln dequant` CLI, `scripts/oracle_q4k.py` (independent check
against the `gguf` Python package).

Tested on real Qwen3-0.6B-Q4_K_M weights (`blk.0.attn_k.weight` [1024×1024],
`blk.0.attn_q.weight` [1024×2048], `blk.5.ffn_gate.weight`, `blk.13.ffn_up.weight`):

| Comparison | max abs err | max rel err |
|---|---|---|
| kiln WGSL kernel vs kiln CPU reference | `0.0` | `0.0` |
| kiln CPU reference vs `gguf` package (independent impl) | `0.0` | `0.0` |

Bit-exact on both legs — not just "within tolerance." The CPU reference is a
trusted oracle (matches a separately-authored implementation), and the WGSL
kernel matches the CPU reference exactly. **The silent-garbage failure mode is
closed for Q4_K.**

Repro:
```
KILN_DUMP=/tmp/k.f32 cargo run -p kiln-cli -- dequant ~/models/Qwen3-0.6B-Q4_K_M.gguf blk.0.attn_k.weight
# oracle needs: uv pip install --python <venv> gguf numpy
python scripts/oracle_q4k.py ~/models/Qwen3-0.6B-Q4_K_M.gguf blk.0.attn_k.weight /tmp/k.f32
```

Synthetic unit tests (`cargo test -p kiln-core`) cover the zero-scale and
identity-nibble cases and run in CI without the model file.

**Notes for the real engine:**
- The GPU path here round-trips through a MAP_READ buffer every call — fine for a
  diff harness, must not survive into the forward pass (dequant output stays on
  the GPU, fused into matmul).
- `d`/`dmin` are decoded with `unpack2x16float` in WGSL — confirmed to match the
  Rust `f16_to_f32`. No `shader-f16` feature needed for the *unpack*.
- Q6_K CPU dequant added later (feat/gguf-metadata-q6k-q8): bit-exact vs the
  `gguf` package on `output.weight` (Q6_K LM head, 155M elems) and `attn_v`.
  Q8_0 covered by unit test (no Q8_0 tensor in Qwen3-0.6B). WGSL kernels for
  Q6_K/Q8_0 land with the forward-pass kernel PRs.

---

## Spike 3 — Approach validation via wgpu-llm

**PASS** (2026-09-04)

`wgpu-llm` built clean (wgpu v29, 12 WGSL shaders). Ran TinyLlama-1.1B-Chat f16
on the RX 9070 XT:

```
Generated 128 tokens in 1.53s (83.6 tok/s)
forward(avg=2.13 ms, steps=141)
argmax/readback(avg=8.71 ms, steps=141)
```

- **A hand-rolled wgpu + WGSL transformer forward pass reaches ~84 tok/s native**
  on this GPU (README: 66 on RTX 3090). The architecture kiln plans — per-layer
  WGSL dispatches, KV paging, f16 storage / f32 compute, dequant-in-shader — is
  proven viable and fast.
- **Readback is 4× the compute.** 8.71 ms per token pulling logits to CPU for
  sampling vs 2.13 ms for the forward pass. Confirms Maczan (arXiv 2608.08730)
  and the CLAUDE.md rule: sample on-GPU (GPU argmax for greedy), never sync
  per-token. Removing readback would lift the native ceiling to ~470 tok/s →
  ~47 tok/s even at a 10× browser penalty, inside the 40–100 target.
- **wgpu-llm has no wasm target** (`cli` + `core` crates only, no web-sys). The
  "run it in-browser" half of this spike isn't possible without porting it —
  which is kiln. Spike #1 already proved wgpu compute + f16 run in Zen and
  Chromium, so that risk is covered.
- wgpu-llm uses **safetensors**, not GGUF, and is **Llama-arch only** — it's a
  reference to read (shader structure, KV paging, f16 packing), not a base to
  fork for a GGUF + Qwen3 engine.

## All four spikes clear — greenlight to build

| Spike | Result |
|---|---|
| #1 WebGPU reality | wgpu compute + f16 work native, in Zen, in Chromium. No subgroups via wgpu-wasm. 128 MiB binding baseline. |
| #2 Q4_K dequant | bit-exact GPU == CPU == gguf package on real weights |
| #3 approach validation | hand-rolled wgpu+WGSL LLM = 84 tok/s native; readback is the bottleneck, not kernels |
| #4 HF Range/CORS | plain browser fetch() Range works cross-origin, no proxy |

Design constraints locked: GGUF Q4_K_M + Q8_0, f16 storage / f32 compute,
scalar reductions (no subgroups), single CommandEncoder per token, GPU-side
sampling, 128 MiB max binding, OPFS streaming loader.

---

## Phase 1 — CPU reference forward pass

**PASS — matches llama.cpp** (2026-09-04)

`kiln-core::model::Model` — scalar-f32 Qwen3 forward pass (GQA, per-head
QK-RMSNorm, NEOX RoPE θ=1e6, SwiGLU, no biases, untied LM head). Weights
dequantized to f32 up front (~2.5 GB RAM for Qwen3-0.6B).

Validation: token ids from `llama-tokenize`, greedy continuation vs
`llama-simple` on the same GGUF.

```
prompt : "The capital of France is"  → ids [785, 6722, 315, 9625, 374]
llama.cpp : "Paris. The capital of"
kiln      : [12095, 13, 576, 6722, 315]  = " Paris" "." " The" " capital" " of"
```

5/5 greedy tokens identical. The Qwen3 architecture port is correct.

- `ops.rs` unit tests (rmsnorm, matmul_vec, rope, softmax, silu) + dequant tests:
  10 pass in CI without the model file.
- Perf: ~10 s/forward for a 5-token prompt in release (no KV cache, naive scalar
  matmul). Fine for an oracle; this is the thing the GPU replaces.
- llama.cpp oracle now at `~/Projects/_ref/llama.cpp` — see CLAUDE.md.
