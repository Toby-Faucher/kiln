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

**TODO.** The highest-risk item. Plan:

1. Minimal GGUF v3 reader: header + tensor directory + locate one `Q4_K` tensor
   (e.g. a `blk.0.ffn_down.weight`) in the Qwen3-0.6B file.
2. CPU reference dequant of Q4_K (256-elem super-blocks: 8×6-bit scales +
   8×6-bit mins packed, `d`/`dmin` f16, 4-bit quants). Port from the ggml
   `dequantize_row_q4_K` spec, commented.
3. WGSL dequant kernel, same math.
4. `kiln dequant` diffs GPU vs CPU element-wise; assert max relative error
   < 1e-3.
5. Cross-check the CPU reference against `llama.cpp`'s own dequant
   (`llama-quantize --dry-run` style, or a tiny C harness) so the oracle is
   trusted, not just self-consistent.

---

## Spike 3 — Approach validation via wgpu-llm

**TODO.** Fork `Beledarian/wgpu-llm`, run native (expect ~25–66 tok/s per its
README), then get it in-browser Chrome with one `CommandEncoder` per token.
Confirms the whole architecture before investing in Q4_K.
