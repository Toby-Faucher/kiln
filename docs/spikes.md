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

**Browser half: TODO.** Needs `rustup` + `wasm32-unknown-unknown` + `wasm-pack`,
then:

```
wasm-pack build crates/kiln-wasm --target web --out-dir ../../web/pkg
python -m http.server -d web    # or any static server
```

Open `localhost:8000` in **Chrome** and in **Zen**. Record both `probe()` JSON
blobs here. Key question: does Zen-on-Linux expose WebGPU at all? (Firefox Linux
WebGPU is Nightly-only as of Sept 2026; Zen tracks Firefox stable.)

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
- Q6_K (used for ~half the Qwen3-0.6B tensors — `attn_v`, `ffn_down` on many
  layers) is still TODO. Same harness, different block layout (210 bytes).

---

## Spike 3 — Approach validation via wgpu-llm

**TODO.** Fork `Beledarian/wgpu-llm`, run native (expect ~25–66 tok/s per its
README), then get it in-browser Chrome with one `CommandEncoder` per token.
Confirms the whole architecture before investing in Q4_K.
