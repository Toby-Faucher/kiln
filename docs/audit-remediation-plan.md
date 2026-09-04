# Audit remediation plan — handoff

**Date:** 2026-09-04
**Source:** `docs/audit-2026-09-04.md` (8-subagent orchestrated review; 1 Critical,
13 High, 33 Medium, 24 Low).

This document turns the audit into an ordered set of PRs with concrete proposed
changes. Pick up here; each PR section is self-contained.

---

## Repo state at handoff

- `dev` HEAD: `b6f54e4` — through PR #8 (CPU reference forward pass).
- **PR #9 open, unmerged** (`feat/wgsl-kernels-embed-rmsnorm`): adds
  `gpu_ops.rs` + `shaders/rmsnorm.wgsl` + the `#[ignore]`d diff test. CI green.
  Several audit findings are *about* this branch's code (H13, H14, M18, M19,
  M33). **Merge #9 first**, then address in the remediation PRs below — do not
  rebase the fixes into #9.
- `~/Projects/_ref/llama.cpp` — built, the numeric oracle.
- `~/Projects/_ref/wgpu-llm` — reference WGSL engine.

## The one thing the audit confirmed is *fine*

The numeric core. Q4_K/Q6_K/Q8_0 dequant match ggml statement-for-statement;
`f16_to_f32` is bit-exact over all 65536 inputs; the Qwen3 forward pass (NEOX
RoPE, QK-RMSNorm-before-RoPE, GQA head mapping, attention scale, SwiGLU, LM-head
orientation, no biases) matches the reference graph. **No numeric bug.** Every
PR below is about robustness, error handling, browser limits, and test coverage
— not correctness of the math.

---

## PR order and rationale

| # | PR | Why this slot |
|---|---|---|
| R1 | Error paths → `Result` (GPU + wasm + model) | Signature changes — do before more code piles on; unblocks the wasm error story |
| R2 | Test coverage + CI | Pins prose-only correctness before regressions creep in; all additive |
| R3 | GGUF parser hardening (untrusted input) | Needed before the OPFS/network loader (Phase 2); safe to do now |
| R4 | 128 MiB / dispatch-limit guards | Can also fold into the matmul-kernel PR (it must chunk anyway) |
| R5 | Module boundary enforcement (`cfg` wasm) | Small, preventive |
| R6 | Docs / consistency cleanup | Quick, do anytime |

---

## R1 — Error paths return `Result`, no panics on the GPU/wasm path

**Findings:** H1, H2, H11, M13, M14, M18 (F6), M20, M21, M29
**Files:** `error.rs`, `compute.rs`, `gpu_ops.rs`, `model.rs`, `kiln-wasm/src/lib.rs`, `kiln-cli/src/main.rs`
**Size:** M

Under workspace `panic = "abort"`, any `expect()`/`assert!` on a GPU or wasm path
traps the Dedicated Worker instead of rejecting the JS promise that
`web/worker.js` is already set up to handle.

Proposed changes:

1. **`error.rs`** — add variants:
   ```rust
   #[error("GPU error: {0}")]
   Gpu(String),
   #[error("map failed: {0}")]
   Map(#[from] wgpu::BufferAsyncError),
   ```
   (M29 — the missing GPU variant is *why* `compute.rs` reaches for `expect`.)

2. **`compute.rs::read_buffer`** → `pub(crate) async fn read_buffer(..) -> Result<Vec<u8>>`.
   - `rx.await` cancel → `Error::Gpu("map callback dropped")`.
   - callback `Result` → `?` via `Error::Map`.
   - `get_mapped_range()` → `Error::Gpu(...)`.
   - Keep `device.poll(...)` but `#[cfg(not(target_arch = "wasm32"))]`-gate it
     (M14 — on WebGPU it's a documented no-op; gating removes the misleading
     "poll then await" on the wasm build).

3. **`compute.rs::hello_compute`** → `-> Result<Vec<f32>>` (H2). Add an
   `if input.is_empty() { return Err(Error::Shape("empty input".into())) }`
   guard (M13 — 0-byte buffers violate `map_async`'s multiple-of-4 contract).

4. **`gpu_ops.rs::rmsnorm`** → `-> Result<Vec<f32>>`, replace the `assert!` with
   `Error::Shape` (M18/F6). Same empty guard.

5. **`model.rs`**:
   - `forward(&self, tokens: &[u32]) -> Result<Vec<f32>>` (H11). Guard
     `tokens.is_empty()` (M20) and `tokens.iter().any(|&t| t as usize >= cfg.vocab_size)`.
   - `generate` propagates the `Result`.
   - `Model::load` — validate `cfg` after `g.config()?`: `n_heads > 0`,
     `n_kv_heads > 0`, `n_heads % n_kv_heads == 0`, `d_ff > 0`, `d_model > 0`
     (M21 — GQA math at `model.rs:141` panics on `n_kv_heads == 0` or uneven
     division). Return `Error::Gguf(...)`.

6. **Callers**: `kiln-wasm/src/lib.rs::probe` — `hello_compute(..).await?` maps
   to `JsError` (already the pattern for `gpu::acquire`). `kiln-cli/src/main.rs`
   — `?` through `anyhow`.

**Verify:** `cargo test`, `cargo clippy -D warnings`, `kiln probe`,
`kiln forward … --tokens <in-range>` still works; `kiln forward … --tokens 9999999`
now prints an error instead of panicking.

---

## R2 — Test coverage + CI

**Findings:** H12, H13, H14, M6, M11, M12, M24, M25, M33, M35, M38
**Files:** `dequant.rs`, `model.rs` (new `#[cfg(test)]`), `gguf.rs` (new tests),
`gpu_ops.rs`, `ops.rs`, `.github/workflows/ci.yml`
**Size:** L (all additive)

The correctness story currently rests on manual runs and prose ("5/5 greedy
tokens"). Pin it.

Proposed changes:

1. **`gpu_ops.rs`** — parameterise `rmsnorm_matches_cpu` over a shape table
   `[(1,1),(3,1024),(7,1),(2,255),(1,4096)]` × 3 seeds; add `eps = 0.0` case
   (M33).
2. **`dequant.rs`** — add `#[ignore] fn q4k_matches_cpu()` mirroring
   `rmsnorm_matches_cpu` (H14). Extend `identity_low_nibbles` → distinct
   per-sub-block scales asserting all 256 outputs (M38). Add error-path tests
   for the `Err` branches (M12). Add `dequant_cpu` dispatcher tests routing
   F32/F16/Q8_0 (M11).
3. **`dequant.rs::max_error`** — `assert_eq!(a.len(), b.len())` first (M9 — it
   currently `zip`-truncates and returns `(0,0)` on a shape mismatch, which is
   *exactly* the most likely kernel bug). Add a test.
4. **`model.rs`** — new `#[cfg(test)] mod tests`: build a **1-layer synthetic
   model in memory** (hand-set tiny weights, no GGUF) and assert golden logits /
   that `forward` is deterministic and finite (H14, M25). Does not need a model
   file → runs in CI.
5. **`gguf.rs`** — new tests: hand-assemble a minimal valid GGUF v3 blob in a
   `Vec<u8>`; assert `parse` handles bad magic, bad version, truncation,
   `alignment == 0`, missing required keys, duplicate keys (M6). These also
   cover R3's fixes.
6. **`ops.rs`** — tests for `dot` (incl. empty) and `argmax` (ties → first,
   NaN via `total_cmp`) (M24). `f16_to_f32` table test:
   `0x0000/0x8000/0x3C00/0xBC00/0x0001/0x7C00/0xFC00/0x7E00` (M25).
7. **`ci.yml`**:
   - pin `wasm-pack` to `0.15.0` in the install step (M35).
   - add a `wasm-pack build crates/kiln-wasm --target web --release` step so the
     release `wasm-opt` `--enable-simd` path is actually exercised (H12).
   - GPU tests stay `#[ignore]`d; add a comment that they run via
     `cargo test -p kiln-core -- --ignored` on a machine with an adapter, and
     open a tracking note for a Lavapipe (`WGPU_BACKEND=vulkan`,
     `LIBGL_ALWAYS_SOFTWARE=1`) CI job (H13).

**Verify:** `cargo test --workspace --exclude kiln-wasm` count goes up
substantially; `cargo test -p kiln-core -- --ignored` passes locally on the dev
GPU; CI wasm-release step green.

---

## R3 — GGUF parser hardening (untrusted input)

**Findings:** H3, H4, H5, H6, H7, H8, H9, H10, M1, M2, M3, M4, M5, L2, L7, L8, L9
**Files:** `gguf.rs`
**Size:** M

GGUF files will come from the network (HF) into a browser. The parser must treat
every length/count/offset field as hostile.

Proposed changes:

1. **`Cursor::take`** — `self.p.checked_add(n).and_then(|end| self.b.get(self.p..end))`
   → `Error::Gguf("truncated")` (H3).
2. **Every `u64 → usize`** (`tensor_count`, `kv_count`, `n_dims`, string len,
   array count, `t.offset`) → `usize::try_from(v).map_err(|_| Error::Gguf(...))`
   (H4 — silently truncates on wasm32).
3. **Count caps before `with_capacity`** (H5): `kv_count`/`tensor_count ≤ 1<<20`,
   `n_dims ≤ 4` (ggml `GGML_MAX_DIMS`), array `count ≤ 1<<28`. Or use
   `Vec::try_reserve` → `Err`.
4. **`raw()`** — `checked_add` on `data_start + offset` and `start + n_bytes`;
   verify against `self.bytes.len()` (H6 — wrap currently can return header
   bytes as "weights", silently).
5. **`TensorInfo::n_elements` / `n_bytes`** — `checked_mul` fold over `dims`,
   `checked` div-ceil, `try_from` (H7 — `[2^32, 2^32, 64]` wraps to 0 →
   `raw()` returns empty slice as success). Make these `-> Result<usize>` or
   return `Option` and propagate.
6. **`alignment`** — reject `a == 0 || a > (1 << 30)`; default 32 (H8 —
   `div_ceil(0)` panics).
7. **`MetaValue::as_i64`** — the `U64` arm: `i64::try_from(v).ok()` instead of
   `v as i64` (H9 — `U64 > i64::MAX` wraps negative → `usize::MAX` capacity).
8. **`MetaValue::as_i64`** — drop the `Bool` arm (L7 — `block_count = true`
   currently parses as 1 layer).
9. **`config()`** (H10, M2–M5):
   - require `n_heads > 0` before the `d_model / n_heads` fallback;
   - `t.dims.get(1)` not `t.dims[1]` for the vocab fallback;
   - make `attention.head_count_kv` **required** (M5 — missing silently
     degrades GQA→MHA);
   - default `rope.freq_base` to `1_000_000.0` **and** emit a warning, or
     require it (M2 — 10k default is wrong for Qwen3);
   - consider making `attention.key_length` required (M4 — the `d_model/n_heads`
     fallback gives 64 vs Qwen3's true 128; wrong logits, in-bounds).
     *Check a real Qwen3 GGUF metadata dump first — if the key is always
     present, keep the fallback but log when it's used.*
   - validate `n_heads % n_kv_heads == 0` and all dims `> 0`.
10. **`data_start` past EOF** — one comparison at end of `parse` (M1).
11. **Duplicate keys** — `HashMap::entry` check → `Error::Gguf` (L8).
12. **Error detail** — split "missing key" vs "wrong type"; include
    `Cursor::p` where useful (L9). `(g.config()` errors currently all say
    `missing …`.)
13. **`gguf_string`** — `std::str::from_utf8(s).map(str::to_owned)` (L-nit,
    skips an intermediate `Vec`).

**Verify:** the R2 in-memory GGUF tests (bad magic / truncation / alignment 0 /
missing keys / hostile counts) all return `Err`, never panic; `kiln config`,
`kiln tensors`, `kiln forward` on the real Qwen3-0.6B GGUF unchanged.

---

## R4 — 128 MiB binding + dispatch-limit guards

**Findings:** C1, M15, M16, M17, M19
**Files:** `gpu.rs`, `dequant.rs`, `gpu_ops.rs`, `compute.rs`
**Size:** S–M

Real Chrome/Safari cap `maxStorageBufferBindingSize` at 128 MiB and
`maxComputeWorkgroupsPerDimension` at 65535. The dev GPU reports multi-GiB and
hides the cliff. The embed / LM-head tensors are ~600 MB f32.

Proposed changes:

1. **`gpu.rs`** — add `pub async fn acquire_pinned(want_f16: bool) -> Result<GpuContext>`
   that requests `wgpu::Limits::default()` (the WebGPU spec baseline) field-wise
   `.min()`-clamped against `adapter.limits()`, returning `Error::Gpu` if the
   adapter can't meet baseline (M15, M16). Keep `acquire` (raw adapter limits)
   for the probe/spike. Engine code uses `acquire_pinned`.
2. **Buffer-size guard** — a `pub(crate) fn check_binding(ctx, bytes) -> Result<()>`
   comparing against `ctx.report.max_storage_buffer_binding_size` /
   `max_buffer_size`; call it in `q4k_gpu` (C1), `gpu_ops::rmsnorm` (M19), and
   every future kernel before `create_buffer`. Return `Error::Shape("tensor
   exceeds 128 MiB binding — needs chunking")`.
3. **Dispatch guard** — assert/return-`Err` when any `dispatch_workgroups`
   dimension would exceed 65535 (M17): `q4k_gpu` at `n_blocks > 4_194_240`,
   `rmsnorm` at `n_rows > 65535`, `hello_compute` at `len > 4_194_240`.

**Note:** the real fix for C1 is the chunked matmul-with-fused-dequant kernel
(upcoming Phase-1 work). This PR just makes the current code *fail cleanly* on
a big tensor instead of silently working on the dev box and breaking in Chrome.
It is fine to fold R4 into that kernel PR instead of shipping it standalone.

---

## R5 — Module boundary enforcement

**Findings:** M26, M27 (F-1)
**Files:** `lib.rs`, `gguf.rs`, `model.rs`
**Size:** S

"Never load the model into the WASM heap" (CLAUDE.md) holds today only because
nothing imports `model`/`Gguf::open` from `kiln-wasm` — not because the type
system forbids it.

Proposed changes:

1. **`lib.rs`** — `#[cfg(not(target_arch = "wasm32"))] pub mod model;` (the
   full-model f32 oracle has no place in a 4 GiB no-Memory64 linear address
   space). Keep it `pub(crate)`-visible to in-crate tests via
   `#[cfg(any(not(target_arch = "wasm32"), test))]` if needed.
2. **`gguf.rs`** — `#[cfg(not(target_arch = "wasm32"))] pub fn open(path)` (it's
   the only `std::fs` in the crate). Add
   `pub fn from_bytes(bytes: Vec<u8>) -> Result<Self>` (rename/expose the
   existing private `parse`) so the future OPFS/streaming loader has the right
   API (M26).
3. Keep `Config`, `TensorInfo`, `GgmlType`, `MetaValue` unconditionally `pub` —
   the wasm loader will want them.
4. Add a `cargo build --target wasm32-unknown-unknown -p kiln-core` check to CI
   (or a note) proving `model`/fs don't leak.

---

## R6 — Docs / consistency cleanup

**Findings:** M31, M32, M36, L4, L5, L6, L16–L23, plus the CLAUDE.md deviations §6
**Files:** `CLAUDE.md`, `CONTRIBUTING.md`, `docs/spikes.md`, `oracle_dequant.py`,
`lib.rs`, `crates/kiln-cli/src/main.rs` (module doc), `gguf.rs`/`dequant.rs`/`gpu.rs` (doc comments)
**Size:** S

1. **One tolerance table** in `CLAUDE.md` (leg × threshold × rationale) and
   reference it from every assert site (M31): CPU-vs-gguf-package `0.0`;
   WGSL-vs-CPU isolated op `< 1e-5`; end-to-end late-layer activation `< 1e-2`;
   greedy-token match exact. Fix the "bit-identity only for isolated fp32 ops"
   line that `rmsnorm`-at-`1e-5` contradicts (M32).
2. **`CLAUDE.md`** — rewrite the stale "Open: #1 browser half, #3, Q6_K" line
   (all PASS now); drop the `(todo)` from the repo-layout entries that exist
   (L18, L19); fix the "not compile-verified" `gpu.rs` warning — spike #1
   compiled and ran against 30.0.1 (L4).
3. **Scope the kernel rule** (M36) — "every *inference* WGSL kernel is diffed
   against `ops.rs`; the `hello_compute` probe is exempt." Fix `gpu_ops.rs` /
   `ops.rs` module docs that overclaim.
4. **`docs/spikes.md`** — `oracle_q4k.py` → `oracle_dequant.py` (L16); note
   Q6_K "bit-exact" is CPU-only, no WGSL leg yet (L16/M32); scope the "bit-exact
   from 4 tensors" claim with date + sample size.
5. **`oracle_dequant.py:11`** docstring — "GPU comparison is Q4_K-only" not
   "only dumps Q4_K" (L17).
6. **`lib.rs`** — either wire `shaders::RMSNORM` (route `gpu_ops.rs` /
   `dequant.rs` `include_str!` through the `shaders` module) or delete the dead
   `DEQUANT_Q4K` const (L6).
7. **`gguf.rs:1-3`** doc — "metadata skipped, not decoded" is now false
   (`config()` decodes plenty) (L4).
8. **`ops.rs:20-22`** — the ggml layout is not "row-major"; reword (L5, code is
   correct).
9. **`CONTRIBUTING.md` / `CLAUDE.md`** — reconcile the commit-area list (`web:`
   in one, not the other) (L21).
10. **CLI module doc** (`main.rs:1-8`) — add the `config` / `forward` /
    `dequant --all` lines (L20).

---

## Deferred / non-goals

- **Lavapipe GPU CI job** — tracked, not in these PRs. Until then GPU tests are
  `#[ignore]`d + run manually (documented).
- **`Vec<Vec<f32>>` → flat buffers in `model.rs`** (L11) — intentional oracle
  perf debt; the GPU engine replaces this code path entirely. Leave it.
- **`half` crate for `f16_to_f32`** (L3) — the hand-rolled version is
  exhaustively bit-exact; a new direct dep buys nothing. Leave it.
- **YaRN / rope-scaling** (M30) — no in-scope Qwen3 GGUF ships scaling metadata;
  add "error on non-`none` scaling type" as a one-liner in R3's `config()` and
  implement YaRN only if a target model needs it.
- **`panic = "abort"` scoping** (M28, L23) — revisit when `cargo test --release`
  or a wasm-unwind decision actually comes up; not blocking.
- **`Backend` trait has no impls** (L12) — deliberate seam; leave.

---

## Suggested execution

1. Merge PR #9 as-is.
2. R1 → R2 → R3 as separate PRs into `dev` (each with its own CI run).
3. R4 folded into the matmul-kernel PR, or standalone if that's further out.
4. R5, R6 anytime (small).
5. After R1–R3 land, tag `alpha` — this is a natural "hardened foundation"
   milestone before the Phase-1 kernel push resumes.
