# kiln

**Fire a language model in your browser.**

An in-browser LLM chat runtime in Rust → WASM + WebGPU. Hand-written WGSL compute
kernels, GGUF Q4_K_M weights, no framework.

The maintained browser-LLM runtimes (WebLLM, wllama, transformers.js) are all
JS/C++/WASM. HuggingFace's Rust attempt, `ratchet`, has been dormant since
November 2024. `kiln` is a from-scratch Rust take on the same problem.

## Status

Pre-code. De-risking spikes in progress — see [`CLAUDE.md`](CLAUDE.md) for the
architecture, the model ladder, and what's been decided.

## Non-goals

- Being the fastest option. If you need in-browser inference *now*, use
  [`wllama`](https://github.com/ngxson/wllama) or
  [transformers.js](https://github.com/huggingface/transformers.js).
- Firefox parity. Firefox/Zen WebGPU is 10–50× slower for this workload; kiln is
  Chrome-first with a fallback message elsewhere.

## License

MIT OR Apache-2.0
