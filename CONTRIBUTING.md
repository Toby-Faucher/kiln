# Contributing to kiln

## Branch flow

```
feature branch --PR--> dev --PR--> alpha --PR--> prod
```

- **`dev`** — default branch. Feature branches PR here. Squash-merge. CI green required.
- **`alpha`** — pre-release integration. `dev` promotes up via merge-commit PR at a milestone.
- **`prod`** — release. Fast-forward from `alpha` only. Tagged per release.

Never push directly to `dev`, `alpha`, or `prod`.

## Local checks (what CI runs)

```
cargo fmt --all --check
cargo clippy --workspace --exclude kiln-wasm --all-targets -- -D warnings
cargo test --workspace --exclude kiln-wasm
wasm-pack build crates/kiln-wasm --target web --dev
```

## Kernel rule

Every WGSL kernel ships with a CPU reference implementation and a test that diffs
the two (relative error < 1e-3 for dequant, < 1e-2 for late-layer activations).
No exceptions — this is the project's whole correctness story.

## Commits

`area: summary` — areas are `core`, `wasm`, `cli`, `web`, `docs`, `ci`.
