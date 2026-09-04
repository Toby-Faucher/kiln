#!/usr/bin/env python3
"""Independent dequant oracle.

Uses the `gguf` Python package (llama.cpp project, separate codebase from kiln's
Rust ports) to dequantize a tensor, then compares against kiln's CPU output
dumped via KILN_DUMP.

  KILN_DUMP=/tmp/kiln.f32 cargo run -p kiln-cli -- dequant model.gguf TENSOR
  ./scripts/oracle_dequant.py model.gguf TENSOR /tmp/kiln.f32

(`kiln dequant` only dumps Q4_K today; for other dtypes dump from a scratch
harness or extend the CLI.)
"""
import sys
import numpy as np
import gguf
from gguf.quants import dequantize

model, tensor_name, kiln_dump = sys.argv[1], sys.argv[2], sys.argv[3]

reader = gguf.GGUFReader(model)
t = next(t for t in reader.tensors if t.name == tensor_name)

ref = dequantize(t.data, t.tensor_type).astype(np.float32).ravel()
kiln = np.fromfile(kiln_dump, dtype="<f4")

assert kiln.size == ref.size, f"size: kiln {kiln.size} vs gguf {ref.size}"

abs_err = np.abs(kiln - ref)
denom = np.maximum(np.maximum(np.abs(kiln), np.abs(ref)), 1e-6)
rel_err = abs_err / denom

print(f"tensor         : {tensor_name} ({t.tensor_type.name})")
print(f"elements       : {ref.size}")
print(f"gguf[:6]       : {ref[:6]}")
print(f"kiln[:6]       : {kiln[:6]}")
print(f"max abs error  : {abs_err.max():.3e}")
print(f"max rel error  : {rel_err.max():.3e}")

if rel_err.max() > 1e-4:
    sys.exit("FAIL: kiln CPU reference disagrees with the gguf package")
print("PASS: kiln CPU reference matches the independent gguf implementation")
