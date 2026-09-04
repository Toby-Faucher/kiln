#!/usr/bin/env python3
"""Independent Q4_K dequant oracle.

Uses the `gguf` Python package (llama.cpp project, separate codebase from kiln's
Rust port) to dequantize a tensor, then compares against kiln's CPU output dumped
via KILN_DUMP.

  KILN_DUMP=/tmp/kiln_cpu.f32 cargo run -p kiln-cli -- dequant model.gguf TENSOR
  ./scripts/oracle_q4k.py model.gguf TENSOR /tmp/kiln_cpu.f32
"""
import sys
import numpy as np
import gguf
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

model, tensor_name, kiln_dump = sys.argv[1], sys.argv[2], sys.argv[3]

reader = gguf.GGUFReader(model)
t = next(t for t in reader.tensors if t.name == tensor_name)
assert t.tensor_type == GGMLQuantizationType.Q4_K, f"{tensor_name} is {t.tensor_type}"

# t.data is the raw quantized bytes; dequantize() is gguf's own reference impl.
ref = dequantize(t.data, GGMLQuantizationType.Q4_K).astype(np.float32).ravel()
kiln = np.fromfile(kiln_dump, dtype="<f4")

assert kiln.size == ref.size, f"size mismatch: kiln {kiln.size} vs gguf {ref.size}"

abs_err = np.abs(kiln - ref)
denom = np.maximum(np.maximum(np.abs(kiln), np.abs(ref)), 1e-6)
rel_err = abs_err / denom

print(f"elements       : {ref.size}")
print(f"gguf[:6]       : {ref[:6]}")
print(f"kiln[:6]       : {kiln[:6]}")
print(f"max abs error  : {abs_err.max():.3e}")
print(f"max rel error  : {rel_err.max():.3e}")
print(f"mismatched >1e-4: {(rel_err > 1e-4).sum()}")

if rel_err.max() > 1e-4:
    sys.exit("FAIL: kiln CPU reference disagrees with the gguf package")
print("PASS: kiln CPU reference matches the independent gguf implementation")
