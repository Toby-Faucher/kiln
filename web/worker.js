// Dedicated Worker that hosts the kiln WASM module.
// Build the module first:  wasm-pack build crates/kiln-wasm --target web --out-dir ../../web/pkg
import init, { probe } from "./pkg/kiln_wasm.js";

let ready = init();

self.onmessage = async (e) => {
  if (e.data !== "probe") return;
  try {
    await ready;
    self.postMessage(await probe());
  } catch (err) {
    self.postMessage("probe failed: " + (err?.message ?? err));
  }
};
