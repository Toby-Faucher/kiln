//! CPU reference forward pass for Qwen3-style decoder models.
//!
//! This is the oracle, not the engine. Weights are dequantized to f32 up front;
//! everything runs in scalar f32 with the [`ops`](crate::ops) functions. Slow
//! and obvious on purpose — the WGSL kernels are diffed against this.
//!
//! Qwen3 specifics handled here:
//! - separate Q/KV head counts (GQA), explicit `head_dim` (not `d_model/n_heads`)
//! - **QK-RMSNorm**: per-head RMSNorm on Q and K (over `head_dim`) before RoPE
//! - NEOX rotary embedding, `rope_theta` from metadata
//! - SwiGLU FFN, RMSNorm pre-norm, no biases anywhere
//! - untied LM head (`output.weight`), falls back to `token_embd.weight` if absent

use crate::dequant::dequant_cpu;
use crate::gguf::{Config, Gguf};
use crate::ops::{argmax, dot, matmul_vec, rmsnorm, rope_neox, silu, softmax_inplace};
use crate::{Error, Result};

struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

pub struct Model {
    pub cfg: Config,
    tok_embd: Vec<f32>,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    lm_head: Vec<f32>,
}

impl Model {
    /// Dequantize every weight to f32 and build the model. Allocates the full
    /// model in RAM as f32 (~2.5 GB for Qwen3-0.6B — a reference, not the engine).
    pub fn load(g: &Gguf) -> Result<Self> {
        let cfg = g.config()?;
        let deq = |name: &str| -> Result<Vec<f32>> {
            let t = g
                .tensor(name)
                .ok_or_else(|| Error::Gguf(format!("missing tensor {name}")))?;
            dequant_cpu(t.ggml_type, g.raw(name)?, t.n_elements())
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            layers.push(Layer {
                attn_norm: deq(&p("attn_norm.weight"))?,
                wq: deq(&p("attn_q.weight"))?,
                wk: deq(&p("attn_k.weight"))?,
                wv: deq(&p("attn_v.weight"))?,
                wo: deq(&p("attn_output.weight"))?,
                q_norm: deq(&p("attn_q_norm.weight"))?,
                k_norm: deq(&p("attn_k_norm.weight"))?,
                ffn_norm: deq(&p("ffn_norm.weight"))?,
                w_gate: deq(&p("ffn_gate.weight"))?,
                w_up: deq(&p("ffn_up.weight"))?,
                w_down: deq(&p("ffn_down.weight"))?,
            });
        }

        let tok_embd = deq("token_embd.weight")?;
        let lm_head = if g.tensor("output.weight").is_some() {
            deq("output.weight")?
        } else {
            tok_embd.clone() // tied embeddings
        };

        Ok(Self {
            cfg,
            tok_embd,
            layers,
            out_norm: deq("output_norm.weight")?,
            lm_head,
        })
    }

    /// Run the full prompt through the model and return the logits
    /// (`vocab_size` of them) for the **last** position.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let seq = tokens.len();
        let hd = c.head_dim;
        let q_dim = c.n_heads * hd;
        let kv_dim = c.n_kv_heads * hd;
        let gqa = c.n_heads / c.n_kv_heads;
        let scale = (hd as f32).sqrt().recip();

        // Residual stream: one d_model vector per position.
        let mut h: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                let off = t as usize * c.d_model;
                self.tok_embd[off..off + c.d_model].to_vec()
            })
            .collect();

        let mut scratch_d = vec![0.0f32; c.d_model];
        let mut scratch_hd = vec![0.0f32; hd];

        for layer in &self.layers {
            let mut q_all = vec![vec![0.0f32; q_dim]; seq];
            let mut k_all = vec![vec![0.0f32; kv_dim]; seq];
            let mut v_all = vec![vec![0.0f32; kv_dim]; seq];

            // --- QKV projection + QK-norm + RoPE, per position ---
            for pos in 0..seq {
                rmsnorm(&h[pos], &layer.attn_norm, c.rms_eps, &mut scratch_d);
                matmul_vec(&layer.wq, &scratch_d, c.d_model, q_dim, &mut q_all[pos]);
                matmul_vec(&layer.wk, &scratch_d, c.d_model, kv_dim, &mut k_all[pos]);
                matmul_vec(&layer.wv, &scratch_d, c.d_model, kv_dim, &mut v_all[pos]);

                for head in 0..c.n_heads {
                    let qh = &mut q_all[pos][head * hd..(head + 1) * hd];
                    rmsnorm(qh, &layer.q_norm, c.rms_eps, &mut scratch_hd);
                    qh.copy_from_slice(&scratch_hd);
                    rope_neox(qh, pos, c.rope_theta);
                }
                for head in 0..c.n_kv_heads {
                    let kh = &mut k_all[pos][head * hd..(head + 1) * hd];
                    rmsnorm(kh, &layer.k_norm, c.rms_eps, &mut scratch_hd);
                    kh.copy_from_slice(&scratch_hd);
                    rope_neox(kh, pos, c.rope_theta);
                }
            }

            // --- causal GQA attention ---
            let mut attn_out = vec![0.0f32; q_dim];
            for pos in 0..seq {
                attn_out.iter_mut().for_each(|x| *x = 0.0);
                for head in 0..c.n_heads {
                    let kv_head = head / gqa;
                    let qh = &q_all[pos][head * hd..(head + 1) * hd];

                    let mut scores = vec![0.0f32; pos + 1];
                    for (j, s) in scores.iter_mut().enumerate() {
                        let kh = &k_all[j][kv_head * hd..(kv_head + 1) * hd];
                        *s = dot(qh, kh) * scale;
                    }
                    softmax_inplace(&mut scores);

                    let out = &mut attn_out[head * hd..(head + 1) * hd];
                    for (j, &w) in scores.iter().enumerate() {
                        let vh = &v_all[j][kv_head * hd..(kv_head + 1) * hd];
                        for (o, &vd) in out.iter_mut().zip(vh) {
                            *o += w * vd;
                        }
                    }
                }
                // output projection + residual
                matmul_vec(&layer.wo, &attn_out, q_dim, c.d_model, &mut scratch_d);
                for (hi, &o) in h[pos].iter_mut().zip(&scratch_d) {
                    *hi += o;
                }
            }

            // --- SwiGLU FFN + residual ---
            let mut gate = vec![0.0f32; c.d_ff];
            let mut up = vec![0.0f32; c.d_ff];
            for hp in h.iter_mut() {
                rmsnorm(hp, &layer.ffn_norm, c.rms_eps, &mut scratch_d);
                matmul_vec(&layer.w_gate, &scratch_d, c.d_model, c.d_ff, &mut gate);
                matmul_vec(&layer.w_up, &scratch_d, c.d_model, c.d_ff, &mut up);
                for (g, &u) in gate.iter_mut().zip(&up) {
                    *g = silu(*g) * u;
                }
                matmul_vec(&layer.w_down, &gate, c.d_ff, c.d_model, &mut scratch_d);
                for (hi, &d) in hp.iter_mut().zip(&scratch_d) {
                    *hi += d;
                }
            }
        }

        // --- final norm + LM head on the last position ---
        rmsnorm(&h[seq - 1], &self.out_norm, c.rms_eps, &mut scratch_d);
        let mut logits = vec![0.0f32; c.vocab_size];
        matmul_vec(
            &self.lm_head,
            &scratch_d,
            c.d_model,
            c.vocab_size,
            &mut logits,
        );
        logits
    }

    /// Greedy generate `n` tokens after the prompt.
    pub fn generate(&self, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut toks = prompt.to_vec();
        for _ in 0..n {
            let logits = self.forward(&toks);
            toks.push(argmax(&logits) as u32);
        }
        toks[prompt.len()..].to_vec()
    }
}
