//! Scalar f32 CPU ops for the reference forward pass.
//!
//! These are the oracle: correctness over speed, one obvious implementation per
//! op. Every WGSL kernel is diffed against the matching function here.

/// RMSNorm: `y = x / sqrt(mean(x^2) + eps) * w`. Qwen-style (no `+1` on `w`).
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = (mean_sq + eps).sqrt().recip();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * inv * wi;
    }
}

/// `out[o] = sum_k w[o * k_in + k] * x[k]`.
///
/// Matches ggml `mul_mat`: the weight tensor is laid out `[k_in, n_out]`
/// (ne[0] = input features, ne[1] = output features), row-major, so output
/// row `o` is the contiguous slice `w[o*k_in .. (o+1)*k_in]`.
pub fn matmul_vec(w: &[f32], x: &[f32], k_in: usize, n_out: usize, out: &mut [f32]) {
    debug_assert_eq!(w.len(), k_in * n_out);
    debug_assert_eq!(x.len(), k_in);
    debug_assert_eq!(out.len(), n_out);
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &w[o * k_in..(o + 1) * k_in];
        *slot = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// NEOX-style rotary embedding, in place, on one head vector at `pos`.
///
/// Pairs dimension `i` with `i + head_dim/2` (the "rotate-half" layout Qwen2/3
/// and GPT-NeoX use — *not* the interleaved LLaMA-original layout).
pub fn rope_neox(v: &mut [f32], pos: usize, theta: f32) {
    let head_dim = v.len();
    let half = head_dim / 2;
    for i in 0..half {
        let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
        let (sin, cos) = (pos as f32 * freq).sin_cos();
        let a = v[i];
        let b = v[i + half];
        v[i] = a * cos - b * sin;
        v[i + half] = b * cos + a * sin;
    }
}

/// Numerically-stable softmax, in place.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SiLU / swish: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Plain dot product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Index of the largest element.
pub fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weight() {
        // x with mean-square 1 and unit weight, eps 0 => y == x.
        let x = [1.0, -1.0, 1.0, -1.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        rmsnorm(&x, &w, 0.0, &mut out);
        for (o, xi) in out.iter().zip(x) {
            assert!((o - xi).abs() < 1e-6);
        }
    }

    #[test]
    fn matmul_vec_identity() {
        // 3x3 identity as [k_in=3, n_out=3] row-major.
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = [7.0, 8.0, 9.0];
        let mut out = [0.0; 3];
        matmul_vec(&w, &x, 3, 3, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let orig = v;
        rope_neox(&mut v, 0, 1_000_000.0);
        for (a, b) in v.iter().zip(orig) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_pair_norm() {
        let mut v: [f32; 6] = [1.0, 2.0, 3.0, 0.5, -1.0, 2.5];
        let n0 = (v[0] * v[0] + v[3] * v[3]).sqrt();
        rope_neox(&mut v, 5, 1_000_000.0);
        let n1 = (v[0] * v[0] + v[3] * v[3]).sqrt();
        assert!((n0 - n1).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > x[0]);
    }

    #[test]
    fn silu_zero() {
        assert!(silu(0.0).abs() < 1e-9);
    }
}
