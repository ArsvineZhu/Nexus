// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::RunOptions;
use crate::csr::TransportGraphView;
use crate::error::Error;
use wide::{f32x8, f64x4};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionMode {
    StrictLeftFold,
    #[default]
    FastSimd,
}

/// Checks finite inputs in contiguous blocks, independent of arithmetic precision.
pub fn all_finite_f32(values: &[f32]) -> bool {
    let (blocks, tail) = values.as_chunks::<8>();
    blocks
        .iter()
        .all(|&block| f32x8::new(block).is_finite().all())
        && tail.iter().all(|value| value.is_finite())
}

#[inline]
fn dot_f32_strict(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |acc, (&x, &y)| acc + f64::from(x) * f64::from(y))
}

#[inline]
fn dot_f32_simd(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    let mut acc0 = f64x4::ZERO;
    let mut acc1 = f64x4::ZERO;
    let (a_blocks, a_tail) = a.as_chunks::<8>();
    let (b_blocks, b_tail) = b.as_chunks::<8>();

    for (xa, xb) in a_blocks.iter().zip(b_blocks) {
        let va0 = f64x4::new([
            f64::from(xa[0]),
            f64::from(xa[1]),
            f64::from(xa[2]),
            f64::from(xa[3]),
        ]);
        let vb0 = f64x4::new([
            f64::from(xb[0]),
            f64::from(xb[1]),
            f64::from(xb[2]),
            f64::from(xb[3]),
        ]);
        acc0 = va0.mul_add(vb0, acc0);

        let va1 = f64x4::new([
            f64::from(xa[4]),
            f64::from(xa[5]),
            f64::from(xa[6]),
            f64::from(xa[7]),
        ]);
        let vb1 = f64x4::new([
            f64::from(xb[4]),
            f64::from(xb[5]),
            f64::from(xb[6]),
            f64::from(xb[7]),
        ]);
        acc1 = va1.mul_add(vb1, acc1);
    }

    let mut sum = (acc0 + acc1).reduce_add();
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        sum += f64::from(x) * f64::from(y);
    }
    sum
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], mode: PrecisionMode) -> f64 {
    match mode {
        PrecisionMode::StrictLeftFold => dot_f32_strict(a, b),
        PrecisionMode::FastSimd => dot_f32_simd(a, b),
    }
}

#[inline]
fn norm_sq_f32_strict(v: &[f32]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| {
        let xf = f64::from(x);
        acc + xf * xf
    })
}

#[inline]
fn norm_sq_f32_simd(v: &[f32]) -> f64 {
    let mut acc0 = f64x4::ZERO;
    let mut acc1 = f64x4::ZERO;
    let (blocks, tail) = v.as_chunks::<8>();

    for c in blocks {
        let v0 = f64x4::new([
            f64::from(c[0]),
            f64::from(c[1]),
            f64::from(c[2]),
            f64::from(c[3]),
        ]);
        acc0 = v0.mul_add(v0, acc0);

        let v1 = f64x4::new([
            f64::from(c[4]),
            f64::from(c[5]),
            f64::from(c[6]),
            f64::from(c[7]),
        ]);
        acc1 = v1.mul_add(v1, acc1);
    }

    let mut sum = (acc0 + acc1).reduce_add();
    for &x in tail {
        let xf = f64::from(x);
        sum += xf * xf;
    }
    sum
}

#[inline]
pub fn norm_sq_f32(v: &[f32], mode: PrecisionMode) -> f64 {
    match mode {
        PrecisionMode::StrictLeftFold => norm_sq_f32_strict(v),
        PrecisionMode::FastSimd => norm_sq_f32_simd(v),
    }
}

#[inline]
pub fn dot_f64(a: &[f64], b: &[f64], mode: PrecisionMode) -> f64 {
    match mode {
        PrecisionMode::StrictLeftFold => a.iter().zip(b).fold(0.0_f64, |acc, (&x, &y)| acc + x * y),
        PrecisionMode::FastSimd => {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];

            let mut acc0 = f64x4::ZERO;
            let mut acc1 = f64x4::ZERO;
            let (a_blocks, a_tail) = a.as_chunks::<8>();
            let (b_blocks, b_tail) = b.as_chunks::<8>();

            for (xa, xb) in a_blocks.iter().zip(b_blocks) {
                let va0 = f64x4::new([xa[0], xa[1], xa[2], xa[3]]);
                let vb0 = f64x4::new([xb[0], xb[1], xb[2], xb[3]]);
                acc0 = va0.mul_add(vb0, acc0);

                let va1 = f64x4::new([xa[4], xa[5], xa[6], xa[7]]);
                let vb1 = f64x4::new([xb[4], xb[5], xb[6], xb[7]]);
                acc1 = va1.mul_add(vb1, acc1);
            }

            let mut sum = (acc0 + acc1).reduce_add();
            for (&x, &y) in a_tail.iter().zip(b_tail) {
                sum += x * y;
            }
            sum
        }
    }
}

#[inline]
pub fn norm_sq_f64(v: &[f64], mode: PrecisionMode) -> f64 {
    match mode {
        PrecisionMode::StrictLeftFold => v.iter().fold(0.0_f64, |acc, &x| acc + x * x),
        PrecisionMode::FastSimd => {
            let mut acc0 = f64x4::ZERO;
            let mut acc1 = f64x4::ZERO;
            let (blocks, tail) = v.as_chunks::<8>();

            for c in blocks {
                let v0 = f64x4::new([c[0], c[1], c[2], c[3]]);
                acc0 = v0.mul_add(v0, acc0);

                let v1 = f64x4::new([c[4], c[5], c[6], c[7]]);
                acc1 = v1.mul_add(v1, acc1);
            }

            let mut sum = (acc0 + acc1).reduce_add();
            for &x in tail {
                sum += x * x;
            }
            sum
        }
    }
}

/// Cosine similarity between two `f32` slices given precomputed squared norms.
/// Returns `0.0` on length mismatch, empty input, or when either squared norm fails `> norm_sq_floor`.
/// Does NOT clamp the ratio, preserving bit-exact `DTSC` / `River` reference semantics (`NaN`/`Inf` propagate).
#[inline]
pub fn cosine_f32_with_norms(
    a: &[f32],
    b: &[f32],
    aa: f64,
    bb: f64,
    mode: PrecisionMode,
    norm_sq_floor: f64,
) -> f64 {
    if a.len() != b.len() || a.is_empty() || !(aa > norm_sq_floor && bb > norm_sq_floor) {
        return 0.0;
    }
    dot_f32(a, b, mode) / (aa.sqrt() * bb.sqrt())
}

#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32], mode: PrecisionMode) -> f64 {
    let aa = norm_sq_f32(a, mode);
    let bb = norm_sq_f32(b, mode);
    cosine_f32_with_norms(a, b, aa, bb, mode, 1e-12)
}

#[inline]
pub fn cosine_f64(a: &[f64], b: &[f64], mode: PrecisionMode) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let aa = norm_sq_f64(a, mode);
    let bb = norm_sq_f64(b, mode);
    if !(aa > 1e-12 && bb > 1e-12) {
        return 0.0;
    }
    dot_f64(a, b, mode) / (aa.sqrt() * bb.sqrt())
}

#[inline]
pub(crate) fn cosine_from_dot(dot: f64, row_norm: f64, query_norm: f64, floor: f64) -> f64 {
    if row_norm > floor && query_norm > floor {
        dot / (row_norm.sqrt() * query_norm.sqrt())
    } else {
        0.0
    }
}

/// Computes 4 inner products `<target, vecs[k]>` in a single pass over `target`.
/// `StrictLeftFold` preserves exact coordinate addition order (`ags::dot_four`).
#[inline]
pub fn batch_dot_4_f64(target: &[f64], vecs: [&[f64]; 4], mode: PrecisionMode) -> [f64; 4] {
    let [v0, v1, v2, v3] = vecs;
    let len = target
        .len()
        .min(v0.len())
        .min(v1.len())
        .min(v2.len())
        .min(v3.len());
    match mode {
        PrecisionMode::StrictLeftFold => {
            let mut sums = [0.0_f64; 4];
            for i in 0..len {
                let q = target[i];
                sums[0] += q * v0[i];
                sums[1] += q * v1[i];
                sums[2] += q * v2[i];
                sums[3] += q * v3[i];
            }
            sums
        }
        PrecisionMode::FastSimd => {
            let mut acc0 = f64x4::ZERO;
            let mut acc1 = f64x4::ZERO;
            let mut acc2 = f64x4::ZERO;
            let mut acc3 = f64x4::ZERO;
            let chunks = len / 4;
            let rem_start = chunks * 4;

            for c in 0..chunks {
                let off = c * 4;
                let vt = f64x4::new([
                    target[off],
                    target[off + 1],
                    target[off + 2],
                    target[off + 3],
                ]);
                acc0 = vt.mul_add(
                    f64x4::new([v0[off], v0[off + 1], v0[off + 2], v0[off + 3]]),
                    acc0,
                );
                acc1 = vt.mul_add(
                    f64x4::new([v1[off], v1[off + 1], v1[off + 2], v1[off + 3]]),
                    acc1,
                );
                acc2 = vt.mul_add(
                    f64x4::new([v2[off], v2[off + 1], v2[off + 2], v2[off + 3]]),
                    acc2,
                );
                acc3 = vt.mul_add(
                    f64x4::new([v3[off], v3[off + 1], v3[off + 2], v3[off + 3]]),
                    acc3,
                );
            }

            let mut sums = [
                acc0.reduce_add(),
                acc1.reduce_add(),
                acc2.reduce_add(),
                acc3.reduce_add(),
            ];
            for i in rem_start..len {
                let q = target[i];
                sums[0] += q * v0[i];
                sums[1] += q * v1[i];
                sums[2] += q * v2[i];
                sums[3] += q * v3[i];
            }
            sums
        }
    }
}

/// Subtracts 4 basis projections in one pass over `residual` (`ags::sub_four`).
#[inline]
pub fn sub_4_f64(residual: &mut [f64], [c0, c1, c2, c3]: [f64; 4], [a0, a1, a2, a3]: [&[f64]; 4]) {
    let len = residual
        .len()
        .min(a0.len())
        .min(a1.len())
        .min(a2.len())
        .min(a3.len());
    for idx in 0..len {
        let mut v = residual[idx];
        v -= c0 * a0[idx];
        v -= c1 * a1[idx];
        v -= c2 * a2[idx];
        v -= c3 * a3[idx];
        residual[idx] = v;
    }
}

/// Single-pass dual-field CSR SpMV (`apply_transport_pair` from `Nous/native/vcp/src/pipeline/fields.rs:65-93`).
/// Pushes mass from `source` to `targets[cursor]` scaled by `graph.effective_weight(cursor)` using the request's effective graph.
pub fn csr_spmv_dual(
    graph: &TransportGraphView<'_>,
    local: &[f64],
    transfer: &[f64],
    local_out: &mut [f64],
    transfer_out: &mut [f64],
    run: &RunOptions,
) -> Result<(), Error> {
    const STAGE: &str = "kernel::csr_spmv_dual";
    run.check_cancelled()?;
    let n = graph.node_count;
    for (name, len) in [
        ("local", local.len()),
        ("transfer", transfer.len()),
        ("local_out", local_out.len()),
        ("transfer_out", transfer_out.len()),
    ] {
        if len != n {
            return Err(Error::Input {
                field: STAGE,
                id: None,
                message: format!("{name} length {len} does not match node_count {n}"),
            });
        }
    }
    if graph.row_offsets.len() != n + 1 {
        return Err(Error::dimension(STAGE, n + 1, graph.row_offsets.len()));
    }

    local_out.fill(0.0);
    transfer_out.fill(0.0);

    for source in 0..n {
        if source.is_multiple_of(64) {
            run.check_cancelled()?;
        }
        let local_mass = local[source];
        let transfer_mass = transfer[source];
        let local_active = local_mass != 0.0;
        let transfer_active = transfer_mass != 0.0;
        if !local_active && !transfer_active {
            continue;
        }
        for cursor in graph.row_offsets[source]..graph.row_offsets[source + 1] {
            if cursor.is_multiple_of(256) {
                run.check_cancelled()?;
            }
            let weight = graph.effective_weight(cursor);
            let target = graph.targets[cursor].index();
            if local_active {
                local_out[target] += local_mass * weight;
            }
            if transfer_active {
                transfer_out[target] += transfer_mass * weight;
            }
        }
    }

    run.check_cancelled()
}

/// `next = (1 - alpha) * source + alpha * propagated` and L1 delta vs `current` (`fields.rs:98-111`).
#[inline]
pub fn relax_field(
    source: &[f64],
    propagated: &[f64],
    alpha: f64,
    current: &[f64],
    next: &mut [f64],
) -> f64 {
    let mut residual = 0.0;
    let one_minus_alpha = 1.0 - alpha;
    for index in 0..source.len() {
        let val = one_minus_alpha * source[index] + alpha * propagated[index];
        next[index] = val;
        residual += (val - current[index]).abs();
    }
    residual
}

pub fn normalize_f64(v: &mut [f64], min_norm: f64, mode: PrecisionMode) -> bool {
    let norm = norm_sq_f64(v, mode).sqrt();
    if !norm.is_finite() || norm <= min_norm {
        return false;
    }
    for val in v.iter_mut() {
        *val /= norm;
    }
    true
}
