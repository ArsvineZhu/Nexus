// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::csr::CsrGraphView;
use crate::error::VcpError;
use crate::operator::{ParallelPolicy, PrecisionMode};
use crate::view::VectorMatrixView;
use rayon::prelude::*;
use wide::{f32x8, f64x4};

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
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);

    for (xa, xb) in (&mut ca).zip(&mut cb) {
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
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
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
    let mut chunks = v.chunks_exact(8);

    for c in &mut chunks {
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
    for &x in chunks.remainder() {
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
            let mut ca = a.chunks_exact(8);
            let mut cb = b.chunks_exact(8);

            for (xa, xb) in (&mut ca).zip(&mut cb) {
                let va0 = f64x4::new([xa[0], xa[1], xa[2], xa[3]]);
                let vb0 = f64x4::new([xb[0], xb[1], xb[2], xb[3]]);
                acc0 = va0.mul_add(vb0, acc0);

                let va1 = f64x4::new([xa[4], xa[5], xa[6], xa[7]]);
                let vb1 = f64x4::new([xb[4], xb[5], xb[6], xb[7]]);
                acc1 = va1.mul_add(vb1, acc1);
            }

            let mut sum = (acc0 + acc1).reduce_add();
            for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
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
            let mut chunks = v.chunks_exact(8);

            for c in &mut chunks {
                let v0 = f64x4::new([c[0], c[1], c[2], c[3]]);
                acc0 = v0.mul_add(v0, acc0);

                let v1 = f64x4::new([c[4], c[5], c[6], c[7]]);
                acc1 = v1.mul_add(v1, acc1);
            }

            let mut sum = (acc0 + acc1).reduce_add();
            for &x in chunks.remainder() {
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
fn dot_4_f32_row(
    row: &[f32],
    q0: &[f32],
    q1: &[f32],
    q2: &[f32],
    q3: &[f32],
    mode: PrecisionMode,
) -> [f64; 4] {
    match mode {
        PrecisionMode::StrictLeftFold => {
            // Vectorize across queries so each lane retains its own left fold.
            let mut sums = f64x4::ZERO;
            for ((((value, a), b), c), d) in row.iter().zip(q0).zip(q1).zip(q2).zip(q3) {
                let queries = f64x4::new([*a, *b, *c, *d].map(f64::from));
                sums += f64x4::splat(f64::from(*value)) * queries;
            }
            sums.to_array()
        }
        PrecisionMode::FastSimd => {
            let mut accumulators = [f64x4::ZERO; 4];
            let (rows, remainder) = row.as_chunks::<4>();
            let [a, b, c, d] = [q0, q1, q2, q3].map(|query| query.as_chunks::<4>().0);
            for ((((row, a), b), c), d) in rows.iter().zip(a).zip(b).zip(c).zip(d) {
                let vector = f64x4::new(row.map(f64::from));
                for (acc, query) in accumulators.iter_mut().zip([a, b, c, d]) {
                    *acc = vector.mul_add(f64x4::new(query.map(f64::from)), *acc);
                }
            }
            let mut result = accumulators.map(f64x4::reduce_add);
            let offset = rows.len() * 4;
            for (i, &value) in remainder.iter().enumerate() {
                let position = offset + i;
                for (sum, query) in result.iter_mut().zip([q0, q1, q2, q3]) {
                    *sum += f64::from(value) * f64::from(query[position]);
                }
            }
            result
        }
    }
}

/// Computes batch cosine similarities between every row of `matrix` and each query in `queries`.
/// Uses a single-pass 4-way fused loop when `queries.len() == 4`, and uses `parallel` when enabled.
pub fn batch_cosines_f32(
    matrix: &VectorMatrixView<'_, f32>,
    queries: &[&[f32]],
    out_cols: &mut [&mut [f64]],
    mode: PrecisionMode,
    norm_sq_floor: f64,
    parallel: ParallelPolicy<'_>,
) -> Result<(), VcpError> {
    const STAGE: &str = "kernel::batch_cosines_f32";
    matrix.validate(STAGE)?;
    if queries.len() != out_cols.len() {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: queries.len(),
            actual: out_cols.len(),
        });
    }
    for q in queries {
        if q.len() != matrix.dim {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: matrix.dim,
                actual: q.len(),
            });
        }
    }
    for col in out_cols.iter() {
        if col.len() != matrix.rows {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: matrix.rows,
                actual: col.len(),
            });
        }
    }
    if matrix.rows == 0 || queries.is_empty() {
        return Ok(());
    }
    if matrix.dim == 0 {
        for col in out_cols.iter_mut() {
            col.fill(0.0);
        }
        return Ok(());
    }

    let dim = matrix.dim;
    let use_par = parallel.is_parallel() && matrix.rows >= 64;

    if queries.len() == 4 {
        let [q0, q1, q2, q3] = [queries[0], queries[1], queries[2], queries[3]];
        let [bb0, bb1, bb2, bb3] = [q0, q1, q2, q3].map(|q| norm_sq_f32(q, mode));
        let [nb0, nb1, nb2, nb3] = [bb0, bb1, bb2, bb3].map(f64::sqrt);
        let valid0 = !q0.is_empty() && bb0 > norm_sq_floor;
        let valid1 = !q1.is_empty() && bb1 > norm_sq_floor;
        let valid2 = !q2.is_empty() && bb2 > norm_sq_floor;
        let valid3 = !q3.is_empty() && bb3 > norm_sq_floor;

        let [out0, out1, out2, out3] = out_cols else {
            unreachable!()
        };

        let compute_row = |row_idx: usize| -> [f64; 4] {
            let row = &matrix.data[row_idx * dim..(row_idx + 1) * dim];
            if row.is_empty() {
                return [0.0; 4];
            }
            let aa = matrix
                .sq_norm_f64(row_idx)
                .unwrap_or_else(|| norm_sq_f32(row, mode));
            if aa.is_nan() || aa <= norm_sq_floor {
                return [0.0; 4];
            }
            let na = aa.sqrt();
            let dots = dot_4_f32_row(row, q0, q1, q2, q3, mode);
            [
                if valid0 { dots[0] / (na * nb0) } else { 0.0 },
                if valid1 { dots[1] / (na * nb1) } else { 0.0 },
                if valid2 { dots[2] / (na * nb2) } else { 0.0 },
                if valid3 { dots[3] / (na * nb3) } else { 0.0 },
            ]
        };

        if use_par {
            parallel.install(|| {
                let chunk = (matrix.rows / rayon::current_num_threads().max(1)).max(16);
                (
                    out0.par_chunks_mut(chunk),
                    out1.par_chunks_mut(chunk),
                    out2.par_chunks_mut(chunk),
                    out3.par_chunks_mut(chunk),
                )
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(chunk_idx, (c0, c1, c2, c3))| {
                        let base = chunk_idx * chunk;
                        for offset in 0..c0.len() {
                            let res = compute_row(base + offset);
                            c0[offset] = res[0];
                            c1[offset] = res[1];
                            c2[offset] = res[2];
                            c3[offset] = res[3];
                        }
                    });
            });
        } else {
            for r in 0..matrix.rows {
                let res = compute_row(r);
                out0[r] = res[0];
                out1[r] = res[1];
                out2[r] = res[2];
                out3[r] = res[3];
            }
        }
        return Ok(());
    }

    if queries.len() == 1 {
        let q = queries[0];
        let bb = norm_sq_f32(q, mode);
        let out = &mut out_cols[0];
        if use_par {
            parallel.install(|| {
                out.par_iter_mut().enumerate().for_each(|(r, slot)| {
                    let row = &matrix.data[r * dim..(r + 1) * dim];
                    let aa = matrix
                        .sq_norm_f64(r)
                        .unwrap_or_else(|| norm_sq_f32(row, mode));
                    *slot = cosine_f32_with_norms(row, q, aa, bb, mode, norm_sq_floor);
                });
            });
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                let row = &matrix.data[r * dim..(r + 1) * dim];
                let aa = matrix
                    .sq_norm_f64(r)
                    .unwrap_or_else(|| norm_sq_f32(row, mode));
                *slot = cosine_f32_with_norms(row, q, aa, bb, mode, norm_sq_floor);
            }
        }
        return Ok(());
    }

    if let [q0, q1] = queries
        && let [out0, out1] = &mut *out_cols
    {
        let [bb0, bb1] = [q0, q1].map(|query| norm_sq_f32(query, mode));
        let row_cosines = |index: usize| {
            let row = matrix.row(index).unwrap();
            let aa = matrix
                .sq_norm_f64(index)
                .unwrap_or_else(|| norm_sq_f32(row, mode));
            [
                cosine_f32_with_norms(row, q0, aa, bb0, mode, norm_sq_floor),
                cosine_f32_with_norms(row, q1, aa, bb1, mode, norm_sq_floor),
            ]
        };
        if use_par {
            parallel.install(|| {
                out0.par_iter_mut()
                    .zip(out1.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, (a, b))| {
                        [*a, *b] = row_cosines(index);
                    });
            });
        } else {
            for (index, (a, b)) in out0.iter_mut().zip(out1.iter_mut()).enumerate() {
                [*a, *b] = row_cosines(index);
            }
        }
        return Ok(());
    }

    let q_sq_norms: Vec<f64> = queries.iter().map(|q| norm_sq_f32(q, mode)).collect();
    for (r, row) in matrix.data.chunks_exact(dim).enumerate() {
        let aa = matrix
            .sq_norm_f64(r)
            .unwrap_or_else(|| norm_sq_f32(row, mode));
        for ((q, q_sq), out_col) in queries.iter().zip(&q_sq_norms).zip(out_cols.iter_mut()) {
            out_col[r] = cosine_f32_with_norms(row, q, aa, *q_sq, mode, norm_sq_floor);
        }
    }
    Ok(())
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
/// Pushes mass from `source` to `targets[cursor]` scaled by `graph.effective_weight(cursor)` (respecting `edge_gates`).
pub fn csr_spmv_dual(
    graph: &CsrGraphView<'_>,
    local: &[f64],
    transfer: &[f64],
    local_out: &mut [f64],
    transfer_out: &mut [f64],
    parallel: ParallelPolicy<'_>,
) -> Result<(), VcpError> {
    const STAGE: &str = "kernel::csr_spmv_dual";
    let n = graph.node_count;
    for (name, len) in [
        ("local", local.len()),
        ("transfer", transfer.len()),
        ("local_out", local_out.len()),
        ("transfer_out", transfer_out.len()),
    ] {
        if len != n {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: format!("{name} length {len} does not match node_count {n}"),
            });
        }
    }
    if graph.row_offsets.len() != n + 1 {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: n + 1,
            actual: graph.row_offsets.len(),
        });
    }

    local_out.fill(0.0);
    transfer_out.fill(0.0);

    parallel.install(|| {
        for source in 0..n {
            let local_mass = local[source];
            let transfer_mass = transfer[source];
            let local_active = local_mass != 0.0;
            let transfer_active = transfer_mass != 0.0;
            if !local_active && !transfer_active {
                continue;
            }
            for cursor in graph.row_offsets[source]..graph.row_offsets[source + 1] {
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
    });

    Ok(())
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

pub fn normalize_f32_via_f64(
    v: &[f32],
    out: &mut [f32],
    min_norm: f64,
    mode: PrecisionMode,
) -> bool {
    if v.len() != out.len() || v.is_empty() {
        return false;
    }
    let norm = norm_sq_f32(v, mode).sqrt();
    if !norm.is_finite() || norm <= min_norm {
        return false;
    }
    for (dst, &src) in out.iter_mut().zip(v) {
        *dst = (f64::from(src) / norm) as f32;
    }
    true
}
