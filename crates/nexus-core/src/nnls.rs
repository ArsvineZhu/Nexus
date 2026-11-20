// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::error::VcpError;
use crate::execution::{CancellationToken, PrecisionMode};
use crate::kernel::{dot_f64, norm_sq_f64};
use faer::Mat;

pub const DUAL_TOL: f64 = 1e-10;
pub const RANK_TOL: f64 = 1e-12;
pub const CHECK_TOL: f64 = 2e-8;

#[derive(Default, Debug)]
pub struct NnlsWorkspace {
    target: Vec<f64>,
    projected: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NnlsSolution {
    /// Unit-space NNLS coefficients (solving `min ||A_unit * w - y_unit||_2` s.t. `w >= 0`).
    pub direction_weights: Vec<f64>,
    column_norms: Vec<f64>,
    /// Unscaled projection `A_unit * (direction_weights * target_norm)` in original target space.
    pub projection: Vec<f64>,
    /// Unscaled residual `target - projection`.
    pub residual: Vec<f64>,
    /// Euclidean norm of `residual`.
    pub residual_norm: f64,
    /// Euclidean norm of `target`.
    pub target_norm: f64,
    /// Squared residual energy ratio `||residual||^2 / ||target||^2`.
    pub residual_ratio: f64,
    /// Explained energy ratio `(1.0 - residual_ratio).clamp(0.0, 1.0)`.
    pub explained_ratio: f64,
    /// Count of active positive direction weights (`direction_weights[i] > 0.0`).
    pub active_columns: usize,
    /// Number of Lawson-Hanson active-set iterations performed.
    pub iterations: usize,
    /// Maximum KKT violation across dual, stationarity, complementarity, and energy closure.
    pub max_kkt_violation: f64,
}

impl NnlsSolution {
    /// Coefficients in the original matrix and target units.
    pub fn coefficients(&self) -> Result<Vec<f64>, VcpError> {
        self.direction_weights
            .iter()
            .zip(&self.column_norms)
            .map(|(&weight, &norm)| {
                if weight == 0.0 || norm == 0.0 {
                    return Ok(0.0);
                }
                let coefficient = weight * (self.target_norm / norm);
                let coefficient = if coefficient.is_finite() && coefficient > 0.0 {
                    coefficient
                } else {
                    (weight * self.target_norm) / norm
                };
                if coefficient.is_finite() {
                    Ok(coefficient)
                } else {
                    Err(VcpError::NumericDivergence {
                        stage: "nnls.coefficients",
                        detail: "original matrix coefficient overflowed".into(),
                    })
                }
            })
            .collect()
    }
}

#[inline]
fn col_slice(cols: &[f64], dim: usize, col: usize) -> &[f64] {
    let start = col * dim;
    &cols[start..start + dim]
}

#[inline]
fn euclidean_norm_f64(v: &[f64]) -> f64 {
    let norm_sq = norm_sq_f64(v, PrecisionMode::StrictLeftFold);
    if norm_sq.is_finite() && norm_sq >= f64::MIN_POSITIVE {
        return norm_sq.sqrt();
    }
    let max_abs = v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return max_abs;
    }
    let scaled_sq = v.iter().fold(0.0_f64, |acc, &x| {
        let r = x / max_abs;
        acc + r * r
    });
    max_abs * scaled_sq.sqrt()
}

fn mat_vec_mul(cols: &[f64], dim: usize, count: usize, x: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for (c, &xc) in x.iter().enumerate().take(count) {
        if xc == 0.0 {
            continue;
        }
        let col = col_slice(cols, dim, c);
        for (dst, &val) in out.iter_mut().zip(col) {
            *dst += val * xc;
        }
    }
}

fn compute_kkt_and_check(
    stage: &'static str,
    unit_cols: &[f64],
    y_unit: &[f64],
    x: &[f64],
    p_unit: &mut [f64],
    r_unit: &mut [f64],
) -> Result<(f64, f64), VcpError> {
    let dim = y_unit.len();
    let count = x.len();
    mat_vec_mul(unit_cols, dim, count, x, p_unit);
    for i in 0..dim {
        r_unit[i] = y_unit[i] - p_unit[i];
    }

    let mut dual = 0.0_f64;
    let mut stationarity = 0.0_f64;
    let mut complementarity = 0.0_f64;
    for (i, &xi) in x.iter().enumerate() {
        let g = dot_f64(
            col_slice(unit_cols, dim, i),
            r_unit,
            PrecisionMode::StrictLeftFold,
        );
        dual = dual.max(g.max(0.0));
        if xi > 0.0 {
            stationarity = stationarity.max(g.abs());
        }
        complementarity = complementarity.max((xi * g).abs());
    }

    let yy = dot_f64(y_unit, y_unit, PrecisionMode::StrictLeftFold);
    let pp = dot_f64(p_unit, p_unit, PrecisionMode::StrictLeftFold);
    let rr = dot_f64(r_unit, r_unit, PrecisionMode::StrictLeftFold);
    let closure = (yy - pp - rr).abs();

    let invalid_weight = x.iter().any(|&v| !v.is_finite() || v < 0.0);
    let max_violation = dual.max(stationarity).max(complementarity).max(closure);

    if invalid_weight
        || !max_violation.is_finite()
        || dual > CHECK_TOL
        || stationarity > CHECK_TOL
        || complementarity > CHECK_TOL
        || closure > CHECK_TOL
        || rr > yy + CHECK_TOL
    {
        return Err(VcpError::NumericDivergence {
            stage,
            detail: format!(
                "NNLS KKT check failed: dual={dual:e}, stationarity={stationarity:e}, complementarity={complementarity:e}, closure={closure:e}, rr={rr:e}, yy={yy:e}"
            ),
        });
    }
    Ok((max_violation, rr))
}

/// Core Lawson-Hanson active-set solver on already unit-normalized `y_unit` and `unit_cols` (`dim x count`).
fn lawson_hanson_core(
    stage: &'static str,
    dim: usize,
    count: usize,
    unit_cols: &[f64],
    y_unit: &[f64],
    cancel: &CancellationToken<'_>,
) -> Result<(Vec<f64>, usize), VcpError> {
    if count == 0 {
        return Ok((Vec::new(), 0));
    }

    let limit = 32 * count.max(1) + 128;
    let mut x = vec![0.0_f64; count];
    let mut z = vec![0.0_f64; count];
    let mut passive = vec![false; count];
    let mut p_work = vec![0.0_f64; dim];
    let mut r_work = vec![0.0_f64; dim];
    let mut indices = Vec::with_capacity(count);
    let mut iterations = 0usize;

    loop {
        cancel.check(stage)?;
        mat_vec_mul(unit_cols, dim, count, &x, &mut p_work);
        for i in 0..dim {
            r_work[i] = y_unit[i] - p_work[i];
        }

        let mut best_idx = None;
        let mut max_grad = DUAL_TOL;
        for (i, &is_passive) in passive.iter().enumerate() {
            if is_passive {
                continue;
            }
            let grad = dot_f64(
                col_slice(unit_cols, dim, i),
                &r_work,
                PrecisionMode::StrictLeftFold,
            );
            if grad > max_grad {
                max_grad = grad;
                best_idx = Some(i);
            }
        }

        let Some(entering) = best_idx else {
            break;
        };
        passive[entering] = true;

        loop {
            iterations += 1;
            if iterations > limit {
                return Err(VcpError::NumericDivergence {
                    stage,
                    detail: format!("Lawson-Hanson iteration limit exceeded ({limit})"),
                });
            }

            indices.clear();
            for (i, &is_passive) in passive.iter().enumerate() {
                if is_passive {
                    indices.push(i);
                }
            }

            let sub_cols = indices.len();
            let sub = Mat::from_fn(dim, sub_cols, |r, c| unit_cols[indices[c] * dim + r]);
            let svd = sub
                .as_ref()
                .thin_svd()
                .map_err(|err| VcpError::NumericDivergence {
                    stage,
                    detail: format!("faer thin_svd failed: {err:?}"),
                })?;

            let u = svd.U();
            let v = svd.V();
            let s = svd.S().column_vector();
            let s_len = s.nrows();

            let mut coeff = vec![0.0_f64; s_len];
            for k in 0..s_len {
                let sigma = s[k];
                if sigma > RANK_TOL {
                    let mut ut_y = 0.0_f64;
                    for r in 0..dim {
                        ut_y += u[(r, k)] * y_unit[r];
                    }
                    coeff[k] = ut_y / sigma;
                }
            }

            z.fill(0.0);
            for (local_j, &orig_idx) in indices.iter().enumerate() {
                let mut val = 0.0_f64;
                for k in 0..s_len {
                    val += v[(local_j, k)] * coeff[k];
                }
                if !val.is_finite() {
                    return Err(VcpError::NumericDivergence {
                        stage,
                        detail: "non-finite unconstrained least-squares fit in NNLS".into(),
                    });
                }
                z[orig_idx] = val;
            }

            if indices.iter().all(|&i| z[i] > 0.0) {
                x.copy_from_slice(&z);
                break;
            }

            let mut alpha = 1.0_f64;
            for &i in &indices {
                if z[i] <= 0.0 {
                    let denom = x[i] - z[i];
                    if denom > 0.0 {
                        alpha = alpha.min(x[i] / denom);
                    } else {
                        alpha = 0.0;
                    }
                }
            }

            for i in 0..count {
                x[i] += alpha * (z[i] - x[i]);
            }

            let boundary = 32.0 * f64::EPSILON * x.iter().copied().fold(1.0_f64, f64::max);
            for i in 0..count {
                if passive[i] && x[i] <= boundary {
                    x[i] = 0.0;
                    passive[i] = false;
                }
            }
        }
    }

    Ok((x, iterations))
}

fn solve_nnls_indexed_impl<'a, I>(
    stage: &'static str,
    target: &[f64],
    tagged_columns: I,
    num_cols: usize,
    max_columns: usize,
    cancel: &CancellationToken<'_>,
    scratch: &mut NnlsWorkspace,
) -> Result<NnlsSolution, VcpError>
where
    I: Iterator<Item = (i64, usize, &'a [f64])>,
{
    cancel.check(stage)?;
    let dim = target.len();
    if dim == 0 || target.iter().any(|v| !v.is_finite()) {
        return Err(VcpError::InvalidInput {
            stage,
            detail: "target must be non-empty and finite".into(),
        });
    }
    if num_cols > max_columns {
        return Err(VcpError::ResourceLimitExceeded {
            stage,
            limit_name: "max_columns",
            actual: num_cols,
            limit: max_columns,
        });
    }

    let target_norm = euclidean_norm_f64(target);
    if !target_norm.is_finite() {
        return Err(VcpError::InvalidInput {
            stage,
            detail: "target vector norm overflowed".into(),
        });
    }

    let mut entries: Vec<(i64, usize, &[f64])> = Vec::with_capacity(num_cols);
    for (id, slot, col) in tagged_columns {
        if col.len() != dim {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: dim,
                actual: col.len(),
            });
        }
        if col.iter().any(|v| !v.is_finite()) {
            return Err(VcpError::InvalidInput {
                stage,
                detail: format!("column {slot} contains non-finite values"),
            });
        }
        entries.push((id, slot, col));
    }
    entries.sort_by_key(|(id, slot, _)| (*id, *slot));

    let mut unit_cols = Vec::with_capacity(entries.len() * dim);
    let mut active_slots = Vec::with_capacity(entries.len());
    let mut norm_col = Vec::with_capacity(dim);

    let mut column_norms = vec![0.0; num_cols];
    for (_id, slot, col) in entries {
        let col_norm = euclidean_norm_f64(col);
        if !col_norm.is_finite() {
            return Err(VcpError::NumericDivergence {
                stage,
                detail: format!("column {slot} norm overflow"),
            });
        }
        column_norms[slot] = col_norm;
        if target_norm > 0.0 && col_norm > 0.0 {
            norm_col.clear();
            norm_col.extend(col.iter().map(|&val| val / col_norm));
            let is_duplicate = (0..active_slots.len())
                .any(|existing| col_slice(&unit_cols, dim, existing) == norm_col.as_slice());
            if !is_duplicate {
                unit_cols.extend_from_slice(&norm_col);
                active_slots.push(slot);
            }
        }
    }

    if target_norm == 0.0 {
        return Ok(NnlsSolution {
            direction_weights: vec![0.0; num_cols],
            column_norms,
            projection: vec![0.0; dim],
            residual: target.to_vec(),
            residual_norm: 0.0,
            target_norm: 0.0,
            residual_ratio: 0.0,
            explained_ratio: 0.0,
            active_columns: 0,
            iterations: 0,
            max_kkt_violation: 0.0,
        });
    }

    let count = active_slots.len();
    if count == 0 {
        return Ok(NnlsSolution {
            direction_weights: vec![0.0; num_cols],
            column_norms,
            projection: vec![0.0; dim],
            residual: target.to_vec(),
            residual_norm: target_norm,
            target_norm,
            residual_ratio: 1.0,
            explained_ratio: 0.0,
            active_columns: 0,
            iterations: 0,
            max_kkt_violation: 0.0,
        });
    }

    let y_unit = {
        scratch.target.resize(dim, 0.0);
        scratch.target.as_mut_slice()
    };
    for (dst, &src) in y_unit.iter_mut().zip(target) {
        *dst = src / target_norm;
    }

    let (x_unit, iterations) = if dim > count {
        // Householder QR compression: Q is dim x count, R is count x count.
        let a_mat = Mat::from_fn(dim, count, |r, c| unit_cols[c * dim + r]);
        let qr = a_mat.as_ref().qr();
        let q_mat = qr.compute_thin_Q();
        let r_mat = qr.thin_R();

        let z = {
            scratch.projected.resize(count, 0.0);
            scratch.projected.fill(0.0);
            scratch.projected.as_mut_slice()
        };
        for k in 0..count {
            let mut sum = 0.0_f64;
            for r in 0..dim {
                sum += q_mat[(r, k)] * y_unit[r];
            }
            z[k] = sum;
        }

        let z_norm = norm_sq_f64(z, PrecisionMode::StrictLeftFold).sqrt();
        if !z_norm.is_finite() {
            return Err(VcpError::NumericDivergence {
                stage,
                detail: "QR-compressed target norm is non-finite".into(),
            });
        }

        if z_norm <= 0.0 {
            (vec![0.0; count], 0)
        } else {
            let mut z_unit = vec![0.0_f64; count];
            for k in 0..count {
                z_unit[k] = z[k] / z_norm;
            }

            let mut r_unit_cols = Vec::with_capacity(count * count);
            let mut r_col_norms = Vec::with_capacity(count);
            let mut r_nonzero_map = Vec::with_capacity(count);
            for j in 0..count {
                let mut norm_sq = 0.0_f64;
                for i in 0..count {
                    let v = r_mat[(i, j)];
                    norm_sq += v * v;
                }
                let r_norm = norm_sq.sqrt();
                if r_norm > 0.0 && r_norm.is_finite() {
                    for i in 0..count {
                        r_unit_cols.push(r_mat[(i, j)] / r_norm);
                    }
                    r_col_norms.push(r_norm);
                    r_nonzero_map.push(j);
                }
            }

            let (x_small, iters) = lawson_hanson_core(
                stage,
                count,
                r_nonzero_map.len(),
                &r_unit_cols,
                &z_unit,
                cancel,
            )?;

            let mut x_unscaled = vec![0.0_f64; count];
            for (small_idx, &orig_c) in r_nonzero_map.iter().enumerate() {
                x_unscaled[orig_c] = x_small[small_idx] * z_norm / r_col_norms[small_idx];
            }
            (x_unscaled, iters)
        }
    } else {
        lawson_hanson_core(stage, dim, count, &unit_cols, y_unit, cancel)?
    };

    let mut p_unit = vec![0.0_f64; dim];
    let mut r_unit = vec![0.0_f64; dim];
    let (max_kkt_violation, residual_ratio) =
        compute_kkt_and_check(stage, &unit_cols, y_unit, &x_unit, &mut p_unit, &mut r_unit)?;

    let mut direction_weights = vec![0.0_f64; num_cols];
    let mut active_columns = 0usize;
    for (k, &orig_slot) in active_slots.iter().enumerate() {
        direction_weights[orig_slot] = x_unit[k];
        if x_unit[k] > 0.0 {
            active_columns += 1;
        }
    }

    let mut projection = vec![0.0_f64; dim];
    let mut residual = vec![0.0_f64; dim];
    for i in 0..dim {
        let pv = p_unit[i] * target_norm;
        let rv = r_unit[i] * target_norm;
        if !pv.is_finite() || !rv.is_finite() {
            return Err(VcpError::NumericDivergence {
                stage,
                detail: "NNLS rescaled output overflow".into(),
            });
        }
        projection[i] = pv;
        residual[i] = rv;
    }

    let residual_norm = euclidean_norm_f64(&residual);
    let explained_ratio = (1.0 - residual_ratio).clamp(0.0, 1.0);

    Ok(NnlsSolution {
        direction_weights,
        column_norms,
        projection,
        residual,
        residual_norm,
        target_norm,
        residual_ratio,
        explained_ratio,
        active_columns,
        iterations,
        max_kkt_violation,
    })
}

/// Solves NNLS with explicit `(id, column)` pairs for deterministic duplicate tie-breaking by `(id, slot)`.
pub fn solve_nnls_with_ids_f64(
    target: &[f64],
    columns: &[(i64, &[f64])],
    max_columns: usize,
    cancel: &CancellationToken<'_>,
    scratch: &mut NnlsWorkspace,
) -> Result<NnlsSolution, VcpError> {
    solve_nnls_indexed_impl(
        "nnls::solve_nnls_with_ids_f64",
        target,
        columns
            .iter()
            .enumerate()
            .map(|(slot, &(id, col))| (id, slot, col)),
        columns.len(),
        max_columns,
        cancel,
        scratch,
    )
}

/// Solves NNLS over `columns: &[&[f64]]` (tie-broken deterministically by ascending column index).
pub fn solve_nnls_f64(
    target: &[f64],
    columns: &[&[f64]],
    max_columns: usize,
    cancel: &CancellationToken<'_>,
    scratch: &mut NnlsWorkspace,
) -> Result<NnlsSolution, VcpError> {
    solve_nnls_indexed_impl(
        "nnls::solve_nnls_f64",
        target,
        columns
            .iter()
            .enumerate()
            .map(|(slot, &col)| (slot as i64, slot, col)),
        columns.len(),
        max_columns,
        cancel,
        scratch,
    )
}

/// Solves NNLS over a flat column-major buffer `cols_flat` (`num_cols * dim` elements).
pub fn solve_nnls_flat_f64(
    target: &[f64],
    cols_flat: &[f64],
    num_cols: usize,
    dim: usize,
    max_columns: usize,
    cancel: &CancellationToken<'_>,
    scratch: &mut NnlsWorkspace,
) -> Result<NnlsSolution, VcpError> {
    const STAGE: &str = "nnls::solve_nnls_flat_f64";
    if target.len() != dim {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: dim,
            actual: target.len(),
        });
    }
    let expected_flat = num_cols
        .checked_mul(dim)
        .ok_or_else(|| VcpError::InvalidInput {
            stage: STAGE,
            detail: "num_cols * dim overflows usize".into(),
        })?;
    if cols_flat.len() != expected_flat {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: expected_flat,
            actual: cols_flat.len(),
        });
    }
    solve_nnls_indexed_impl(
        STAGE,
        target,
        (0..num_cols).map(|slot| (slot as i64, slot, &cols_flat[slot * dim..(slot + 1) * dim])),
        num_cols,
        max_columns,
        cancel,
        scratch,
    )
}

/// Widens `f32` inputs to `f64` and delegates to `solve_nnls_f64`.
pub fn solve_nnls_f32(
    target: &[f32],
    columns: &[&[f32]],
    max_columns: usize,
    cancel: &CancellationToken<'_>,
    scratch: &mut NnlsWorkspace,
) -> Result<NnlsSolution, VcpError> {
    const STAGE: &str = "nnls::solve_nnls_f32";
    let dim = target.len();
    if dim == 0 || target.iter().any(|v| !v.is_finite()) {
        return Err(VcpError::InvalidInput {
            stage: STAGE,
            detail: "target must be non-empty and finite".into(),
        });
    }
    if columns.len() > max_columns {
        return Err(VcpError::ResourceLimitExceeded {
            stage: STAGE,
            limit_name: "max_columns",
            actual: columns.len(),
            limit: max_columns,
        });
    }
    for (slot, col) in columns.iter().enumerate() {
        if col.len() != dim {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: dim,
                actual: col.len(),
            });
        }
        if col.iter().any(|v| !v.is_finite()) {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: format!("column {slot} contains non-finite values"),
            });
        }
    }

    let target_f64: Vec<f64> = target.iter().map(|&v| f64::from(v)).collect();
    let cols_flat: Vec<f64> = columns
        .iter()
        .flat_map(|c| c.iter().map(|&v| f64::from(v)))
        .collect();

    solve_nnls_flat_f64(
        &target_f64,
        &cols_flat,
        columns.len(),
        dim,
        max_columns,
        cancel,
        scratch,
    )
}
