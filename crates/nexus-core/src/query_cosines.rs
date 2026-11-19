// SPDX-License-Identifier: Apache-2.0

use crate::kernel::{all_finite_f32, cosine_f32_with_norms, norm_sq_f32};
use crate::{
    CandidateRowsView, ExecutionContext, PrecisionMode::StrictLeftFold, VcpError, VcpResult,
};
use rayon::prelude::*;
use std::sync::Arc;

const STAGE: &str = "query.cosines";
const FLOOR: f64 = 1e-12;

/// Strict cosine columns with the exact rows and queries that produced them.
#[derive(Debug)]
pub struct QueryCosines<'rows, 'query> {
    rows: CandidateRowsView<'rows>,
    original_query: &'query [f32],
    enhanced_query: &'query [f32],
    candidate_norms: Arc<[f64]>,
    original_cosines: Arc<[f64]>,
    enhanced_cosines: Vec<f64>,
}

impl<'rows, 'query> QueryCosines<'rows, 'query> {
    pub fn prepare(
        ctx: &ExecutionContext<'_>,
        rows: CandidateRowsView<'rows>,
        original_query: &'query [f32],
        enhanced_query: &'query [f32],
    ) -> VcpResult<Self> {
        ctx.check_cancelled(STAGE)?;
        let mut rows = rows;
        rows.candidate_vectors.sq_norms_f64 = None;
        rows.validate(STAGE)?;
        let dim = rows.candidate_vectors.dim;
        let original_norm = query_norm(original_query, dim)?;
        let enhanced_norm = query_norm(enhanced_query, dim)?;
        let calculate = |i| {
            ctx.check_cancelled(STAGE)?;
            let row = rows
                .candidate_vectors
                .row(i)
                .expect("validated candidate matrix");
            let (norm, raw_dot, enhanced_dot) =
                row.iter().zip(original_query).zip(enhanced_query).fold(
                    (0.0, 0.0, 0.0),
                    |(norm, raw_dot, enhanced_dot), ((&value, &original), &enhanced)| {
                        let value = f64::from(value);
                        (
                            norm + value * value,
                            raw_dot + value * f64::from(original),
                            enhanced_dot + value * f64::from(enhanced),
                        )
                    },
                );
            if !norm.is_finite() {
                return Err(invalid(format!(
                    "candidate row {i} contains nonfinite values"
                )));
            }
            Ok((
                norm,
                from_dot(raw_dot, norm, original_norm),
                from_dot(enhanced_dot, norm, enhanced_norm),
            ))
        };
        let values: Vec<_> = if ctx.parallel.is_parallel() && rows.candidate_count >= 64 {
            ctx.parallel.install(|| {
                (0..rows.candidate_count)
                    .into_par_iter()
                    .map(calculate)
                    .collect::<VcpResult<_>>()
            })?
        } else {
            (0..rows.candidate_count)
                .map(calculate)
                .collect::<VcpResult<_>>()?
        };
        let mut candidate_norms = Vec::with_capacity(values.len());
        let mut original_cosines = Vec::with_capacity(values.len());
        let mut enhanced_cosines = Vec::with_capacity(values.len());
        for (norm, original, enhanced) in values {
            candidate_norms.push(norm);
            original_cosines.push(original);
            enhanced_cosines.push(enhanced);
        }
        Ok(Self {
            rows,
            original_query,
            enhanced_query,
            candidate_norms: candidate_norms.into(),
            original_cosines: original_cosines.into(),
            enhanced_cosines,
        })
    }

    pub fn reuse_original(
        ctx: &ExecutionContext<'_>,
        previous: &QueryCosines<'_, '_>,
        rows: CandidateRowsView<'rows>,
        original_query: &'query [f32],
        enhanced_query: &'query [f32],
    ) -> VcpResult<Self> {
        ctx.check_cancelled(STAGE)?;
        let mut rows = rows;
        rows.candidate_vectors.sq_norms_f64 = None;
        rows.validate(STAGE)?;
        let matrix = rows.candidate_vectors;
        let prior = previous.rows.candidate_vectors;
        if !std::ptr::eq(matrix.data, prior.data)
            || matrix.rows != prior.rows
            || matrix.dim != prior.dim
            || original_query != previous.original_query
        {
            return Err(invalid("original query or candidate matrix differs".into()));
        }
        let enhanced_norm = query_norm(enhanced_query, matrix.dim)?;
        let calculate = |i| {
            ctx.check_cancelled(STAGE)?;
            let row = matrix.row(i).expect("validated candidate matrix");
            Ok(cosine(
                row,
                enhanced_query,
                previous.candidate_norms[i],
                enhanced_norm,
            ))
        };
        let enhanced_cosines = if ctx.parallel.is_parallel() && matrix.rows >= 64 {
            ctx.parallel.install(|| {
                (0..matrix.rows)
                    .into_par_iter()
                    .map(calculate)
                    .collect::<VcpResult<_>>()
            })?
        } else {
            (0..matrix.rows).map(calculate).collect::<VcpResult<_>>()?
        };
        Ok(Self {
            rows,
            original_query,
            enhanced_query,
            candidate_norms: Arc::clone(&previous.candidate_norms),
            original_cosines: Arc::clone(&previous.original_cosines),
            enhanced_cosines,
        })
    }

    pub fn rows(&self) -> CandidateRowsView<'_> {
        let mut rows = self.rows;
        rows.candidate_vectors.sq_norms_f64 = Some(&self.candidate_norms);
        rows
    }

    pub fn original_query(&self) -> &'query [f32] {
        self.original_query
    }
    pub fn enhanced_query(&self) -> &'query [f32] {
        self.enhanced_query
    }
    pub fn original_cosines(&self) -> &[f64] {
        &self.original_cosines
    }
    pub fn enhanced_cosines(&self) -> &[f64] {
        &self.enhanced_cosines
    }
}

fn invalid(detail: String) -> VcpError {
    VcpError::InvalidInput {
        stage: STAGE,
        detail,
    }
}

fn query_norm(query: &[f32], dim: usize) -> VcpResult<f64> {
    if query.len() != dim {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: dim,
            actual: query.len(),
        });
    }
    if !all_finite_f32(query) {
        return Err(invalid("query must be finite".into()));
    }
    Ok(norm_sq_f32(query, StrictLeftFold))
}

fn cosine(row: &[f32], query: &[f32], row_norm: f64, query_norm: f64) -> f64 {
    cosine_f32_with_norms(row, query, row_norm, query_norm, StrictLeftFold, FLOOR)
}

fn from_dot(dot: f64, row_norm: f64, query_norm: f64) -> f64 {
    if row_norm > FLOOR && query_norm > FLOOR {
        dot / (row_norm.sqrt() * query_norm.sqrt())
    } else {
        0.0
    }
}
