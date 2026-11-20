// SPDX-License-Identifier: Apache-2.0

use crate::kernel::{StrictDotPair, all_finite_f32, cosine_from_dot, norm_sq_f32};
use crate::{
    CandidateRowsView, ExecutionContext, PrecisionMode::StrictLeftFold, VcpError, VcpResult,
};
use rayon::prelude::*;

const STAGE: &str = "query.cosines";
const FLOOR: f64 = 1e-12;

/// Strict cosine columns with the exact rows and queries that produced them.
#[derive(Debug)]
pub struct QueryCosines<'rows, 'query> {
    rows: CandidateRowsView<'rows>,
    original_query: &'query [f32],
    enhanced_query: &'query [f32],
    candidate_norms: Vec<f64>,
    original_cosines: Vec<f64>,
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
        let pair = StrictDotPair::new(original_query, enhanced_query);
        let calculate = |i| {
            ctx.check_cancelled(STAGE)?;
            let row = rows
                .candidate_vectors
                .row(i)
                .expect("validated candidate matrix");
            let ([raw_dot, enhanced_dot], norm) = pair.accumulate::<true>(row);
            if !norm.is_finite() {
                return Err(invalid(format!(
                    "candidate row {i} contains nonfinite values"
                )));
            }
            Ok((
                norm,
                cosine_from_dot(raw_dot, norm, original_norm, FLOOR),
                cosine_from_dot(enhanced_dot, norm, enhanced_norm, FLOOR),
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
            candidate_norms,
            original_cosines,
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
