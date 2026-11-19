// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::FieldState;
use crate::observation::{finite_vector, invalid};
use nexus_core::kernel::{all_finite_f32, batch_cosines_f32};
use nexus_core::{
    CandidateId, CandidatePoolView, ExecutionContext, Operator, PathId, PrecisionMode, VcpResult,
    VectorMatrixView,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoolCalibrateConfig {
    pub compute_anchor_contacts: bool,
    /// Override the context when the pool feeds a readout requiring exact left folds.
    pub precision: Option<PrecisionMode>,
}

impl Default for PoolCalibrateConfig {
    fn default() -> Self {
        Self {
            compute_anchor_contacts: true,
            precision: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PoolCalibrateInput<'asset, 'query> {
    pub fields: &'query FieldState,
    pub candidates: CandidatePoolView<'asset>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreparedCandidatePool {
    pub candidate_count: usize,
    pub precision: PrecisionMode,
    pub query_cosines: Vec<f64>,
    pub denoised_cosines: Vec<f64>,
    pub local_cosines: Vec<f64>,
    pub transfer_cosines: Vec<f64>,
    pub normalized_bm25: Vec<f64>,
    pub domain_anchor_counts: Vec<u32>,
}

pub struct PoolCalibrateOperator;

impl PoolCalibrateOperator {
    /// Calibrate four cosine lanes in one matrix pass, retaining caller buffers.
    pub fn execute_into(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &PoolCalibrateConfig,
        input: PoolCalibrateInput<'_, '_>,
        output: &mut PreparedCandidatePool,
    ) -> VcpResult<()> {
        let stage = self.name();
        output.candidate_count = 0;
        ctx.check_cancelled(stage)?;
        let pool = input.candidates;
        pool.validate(stage)?;
        let fields = input.fields;
        let observation = fields.observation();
        let queries = [
            observation.original_query.as_slice(),
            observation.fused.enhanced_query.as_slice(),
            fields.basins.local_vector.as_slice(),
            fields.basins.transfer_vector.as_slice(),
        ];
        for query in queries {
            finite_vector(stage, query, pool.candidate_vectors.dim)?;
        }
        let values = pool.candidate_vectors.data;
        let finite = if ctx.parallel.is_parallel() && values.len() >= 262_144 {
            ctx.parallel
                .install(|| values.par_chunks(65_536).all(all_finite_f32))
        } else {
            all_finite_f32(values)
        };
        if !finite || pool.bm25_scores.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(invalid(
                stage,
                "candidate vectors and BM25 scores must be finite",
            ));
        }
        let count = pool.candidate_count;
        let precision = config.precision.unwrap_or(ctx.precision);
        let matrix = VectorMatrixView {
            sq_norms_f64: if precision == PrecisionMode::StrictLeftFold {
                None
            } else {
                pool.candidate_vectors.sq_norms_f64
            },
            ..pool.candidate_vectors
        };
        let lanes = [
            &mut output.query_cosines,
            &mut output.denoised_cosines,
            &mut output.local_cosines,
            &mut output.transfer_cosines,
        ];
        let mut columns = lanes.map(|lane| {
            lane.resize(count, 0.0);
            lane.as_mut_slice()
        });
        batch_cosines_f32(
            &matrix,
            &queries,
            &mut columns,
            precision,
            1e-12,
            ctx.parallel,
        )?;
        ctx.check_cancelled(stage)?;
        output.normalized_bm25.clear();
        output
            .normalized_bm25
            .extend(pool.bm25_scores.iter().map(|&score| {
                if pool.bm25_reference > 0.0 {
                    (score / pool.bm25_reference).min(1.0)
                } else {
                    0.0
                }
            }));
        output.domain_anchor_counts.resize(count, 0);
        output.domain_anchor_counts.fill(0);
        if config.compute_anchor_contacts {
            let dim = fields.dual_fields.local_field.len();
            let mut anchor_table = nexus_core::EpochTable::default();
            let mut counted_table = nexus_core::EpochTable::default();
            let (anchors, anchor_epoch) = anchor_table.next_epoch(dim);
            for &node in fields
                .dual_fields
                .local_domain
                .iter()
                .chain(&fields.dual_fields.transfer_domain)
            {
                *anchors
                    .get_mut(node.index())
                    .ok_or_else(|| invalid(stage, "anchor outside the observation"))? =
                    anchor_epoch;
            }
            for (index, result) in output.domain_anchor_counts.iter_mut().enumerate() {
                ctx.check_cancelled(stage)?;
                let candidate = CandidateId::from_index(index, stage)?;
                let mut best = 0;
                for path in pool.candidate_paths(candidate) {
                    let (counted, epoch) = counted_table.next_epoch(dim);
                    let mut count = 0;
                    for &node in pool.path_steps(PathId::from_index(path, stage)?).0 {
                        let slot = node.index();
                        if anchors.get(slot) == Some(&anchor_epoch) && counted[slot] != epoch {
                            counted[slot] = epoch;
                            count += 1;
                        }
                    }
                    best = best.max(count);
                }
                *result = best;
            }
        }
        output.candidate_count = count;
        output.precision = precision;
        Ok(())
    }
}

impl Operator for PoolCalibrateOperator {
    type Config = PoolCalibrateConfig;
    type Input<'asset, 'query> = PoolCalibrateInput<'asset, 'query>;
    type Output = PreparedCandidatePool;

    fn name(&self) -> &'static str {
        "pool.calibrate"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let mut output = PreparedCandidatePool::default();
        self.execute_into(ctx, config, input, &mut output)?;
        Ok(output)
    }
}
