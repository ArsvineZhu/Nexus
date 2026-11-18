// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::observation::{finite_vector, invalid};
use nexus_core::kernel::{dot_f32, norm_sq_f32};
use nexus_core::{
    CandidateId, Diagnostics, ExecutionContext, Operator, PrecisionMode, ScratchPad, VcpResult,
    VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CosineDedupConfig {
    pub threshold: f64,
    pub bound_block: usize,
    pub bound_slow_budget: usize,
}

impl Default for CosineDedupConfig {
    fn default() -> Self {
        Self {
            threshold: 0.92,
            bound_block: 256,
            bound_slow_budget: 64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CosineDedupInput<'asset, 'query> {
    pub ranked: &'query [CandidateId],
    pub candidate_vectors: VectorMatrixView<'asset, f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FoldedMapping {
    pub folded: CandidateId,
    pub representative: CandidateId,
    pub cosine: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DedupResult {
    pub survivors: Vec<CandidateId>,
    pub folded: Vec<FoldedMapping>,
    pub diagnostics: Diagnostics,
}

struct Representative<'a> {
    candidate: CandidateId,
    vector: &'a [f32],
    norm: f64,
    suffix: Vec<f64>,
}

fn fill_suffix(vector: &[f32], block: usize, norms: &mut Vec<f64>) {
    norms.clear();
    norms.push(0.0);
    let mut sum = 0.0;
    for values in vector.chunks(block).rev() {
        sum += norm_sq_f32(values, PrecisionMode::StrictLeftFold);
        norms.push(sum.sqrt());
    }
    norms.reverse();
}

/// Pruning is conservative under FP64 roundoff. Surviving comparisons retain
/// the same left-fold as the unbounded path.
fn bounded_cosine(
    prior: &Representative<'_>,
    vector: &[f32],
    norm: f64,
    suffix: &[f64],
    block: usize,
    threshold: f64,
) -> Option<f64> {
    let denominator = prior.norm * norm;
    let floor = threshold * denominator;
    let guard = 16.0 * (vector.len() as f64 + 1.0) * f64::EPSILON * denominator;
    let mut dot = -0.0;
    for (index, (a, b)) in vector
        .chunks(block)
        .zip(prior.vector.chunks(block))
        .enumerate()
    {
        for (&a, &b) in a.iter().zip(b) {
            dot += f64::from(a) * f64::from(b);
        }
        if dot + suffix[index + 1] * prior.suffix[index + 1] + guard < floor {
            return None;
        }
    }
    Some(dot / denominator)
}

pub struct CosineDedupOperator;

impl Operator for CosineDedupOperator {
    type Config = CosineDedupConfig;
    type Input<'asset, 'query> = CosineDedupInput<'asset, 'query>;
    type Output = DedupResult;

    fn name(&self) -> &'static str {
        "cosine.dedup"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        scratch: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        input.candidate_vectors.validate(stage)?;
        if !(-1.0..=1.0).contains(&config.threshold) || config.bound_block == 0 {
            return Err(invalid(stage, "invalid cosine cutoff or block size"));
        }
        let mut result = DedupResult::default();
        let mut unique = HashSet::new();
        let mut representatives = Vec::<Representative<'_>>::new();
        let block = config.bound_block;
        let bounded = input.candidate_vectors.dim > block && config.bound_slow_budget > 0;
        let mode = PrecisionMode::StrictLeftFold;
        for &candidate in input.ranked {
            ctx.check_cancelled(stage)?;
            if !unique.insert(candidate) {
                continue;
            }
            let vector = input
                .candidate_vectors
                .row(candidate.index())
                .ok_or_else(|| invalid(stage, "rank contains a candidate outside the matrix"))?;
            finite_vector(stage, vector, input.candidate_vectors.dim)?;
            let norm = norm_sq_f32(vector, mode).sqrt();
            if bounded {
                fill_suffix(vector, block, &mut scratch.f64_buf_a);
            }
            let mut matched = None;
            for (index, prior) in representatives.iter().enumerate() {
                ctx.check_cancelled(stage)?;
                if norm <= 1e-12 || prior.norm <= 1e-12 {
                    continue;
                }
                let cosine = if bounded && index < config.bound_slow_budget {
                    bounded_cosine(
                        prior,
                        vector,
                        norm,
                        &scratch.f64_buf_a,
                        block,
                        config.threshold,
                    )
                } else {
                    Some(dot_f32(vector, prior.vector, mode) / (norm * prior.norm))
                };
                if let Some(cosine) = cosine.filter(|cosine| *cosine >= config.threshold) {
                    matched = Some((prior.candidate, cosine));
                    break;
                }
            }
            if let Some((representative, cosine)) = matched {
                result.folded.push(FoldedMapping {
                    folded: candidate,
                    representative,
                    cosine,
                });
            } else {
                result.survivors.push(candidate);
                representatives.push(Representative {
                    candidate,
                    vector,
                    norm,
                    suffix: if bounded && representatives.len() < config.bound_slow_budget {
                        scratch.f64_buf_a.clone()
                    } else {
                        Vec::new()
                    },
                });
            }
        }
        Ok(result)
    }
}
