// SPDX-License-Identifier: Apache-2.0

use super::curve::{DtscCurveContext, empty_eval, eval_path};
use super::{
    AuxiliaryEvidence, DtscEvidenceConfig, DtscFieldSampler, DtscFieldValues, DtscGeometryState,
    unit,
};
use nexus_core::kernel::norm_sq_f32;
use nexus_core::{
    CandidateId, ExecutionContext, Operator, PathId, PrecisionMode, QueryCosines,
    TransportGraphView, VcpError, VcpResult, VectorMatrixView,
};
use std::cmp::Ordering;

pub struct DtscEvidenceInput<'call> {
    pub field: &'call DtscFieldValues,
    pub query: &'call QueryCosines<'call, 'call>,
    pub graph: TransportGraphView<'call>,
    pub tag_vectors: VectorMatrixView<'call, f32>,
    pub geometry: DtscGeometryState,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DtscItem {
    pub candidate: CandidateId,
    pub chosen_path: Option<PathId>,
    pub score: f64,
    pub original_cosine: f64,
    pub geo_score: f64,
    pub normalized_geo: f64,
    pub geo_bonus: f64,
    pub geo_base_bonus: f64,
    pub geo_aux_bonus: f64,
    pub auxiliary_evidence: Option<AuxiliaryEvidence>,
    pub geo_effect: &'static str,
    pub geo_evidence_class: &'static str,
    pub geo_reward_eligible: bool,
    pub geo_confidence: f64,
    pub geo_exact_hits: usize,
    pub geo_strong_hits: usize,
    pub geo_hit_count: usize,
    pub geo_continuity: f64,
    pub geo_isolated_ratio: f64,
}

/// Unmodified geometric scores in candidate input order, with pool statistics for readout.
#[derive(Clone, Debug)]
pub struct DtscEvidenceValues {
    items: Vec<DtscItem>,
    field_entropy: f64,
    field_has_support: bool,
    min_field_entropy: f64,
    pub(super) stats: CandidateEvidenceStats,
}

impl DtscEvidenceValues {
    pub fn items(&self) -> &[DtscItem] {
        &self.items
    }

    pub fn field_entropy(&self) -> f64 {
        self.field_entropy
    }

    pub(super) fn field_trusted(&self, enforce_entropy: bool) -> bool {
        self.field_has_support && (!enforce_entropy || self.field_entropy >= self.min_field_entropy)
    }
}

#[derive(Clone, Debug)]
pub(super) struct CandidateEvidenceStats {
    pub contributors: usize,
    pub maximum: f64,
    pub minimum: f64,
    pub strong_evidence: f64,
}

impl CandidateEvidenceStats {
    fn from_items(items: &[DtscItem]) -> Self {
        let mut stats = Self {
            contributors: 0,
            maximum: 0.0,
            minimum: f64::INFINITY,
            strong_evidence: 0.0,
        };
        for item in items.iter().filter(|item| item.geo_score > 0.0) {
            stats.contributors += 1;
            stats.maximum = stats.maximum.max(item.geo_score);
            stats.minimum = stats.minimum.min(item.geo_score);
            stats.strong_evidence +=
                item.geo_strong_hits as f64 + 0.75 * item.geo_exact_hits as f64;
        }
        stats
    }
}

pub struct DtscEvidenceOperator;

impl Operator for DtscEvidenceOperator {
    type Config = DtscEvidenceConfig;
    type Input<'call> = DtscEvidenceInput<'call>;
    type Output = DtscEvidenceValues;

    fn name(&self) -> &'static str {
        "dtsc_evidence"
    }

    fn execute<'call>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'call>,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        config.validate()?;
        input.tag_vectors.validate(stage)?;
        let rows = input.query.rows();
        let matrix = rows.candidate_vectors;
        let dim = input.tag_vectors.dim;
        if dim == 0 {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "DTSC dimension must be positive".into(),
            });
        }
        for (expected, actual) in [
            (dim, matrix.dim),
            (input.tag_vectors.rows, input.graph.node_count),
        ] {
            if actual != expected {
                return Err(VcpError::DimensionMismatch {
                    stage,
                    expected,
                    actual,
                });
            }
        }

        let sampler = DtscFieldSampler::new(input.field, input.graph.inbound);
        let mut field_samples = vec![None; input.tag_vectors.rows];
        let mut neighbors = Vec::new();
        for &node in rows.path_tags {
            if let Some(slot) = field_samples.get_mut(node.index())
                && slot.is_none()
            {
                ctx.check_cancelled(stage)?;
                let vector = input
                    .tag_vectors
                    .row(node.index())
                    .expect("validated tag row");
                *slot = Some((
                    sampler.sample_with_buf(node, vector, &mut neighbors),
                    norm_sq_f32(vector, PrecisionMode::StrictLeftFold),
                ));
            }
        }

        let field_config = input.field.config();
        let curve_ctx = DtscCurveContext {
            config,
            hub_floor: field_config.public_hub_floor,
            field_samples: &field_samples,
            graph: input.graph,
            tag_vectors: &input.tag_vectors,
            geometry: &input.geometry,
            max_inbound: sampler.max_inbound,
        };
        let original = input.query.original_cosines();
        let enhanced = input.query.enhanced_cosines();
        let norms = matrix.sq_norms_f64.expect("strict query cosine norms");
        let mut items = Vec::with_capacity(rows.candidate_count);
        let mut samples = Vec::new();
        let mut isolated = Vec::new();
        let mut connected = Vec::new();
        for i in 0..rows.candidate_count {
            ctx.check_cancelled(stage)?;
            let candidate = CandidateId(i as u32);
            let closures = [original[i], enhanced[i]].map(|cosine| unit((cosine + 1.0) / 2.0));
            let base = empty_eval(candidate, original[i]);
            let mut best = base;
            let vector = matrix.row(i).expect("validated candidate row");
            for path in rows.candidate_paths(candidate).map(|p| PathId(p as u32)) {
                ctx.check_cancelled(stage)?;
                let (tags, positions) = rows.path_steps(path);
                let mut eval = eval_path(
                    base,
                    closures,
                    vector,
                    norms[i],
                    tags,
                    positions,
                    &curve_ctx,
                    &mut samples,
                    &mut isolated,
                    &mut connected,
                );
                eval.chosen_path = Some(path);
                let comparison = eval
                    .score
                    .total_cmp(&best.score)
                    .then_with(|| eval.geo_score.total_cmp(&best.geo_score));
                if best.chosen_path.is_none() || comparison == Ordering::Greater {
                    best = eval;
                }
            }
            items.push(best.into_item());
        }

        Ok(DtscEvidenceValues {
            stats: CandidateEvidenceStats::from_items(&items),
            items,
            field_entropy: input.field.field_entropy(),
            field_has_support: input.field.input_node_count() >= field_config.min_field_tags
                && input.field.retained_node_count() >= field_config.min_field_tags
                && input.field.total_energy() > 0.0,
            min_field_entropy: field_config.min_field_entropy,
        })
    }
}
