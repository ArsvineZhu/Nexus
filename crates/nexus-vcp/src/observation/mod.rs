// SPDX-License-Identifier: CC-BY-NC-SA-4.0

mod analysis;
mod fields;
mod fusion;
mod gating;
mod projection;
mod sensing;

pub use analysis::*;
pub use fields::*;
pub use fusion::*;
pub use gating::*;
pub use projection::*;
pub use sensing::*;

use nexus_core::{
    CsrGraphView, EpaBasisView, ExecutionContext, NodeId, Operator, TagLexicalMetaView, VcpError,
    VcpResult, VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub(crate) fn invalid(stage: &'static str, detail: &'static str) -> VcpError {
    VcpError::InvalidInput {
        stage,
        detail: detail.into(),
    }
}

pub(crate) fn numeric_failure(stage: &'static str, detail: &'static str) -> VcpError {
    VcpError::NumericDivergence {
        stage,
        detail: detail.into(),
    }
}

pub(crate) fn finite_vector(stage: &'static str, vector: &[f32], dim: usize) -> VcpResult<()> {
    if vector.len() != dim {
        return Err(VcpError::DimensionMismatch {
            stage,
            expected: dim,
            actual: vector.len(),
        });
    }
    if !nexus_core::kernel::all_finite_f32(vector) {
        return Err(invalid(stage, "vector must be finite"));
    }
    Ok(())
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservationConfig {
    pub epa: EpaConfig,
    pub pyramid: PyramidConfig,
    pub hypernym: HypernymConfig,
    pub gating: TagGatingConfig,
    pub sense: SenseConfig,
    pub fusion: ObservationFuseConfig,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            epa: EpaConfig::default(),
            pyramid: PyramidConfig::default(),
            hypernym: HypernymConfig::default(),
            gating: TagGatingConfig::default(),
            sense: SenseConfig {
                max_transition_records: 16_000,
                ..SenseConfig::default()
            },
            fusion: ObservationFuseConfig::default(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct ObservationInput<'asset> {
    pub query: &'asset [f32],
    pub query_text: &'asset str,
    pub candidates: &'asset PyramidCandidates<'asset>,
    pub graph: CsrGraphView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub lexical_meta: TagLexicalMetaView<'asset>,
    pub epa_basis: Option<EpaBasisView<'asset>>,
    pub core_tags: &'asset [NodeId],
    pub ghost_vectors: Option<VectorMatrixView<'asset, f32>>,
    pub ghost_is_core: &'asset [bool],
}

/// The owned intermediate state shared by readouts and planning for one query.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationState {
    pub original_query: Vec<f32>,
    pub epa: EpaAnalysis,
    pub pyramid: PyramidResidual,
    pub disambiguated: DisambiguatedTags,
    pub gated_seeds: GatedSeeds,
    pub sense: SenseOutput,
    pub fused: FusedObservation,
}

/// Composes query geometry, tag selection, propagation and enhancement.
pub struct Observe;

impl Operator for Observe {
    type Config = ObservationConfig;
    type Input<'asset, 'query> = ObservationInput<'asset>;
    type Output = Arc<ObservationState>;

    fn name(&self) -> &'static str {
        "observe"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        cfg: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let epa = EpaAnalyzeOperator.execute(
            ctx,
            &cfg.epa,
            EpaAnalyzeInput {
                query: input.query,
                basis: input.epa_basis,
            },
        )?;
        let pyramid = PyramidAnalyzeOperator.execute(
            ctx,
            &cfg.pyramid,
            PyramidAnalyzeInput {
                query: input.query,
                candidates: input.candidates,
                epa: Some(&epa),
                tag_vectors: input.tag_vectors,
            },
        )?;
        let disambiguated = HypernymPruneOperator.execute(
            ctx,
            &cfg.hypernym,
            HypernymPruneInput {
                query: input.query,
                pyramid: &pyramid,
                tag_vectors: input.tag_vectors,
                lexical_meta: input.lexical_meta,
                core_tags: input.core_tags,
                seed_contribution_floor: cfg.gating.seed_contribution_floor,
                seed_contribution_ratio: cfg.gating.seed_contribution_ratio,
            },
        )?;
        let gated_seeds = TagGatingOperator.execute(
            ctx,
            &cfg.gating,
            TagGatingInput {
                epa: &epa,
                pyramid: &pyramid,
                disambiguated: &disambiguated,
                query_text: input.query_text,
                core_tags: input.core_tags,
                tag_vectors: input.tag_vectors,
                lexical_meta: input.lexical_meta,
                graph_node_count: input.graph.node_count,
            },
        )?;
        let sense = SpikeSenseOperator.execute(
            ctx,
            &cfg.sense,
            SenseInput {
                graph: input.graph,
                tag_vectors: input.tag_vectors,
                seeds: &gated_seeds,
                residuals: Some(&pyramid.residuals),
            },
        )?;
        let fused = ObservationFuseOperator.execute(
            ctx,
            &cfg.fusion,
            ObservationFuseInput {
                query: input.query,
                gated: &gated_seeds,
                sense: &sense,
                core_tags: input.core_tags,
                ghost_vectors: input.ghost_vectors,
                ghost_is_core: input.ghost_is_core,
                tag_vectors: input.tag_vectors,
            },
        )?;
        Ok(Arc::new(ObservationState {
            original_query: input.query.to_vec(),
            epa,
            pyramid,
            disambiguated,
            gated_seeds,
            sense,
            fused,
        }))
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldConfig {
    pub solve: DualFieldConfig,
    pub basins: BasinProjectionConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldState {
    observation: Arc<ObservationState>,
    pub dual_fields: DualFields,
    pub basins: BasinSupports,
}

impl FieldState {
    /// Import a field solution and basin projection alongside the observation they used.
    pub fn from_parts(
        observation: Arc<ObservationState>,
        dual_fields: DualFields,
        basins: BasinSupports,
    ) -> Self {
        Self {
            observation,
            dual_fields,
            basins,
        }
    }

    pub fn observation(&self) -> &ObservationState {
        &self.observation
    }
}

#[derive(Clone, Copy)]
pub struct FieldInput<'asset, 'query> {
    pub graph: CsrGraphView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset>,
    pub observation: &'query Arc<ObservationState>,
}

pub struct ComputeFields;

impl Operator for ComputeFields {
    type Config = FieldConfig;
    type Input<'asset, 'query> = FieldInput<'asset, 'query>;
    type Output = FieldState;

    fn name(&self) -> &'static str {
        "fields"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        cfg: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let dual_fields = DualFieldSolveOperator.execute(
            ctx,
            &cfg.solve,
            DualFieldSolveInput {
                graph: input.graph,
                fused: &input.observation.fused,
            },
        )?;
        let basins = BasinProjectionOperator.execute(
            ctx,
            &cfg.basins,
            BasinProjectionInput {
                graph: input.graph,
                tag_vectors: input.tag_vectors,
                sense: &input.observation.sense,
                dual_fields: &dual_fields,
            },
        )?;
        Ok(FieldState::from_parts(
            Arc::clone(input.observation),
            dual_fields,
            basins,
        ))
    }
}
