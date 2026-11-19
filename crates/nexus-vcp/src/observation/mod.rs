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
    EpaBasisView, ExecutionContext, NodeId, Operator, TagLexicalMetaView, TransportGraphView,
    VcpError, VcpResult, VectorMatrixView,
};
use serde::{Deserialize, Serialize};

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

pub(crate) fn unit_direction(stage: &'static str, vector: &[f32], dim: usize) -> VcpResult<()> {
    if vector.len() != dim {
        return Err(VcpError::DimensionMismatch {
            stage,
            expected: dim,
            actual: vector.len(),
        });
    }
    nexus_core::kernel::validate_unit_direction(vector, stage)
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
pub struct ObservationInput<'call> {
    pub query: &'call [f32],
    pub query_text: &'call str,
    pub candidates: &'call PyramidCandidates<'call>,
    pub graph: TransportGraphView<'call>,
    pub tag_vectors: VectorMatrixView<'call, f32>,
    pub lexical_meta: TagLexicalMetaView<'call>,
    pub epa_basis: Option<EpaBasisView<'call>>,
    pub core_tags: &'call [NodeId],
    pub ghost_vectors: Option<VectorMatrixView<'call, f32>>,
    pub ghost_is_core: &'call [bool],
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
    type Input<'call> = ObservationInput<'call>;
    type Output = ObservationState;

    fn name(&self) -> &'static str {
        "observe"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        cfg: &Self::Config,
        input: Self::Input<'_>,
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
        Ok(ObservationState {
            original_query: input.query.to_vec(),
            epa,
            pyramid,
            disambiguated,
            gated_seeds,
            sense,
            fused,
        })
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
pub struct FieldValues {
    pub dual_fields: DualFields,
    pub basins: BasinSupports,
}

#[derive(Clone, Copy)]
pub struct FieldInput<'call> {
    pub graph: TransportGraphView<'call>,
    pub tag_vectors: VectorMatrixView<'call>,
    pub observation: &'call ObservationState,
}

pub struct ComputeFields;

impl Operator for ComputeFields {
    type Config = FieldConfig;
    type Input<'call> = FieldInput<'call>;
    type Output = FieldValues;

    fn name(&self) -> &'static str {
        "fields"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        cfg: &Self::Config,
        input: Self::Input<'_>,
    ) -> VcpResult<Self::Output> {
        let dual_fields = DualFieldSolveOperator.execute(
            ctx,
            &cfg.solve,
            DualFieldSolveInput {
                graph: input.graph,
                source_field: &input.observation.sense.source_field,
            },
        )?;
        for (stage, converged, residual, tolerance) in [
            (
                "field.local",
                dual_fields.local_converged,
                dual_fields.local_residual,
                cfg.solve.local_tolerance,
            ),
            (
                "field.transfer",
                dual_fields.transfer_converged,
                dual_fields.transfer_residual,
                cfg.solve.transfer_tolerance,
            ),
        ] {
            if !converged {
                return Err(VcpError::NotConverged {
                    stage,
                    iterations: dual_fields.iterations,
                    residual,
                    tolerance,
                });
            }
        }
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
        Ok(FieldValues {
            dual_fields,
            basins,
        })
    }
}
