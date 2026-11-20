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
    EpaBasisView, ExecutionContext, NodeId, TagLexicalMetaView, TransportGraphView, VcpError,
    VcpResult, VectorMatrixView,
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
    pub pyramid: Option<PyramidConfig>,
    pub seeds: SeedSelectionConfig,
    pub sense: SenseConfig,
    pub fusion: ObservationFuseConfig,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            epa: EpaConfig::default(),
            pyramid: Some(PyramidConfig::default()),
            seeds: SeedSelectionConfig::default(),
            sense: SenseConfig {
                max_transition_records: Some(16_000),
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
    pub candidates: Option<&'call PyramidCandidates<'call>>,
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

pub fn observe(
    input: ObservationInput<'_>,
    cfg: &ObservationConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<ObservationState> {
    let epa = analyze_epa(
        EpaAnalyzeInput {
            query: input.query,
            basis: input.epa_basis,
        },
        &cfg.epa,
        ctx,
    )?;
    let pyramid = if let Some(config) = &cfg.pyramid {
        analyze_pyramid(
            PyramidAnalyzeInput {
                query: input.query,
                candidates: input.candidates,
                epa: Some(&epa),
                tag_vectors: input.tag_vectors,
            },
            config,
            ctx,
        )?
    } else {
        PyramidResidual {
            novelty: 1.0,
            ..PyramidResidual::default()
        }
    };
    let SeedSelection {
        disambiguated,
        gated_seeds,
    } = select_seeds(
        SeedSelectionInput {
            query: input.query,
            epa: &epa,
            pyramid: &pyramid,
            query_text: input.query_text,
            core_tags: input.core_tags,
            tag_vectors: input.tag_vectors,
            lexical_meta: input.lexical_meta,
        },
        &cfg.seeds,
        ctx,
    )?;
    let sense = sense(
        SenseInput {
            graph: input.graph,
            tag_vectors: input.tag_vectors,
            seeds: &gated_seeds,
            residuals: Some(&pyramid.residuals),
        },
        &cfg.sense,
        ctx,
    )?;
    let fused = fuse_observation(
        ObservationFuseInput {
            query: input.query,
            gated: &gated_seeds,
            sense: &sense,
            core_tags: input.core_tags,
            ghost_vectors: input.ghost_vectors,
            ghost_is_core: input.ghost_is_core,
            tag_vectors: input.tag_vectors,
        },
        &cfg.fusion,
        ctx,
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

pub fn compute_fields(
    input: FieldInput<'_>,
    cfg: &FieldConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<FieldValues> {
    let source_field: Vec<_> = input
        .observation
        .sense
        .nodes
        .iter()
        .map(|node| (node.node, node.energy))
        .collect();
    let dual_fields = solve_dual_fields(
        DualFieldSolveInput {
            graph: input.graph,
            source_field: &source_field,
        },
        &cfg.solve,
        ctx,
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
    let basins = project_basins(
        BasinProjectionInput {
            graph: input.graph,
            tag_vectors: input.tag_vectors,
            sense: &input.observation.sense,
            dual_fields: &dual_fields,
        },
        &cfg.basins,
        ctx,
    )?;
    Ok(FieldValues {
        dual_fields,
        basins,
    })
}
