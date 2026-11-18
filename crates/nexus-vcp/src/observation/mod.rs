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
    CsrGraphView, EpaBasisView, ExecutionContext, NodeId, Operator, ScratchPad, TagLexicalMetaView,
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

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservationConfig {
    pub epa: EpaConfig,
    pub pyramid: PyramidConfig,
    pub hypernym: HypernymConfig,
    pub gating: TagGatingConfig,
    pub sense: SenseConfig,
    pub fusion: ObservationFuseConfig,
    pub fields: DualFieldConfig,
    pub basins: BasinProjectionConfig,
}

#[derive(Clone, Copy)]
pub struct ObservationInput<'asset, 'query> {
    pub query: &'query [f32],
    pub query_text: &'query str,
    pub candidates: &'query PyramidCandidates<'query>,
    pub graph: CsrGraphView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub lexical_meta: TagLexicalMetaView<'asset>,
    pub epa_basis: Option<EpaBasisView<'asset>>,
    pub core_tags: &'query [NodeId],
    pub ghost_vectors: Option<VectorMatrixView<'query, f32>>,
    pub ghost_is_core: &'query [bool],
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
    pub dual_fields: DualFields,
    pub basins: BasinSupports,
}

/// Runs the standard observation chain on the caller's current graph view.
pub fn observe(
    ctx: &ExecutionContext<'_>,
    cfg: &ObservationConfig,
    input: ObservationInput<'_, '_>,
    scratch: &mut ScratchPad,
) -> VcpResult<ObservationState> {
    let epa = EpaAnalyzeOperator.execute(
        ctx,
        &cfg.epa,
        EpaAnalyzeInput {
            query: input.query,
            basis: input.epa_basis,
        },
        scratch,
    )?;
    let pyramid = PyramidAnalyzeOperator.execute(
        ctx,
        &cfg.pyramid,
        PyramidAnalyzeInput {
            query: input.query,
            candidates: input.candidates,
            tag_vectors: input.tag_vectors,
        },
        scratch,
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
        scratch,
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
        scratch,
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
        scratch,
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
        scratch,
    )?;
    let dual_fields = DualFieldSolveOperator.execute(
        ctx,
        &cfg.fields,
        DualFieldSolveInput {
            graph: input.graph,
            fused: &fused,
        },
        scratch,
    )?;
    let basins = BasinProjectionOperator.execute(
        ctx,
        &cfg.basins,
        BasinProjectionInput {
            graph: input.graph,
            tag_vectors: input.tag_vectors,
            sense: &sense,
            dual_fields: &dual_fields,
        },
        scratch,
    )?;
    Ok(ObservationState {
        original_query: input.query.to_vec(),
        epa,
        pyramid,
        disambiguated,
        gated_seeds,
        sense,
        fused,
        dual_fields,
        basins,
    })
}
