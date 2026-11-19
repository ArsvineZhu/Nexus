// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::{invalid, slot};
use crate::query::PyramidRecall;
use crate::{Nexus, Query};
use nexus_core::kernel::normalize_direction;
use nexus_core::{ExecutionContext, NodeId, Operator, VcpResult, VectorMatrixView};
use nexus_math::{
    DtscFieldBuildInput, DtscFieldBuildOperator, DtscFieldConfig, DtscFieldValues, DtscNodeInput,
    DtscSourceType,
};
use nexus_vcp::{
    ComputeFields, FieldConfig, FieldInput, FieldValues, ObservationConfig, ObservationInput,
    ObservationState, Observe, SeedSourceKind,
};

pub struct Observation<'data> {
    pub(crate) nexus: &'data Nexus<'data>,
    ctx: ExecutionContext<'data>,
    values: ObservationState,
}

pub struct FieldState<'obs, 'data> {
    pub(crate) observation: &'obs Observation<'data>,
    values: FieldValues,
}

pub struct DtscField<'obs, 'data> {
    pub(crate) observation: &'obs Observation<'data>,
    values: DtscFieldValues,
}

impl Nexus<'_> {
    pub fn observe<'data>(
        &'data self,
        query: &Query<'_>,
        config: &ObservationConfig,
        ctx: ExecutionContext<'data>,
    ) -> VcpResult<Observation<'data>> {
        const STAGE: &str = "observe";
        ctx.check_cancelled(STAGE)?;
        let assets = self.prepared();
        let recall = query
            .tag_recall
            .ok_or_else(|| invalid(STAGE, "supply a concept recall provider on Query"))?;
        let original = normalize_direction(query.vector, assets.tags.dim, STAGE)?;
        let core_tags = query
            .core_concepts
            .iter()
            .map(|&id| NodeId::from_index(slot(&self.concept_slots, id)?, STAGE))
            .collect::<VcpResult<Vec<_>>>()?;
        let adapter = PyramidRecall::new(self, &ctx, recall);
        let recall = |layer,
                      residual: &[f32],
                      limit,
                      out: &mut Vec<NodeId>,
                      epa: Option<&nexus_vcp::EpaAnalysis>,
                      excluded: &[NodeId]| {
            adapter.recall(layer, residual, limit, out, epa, excluded)
        };
        let mut ghosts = Vec::new();
        let ghost_vectors = if let Some(matrix) = query.ghost_vectors {
            matrix.validate(STAGE)?;
            if matrix.dim != assets.tags.dim {
                return Err(invalid(
                    STAGE,
                    "ghost dimensions must match the concept vectors",
                ));
            }
            for row in matrix.data.chunks_exact(matrix.dim) {
                ghosts.extend(normalize_direction(row, assets.tags.dim, STAGE)?);
            }
            Some(VectorMatrixView::new(&ghosts, matrix.rows, matrix.dim)?)
        } else {
            None
        };
        let values = Observe.execute(
            &ctx,
            config,
            ObservationInput {
                query: &original,
                query_text: query.text,
                candidates: &recall,
                graph: assets.graph,
                tag_vectors: assets.tags,
                lexical_meta: assets.lexical,
                epa_basis: assets.basis,
                core_tags: &core_tags,
                ghost_vectors,
                ghost_is_core: query.ghost_is_core,
            },
        )?;
        Ok(Observation {
            nexus: self,
            ctx,
            values,
        })
    }
}

impl<'data> Observation<'data> {
    pub fn context(&self) -> &ExecutionContext<'data> {
        &self.ctx
    }
    pub fn values(&self) -> &ObservationState {
        &self.values
    }

    pub fn fields<'obs>(&'obs self, config: &FieldConfig) -> VcpResult<FieldState<'obs, 'data>> {
        let assets = self.nexus.prepared();
        let values = ComputeFields.execute(
            &self.ctx,
            config,
            FieldInput {
                graph: assets.graph,
                tag_vectors: assets.tags,
                observation: &self.values,
            },
        )?;
        Ok(FieldState {
            observation: self,
            values,
        })
    }

    pub fn dtsc_field<'obs>(
        &'obs self,
        config: &DtscFieldConfig,
    ) -> VcpResult<DtscField<'obs, 'data>> {
        let nodes = self
            .values
            .sense
            .nodes
            .iter()
            .map(|node| DtscNodeInput {
                node: node.node,
                energy: node.energy,
                normalized_energy: node.normalized_energy,
                source_type: if node.hop > 0 {
                    DtscSourceType::Emergent
                } else if node.source_kind == SeedSourceKind::Core {
                    DtscSourceType::Core
                } else {
                    DtscSourceType::Seed
                },
            })
            .collect::<Vec<_>>();
        let values = DtscFieldBuildOperator.execute(
            &self.ctx,
            config,
            DtscFieldBuildInput {
                nodes: &nodes,
                tag_vectors: self.nexus.prepared().tags,
            },
        )?;
        Ok(DtscField {
            observation: self,
            values,
        })
    }
}

impl FieldState<'_, '_> {
    pub fn values(&self) -> &FieldValues {
        &self.values
    }
}

impl DtscField<'_, '_> {
    pub fn values(&self) -> &DtscFieldValues {
        &self.values
    }
}
