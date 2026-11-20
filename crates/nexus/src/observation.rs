// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::{invalid, slot};
use crate::query::PyramidRecall;
use crate::{Nexus, Query};
use nexus_core::csr::EffectiveTransport;
use nexus_core::kernel::normalize_direction;
use nexus_core::{ExecutionContext, NodeId, VcpResult, VectorMatrixView};
use nexus_math::{
    DtscFieldBuildInput, DtscFieldConfig, DtscFieldValues, DtscNodeInput, DtscSourceType,
    build_dtsc_field,
};
use nexus_vcp::{
    FieldConfig, FieldInput, FieldValues, ObservationConfig, ObservationInput, ObservationState,
    SeedSourceKind, compute_fields, observe,
};

pub struct Observation<'data> {
    pub(crate) nexus: &'data Nexus<'data>,
    effective: Option<EffectiveTransport<'data>>,
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
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Observation<'data>> {
        const STAGE: &str = "observe";
        ctx.check_cancelled(STAGE)?;
        let assets = self.prepared();
        let effective = query
            .edge_gates
            .map(|gates| assets.graph.attenuate(gates))
            .transpose()?;
        let graph = effective
            .as_ref()
            .map_or(assets.graph, EffectiveTransport::view);
        let original = normalize_direction(query.vector, assets.tags.dim, STAGE)?;
        let core_tags = query
            .core_concepts
            .iter()
            .map(|&id| NodeId::from_index(slot(&self.concept_slots, id)?, STAGE))
            .collect::<VcpResult<Vec<_>>>()?;
        let recall = query.tag_recall.map(|provider| {
            let adapter = PyramidRecall::new(self, ctx, provider);
            move |layer,
                  residual: &[f32],
                  limit,
                  out: &mut Vec<NodeId>,
                  epa: Option<&nexus_vcp::EpaAnalysis>,
                  excluded: &[NodeId]| {
                adapter.recall(layer, residual, limit, out, epa, excluded)
            }
        });
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
        let values = observe(
            ObservationInput {
                query: &original,
                query_text: query.text,
                candidates: recall
                    .as_ref()
                    .map(|callback| callback as &nexus_vcp::PyramidCandidates<'_>),
                graph,
                tag_vectors: assets.tags,
                lexical_meta: assets.lexical,
                epa_basis: assets.basis,
                core_tags: &core_tags,
                ghost_vectors,
                ghost_is_core: query.ghost_is_core,
            },
            config,
            ctx,
        )?;
        Ok(Observation {
            nexus: self,
            effective,
            values,
        })
    }
}

impl<'data> Observation<'data> {
    pub fn graph(&self) -> nexus_core::TransportGraphView<'_> {
        self.effective
            .as_ref()
            .map_or_else(|| self.nexus.prepared().graph, EffectiveTransport::view)
    }
    pub fn values(&self) -> &ObservationState {
        &self.values
    }

    pub fn fields<'obs>(
        &'obs self,
        config: &FieldConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<FieldState<'obs, 'data>> {
        let assets = self.nexus.prepared();
        let values = compute_fields(
            FieldInput {
                graph: self.graph(),
                tag_vectors: assets.tags,
                observation: &self.values,
            },
            config,
            ctx,
        )?;
        Ok(FieldState {
            observation: self,
            values,
        })
    }

    pub fn dtsc_field<'obs>(
        &'obs self,
        config: &DtscFieldConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<DtscField<'obs, 'data>> {
        let nodes = self
            .values
            .sense
            .nodes
            .iter()
            .map(|node| DtscNodeInput {
                node: node.node,
                mass: node.energy,
                source_type: match node.source_kind {
                    SeedSourceKind::Core => DtscSourceType::Core,
                    SeedSourceKind::LexicalSeed => DtscSourceType::Seed,
                    SeedSourceKind::Seed => DtscSourceType::Inferred,
                    SeedSourceKind::Emergent => DtscSourceType::Emergent,
                },
            })
            .collect::<Vec<_>>();
        let values = build_dtsc_field(
            DtscFieldBuildInput {
                nodes: &nodes,
                tag_vectors: self.nexus.prepared().tags,
            },
            config,
            ctx,
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
