// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::{invalid, slot};
use crate::{CandidateSet, Nexus, Observation, TagRecall, TagRecallRequest};
use nexus_core::kernel::norm_sq_f32;
use nexus_core::{ExecutionContext, NodeId, PrecisionMode, VcpResult};
use nexus_vcp::geometry::IntrinsicResidual;
use nexus_vcp::hyperbolic::{
    ChunkField, SearchConfig, WarpedConfig, WarpedIndex, residual_radius, tag_field,
};

const STAGE: &str = "recall";

pub struct RecallConfig {
    pub concepts: Option<WarpedConfig>,
    pub documents: Option<WarpedConfig>,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            concepts: Some(WarpedConfig::default()),
            documents: Some(WarpedConfig::default()),
        }
    }
}

pub struct RecallRequest<'a> {
    pub limit: usize,
    pub allowed: Option<&'a [u64]>,
    pub search: SearchConfig,
}

impl RecallRequest<'_> {
    pub fn top_k(limit: usize) -> Self {
        Self {
            limit,
            allowed: None,
            search: SearchConfig::default(),
        }
    }
}

pub struct RecallIndexes<'data> {
    nexus: &'data Nexus<'data>,
    concepts: Option<WarpedIndex>,
    documents: Option<ChunkField>,
}

impl Nexus<'_> {
    pub fn prepare_recall<'data>(
        &'data self,
        config: &RecallConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RecallIndexes<'data>> {
        ctx.check_cancelled(STAGE)?;
        let assets = self.prepared();
        if config.concepts.is_none() && config.documents.is_none() {
            return Err(invalid(STAGE, "select a concept or document index"));
        }
        let documents = config
            .documents
            .map(|config| {
                let catalog = assets
                    .documents
                    .ok_or_else(|| invalid(STAGE, "document recall requires a catalog"))?;
                ChunkField::build(catalog.vectors, config, ctx)
            })
            .transpose()?;
        let concepts = config
            .concepts
            .map(|config| {
                let default = |node| IntrinsicResidual {
                    node,
                    residual_ratio: 1.0,
                    neighbor_count: 0,
                    basis_used: 0,
                    mgs2_fallback: false,
                };
                let mut residuals = (0..assets.tags.rows)
                    .map(|i| NodeId::from_index(i, STAGE).map(default))
                    .collect::<VcpResult<Vec<_>>>()?;
                for value in self.tag_residuals() {
                    residuals[value.node.index()] = value.clone();
                }
                tag_field(
                    assets.tags,
                    assets.basis.map_or(&[], |basis| basis.mean),
                    assets.graph.inbound,
                    &residuals,
                    config,
                    ctx,
                )
            })
            .transpose()?;
        Ok(RecallIndexes {
            nexus: self,
            concepts,
            documents,
        })
    }
}

impl TagRecall for RecallIndexes<'_> {
    fn recall(
        &self,
        ctx: &ExecutionContext<'_>,
        request: TagRecallRequest<'_>,
        out: &mut Vec<u64>,
    ) -> VcpResult<()> {
        ctx.check_cancelled(STAGE)?;
        let index = self
            .concepts
            .as_ref()
            .ok_or_else(|| invalid(STAGE, "concept index was not prepared"))?;
        let assets = self.nexus.prepared();
        let mut scope = vec![true; assets.tags.rows];
        for id in request.excluded_concepts {
            scope[slot(&self.nexus.concept_slots, *id)?] = false;
        }
        let energy = norm_sq_f32(request.residual, PrecisionMode::FastSimd);
        if request.limit == 0 || energy <= 1e-12 {
            return Ok(());
        }
        let radius = residual_radius(request.epa, request.layer, energy);
        let search = SearchConfig {
            rerank: Some(64),
            ..SearchConfig::default()
        };
        let hits = index.search(
            request.residual,
            radius,
            request.limit.min(64),
            Some(&scope),
            search,
            ctx,
        )?;
        out.extend(hits.into_iter().map(|hit| assets.concept_ids[hit.slot]));
        Ok(())
    }
}

impl<'data> Observation<'data> {
    pub fn recall(
        &self,
        indexes: &RecallIndexes<'data>,
        request: &RecallRequest<'_>,
    ) -> VcpResult<CandidateSet<'data>> {
        if !std::ptr::eq(self.nexus, indexes.nexus) {
            return Err(invalid(STAGE, "recall indexes belong to a different Nexus"));
        }
        self.context().check_cancelled(STAGE)?;
        let documents = indexes
            .documents
            .as_ref()
            .ok_or_else(|| invalid(STAGE, "document index was not prepared"))?;
        let scope = request
            .allowed
            .map(|ids| {
                let mut scope = vec![false; self.nexus.document_slots.len()];
                for &id in ids {
                    scope[slot(&self.nexus.document_slots, id)?] = true;
                }
                Ok(scope)
            })
            .transpose()?;
        if request.limit == 0 {
            return CandidateSet::from_catalog_slots(self.nexus, []);
        }
        let hits = documents.search(
            &self.values().original_query,
            request.limit,
            scope.as_deref(),
            request.search,
            self.context(),
        )?;
        CandidateSet::from_catalog_slots(self.nexus, hits.into_iter().map(|hit| hit.slot))
    }
}
