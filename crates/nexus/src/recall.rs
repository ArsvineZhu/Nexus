// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::{invalid, slot};
use crate::{CandidateSet, Nexus, Observation, Query, TagRecallRequest};
use nexus_core::kernel::{norm_sq_f32, normalize_direction};
use nexus_core::{ExecutionContext, PrecisionMode, VcpResult};
use nexus_vcp::ObservationConfig;
use nexus_vcp::hyperbolic::{
    ChunkField, SearchConfig, SearchHit, WarpedConfig, WarpedIndex, residual_radius, tag_field,
};

const STAGE: &str = "recall";

pub struct RecallConfig {
    pub concepts: Option<WarpedConfig>,
    pub documents: Option<WarpedConfig>,
    /// Concept search settings used at each residual-pyramid layer.
    pub concept_search: SearchConfig,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            concepts: Some(WarpedConfig::default()),
            documents: Some(WarpedConfig::default()),
            concept_search: SearchConfig::default(),
        }
    }
}

pub struct RecallRequest<'a> {
    pub limit: usize,
    /// Restricts this search to the supplied candidate IDs.
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
    concept_search: SearchConfig,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocumentHit {
    pub id: u64,
    pub document_id: u64,
    /// Warped distance. Nearer vectors sort first.
    pub distance: f64,
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
                tag_field(
                    assets.tags,
                    assets.basis.map_or(&[], |basis| basis.mean),
                    assets.graph.inbound,
                    assets.tag_residuals.ok_or_else(|| {
                        invalid(STAGE, "concept recall requires prepared tag residuals")
                    })?,
                    config,
                    ctx,
                )
            })
            .transpose()?;
        Ok(RecallIndexes {
            nexus: self,
            concepts,
            documents,
            concept_search: config.concept_search,
        })
    }
}

impl<'data> RecallIndexes<'data> {
    /// Builds an observation with the prepared concept index or the query's host provider.
    pub fn observe(
        &self,
        query: &Query<'_>,
        config: &ObservationConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Observation<'data>> {
        if query.tag_recall.is_some() || self.concepts.is_none() {
            return self.nexus.observe(query, config, ctx);
        }
        let recall = |ctx: &ExecutionContext<'_>, req: TagRecallRequest<'_>, out: &mut Vec<u64>| {
            self.recall_tags(ctx, req, out)
        };
        self.nexus
            .observe(&query.with_tag_recall(&recall), config, ctx)
    }

    /// Searches the document index with a raw vector from any retrieval lane.
    /// Use the returned IDs to fuse or filter lanes before preparing a candidate set.
    pub fn search_documents(
        &self,
        vector: &[f32],
        request: &RecallRequest<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Vec<DocumentHit>> {
        ctx.check_cancelled(STAGE)?;
        let assets = self.nexus.prepared();
        let unit = normalize_direction(vector, assets.tags.dim, STAGE)?;
        let hits = self.document_hits(&unit, request, ctx)?;
        let catalog = assets.documents.expect("document index requires a catalog");
        Ok(hits
            .into_iter()
            .map(|hit| {
                let id = catalog.ids[hit.slot];
                DocumentHit {
                    id,
                    document_id: catalog.document_ids.map_or(id, |ids| ids[hit.slot]),
                    distance: hit.distance,
                }
            })
            .collect())
    }

    fn recall_tags(
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
        let hits = index.search(
            request.residual,
            radius,
            request.limit,
            Some(&scope),
            self.concept_search,
            ctx,
        )?;
        out.extend(hits.into_iter().map(|hit| assets.concept_ids[hit.slot]));
        Ok(())
    }

    fn document_hits(
        &self,
        unit: &[f32],
        request: &RecallRequest<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Vec<SearchHit>> {
        ctx.check_cancelled(STAGE)?;
        let documents = self
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
            return Ok(Vec::new());
        }
        documents.search(unit, request.limit, scope.as_deref(), request.search, ctx)
    }
}

impl<'data> Observation<'data> {
    /// Uses the original query direction to recall and prepare an ordered candidate set.
    pub fn recall(
        &self,
        indexes: &RecallIndexes<'data>,
        request: &RecallRequest<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<CandidateSet<'data>> {
        if !std::ptr::eq(self.nexus, indexes.nexus) {
            return Err(invalid(STAGE, "recall indexes belong to a different Nexus"));
        }
        let hits = indexes.document_hits(&self.values().original_query, request, ctx)?;
        CandidateSet::from_catalog_slots(self.nexus, hits.into_iter().map(|hit| hit.slot))
    }
}
