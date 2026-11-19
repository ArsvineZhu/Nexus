// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::invalid;
use crate::{Catalog, Nexus, PreparedDataset};
use nexus_core::kernel::{cosine_f32, norm_sq_f32};
use nexus_core::*;
use nexus_math::*;
use nexus_vcp::hyperbolic::{SearchConfig, residual_radius, tag_hits};
use nexus_vcp::*;
use std::collections::HashSet;
use std::sync::Arc;

const QUERY: &str = "query";

#[derive(Clone, Copy)]
pub struct Query<'a> {
    pub vector: &'a [f32],
    pub text: &'a str,
    pub core_concepts: &'a [u64],
    pub ghost_vectors: Option<VectorMatrixView<'a>>,
    pub ghost_is_core: &'a [bool],
    pub tag_recall: Option<&'a PyramidCandidates<'a>>,
}

impl<'a> Query<'a> {
    pub fn new(vector: &'a [f32], text: &'a str) -> Self {
        Self {
            vector,
            text,
            core_concepts: &[],
            ghost_vectors: None,
            ghost_is_core: &[],
            tag_recall: None,
        }
    }

    /// Use the host's Tag recall for each residual layer.
    pub fn with_tag_recall(mut self, recall: &'a PyramidCandidates<'a>) -> Self {
        self.tag_recall = Some(recall);
        self
    }
}

#[derive(Clone, Copy)]
pub enum Candidates<'a> {
    Recall {
        limit: usize,
        allowed: Option<&'a [u64]>,
    },
    Provided(&'a [u64]),
    Borrowed {
        ids: &'a [u64],
        pool: &'a CandidatePoolView<'a>,
    },
}

impl Candidates<'_> {
    pub fn recall(limit: usize) -> Self {
        Self::Recall {
            limit,
            allowed: None,
        }
    }
}

pub struct Stage<'a> {
    pub name: &'a str,
    pub limit: usize,
    pub documents: &'a [u64],
}

/// Candidate slots in the readout index this ID column.
pub struct Readout<T> {
    pub candidate_ids: Vec<u64>,
    pub scores: T,
}

pub type RiverOutput = Readout<RiverReadoutOutput>;
pub type DtscReadout = Readout<DtscOutput>;

#[derive(Clone, Debug)]
pub struct RiverConfig {
    pub scoring: RiverReadoutConfig,
    pub wormhole: CrossFileWormholeConfig,
    pub evidence: EvidencePolicy,
}

impl Default for RiverConfig {
    fn default() -> Self {
        Self {
            scoring: RiverReadoutConfig::default(),
            wormhole: CrossFileWormholeConfig::default(),
            evidence: EvidencePolicy::Vcp,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RiverPipelineConfig {
    pub observation: ObservationConfig,
    pub fields: FieldConfig,
    pub calibration: PoolCalibrateConfig,
    pub readout: RiverConfig,
    pub chunk_search: SearchConfig,
}

impl Default for RiverPipelineConfig {
    fn default() -> Self {
        let mut observation = ObservationConfig::default();
        observation.sense.max_transition_records = 0;
        Self {
            observation,
            fields: FieldConfig::default(),
            calibration: PoolCalibrateConfig::default(),
            readout: RiverConfig::default(),
            chunk_search: SearchConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DtscPipelineConfig {
    pub observation: ObservationConfig,
    pub readout: DtscConfig,
}

#[derive(Clone, Debug, Default)]
pub struct TagMemoConfig {
    pub observation: ObservationConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TagMemoItem {
    pub candidate: CandidateId,
    pub score: f64,
}

pub struct TagMemoOutput {
    pub observation: Arc<ObservationState>,
    pub ranking: Readout<Vec<TagMemoItem>>,
}

type Recipe<R, C> =
    fn(&Nexus<'_>, &Query<'_>, &Candidates<'_>, &C, &ExecutionContext<'_>) -> VcpResult<R>;

pub struct Pipeline<R, C> {
    run: Recipe<R, C>,
}

impl Pipeline<RiverOutput, RiverPipelineConfig> {
    pub fn river() -> Self {
        Self {
            run: |nexus, query, candidates, config, ctx| {
                let observation = nexus.observe_with(query, &config.observation, ctx)?;
                let fields = nexus.fields_with(&observation, &config.fields, ctx)?;
                let batch =
                    nexus.candidates_with(query.vector, candidates, config.chunk_search, ctx)?;
                nexus.river_with(&fields, &batch, &config.readout, &config.calibration, ctx)
            },
        }
    }
}

impl Pipeline<DtscReadout, DtscPipelineConfig> {
    pub fn dtsc() -> Self {
        Self {
            run: |nexus, query, candidates, config, ctx| {
                let observation = nexus.observe_with(query, &config.observation, ctx)?;
                let batch = nexus.candidates_with(
                    query.vector,
                    candidates,
                    SearchConfig::default(),
                    ctx,
                )?;
                nexus.dtsc_with(&observation, &batch, &config.readout, ctx)
            },
        }
    }
}

impl Pipeline<TagMemoOutput, TagMemoConfig> {
    pub fn tag_memo() -> Self {
        Self {
            run: |nexus, query, candidates, config, ctx| {
                let observation = nexus.observe_with(query, &config.observation, ctx)?;
                let pool = nexus.candidates_with(
                    query.vector,
                    candidates,
                    SearchConfig::default(),
                    ctx,
                )?;
                let scores = TagMemoReadout.execute(
                    ctx,
                    &(),
                    TagMemoInput {
                        enhanced: &observation.fused.enhanced_query,
                        candidates: pool.view(),
                    },
                )?;
                Ok(TagMemoOutput {
                    observation,
                    ranking: Readout {
                        candidate_ids: pool.ids().to_vec(),
                        scores,
                    },
                })
            },
        }
    }
}

impl Nexus<'_> {
    pub fn run<R, C>(
        &self,
        query: &Query<'_>,
        candidates: &Candidates<'_>,
        pipeline: Pipeline<R, C>,
        config: &C,
    ) -> VcpResult<R> {
        self.run_with(
            query,
            candidates,
            pipeline,
            config,
            &ExecutionContext::default(),
        )
    }

    pub fn run_with<R, C>(
        &self,
        query: &Query<'_>,
        candidates: &Candidates<'_>,
        pipeline: Pipeline<R, C>,
        config: &C,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<R> {
        (pipeline.run)(self, query, candidates, config, ctx)
    }

    pub fn observe(
        &self,
        query: &Query<'_>,
        config: &ObservationConfig,
    ) -> VcpResult<Arc<ObservationState>> {
        self.observe_with(query, config, &ExecutionContext::default())
    }

    pub fn observe_with(
        &self,
        query: &Query<'_>,
        config: &ObservationConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Arc<ObservationState>> {
        ObserveQuery.execute(
            ctx,
            config,
            QueryInput {
                nexus: self,
                query: *query,
            },
        )
    }

    pub fn fields(
        &self,
        observed: &Arc<ObservationState>,
        config: &FieldConfig,
    ) -> VcpResult<FieldState> {
        self.fields_with(observed, config, &ExecutionContext::default())
    }

    pub fn fields_with(
        &self,
        observed: &Arc<ObservationState>,
        config: &FieldConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<FieldState> {
        ComputeFields.execute(ctx, config, self.field_input(observed))
    }

    pub fn field_input<'a, 'q>(
        &'a self,
        observation: &'q Arc<ObservationState>,
    ) -> FieldInput<'a, 'q> {
        let assets = self.prepared();
        FieldInput {
            graph: assets.graph,
            tag_vectors: assets.tags,
            observation,
        }
    }

    pub fn river(
        &self,
        fields: &FieldState,
        batch: &CandidateBatch<'_>,
        config: &RiverConfig,
    ) -> VcpResult<RiverOutput> {
        self.river_with(
            fields,
            batch,
            config,
            &Default::default(),
            &Default::default(),
        )
    }

    pub fn river_with(
        &self,
        fields: &FieldState,
        batch: &CandidateBatch<'_>,
        config: &RiverConfig,
        calibration: &PoolCalibrateConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RiverOutput> {
        batch.check_query(&fields.observation().original_query)?;
        let candidates = batch.view();
        let pool = PoolCalibrateOperator.execute(
            ctx,
            calibration,
            PoolCalibrateInput { fields, candidates },
        )?;
        let scores = ReadRiver.execute(
            ctx,
            config,
            RiverInput {
                assets: self.prepared(),
                fields,
                candidates,
                pool: &pool,
            },
        )?;
        Ok(Readout {
            candidate_ids: batch.ids().to_vec(),
            scores,
        })
    }

    pub fn dtsc(
        &self,
        observed: &ObservationState,
        batch: &CandidateBatch<'_>,
        config: &DtscConfig,
    ) -> VcpResult<DtscReadout> {
        self.dtsc_with(observed, batch, config, &ExecutionContext::default())
    }

    pub fn dtsc_with(
        &self,
        observed: &ObservationState,
        batch: &CandidateBatch<'_>,
        config: &DtscConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<DtscReadout> {
        batch.check_query(&observed.original_query)?;
        let scores = ReadDtsc.execute(
            ctx,
            config,
            ObservedDtscInput {
                assets: self.prepared(),
                pool: None,
                observation: observed,
                candidates: batch.view(),
            },
        )?;
        Ok(Readout {
            candidate_ids: batch.ids().to_vec(),
            scores,
        })
    }

    pub fn thinking(
        &self,
        observation: &ObservationState,
        batch: &CandidateBatch<'_>,
        stages: &[Stage<'_>],
        config: &ThinkingConfig,
    ) -> VcpResult<Readout<ThinkingPlan>> {
        batch.check_query(&observation.original_query)?;
        let trace = observation.complete_trace()?;
        let ctx = ExecutionContext::default();
        let lookup: std::collections::HashMap<_, _> = batch
            .ids()
            .iter()
            .enumerate()
            .map(|(i, &id)| Ok((id, CandidateId::from_index(i, "thinking")?)))
            .collect::<VcpResult<_>>()?;
        let stages_candidates: Vec<Vec<_>> = stages
            .iter()
            .map(|stage| {
                stage
                    .documents
                    .iter()
                    .map(|id| {
                        lookup.get(id).copied().ok_or_else(|| {
                            invalid(
                                "thinking",
                                format!("document {id} is outside the candidate batch"),
                            )
                        })
                    })
                    .collect::<VcpResult<Vec<_>>>()
            })
            .collect::<VcpResult<_>>()?;
        let stages: Vec<_> = stages
            .iter()
            .zip(&stages_candidates)
            .map(|(stage, candidates)| ThinkingStageInputView {
                stage_label: stage.name,
                k: stage.limit,
                candidates,
            })
            .collect();
        let assets = self.prepared();
        let scores = ThinkingPlanOperator.execute(
            &ctx,
            config,
            ThinkingInput {
                trace,
                candidates: batch.view(),
                stages: &stages,
                graph: assets.graph,
                provenance: assets.provenance,
                support_groups: assets.groups,
                tag_vectors: assets.tags,
                query_cosines: None,
            },
        )?;
        Ok(Readout {
            candidate_ids: batch.ids().to_vec(),
            scores,
        })
    }

    pub fn candidates<'a>(
        &self,
        query: &'a [f32],
        candidates: &Candidates<'a>,
    ) -> VcpResult<CandidateBatch<'a>> {
        self.candidates_with(
            query,
            candidates,
            SearchConfig::default(),
            &Default::default(),
        )
    }

    pub fn candidates_with<'a>(
        &self,
        query: &'a [f32],
        candidates: &Candidates<'a>,
        config: SearchConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<CandidateBatch<'a>> {
        let assets = self.prepared();
        if query.len() != assets.tags.dim || !nexus_core::kernel::all_finite_f32(query) {
            return Err(invalid(
                QUERY,
                "query must be finite and match the dataset dimension",
            ));
        }
        if let Candidates::Borrowed { ids, pool } = *candidates {
            pool.validate(QUERY)?;
            if ids.len() != pool.candidate_count {
                return Err(invalid(QUERY, "provide one candidate ID per row"));
            }
            return Ok(CandidateBatch {
                query,
                batch: Batch::Borrowed { ids, pool },
            });
        }
        let catalog = assets
            .documents
            .ok_or_else(|| invalid(QUERY, "provide a document catalog or borrowed candidates"))?;
        let document_slot = |key: &u64| {
            self.document_slots
                .get(key)
                .copied()
                .ok_or_else(|| invalid(QUERY, format!("unknown document {key}")))
        };
        let selection = match *candidates {
            Candidates::Recall { limit, allowed } => {
                let scope = allowed
                    .map(|allowed| {
                        let mut mask = vec![false; catalog.ids.len()];
                        for key in allowed {
                            mask[document_slot(key)?] = true;
                        }
                        Ok(mask)
                    })
                    .transpose()?;
                catalog
                    .chunks
                    .search(query, limit, scope.as_deref(), config, ctx)?
                    .into_iter()
                    .map(|hit| (hit.slot, Some(-hit.distance)))
                    .collect()
            }
            Candidates::Provided(ids) => {
                let mut seen = HashSet::new();
                ids.iter()
                    .map(|id| {
                        let slot = document_slot(id)?;
                        if !seen.insert(slot) {
                            return Err(invalid(QUERY, format!("duplicate document {id}")));
                        }
                        Ok((slot, None))
                    })
                    .collect::<VcpResult<Vec<_>>>()?
            }
            Candidates::Borrowed { .. } => unreachable!("borrowed candidates handled above"),
        };
        Ok(CandidateBatch {
            query,
            batch: Batch::Selected(Box::new(Selected::build(catalog, &selection, query, ctx)?)),
        })
    }
}

pub struct QueryInput<'asset> {
    pub nexus: &'asset Nexus<'asset>,
    pub query: Query<'asset>,
}

pub struct ObserveQuery;

impl Operator for ObserveQuery {
    type Config = ObservationConfig;
    type Input<'asset, 'query> = QueryInput<'asset>;
    type Output = Arc<ObservationState>;

    fn name(&self) -> &'static str {
        "query.observe"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let assets = input.nexus.prepared();
        let query = input.query;
        let core_tags = query
            .core_concepts
            .iter()
            .map(|key| {
                let slot = input
                    .nexus
                    .concept_slots
                    .get(key)
                    .copied()
                    .ok_or_else(|| invalid(QUERY, format!("unknown concept {key}")))?;
                NodeId::from_index(slot, QUERY)
            })
            .collect::<VcpResult<Vec<_>>>()?;
        let mask = std::cell::RefCell::new(Vec::new());
        let query_norm = std::cell::OnceCell::new();
        let built_in = |level,
                        residual: &[f32],
                        limit: usize,
                        out: &mut Vec<_>,
                        epa: Option<&EpaAnalysis>,
                        excluded: &[NodeId]| {
            let index = assets.tag_search.ok_or_else(|| {
                invalid(QUERY, "prepare a Tag field or set Query::with_tag_recall")
            })?;
            let energy = norm_sq_f32(residual, ctx.precision);
            if energy <= 1e-12 {
                return Ok(());
            }
            let norm = *query_norm.get_or_init(|| norm_sq_f32(query.vector, ctx.precision));
            let radius = residual_radius(epa, level, energy / norm);
            let mut mask = mask.borrow_mut();
            mask.resize(assets.tags.rows, true);
            mask.fill(true);
            for &node in excluded {
                mask[node.index()] = false;
            }
            tag_hits(
                index.search(
                    residual,
                    radius,
                    limit.min(64),
                    Some(&mask),
                    SearchConfig {
                        rerank: Some(64),
                        ..SearchConfig::default()
                    },
                    ctx,
                )?,
                out,
            )
        };
        Observe.execute(
            ctx,
            config,
            ObservationInput {
                query: query.vector,
                query_text: query.text,
                core_tags: &core_tags,
                candidates: query.tag_recall.unwrap_or(&built_in),
                graph: assets.graph,
                tag_vectors: assets.tags,
                lexical_meta: assets.lexical,
                epa_basis: assets.basis,
                ghost_vectors: query.ghost_vectors,
                ghost_is_core: query.ghost_is_core,
            },
        )
    }
}

pub struct RiverInput<'asset, 'query> {
    pub assets: PreparedDataset<'asset>,
    pub fields: &'query FieldState,
    pub pool: &'query PreparedCandidatePool,
    pub candidates: CandidatePoolView<'asset>,
}

pub struct ReadRiver;

impl Operator for ReadRiver {
    type Config = RiverConfig;
    type Input<'asset, 'query> = RiverInput<'asset, 'query>;
    type Output = RiverReadoutOutput;

    fn name(&self) -> &'static str {
        "river"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let start = std::time::Instant::now();
        let assets = input.assets;
        let mut result = RiverReadoutOperator.execute(
            ctx,
            &config.scoring,
            RiverReadoutInput {
                fields: input.fields,
                evidence: config.evidence,
                pool: input.pool,
                candidates: input.candidates,
                graph: assets.graph,
                provenance: assets.provenance,
                support_groups: assets.groups,
                tag_vectors: assets.tags,
            },
        )?;
        if assets.sources.root_count > 0 {
            result = CrossFileWormholeOperator.execute(
                ctx,
                &config.wormhole,
                CrossFileWormholeInput {
                    observation: input.fields.observation(),
                    readout: &result,
                    candidates: input.candidates,
                    graph: assets.graph,
                    root_facts: assets.sources,
                    tag_vectors: assets.tags,
                },
            )?;
        }
        result.diagnostics.elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        Ok(result)
    }
}

pub struct ObservedDtscInput<'asset, 'query> {
    pub assets: PreparedDataset<'asset>,
    pub observation: &'query ObservationState,
    pub pool: Option<&'query PreparedCandidatePool>,
    pub candidates: CandidatePoolView<'asset>,
}

pub struct ReadDtsc;

impl Operator for ReadDtsc {
    type Config = DtscConfig;
    type Input<'asset, 'query> = ObservedDtscInput<'asset, 'query>;
    type Output = DtscOutput;

    fn name(&self) -> &'static str {
        "observed_dtsc"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let start = std::time::Instant::now();
        let state = input.observation;
        let nodes: Vec<_> = state
            .sense
            .nodes
            .iter()
            .map(|node| DtscNodeInput {
                node: node.node,
                energy: node.energy,
                normalized_energy: node.normalized_energy,
                source_type: if node.hop != 0 {
                    "emergent"
                } else if node.source_kind == SeedSourceKind::Core {
                    "core"
                } else {
                    "seed"
                }
                .into(),
            })
            .collect();
        let field = DtscFieldBuildOperator.execute(
            ctx,
            &config.field,
            DtscFieldBuildInput {
                nodes: &nodes,
                tag_vectors: input.assets.tags,
            },
        )?;
        let readout = DtscReadoutInput {
            field: &field,
            candidates: input.candidates,
            graph: input.assets.graph,
            tag_vectors: input.assets.tags,
            original_query: &state.original_query,
            enhanced_query: &state.fused.enhanced_query,
            geometry: DtscGeometryState {
                epa_logic_depth: state.epa.logic_depth,
                epa_entropy: state.epa.entropy,
                epa_resonance: state.epa.resonance,
                pyramid_coverage: state.pyramid.coverage,
                pyramid_novelty: state.pyramid.novelty,
                pyramid_depth: state.pyramid.levels.len() as f64,
            },
        };
        let mut result = if let Some(pool) = input
            .pool
            .filter(|pool| pool.precision == PrecisionMode::StrictLeftFold)
        {
            DtscReadoutOperator.execute_with_query_cosines(
                ctx,
                config,
                readout,
                [&pool.query_cosines, &pool.denoised_cosines],
            )?
        } else {
            DtscReadoutOperator.execute(ctx, config, readout)?
        };
        result.diagnostics.elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        Ok(result)
    }
}

pub struct TagMemoInput<'query, 'asset> {
    pub enhanced: &'query [f32],
    pub candidates: CandidatePoolView<'asset>,
}

/// Ranks candidate vectors by their cosine against the TagMemo-enhanced query.
pub struct TagMemoReadout;

impl Operator for TagMemoReadout {
    type Config = ();
    type Input<'asset, 'query> = TagMemoInput<'query, 'asset>;
    type Output = Vec<TagMemoItem>;

    fn name(&self) -> &'static str {
        "tag_memo"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        _: &(),
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        input.candidates.validate(QUERY)?;
        let vectors = input.candidates.candidate_vectors;
        if input.enhanced.len() != vectors.dim
            || !nexus_core::kernel::all_finite_f32(input.enhanced)
        {
            return Err(invalid(
                QUERY,
                "enhanced query must match the candidate matrix",
            ));
        }
        let mut results = Vec::with_capacity(vectors.rows);
        for (slot, row) in vectors.data.chunks_exact(vectors.dim).enumerate() {
            ctx.check_cancelled(QUERY)?;
            results.push(TagMemoItem {
                candidate: CandidateId::from_index(slot, QUERY)?,
                score: cosine_f32(input.enhanced, row, ctx.precision),
            });
        }
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(results)
    }
}

/// An ordered candidate selection for one query, shared across readouts.
pub struct CandidateBatch<'a> {
    query: &'a [f32],
    batch: Batch<'a>,
}

enum Batch<'a> {
    Borrowed {
        ids: &'a [u64],
        pool: &'a CandidatePoolView<'a>,
    },
    Selected(Box<Selected>),
}

impl CandidateBatch<'_> {
    pub fn ids(&self) -> &[u64] {
        match &self.batch {
            Batch::Borrowed { ids, .. } => ids,
            Batch::Selected(rows) => &rows.ids,
        }
    }

    pub fn view(&self) -> CandidatePoolView<'_> {
        match &self.batch {
            Batch::Borrowed { pool, .. } => **pool,
            Batch::Selected(rows) => rows.view(),
        }
    }

    fn check_query(&self, query: &[f32]) -> VcpResult<()> {
        if self.query == query {
            Ok(())
        } else {
            Err(invalid(
                QUERY,
                "candidate batch belongs to a different query vector",
            ))
        }
    }
}

struct Selected {
    dim: usize,
    ids: Vec<u64>,
    docs: Vec<DocumentId>,
    vectors: Vec<f32>,
    baseline: Vec<f64>,
    union: Vec<f64>,
    ranks: Vec<u32>,
    zeros: Vec<f64>,
    root_offsets: Vec<u32>,
    roots: Vec<RootId>,
    path_offsets: Vec<u32>,
    tag_offsets: Vec<u32>,
    tags: Vec<NodeId>,
    positions: Vec<u32>,
}

fn offsets(slice: &[u32], slot: usize) -> std::ops::Range<usize> {
    slice[slot] as usize..slice[slot + 1] as usize
}

impl Selected {
    fn build(
        catalog: Catalog<'_>,
        rows: &[(usize, Option<f64>)],
        query: &[f32],
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Self> {
        let count = rows.len();
        let mut selected = Self {
            dim: catalog.vectors.dim,
            ids: Vec::new(),
            docs: Vec::new(),
            vectors: Vec::with_capacity(count * catalog.vectors.dim),
            baseline: Vec::with_capacity(count),
            union: Vec::with_capacity(count),
            ranks: Vec::with_capacity(count),
            zeros: vec![0.0; rows.len()],
            root_offsets: vec![0],
            roots: Vec::new(),
            path_offsets: vec![0],
            tag_offsets: vec![0],
            tags: Vec::new(),
            positions: Vec::new(),
        };
        let offset = |count| {
            u32::try_from(count).map_err(|_| invalid(QUERY, "candidate offsets exceed u32"))
        };
        for (i, &(slot, union)) in rows.iter().enumerate() {
            ctx.check_cancelled(QUERY)?;
            let vector = catalog
                .vectors
                .row(slot)
                .ok_or_else(|| invalid(QUERY, "document slot outside catalog"))?;
            selected.ids.push(catalog.ids[slot]);
            selected.docs.push(DocumentId::from_index(slot, QUERY)?);
            selected.vectors.extend_from_slice(vector);
            let baseline = cosine_f32(query, vector, ctx.precision);
            selected.baseline.push(baseline);
            selected.union.push(union.unwrap_or(baseline));
            selected.ranks.push(offset(i + 1)?);
            selected
                .roots
                .extend_from_slice(&catalog.roots[offsets(catalog.root_offsets, slot)]);
            selected.root_offsets.push(offset(selected.roots.len())?);
            for path in offsets(catalog.path_offsets, slot) {
                let steps = offsets(catalog.tag_offsets, path);
                selected
                    .tags
                    .extend_from_slice(&catalog.tags[steps.clone()]);
                selected
                    .positions
                    .extend_from_slice(&catalog.positions[steps]);
                selected.tag_offsets.push(offset(selected.tags.len())?);
            }
            selected
                .path_offsets
                .push(offset(selected.tag_offsets.len() - 1)?);
        }
        Ok(selected)
    }

    fn view(&self) -> CandidatePoolView<'_> {
        CandidatePoolView {
            candidate_count: self.ids.len(),
            baseline_scores: &self.baseline,
            union_scores: &self.union,
            union_ranks: &self.ranks,
            bm25_scores: &self.zeros,
            bm25_reference: 0.0,
            anchor_scores: &self.zeros,
            curve_document_ids: &self.docs,
            candidate_vectors: VectorMatrixView::new(&self.vectors, self.ids.len(), self.dim)
                .expect("selected candidate vectors have a fixed dimension"),
            candidate_root_offsets: &self.root_offsets,
            candidate_roots: &self.roots,
            candidate_path_offsets: &self.path_offsets,
            path_tag_offsets: &self.tag_offsets,
            path_tags: &self.tags,
            path_positions: &self.positions,
        }
    }
}
