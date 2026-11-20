// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::invalid;
use crate::readout::{CandidateIdentity, dtsc_readout, river_readout};
use crate::{
    CandidateInput, CandidateSet, DtscField, DtscReadout, FieldState, Nexus, Observation, PathRef,
    Query, RiverReadout, ScoreItem, Stage, TagMemoReadout, ThinkingPlan,
};
use nexus_core::{CandidateId, ExecutionContext, QueryCosines, ThinkingStageInputView, VcpResult};
use nexus_math::{
    DtscConfig, DtscEvidenceConfig, DtscEvidenceInput, DtscEvidenceValues, DtscGeometryState,
    DtscReadoutConfig, prepare_dtsc_evidence, readout_dtsc,
};
use nexus_vcp::{
    FieldConfig, ObservationConfig, PathEvidence, PropagationCompletion, RiverAnalysisConfig,
    RiverAnalysisInput, RiverScoringConfig, ThinkingConfig, ThinkingInput, plan_thinking,
    prepare_river_evidence,
};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct RiverQueryConfig {
    pub observation: ObservationConfig,
    pub river: RiverConfig,
}

#[derive(Clone, Debug, Default)]
pub struct RiverConfig {
    pub fields: FieldConfig,
    pub analysis: RiverAnalysisConfig,
    pub scoring: RiverScoringConfig,
}

#[derive(Clone, Debug, Default)]
pub struct DtscQueryConfig {
    pub observation: ObservationConfig,
    pub dtsc: DtscConfig,
}

pub struct Evaluation<'obs, 'set, 'data> {
    observation: &'obs Observation<'data>,
    candidates: &'set CandidateSet<'data>,
    cosines: QueryCosines<'set, 'obs>,
}

pub struct RiverEvidence {
    ids: Vec<CandidateIdentity>,
    path_sources: Vec<Option<u64>>,
    values: nexus_vcp::RiverEvidence,
}

pub struct DtscEvidence {
    ids: Vec<CandidateIdentity>,
    path_sources: Vec<Option<u64>>,
    values: DtscEvidenceValues,
    propagation: PropagationCompletion,
}

impl<'data> Observation<'data> {
    pub fn evaluate<'obs, 'set>(
        &'obs self,
        candidates: &'set CandidateSet<'data>,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Evaluation<'obs, 'set, 'data>> {
        self.check_candidates(candidates)?;
        let values = self.values();
        let cosines = QueryCosines::prepare(
            ctx,
            candidates.rows(),
            &values.original_query,
            &values.fused.enhanced_query,
        )?;
        Ok(Evaluation {
            observation: self,
            candidates,
            cosines,
        })
    }

    fn check_candidates(&self, set: &CandidateSet<'data>) -> VcpResult<()> {
        if std::ptr::eq(self.nexus, set.nexus) {
            Ok(())
        } else {
            Err(invalid(
                "evaluation",
                "candidate set belongs to a different Nexus",
            ))
        }
    }
}

impl<'obs, 'set, 'data> Evaluation<'obs, 'set, 'data> {
    /// Runs River on the same observation and candidates used by this evaluation.
    pub fn river(
        &self,
        config: &RiverConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RiverReadout> {
        let fields = self.observation.fields(&config.fields, ctx)?;
        self.river_evidence(&fields, &config.analysis, ctx)?
            .readout(&config.scoring, ctx)
    }

    /// Runs DTSC on the same observation and candidates used by this evaluation.
    pub fn dtsc(&self, config: &DtscConfig, ctx: &ExecutionContext<'_>) -> VcpResult<DtscReadout> {
        let field = self.observation.dtsc_field(&config.field, ctx)?;
        self.dtsc_evidence(&field, &config.scoring.evidence, ctx)?
            .readout(&config.scoring.readout, ctx)
    }

    pub fn tag_memo(&self, ctx: &ExecutionContext<'_>) -> VcpResult<TagMemoReadout> {
        let start = Instant::now();
        ctx.check_cancelled("tag_memo")?;
        let mut items = self
            .candidates
            .ids()
            .iter()
            .zip(self.candidates.document_ids())
            .zip(self.cosines.enhanced_cosines())
            .map(|((&id, &document_id), &score)| ScoreItem {
                id,
                document_id,
                score,
            })
            .collect::<Vec<_>>();
        items.sort_by(|a, b| b.score.total_cmp(&a.score));
        ctx.check_cancelled("tag_memo")?;
        Ok(TagMemoReadout {
            items,
            propagation: self.observation.values().sense.propagation,
            elapsed_ms: elapsed_ms(start),
        })
    }

    pub fn river_evidence(
        &self,
        fields: &FieldState<'obs, 'data>,
        config: &RiverAnalysisConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RiverEvidence> {
        let observation = self.observation;
        if !std::ptr::eq(observation, fields.observation) {
            return Err(invalid(
                "river.prepare",
                "field state belongs to a different observation",
            ));
        }
        let assets = observation.nexus.prepared();
        let values = prepare_river_evidence(
            RiverAnalysisInput {
                observation: observation.values(),
                fields: fields.values(),
                query: &self.cosines,
                graph: observation.graph(),
                provenance: assets.provenance,
                support_groups: assets.groups,
                root_facts: (assets.sources.root_count > 0).then_some(assets.sources),
                tag_vectors: assets.tags,
            },
            config,
            ctx,
        )?;
        Ok(RiverEvidence {
            ids: self.candidates.identity_snapshot(),
            path_sources: self.candidates.path_sources().to_vec(),
            values,
        })
    }

    pub fn dtsc_evidence(
        &self,
        field: &DtscField<'obs, 'data>,
        config: &DtscEvidenceConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<DtscEvidence> {
        let observation = self.observation;
        if !std::ptr::eq(observation, field.observation) {
            return Err(invalid(
                "dtsc.evidence",
                "DTSC field belongs to a different observation",
            ));
        }
        let state = observation.values();
        let assets = observation.nexus.prepared();
        let values = prepare_dtsc_evidence(
            DtscEvidenceInput {
                field: field.values(),
                query: &self.cosines,
                graph: observation.graph(),
                tag_vectors: assets.tags,
                geometry: DtscGeometryState {
                    epa_spectral_concentration: state.epa.spectral_concentration,
                    epa_entropy: state.epa.entropy,
                    epa_resonance: state.epa.resonance,
                    pyramid_coverage: state.pyramid.coverage,
                    pyramid_novelty: state.pyramid.novelty,
                    pyramid_depth: state.pyramid.levels.len() as f64,
                },
            },
            config,
            ctx,
        )?;
        Ok(DtscEvidence {
            ids: self.candidates.identity_snapshot(),
            path_sources: self.candidates.path_sources().to_vec(),
            values,
            propagation: state.sense.propagation,
        })
    }

    pub fn thinking(
        &self,
        stages: &[Stage<'_>],
        config: &ThinkingConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<ThinkingPlan> {
        const STAGE: &str = "thinking.plan";
        let start = Instant::now();
        let observation = self.observation;
        ctx.check_cancelled(STAGE)?;
        let trace = observation.values().sense.complete_trace()?;
        let mut seen = HashSet::new();
        let mut slots_for_stages = Vec::with_capacity(stages.len());
        for stage in stages {
            seen.clear();
            slots_for_stages.push(
                stage
                    .candidates
                    .iter()
                    .map(|id| {
                        if !seen.insert(id) {
                            return Err(invalid(
                                STAGE,
                                format!("candidate {id} is repeated in stage {}", stage.name),
                            ));
                        }
                        let slot = self
                            .candidates
                            .slots
                            .get(id)
                            .ok_or_else(|| invalid(STAGE, format!("unknown candidate {id}")))?;
                        CandidateId::from_index(*slot, STAGE)
                    })
                    .collect::<VcpResult<Vec<_>>>()?,
            );
        }
        let stages = stages
            .iter()
            .zip(&slots_for_stages)
            .map(|(stage, slots)| ThinkingStageInputView {
                stage_label: stage.name,
                k: stage.limit,
                candidates: slots,
            })
            .collect::<Vec<_>>();
        let assets = observation.nexus.prepared();
        let result = plan_thinking(
            ThinkingInput {
                trace,
                query: &self.cosines,
                stages: &stages,
                graph: observation.graph(),
                provenance: assets.provenance,
                support_groups: assets.groups,
                tag_vectors: assets.tags,
            },
            config,
            ctx,
        )?;
        let mut plan = self.candidates.thinking_plan(result);
        plan.elapsed_ms = elapsed_ms(start);
        Ok(plan)
    }
}

impl RiverEvidence {
    pub fn readout(
        &self,
        config: &RiverScoringConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RiverReadout> {
        Ok(river_readout(
            self.values.readout(config, ctx)?,
            &self.ids,
            &self.path_sources,
        ))
    }

    pub fn candidate_paths(
        &self,
        id: u64,
    ) -> Option<impl Iterator<Item = (Option<PathRef>, &PathEvidence)>> {
        let index = self.ids.iter().position(|identity| identity.id == id)?;
        let identity = self.ids[index];
        Some(
            self.values.paths()[index]
                .iter()
                .map(move |path| (identity.path_ref(path.path, &self.path_sources), path)),
        )
    }
}

impl DtscEvidence {
    pub fn readout(
        &self,
        config: &DtscReadoutConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<DtscReadout> {
        Ok(dtsc_readout(
            readout_dtsc(&self.values, config, ctx)?,
            &self.ids,
            &self.path_sources,
            self.propagation,
        ))
    }
}

impl Nexus<'_> {
    pub fn rank_river<'data>(
        &'data self,
        query: &Query<'_>,
        candidates: CandidateInput<'data>,
        config: &RiverQueryConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RiverReadout> {
        let set = self.candidates(candidates)?;
        let observation = self.observe(query, &config.observation, ctx)?;
        observation.evaluate(&set, ctx)?.river(&config.river, ctx)
    }

    pub fn rank_dtsc<'data>(
        &'data self,
        query: &Query<'_>,
        candidates: CandidateInput<'data>,
        config: &DtscQueryConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<DtscReadout> {
        let set = self.candidates(candidates)?;
        let observation = self.observe(query, &config.observation, ctx)?;
        observation.evaluate(&set, ctx)?.dtsc(&config.dtsc, ctx)
    }

    pub fn rank_tag_memo<'data>(
        &'data self,
        query: &Query<'_>,
        candidates: CandidateInput<'data>,
        config: &ObservationConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<TagMemoReadout> {
        let set = self.candidates(candidates)?;
        let observation = self.observe(query, config, ctx)?;
        observation.evaluate(&set, ctx)?.tag_memo(ctx)
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}
