// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::invalid;
use crate::{
    CandidateInput, CandidateSet, DtscField, DtscReadout, FieldState, Nexus, Observation, Query,
    RiverReadout, ScoreItem, Stage, TagMemoReadout, ThinkingPlan,
};
use nexus_core::{
    CandidateId, ExecutionContext, Operator, QueryCosines, ThinkingStageInputView, VcpResult,
};
use nexus_math::{
    DtscConfig, DtscEvidenceConfig, DtscEvidenceInput, DtscEvidenceOperator, DtscEvidenceValues,
    DtscGeometryState, DtscReadoutConfig, DtscReadoutInput, DtscReadoutOperator, DtscScoringConfig,
};
use nexus_vcp::{
    CrossFileWormholeConfig, CrossFileWormholeInput, CrossFileWormholeOperator, EvidencePolicy,
    FieldConfig, ObservationConfig, RiverFeatures, RiverReadoutConfig, RiverReadoutInput,
    RiverReadoutOperator, ThinkingConfig, ThinkingInput, ThinkingPlanOperator,
};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct RiverConfig {
    pub scoring: RiverReadoutConfig,
    pub wormhole: CrossFileWormholeConfig,
    pub evidence: EvidencePolicy,
}

#[derive(Clone, Debug, Default)]
pub struct RiverQueryConfig {
    pub observation: ObservationConfig,
    pub fields: FieldConfig,
    pub readout: RiverConfig,
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

pub struct PreparedRiver<'prepared, 'obs, 'set, 'data> {
    evaluation: &'prepared Evaluation<'obs, 'set, 'data>,
    features: RiverFeatures<'prepared>,
}

pub struct DtscEvidence<'prepared, 'obs, 'set, 'data> {
    evaluation: &'prepared Evaluation<'obs, 'set, 'data>,
    _field: &'prepared DtscField<'obs, 'data>,
    values: DtscEvidenceValues,
}

impl<'data> Observation<'data> {
    pub fn evaluate<'obs, 'set>(
        &'obs self,
        candidates: &'set CandidateSet<'data>,
    ) -> VcpResult<Evaluation<'obs, 'set, 'data>> {
        self.check_candidates(candidates)?;
        let values = self.values();
        let cosines = QueryCosines::prepare(
            self.context(),
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

    pub fn evaluate_reusing<'obs, 'set>(
        &'obs self,
        previous: &Evaluation<'_, 'set, 'data>,
    ) -> VcpResult<Evaluation<'obs, 'set, 'data>> {
        let candidates = previous.candidates;
        self.check_candidates(candidates)?;
        let values = self.values();
        let cosines = QueryCosines::reuse_original(
            self.context(),
            &previous.cosines,
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
    pub fn tag_memo(&self) -> VcpResult<TagMemoReadout> {
        let start = Instant::now();
        let ctx = self.observation.context();
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
            diagnostics: nexus_core::Diagnostics {
                elapsed_ms: elapsed_ms(start),
                ..Default::default()
            },
        })
    }

    pub fn prepare_river<'prepared>(
        &'prepared self,
        fields: &'prepared FieldState<'obs, 'data>,
    ) -> VcpResult<PreparedRiver<'prepared, 'obs, 'set, 'data>> {
        if !std::ptr::eq(self.observation, fields.observation) {
            return Err(invalid(
                "river.prepare",
                "field state belongs to a different observation",
            ));
        }
        let features =
            RiverFeatures::prepare(self.observation.context(), &self.cosines, fields.values())?;
        Ok(PreparedRiver {
            evaluation: self,
            features,
        })
    }

    pub fn river(
        &self,
        fields: &FieldState<'obs, 'data>,
        config: &RiverConfig,
    ) -> VcpResult<RiverReadout> {
        let start = Instant::now();
        let mut result = self.prepare_river(fields)?.readout(config)?;
        result.diagnostics.elapsed_ms = elapsed_ms(start);
        Ok(result)
    }

    pub fn dtsc_evidence<'prepared>(
        &'prepared self,
        field: &'prepared DtscField<'obs, 'data>,
        config: &DtscEvidenceConfig,
    ) -> VcpResult<DtscEvidence<'prepared, 'obs, 'set, 'data>> {
        let observation = self.observation;
        if !std::ptr::eq(observation, field.observation) {
            return Err(invalid(
                "dtsc.evidence",
                "DTSC field belongs to a different observation",
            ));
        }
        let state = observation.values();
        let assets = observation.nexus.prepared();
        let values = DtscEvidenceOperator.execute(
            observation.context(),
            config,
            DtscEvidenceInput {
                field: field.values(),
                query: &self.cosines,
                graph: assets.graph,
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
        )?;
        Ok(DtscEvidence {
            evaluation: self,
            _field: field,
            values,
        })
    }

    pub fn dtsc(
        &self,
        field: &DtscField<'obs, 'data>,
        config: &DtscScoringConfig,
    ) -> VcpResult<DtscReadout> {
        let start = Instant::now();
        let mut result = self
            .dtsc_evidence(field, &config.evidence)?
            .readout(&config.readout)?;
        result.diagnostics.elapsed_ms = elapsed_ms(start);
        Ok(result)
    }

    pub fn thinking(
        &self,
        stages: &[Stage<'_>],
        config: &ThinkingConfig,
    ) -> VcpResult<ThinkingPlan> {
        const STAGE: &str = "thinking.plan";
        let start = Instant::now();
        let observation = self.observation;
        let ctx = observation.context();
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
        let result = ThinkingPlanOperator.execute(
            ctx,
            config,
            ThinkingInput {
                trace,
                query: &self.cosines,
                stages: &stages,
                graph: assets.graph,
                provenance: assets.provenance,
                support_groups: assets.groups,
                tag_vectors: assets.tags,
            },
        )?;
        let mut plan = self.candidates.thinking_plan(result);
        plan.diagnostics.elapsed_ms = elapsed_ms(start);
        Ok(plan)
    }
}

impl PreparedRiver<'_, '_, '_, '_> {
    pub fn readout(&self, config: &RiverConfig) -> VcpResult<RiverReadout> {
        let start = Instant::now();
        let observation = self.evaluation.observation;
        let assets = observation.nexus.prepared();
        let ctx = observation.context();
        let mut result = RiverReadoutOperator.execute(
            ctx,
            &config.scoring,
            RiverReadoutInput {
                observation: observation.values(),
                features: &self.features,
                evidence: config.evidence,
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
                    observation: observation.values(),
                    readout: &result,
                    features: &self.features,
                    graph: assets.graph,
                    root_facts: assets.sources,
                    tag_vectors: assets.tags,
                },
            )?;
        }
        let mut result = self.evaluation.candidates.river_readout(result);
        result.diagnostics.elapsed_ms = elapsed_ms(start);
        Ok(result)
    }
}

impl DtscEvidence<'_, '_, '_, '_> {
    pub fn readout(&self, config: &DtscReadoutConfig) -> VcpResult<DtscReadout> {
        let start = Instant::now();
        let result = DtscReadoutOperator.execute(
            self.evaluation.observation.context(),
            config,
            DtscReadoutInput {
                evidence: &self.values,
            },
        )?;
        let mut result = self.evaluation.candidates.dtsc_readout(result);
        result.diagnostics.elapsed_ms = elapsed_ms(start);
        Ok(result)
    }
}

impl Nexus<'_> {
    pub fn rank_river<'data>(
        &'data self,
        query: &Query<'_>,
        candidates: CandidateInput<'data>,
        config: &RiverQueryConfig,
        ctx: ExecutionContext<'data>,
    ) -> VcpResult<RiverReadout> {
        let set = self.candidates(candidates)?;
        let observation = self.observe(query, &config.observation, ctx)?;
        let evaluation = observation.evaluate(&set)?;
        let fields = observation.fields(&config.fields)?;
        evaluation.river(&fields, &config.readout)
    }

    pub fn rank_dtsc<'data>(
        &'data self,
        query: &Query<'_>,
        candidates: CandidateInput<'data>,
        config: &DtscQueryConfig,
        ctx: ExecutionContext<'data>,
    ) -> VcpResult<DtscReadout> {
        let set = self.candidates(candidates)?;
        let observation = self.observe(query, &config.observation, ctx)?;
        let evaluation = observation.evaluate(&set)?;
        let field = observation.dtsc_field(&config.dtsc.field)?;
        evaluation.dtsc(&field, &config.dtsc.scoring)
    }

    pub fn rank_tag_memo<'data>(
        &'data self,
        query: &Query<'_>,
        candidates: CandidateInput<'data>,
        config: &ObservationConfig,
        ctx: ExecutionContext<'data>,
    ) -> VcpResult<TagMemoReadout> {
        let set = self.candidates(candidates)?;
        let observation = self.observe(query, config, ctx)?;
        observation.evaluate(&set)?.tag_memo()
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}
