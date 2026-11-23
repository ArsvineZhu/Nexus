// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use nexus_math::dtsc::{
    self, CurveMetrics, DtscConfig, DtscEvidenceConfig, DtscFieldConfig, DtscReadoutConfig,
    FieldStats,
};
use nexus_math::kernel::{PrecisionMode, cosine_f32, cosine_f32_with_norms, norm_sq_f32};
use nexus_math::{NodeId, VectorRows, cosine_columns};
use rayon::prelude::*;

use crate::candidates::CandidateInfo;
use crate::{
    CandidateId, Candidates, DtscCandidateEvidence, DtscDecision, Error, Observation,
    ReadoutSummary, RecallKind, Result, RunOptions, ScoreEvidence, ScoreRow, Scores, SelectedPath,
    input,
};

#[derive(Clone, Default)]
pub struct EvaluateOptions {
    pub run: RunOptions,
}

#[derive(Clone, Default)]
pub struct DtscOptions {
    pub config: DtscConfig,
    pub run: RunOptions,
}

#[derive(Clone, Default)]
pub struct DtscEvidenceOptions {
    pub field: DtscFieldConfig,
    pub evidence: DtscEvidenceConfig,
    pub run: RunOptions,
}

#[derive(Clone, Default)]
pub struct DtscReadoutOptions {
    pub config: DtscReadoutConfig,
    pub run: RunOptions,
}

const LANES: [RecallKind; 4] = [
    RecallKind::Original,
    RecallKind::Enhanced,
    RecallKind::LocalField,
    RecallKind::TransferField,
];
const FOLD: PrecisionMode = PrecisionMode::StrictLeftFold;

struct Similarities {
    closures: HashMap<NodeId, f64>,
}

pub struct Evaluation<'a> {
    pub(crate) observation: &'a Observation,
    pub(crate) candidates: &'a Candidates<'a>,
    vectors: VectorRows<'a>,
    identity: Arc<()>,
    slots: HashMap<CandidateId, usize>,
    similarities: Vec<Similarities>,
    lanes: [Vec<f64>; 5],
}

impl Observation {
    pub fn evaluate<'a>(
        &'a self,
        candidates: &'a Candidates<'_>,
        options: &EvaluateOptions,
    ) -> Result<Evaluation<'a>> {
        let run = &options.run;
        run.check_cancelled()?;
        if !Arc::ptr_eq(self.model.identity(), candidates.identity()) {
            return Err(Error::Association {
                message: "candidates and observation belong to different models".into(),
            });
        }
        let rows = candidates.vectors();
        let concepts = self.model.concepts();
        let seeds = &self.analysis().seeds;
        let [original, enhanced, local, transfer] = LANES.map(|kind| self.vector(kind));
        let directions = [
            original,
            enhanced,
            local,
            transfer,
            seeds.differentia.as_ref().map(|v| v.as_slice()),
        ];
        let columns = cosine_columns(directions, rows, run)?;
        let lanes = std::array::from_fn(|i| columns.iter().map(|row| row[i]).collect());
        let parent_ids: Vec<_> = seeds
            .pruned_hypernyms
            .iter()
            .map(|id| concepts.node(*id))
            .collect::<Result<_>>()?;
        let calculate = |index| -> Result<Similarities> {
            run.check_cancelled()?;
            let row = rows.row(index).expect("validated candidate index");
            let norm = norm_sq_f32(row, FOLD);
            let nodes: BTreeSet<_> = candidates
                .info(index)
                .paths
                .iter()
                .flat_map(|path| path.nodes.iter().copied())
                .chain(parent_ids.iter().copied())
                .collect();
            let closures = nodes
                .into_iter()
                .map(|node| {
                    run.check_cancelled()?;
                    let tag = concepts.row(node);
                    let cosine =
                        cosine_f32_with_norms(row, tag, norm, norm_sq_f32(tag, FOLD), FOLD, 0.0)
                            .clamp(-1.0, 1.0);
                    Ok((node, cosine))
                })
                .collect::<Result<_>>()?;
            Ok(Similarities { closures })
        };
        let similarities: Vec<_> = if run.parallelism.is_parallel() {
            (0..rows.len())
                .into_par_iter()
                .map(calculate)
                .collect::<Result<_>>()?
        } else {
            (0..rows.len()).map(calculate).collect::<Result<_>>()?
        };
        let slots = rows
            .ids()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        run.check_cancelled()?;
        Ok(Evaluation {
            observation: self,
            candidates,
            vectors: rows,
            identity: Arc::new(()),
            slots,
            similarities,
            lanes,
        })
    }
}

impl Evaluation<'_> {
    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }
    pub fn cosine(&self) -> Scores {
        self.cosine_lane(0)
    }
    pub fn tag_memo(&self) -> Scores {
        self.cosine_lane(1)
    }

    fn cosine_lane(&self, lane: usize) -> Scores {
        let direction = LANES[lane];
        let rows = self.lanes[lane]
            .iter()
            .enumerate()
            .map(|(i, &value)| self.row(i, value, ScoreEvidence::Cosine { direction }))
            .collect();
        self.scores(rows, ReadoutSummary::Cosine { direction })
    }

    pub fn baseline(&self, values: impl IntoIterator<Item = (CandidateId, f64)>) -> Result<Scores> {
        let count = self.candidates.len();
        let mut rows = vec![None; count];
        let mut priorities = vec![0; count];
        for (priority, (id, value)) in values.into_iter().enumerate() {
            let slot = self.slot(id, "baseline")?;
            if !value.is_finite() || rows[slot].is_some() {
                return Err(input(
                    "baseline",
                    Some(id),
                    "expected a unique finite value",
                ));
            }
            rows[slot] = Some(self.row(slot, value, ScoreEvidence::External));
            priorities[slot] = priority;
        }
        let rows = rows
            .into_iter()
            .enumerate()
            .map(|(slot, row)| {
                row.ok_or_else(|| {
                    input(
                        "baseline",
                        Some(self.candidates.ids()[slot]),
                        "missing candidate value",
                    )
                })
            })
            .collect::<Result<_>>()?;
        Ok(Scores {
            rows,
            priorities,
            ..self.scores(Vec::new(), ReadoutSummary::External)
        })
    }

    pub fn dtsc(&self, baseline: &Scores, options: &DtscOptions) -> Result<Scores> {
        if !Arc::ptr_eq(&self.identity, &baseline.evaluation) {
            return Err(Error::Association {
                message: "baseline belongs to a different evaluation".into(),
            });
        }
        self.build_dtsc(
            &options.config.field,
            &options.config.evidence,
            &options.run,
        )?
        .apply_with(baseline, &options.config.readout, &options.run)
    }

    pub fn dtsc_evidence(&self, options: &DtscEvidenceOptions) -> Result<DtscEvidence> {
        self.build_dtsc(&options.field, &options.evidence, &options.run)
    }

    fn build_dtsc(
        &self,
        field_config: &DtscFieldConfig,
        evidence_config: &DtscEvidenceConfig,
        run: &RunOptions,
    ) -> Result<DtscEvidence> {
        let observation = self.observation;
        let data = observation.model.concepts().vectors().as_rows();
        let analysis = observation.analysis();
        let geometry = if evidence_config
            .auxiliary
            .as_ref()
            .is_some_and(|aux| aux.geometry.is_some())
        {
            let epa = analysis.epa.as_ref().ok_or_else(|| {
                input(
                    "dtsc.auxiliary.geometry",
                    None,
                    "geometry auxiliary requires an EPA analysis",
                )
            })?;
            let pyramid = analysis.pyramid.as_ref().ok_or_else(|| {
                input(
                    "dtsc.auxiliary.geometry",
                    None,
                    "geometry auxiliary requires a Pyramid analysis",
                )
            })?;
            Some(dtsc::GeometryState {
                epa_concentration: epa.concentration,
                epa_entropy: epa.entropy,
                epa_resonance: epa.resonance,
                pyramid_coverage: pyramid.coverage,
                pyramid_novelty: pyramid.novelty,
                pyramid_depth: pyramid.levels.len() as f64,
            })
        } else {
            None
        };
        let field = dtsc::build_field(
            data,
            observation.active_concepts(),
            &dtsc::FieldOptions {
                config: field_config.clone(),
                run: run.clone(),
            },
        )?;
        let mut sequences = Vec::new();
        for (i, &id) in self.candidates.ids().iter().enumerate() {
            run.check_cancelled()?;
            for (ordinal, path) in self.info(i).paths.iter().enumerate() {
                let concepts = path
                    .nodes
                    .iter()
                    .map(|node| data.ids()[node.index()])
                    .collect::<Vec<_>>();
                if !concepts.is_empty() {
                    sequences.push((id, ordinal, concepts));
                }
            }
        }
        let paths = sequences
            .iter()
            .map(|(candidate, ordinal, concepts)| dtsc::Path {
                candidate: *candidate,
                ordinal: *ordinal,
                concepts,
            })
            .collect::<Vec<_>>();
        let tag_body = |candidate, concept| {
            let slot = *self.slots.get(&candidate)?;
            let node = observation.model.concepts().node(concept).ok()?;
            self.similarities[slot].closures.get(&node).copied()
        };
        let evidence = dtsc::evidence_with_shared(
            dtsc::EvidenceInput {
                field: &field,
                concepts: data,
                graph: Some(observation.model.transport()),
                query: observation
                    .vector(RecallKind::Original)
                    .expect("original direction"),
                enhanced_query: observation.vector(RecallKind::Enhanced),
                candidates: self.vectors,
                paths: &paths,
                geometry,
            },
            &dtsc::EvidenceOptions {
                config: evidence_config.clone(),
                run: run.clone(),
            },
            dtsc::SharedEvidence {
                effective: observation.effective.as_ref(),
                cosines: Some([&self.lanes[0], &self.lanes[1]]),
                tag_body: Some(&tag_body),
            },
        )?;
        let path_by_id: HashMap<_, _> = evidence
            .items()
            .iter()
            .filter_map(|item| {
                let index = *self.slots.get(&item.candidate)?;
                let path = self.info(index).paths.get(item.path?)?;
                Some((item.candidate, path.selected(&observation.model)))
            })
            .collect();
        let metrics_slots = evidence
            .items()
            .iter()
            .enumerate()
            .map(|(i, item)| (item.candidate, i))
            .collect();
        Ok(DtscEvidence {
            evaluation: self.identity.clone(),
            ids: self.candidates.ids().to_vec(),
            evidence,
            metrics_slots,
            path_by_id,
        })
    }

    pub(crate) fn slot(&self, id: CandidateId, field: &'static str) -> Result<usize> {
        self.slots
            .get(&id)
            .copied()
            .ok_or_else(|| input(field, Some(id), "candidate is outside this evaluation"))
    }

    pub(crate) fn info(&self, slot: usize) -> &CandidateInfo {
        self.candidates.info(slot)
    }
    pub(crate) fn lane(&self, slot: usize, lane: usize) -> f64 {
        self.lanes[lane][slot]
    }
    pub(crate) fn differentia(&self, slot: usize) -> f64 {
        self.lanes[4][slot]
    }

    pub(crate) fn closure(&self, slot: usize, node: NodeId) -> f64 {
        self.similarities[slot]
            .closures
            .get(&node)
            .copied()
            .unwrap_or_else(|| {
                cosine_f32(
                    self.vectors.row(slot).expect("validated candidate"),
                    self.observation.model.concepts().row(node),
                    FOLD,
                )
                .clamp(-1.0, 1.0)
            })
    }

    pub(crate) fn row(&self, slot: usize, value: f64, evidence: ScoreEvidence) -> ScoreRow {
        ScoreRow {
            id: self.candidates.ids()[slot],
            value,
            document_id: self.info(slot).document_id,
            evidence,
        }
    }

    pub(crate) fn scores(&self, rows: Vec<ScoreRow>, summary: ReadoutSummary) -> Scores {
        Scores {
            evaluation: self.identity.clone(),
            priorities: (0..rows.len()).collect(),
            rows,
            summary,
        }
    }
}

pub struct DtscEvidence {
    evaluation: Arc<()>,
    ids: Vec<CandidateId>,
    evidence: dtsc::Evidence,
    metrics_slots: HashMap<CandidateId, usize>,
    path_by_id: HashMap<CandidateId, SelectedPath>,
}

impl DtscEvidence {
    pub fn candidate_ids(&self) -> &[CandidateId] {
        &self.ids
    }
    pub fn metrics(&self, id: CandidateId) -> Option<&CurveMetrics> {
        Some(
            &self
                .evidence
                .items()
                .get(*self.metrics_slots.get(&id)?)?
                .metrics,
        )
    }
    pub fn selected_path(&self, id: CandidateId) -> Option<&SelectedPath> {
        self.path_by_id.get(&id)
    }
    pub fn field_stats(&self) -> &FieldStats {
        self.evidence.field_stats()
    }

    pub fn apply(&self, baseline: &Scores, options: &DtscReadoutOptions) -> Result<Scores> {
        self.apply_with(baseline, &options.config, &options.run)
    }

    fn apply_with(
        &self,
        baseline: &Scores,
        config: &DtscReadoutConfig,
        run: &RunOptions,
    ) -> Result<Scores> {
        run.check_cancelled()?;
        if !Arc::ptr_eq(&self.evaluation, &baseline.evaluation) {
            return Err(Error::Association {
                message: "DTSC evidence and baseline belong to different evaluations".into(),
            });
        }
        let base = nexus_math::Scores::new(baseline.values())?;
        let result = self.evidence.apply(
            &base,
            &dtsc::ReadoutOptions {
                config: config.clone(),
                run: run.clone(),
            },
        )?;
        let rows = result
            .scores
            .values()
            .zip(baseline.iter())
            .map(|((id, value), base)| {
                debug_assert_eq!(id, base.id);
                let geometry = *self.metrics(id).expect("evidence covers candidates");
                let correction = if result.decision == DtscDecision::Applied {
                    config.correction_gain * geometry.geo_bonus
                } else {
                    0.0
                };
                ScoreRow {
                    id,
                    value,
                    document_id: base.document_id,
                    evidence: ScoreEvidence::Dtsc(DtscCandidateEvidence {
                        selected_path: self.selected_path(id).cloned(),
                        baseline_value: base.value,
                        correction_gain: config.correction_gain,
                        correction,
                        geometry,
                    }),
                }
            })
            .collect();
        Ok(Scores {
            evaluation: self.evaluation.clone(),
            rows,
            priorities: baseline.priorities.clone(),
            summary: ReadoutSummary::Dtsc {
                decision: result.decision,
            },
        })
    }
}
