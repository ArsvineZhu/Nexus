// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::sync::Arc;

use crate::{CandidateId, ConceptId, DocumentId, RecallKind, SourceId};
use nexus_math::dtsc::CurveMetrics;
use nexus_math::rank_order;

pub use nexus_math::dtsc::{DtscDecision, DtscFallbackReason};

#[derive(Clone, Debug, PartialEq)]
pub enum ReadoutSummary {
    Cosine { direction: RecallKind },
    External,
    River,
    Dtsc { decision: DtscDecision },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedPath {
    pub source: SourceId,
    pub path: usize,
    pub positions: Vec<usize>,
    pub concepts: Vec<ConceptId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BridgeWitness {
    pub from: ConceptId,
    pub to: ConceptId,
    pub source_a: SourceId,
    pub source_b: SourceId,
    pub third: ConceptId,
    pub other_foci: Vec<ConceptId>,
    pub support: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiverCandidateEvidence {
    pub selected_path: Option<SelectedPath>,
    pub baseline_value: f64,
    pub path_bonus: f64,
    pub graph_bonus: f64,
    pub omega: f64,
    pub genus_gate: f64,
    pub field_support: f64,
    pub node_alignment: f64,
    pub edge_coverage: f64,
    pub ordinary_transport_factor: f64,
    pub cross_source_transport_factor: f64,
    pub witnesses: Vec<BridgeWitness>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DtscCandidateEvidence {
    pub selected_path: Option<SelectedPath>,
    pub baseline_value: f64,
    pub correction_gain: f64,
    pub correction: f64,
    pub geometry: CurveMetrics,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScoreEvidence {
    Cosine { direction: RecallKind },
    External,
    River(RiverCandidateEvidence),
    Dtsc(DtscCandidateEvidence),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoreRow {
    pub id: CandidateId,
    pub value: f64,
    pub document_id: Option<DocumentId>,
    pub evidence: ScoreEvidence,
}

#[derive(Clone, Debug)]
pub struct Scores {
    pub(crate) evaluation: Arc<()>,
    pub(crate) rows: Vec<ScoreRow>,
    pub(crate) priorities: Vec<usize>,
    pub(crate) summary: ReadoutSummary,
}

impl Scores {
    pub fn values(&self) -> impl ExactSizeIterator<Item = (CandidateId, f64)> + '_ {
        self.iter().map(|row| (row.id, row.value))
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ScoreRow> + '_ {
        self.rows.iter()
    }
    pub fn summary(&self) -> &ReadoutSummary {
        &self.summary
    }

    pub fn top(&self, limit: Option<usize>) -> Ranking {
        let order = rank_order(self.rows.len(), limit, |a, b| {
            self.rows[b]
                .value
                .partial_cmp(&self.rows[a].value)
                .expect("finite scores")
                .then_with(|| self.priorities[a].cmp(&self.priorities[b]))
        });
        let rows: Vec<_> = order
            .into_iter()
            .map(|index| self.rows[index].clone())
            .collect();
        let ids = rows.iter().map(|row| row.id).collect();
        Ranking {
            rows,
            ids,
            summary: self.summary.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ranking {
    rows: Vec<ScoreRow>,
    ids: Vec<CandidateId>,
    summary: ReadoutSummary,
}

impl Ranking {
    pub fn rows(&self) -> &[ScoreRow] {
        &self.rows
    }
    pub fn ids(&self) -> &[CandidateId] {
        &self.ids
    }
    pub fn into_rows(self) -> Vec<ScoreRow> {
        self.rows
    }
    pub fn summary(&self) -> &ReadoutSummary {
        &self.summary
    }
}
