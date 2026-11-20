// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::CandidateSet;
use nexus_core::{CandidateId, PathId};
use nexus_math::{DtscDecision, DtscFieldStats, DtscScores};
use nexus_vcp::{
    OmegaOutput, ProjectionLimits, PropagationCompletion, QueryMorphology, RiverScores,
    ThinkingRelation,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreItem {
    pub id: u64,
    pub document_id: u64,
    pub score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathRef {
    pub candidate_id: u64,
    /// Zero-based path index within the candidate, in host input order.
    pub path_index: usize,
    /// Business ID of the source when path attribution was supplied.
    pub source_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConceptTransition {
    pub source: u64,
    pub target: u64,
    pub previous: Option<u64>,
    pub hop: usize,
    pub flow: f64,
    pub wormhole: bool,
    pub immediate_return: bool,
}

pub struct Stage<'a> {
    pub name: &'a str,
    pub limit: usize,
    pub candidates: &'a [u64],
}

#[derive(Clone, Debug)]
pub struct TagMemoReadout {
    pub items: Vec<ScoreItem>,
    pub propagation: PropagationCompletion,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct RiverReadout {
    pub items: Vec<RiverItem>,
    pub omega: OmegaOutput,
    pub morphology: QueryMorphology,
    pub propagation: PropagationCompletion,
    pub projection: ProjectionLimits,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiverItem {
    pub id: u64,
    pub document_id: u64,
    pub chosen_path: Option<PathRef>,
    pub scores: RiverScores,
}

#[derive(Clone, Debug)]
pub struct DtscReadout {
    pub items: Vec<DtscItem>,
    pub field: DtscFieldStats,
    pub decision: DtscDecision,
    pub available: usize,
    pub propagation: PropagationCompletion,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct DtscItem {
    pub id: u64,
    pub document_id: u64,
    pub chosen_path: Option<PathRef>,
    pub scores: DtscScores,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingPlan {
    pub stages: Vec<ThinkingStage>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingStage {
    pub stage_label: String,
    pub k: usize,
    pub results: Vec<ThinkingItem>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingItem {
    pub id: u64,
    pub document_id: u64,
    pub chosen_path: Option<PathRef>,
    pub score: f64,
    pub parents: Vec<u64>,
    pub relation: ThinkingRelation,
    pub route: Option<ConceptTransition>,
}

#[derive(Clone, Copy)]
pub(crate) struct CandidateIdentity {
    pub(crate) id: u64,
    document_id: u64,
    path_start: usize,
}

impl CandidateIdentity {
    pub(crate) fn path_ref(
        self,
        path: Option<PathId>,
        path_sources: &[Option<u64>],
    ) -> Option<PathRef> {
        path.map(|path| PathRef {
            candidate_id: self.id,
            path_index: path.index() - self.path_start,
            source_id: path_sources[path.index()],
        })
    }
}

pub(crate) fn river_readout(
    result: nexus_vcp::RiverReadoutOutput,
    ids: &[CandidateIdentity],
    path_sources: &[Option<u64>],
) -> RiverReadout {
    let items = result
        .items
        .into_iter()
        .map(|item| {
            let id = ids[item.candidate.index()];
            RiverItem {
                id: id.id,
                document_id: id.document_id,
                chosen_path: id.path_ref(item.chosen_path, path_sources),
                scores: item.scores,
            }
        })
        .collect();
    RiverReadout {
        items,
        omega: result.omega,
        morphology: result.morphology,
        propagation: result.propagation,
        projection: result.projection,
        elapsed_ms: result.elapsed_ms,
    }
}

pub(crate) fn dtsc_readout(
    result: nexus_math::DtscOutput,
    ids: &[CandidateIdentity],
    path_sources: &[Option<u64>],
    propagation: PropagationCompletion,
) -> DtscReadout {
    let items = result
        .results
        .into_iter()
        .map(|item| {
            let id = ids[item.candidate.index()];
            DtscItem {
                id: id.id,
                document_id: id.document_id,
                chosen_path: id.path_ref(item.chosen_path, path_sources),
                scores: item.scores,
            }
        })
        .collect();
    DtscReadout {
        items,
        field: result.field,
        decision: result.decision,
        available: result.available,
        propagation,
        elapsed_ms: result.elapsed_ms,
    }
}

impl CandidateSet<'_> {
    fn identity(&self, candidate: CandidateId) -> CandidateIdentity {
        CandidateIdentity {
            id: self.ids()[candidate.index()],
            document_id: self.document_ids()[candidate.index()],
            path_start: self.rows().candidate_paths(candidate).start,
        }
    }

    pub(crate) fn identity_snapshot(&self) -> Vec<CandidateIdentity> {
        (0..self.ids().len())
            .map(|index| {
                self.identity(
                    CandidateId::from_index(index, "readout").expect("validated candidate slot"),
                )
            })
            .collect()
    }

    pub(crate) fn thinking_plan(&self, result: nexus_vcp::ThinkingPlan) -> ThinkingPlan {
        let concepts = self.nexus.prepared().concept_ids;
        let stages = result
            .stages
            .into_iter()
            .map(|stage| ThinkingStage {
                stage_label: stage.stage_label,
                k: stage.k,
                results: stage
                    .results
                    .into_iter()
                    .map(|item| ThinkingItem {
                        id: self.ids()[item.candidate.index()],
                        document_id: self.document_ids()[item.candidate.index()],
                        chosen_path: self
                            .identity(item.candidate)
                            .path_ref(item.chosen_path, self.path_sources()),
                        score: item.score,
                        parents: item
                            .parents
                            .into_iter()
                            .map(|id| self.ids()[id.index()])
                            .collect(),
                        relation: item.relation,
                        route: item.route.map(|route| ConceptTransition {
                            source: concepts[route.source.index()],
                            target: concepts[route.target.index()],
                            previous: route.previous.map(|id| concepts[id.index()]),
                            hop: route.hop,
                            flow: route.flow,
                            wormhole: route.wormhole,
                            immediate_return: route.immediate_return,
                        }),
                    })
                    .collect(),
            })
            .collect();
        ThinkingPlan {
            stages,
            elapsed_ms: result.elapsed_ms,
        }
    }
}
