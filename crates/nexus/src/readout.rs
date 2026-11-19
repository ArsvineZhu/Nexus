// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::CandidateSet;
use nexus_core::{CandidateId, Diagnostics, PathId};
use nexus_math::{AuxiliaryEvidence, DtscOrder};
use nexus_vcp::{OmegaOutput, QueryMorphology, ThinkingRelation};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreItem {
    pub id: u64,
    pub document_id: u64,
    pub score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathRef {
    pub candidate_id: u64,
    pub path_index: usize,
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
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug)]
pub struct RiverReadout {
    pub items: Vec<RiverItem>,
    pub omega: OmegaOutput,
    pub morphology: QueryMorphology,
    pub path_evidence: Vec<CandidatePathEvidence>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiverItem {
    pub id: u64,
    pub document_id: u64,
    pub chosen_path: Option<PathRef>,
    pub score: f64,
    pub base_score: f64,
    pub path_bonus: f64,
    pub graph_bonus: f64,
    pub topology_bonus: f64,
    pub genus_gate: f64,
    pub direct_anchor_contacts: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidatePathEvidence {
    pub id: u64,
    pub paths: Vec<RiverPathEvidence>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverPathEvidence {
    pub path: Option<PathRef>,
    pub base: f64,
    pub path_bonus: f64,
    pub graph_bonus: f64,
    pub path_quality: f64,
    pub field_support: f64,
    pub node_alignment: f64,
    pub edge_coverage: f64,
    pub ordinary_transport: f64,
    pub genus_gate: f64,
    pub anchor_contacts: u8,
}

#[derive(Clone, Debug)]
pub struct DtscReadout {
    pub items: Vec<DtscItem>,
    pub field_trusted: bool,
    pub field_entropy: f64,
    pub order: DtscOrder,
    pub available: usize,
    pub fallback_used: bool,
    pub fallback_reason: Option<&'static str>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug)]
pub struct DtscItem {
    pub id: u64,
    pub document_id: u64,
    pub chosen_path: Option<PathRef>,
    pub score: f64,
    pub original_cosine: f64,
    pub geo_score: f64,
    pub normalized_geo: f64,
    pub geo_bonus: f64,
    pub geo_base_bonus: f64,
    pub geo_aux_bonus: f64,
    pub auxiliary_evidence: Option<AuxiliaryEvidence>,
    pub geo_effect: &'static str,
    pub geo_evidence_class: &'static str,
    pub geo_reward_eligible: bool,
    pub geo_confidence: f64,
    pub geo_exact_hits: usize,
    pub geo_strong_hits: usize,
    pub geo_hit_count: usize,
    pub geo_continuity: f64,
    pub geo_isolated_ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingPlan {
    pub stages: Vec<ThinkingStage>,
    pub diagnostics: Diagnostics,
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

impl CandidateSet<'_> {
    fn path_ref(&self, candidate: CandidateId, path: Option<PathId>) -> Option<PathRef> {
        path.map(|path| PathRef {
            candidate_id: self.ids()[candidate.index()],
            path_index: path.index() - self.rows().candidate_paths(candidate).start,
        })
    }

    pub(crate) fn river_readout(&self, result: nexus_vcp::RiverReadoutOutput) -> RiverReadout {
        let items = result
            .items
            .into_iter()
            .map(|item| RiverItem {
                id: self.ids()[item.candidate.index()],
                document_id: self.document_ids()[item.candidate.index()],
                chosen_path: self.path_ref(item.candidate, item.chosen_path),
                score: item.score,
                base_score: item.base_score,
                path_bonus: item.path_bonus,
                graph_bonus: item.graph_bonus,
                topology_bonus: item.topology_bonus,
                genus_gate: item.genus_gate,
                direct_anchor_contacts: item.direct_anchor_contacts,
            })
            .collect();
        let path_evidence = result
            .path_evidence
            .into_iter()
            .enumerate()
            .map(|(index, paths)| {
                let candidate = CandidateId::from_index(index, "river.readout")
                    .expect("validated candidate slot");
                CandidatePathEvidence {
                    id: self.ids()[index],
                    paths: paths
                        .into_iter()
                        .map(|path| RiverPathEvidence {
                            path: self.path_ref(candidate, path.path),
                            base: path.base,
                            path_bonus: path.path_bonus,
                            graph_bonus: path.graph_bonus,
                            path_quality: path.path_quality,
                            field_support: path.field_support,
                            node_alignment: path.node_alignment,
                            edge_coverage: path.edge_coverage,
                            ordinary_transport: path.ordinary_transport,
                            genus_gate: path.genus_gate,
                            anchor_contacts: path.anchor_contacts,
                        })
                        .collect(),
                }
            })
            .collect();
        RiverReadout {
            items,
            path_evidence,
            omega: result.omega,
            morphology: result.morphology,
            diagnostics: result.diagnostics,
        }
    }

    pub(crate) fn dtsc_readout(&self, result: nexus_math::DtscOutput) -> DtscReadout {
        let items = result
            .results
            .into_iter()
            .map(|item| DtscItem {
                id: self.ids()[item.candidate.index()],
                document_id: self.document_ids()[item.candidate.index()],
                chosen_path: self.path_ref(item.candidate, item.chosen_path),
                score: item.score,
                original_cosine: item.original_cosine,
                geo_score: item.geo_score,
                normalized_geo: item.normalized_geo,
                geo_bonus: item.geo_bonus,
                geo_base_bonus: item.geo_base_bonus,
                geo_aux_bonus: item.geo_aux_bonus,
                auxiliary_evidence: item.auxiliary_evidence,
                geo_effect: item.geo_effect,
                geo_evidence_class: item.geo_evidence_class,
                geo_reward_eligible: item.geo_reward_eligible,
                geo_confidence: item.geo_confidence,
                geo_exact_hits: item.geo_exact_hits,
                geo_strong_hits: item.geo_strong_hits,
                geo_hit_count: item.geo_hit_count,
                geo_continuity: item.geo_continuity,
                geo_isolated_ratio: item.geo_isolated_ratio,
            })
            .collect();
        DtscReadout {
            items,
            field_trusted: result.field_trusted,
            field_entropy: result.field_entropy,
            order: result.order,
            available: result.available,
            fallback_used: result.fallback_used,
            fallback_reason: result.fallback_reason,
            diagnostics: result.diagnostics,
        }
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
                        chosen_path: self.path_ref(item.candidate, item.chosen_path),
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
            diagnostics: result.diagnostics,
        }
    }
}
