// SPDX-License-Identifier: CC-BY-NC-SA-4.0
mod network;
mod path;
mod wormhole;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nexus_math::NodeId;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    BridgeWitness, CandidateId, DocumentId, Evaluation, EvidenceSource, Observation,
    ReadoutSummary, RecallKind, Result, RiverCandidateEvidence, RunOptions, ScoreEvidence,
    ScoreRow, Scores, SelectedPath, config, parameter,
};
use wormhole::{CrossSourceRoute, collect_routes};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverConfig {
    pub analysis: RiverAnalysisConfig,
    pub scoring: RiverScoringConfig,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverAnalysisConfig {
    pub semantic_field_inherit_cos: f64,
    pub tag_align_min_cos: f64,
    pub weak_match_pivot: f64,
    pub genus_gate_floor: f64,
    pub genus_gate_span: f64,
    pub cross_source: Option<CrossSourceConfig>,
}

impl Default for RiverAnalysisConfig {
    fn default() -> Self {
        Self {
            semantic_field_inherit_cos: 0.82,
            tag_align_min_cos: 0.48,
            weak_match_pivot: 0.65,
            genus_gate_floor: 0.50,
            genus_gate_span: 0.22,
            cross_source: Some(Default::default()),
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverScoringConfig {
    pub topology_budget: f64,
    pub omega_gamma: f64,
}

impl Default for RiverScoringConfig {
    fn default() -> Self {
        Self {
            topology_budget: 0.08,
            omega_gamma: 1.0,
        }
    }
}

#[derive(Clone, Default)]
pub struct RiverOptions {
    pub config: RiverConfig,
    pub run: RunOptions,
}
#[derive(Clone, Default)]
pub struct RiverAnalysisOptions {
    pub config: RiverAnalysisConfig,
    pub run: RunOptions,
}
#[derive(Clone, Default)]
pub struct RiverScoringOptions {
    pub config: RiverScoringConfig,
    pub run: RunOptions,
}
#[derive(Clone, Default)]
pub struct CrossSourceOptions {
    pub config: CrossSourceConfig,
    pub run: RunOptions,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CrossSourceConfig {
    pub min_relative_flow: f64,
    pub semantic_node_threshold: f64,
    pub max_river_edges: usize,
}

impl Default for CrossSourceConfig {
    fn default() -> Self {
        Self {
            min_relative_flow: 0.5,
            semantic_node_threshold: 0.48,
            max_river_edges: 96,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryMode {
    Atomic,
    Propositional,
    Narrative,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryMorphology {
    pub atomic_weight: f64,
    pub propositional_weight: f64,
    pub narrative_weight: f64,
    pub confidence: f64,
    pub dominant_mode: QueryMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OmegaOutput {
    pub omega: f64,
    pub omega_edge: f64,
    pub omega_emerge: f64,
    pub omega_flow: f64,
    pub regime: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathEvidence {
    pub selected_path: Option<SelectedPath>,
    pub base_score: f64,
    pub path_topology_factor: f64,
    pub graph_request_factor: f64,
    pub ordinary_transport_factor: f64,
    pub cross_source_transport_factor: f64,
    pub field_support: f64,
    pub node_alignment: f64,
    pub edge_coverage: f64,
    pub genus_gate: f64,
    pub witnesses: Vec<BridgeWitness>,
}

impl PathEvidence {
    fn bonuses(&self, cap: f64, gate: f64) -> (f64, f64, f64) {
        let path = cap * self.path_topology_factor * gate;
        let remaining = cap - path;
        let graph = (cap * self.graph_request_factor * gate).min(remaining);
        let transport = cap
            * gate
            * self
                .ordinary_transport_factor
                .max(self.cross_source_transport_factor);
        let merged = if cap > 0.0 {
            cap * (1.0 - (1.0 - graph / cap) * (1.0 - transport / cap))
        } else {
            0.0
        };
        let mut bonus = merged.min(remaining);
        if path + bonus > cap {
            bonus = bonus.next_down();
        }
        (
            (self.base_score + path + bonus).clamp(0.0, 1.0),
            path,
            bonus,
        )
    }
}

pub struct RiverEvidence {
    evaluation: Arc<()>,
    paths: Vec<(CandidateId, Option<DocumentId>, Vec<PathEvidence>)>,
    omega: OmegaOutput,
    morphology: QueryMorphology,
}

impl RiverEvidence {
    pub fn paths(&self, id: CandidateId) -> Option<&[PathEvidence]> {
        self.paths
            .iter()
            .find(|(candidate, _, _)| *candidate == id)
            .map(|(_, _, path)| path.as_slice())
    }

    pub fn omega(&self) -> &OmegaOutput {
        &self.omega
    }
    pub fn morphology(&self) -> &QueryMorphology {
        &self.morphology
    }

    pub fn readout(&self, options: &RiverScoringOptions) -> Result<Scores> {
        self.read(&options.config, &options.run)
    }

    fn read(&self, cfg: &RiverScoringConfig, run: &RunOptions) -> Result<Scores> {
        parameter("river", "topology_budget", cfg.topology_budget, 0.0, 1.0)?;
        parameter("river", "omega_gamma", cfg.omega_gamma, 0.0, f64::INFINITY)?;
        run.check_cancelled()?;
        let omega = self.omega.omega;
        let gate = if omega > 0.0 {
            omega.powf(cfg.omega_gamma)
        } else {
            0.0
        };
        let rows = self
            .paths
            .iter()
            .map(|(id, document_id, paths)| {
                run.check_cancelled()?;
                let mut best = None::<(&PathEvidence, f64, f64, f64)>;
                for path in paths {
                    run.check_cancelled()?;
                    let (score, path_bonus, graph_bonus) = path.bonuses(cfg.topology_budget, gate);
                    if best.is_none_or(|(_, other, p, g)| {
                        score > other || (score == other && path_bonus + graph_bonus > p + g)
                    }) {
                        best = Some((path, score, path_bonus, graph_bonus));
                    }
                }
                let (path, value, path_bonus, graph_bonus) =
                    best.expect("candidate has path or body-only evidence");
                Ok(ScoreRow {
                    id: *id,
                    value,
                    document_id: *document_id,
                    evidence: ScoreEvidence::River(RiverCandidateEvidence {
                        selected_path: path.selected_path.clone(),
                        baseline_value: path.base_score,
                        path_bonus,
                        graph_bonus,
                        omega,
                        genus_gate: path.genus_gate,
                        field_support: path.field_support,
                        node_alignment: path.node_alignment,
                        edge_coverage: path.edge_coverage,
                        ordinary_transport_factor: path.ordinary_transport_factor,
                        cross_source_transport_factor: path.cross_source_transport_factor,
                        witnesses: path.witnesses.clone(),
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Scores {
            evaluation: self.evaluation.clone(),
            priorities: (0..rows.len()).collect(),
            rows,
            summary: ReadoutSummary::River,
        })
    }
}

type TopologyEdges = HashMap<NodeId, Vec<(usize, NodeId, NodeId, f64)>>;

struct Network {
    nodes: Vec<(NodeId, f64)>,
    peak_energy: f64,
    hops: HashMap<NodeId, usize>,
    max_hop: f64,
    seeds: Vec<(NodeId, f64)>,
    focus_targets: HashSet<NodeId>,
    topology: TopologyEdges,
    edge_mass: f64,
    active: HashMap<(NodeId, NodeId), f64>,
    ordinary: HashMap<NodeId, f64>,
    fields: [Vec<f64>; 2],
    domains: [HashSet<NodeId>; 2],
    field_mix: [bool; 2],
    morphology: QueryMorphology,
    omega: OmegaOutput,
}

impl Network {
    fn new(obs: &Observation, run: &RunOptions) -> Result<Self> {
        let concepts = obs.model.concepts();
        let sense = &obs.analysis().sense;
        let seeds = &obs.analysis().seeds;
        let graph = obs.graph();
        let mut nodes = Vec::with_capacity(sense.nodes.len());
        let mut hops = HashMap::new();
        let mut explicit = HashSet::new();
        let mut seed_energy = Vec::new();
        let mut peak_energy = 0.0_f64;
        let mut max_hop = 1;
        for node in &sense.nodes {
            run.check_cancelled()?;
            let slot = concepts.node(node.concept)?;
            nodes.push((slot, node.energy));
            hops.insert(slot, node.hop);
            max_hop = max_hop.max(node.hop);
            peak_energy = peak_energy.max(node.energy);
            if node.hop == 0 && node.energy > 0.0 {
                seed_energy.push((slot, node.energy));
                if matches!(node.kind, EvidenceSource::Core | EvidenceSource::Lexical) {
                    explicit.insert(slot);
                }
            }
        }
        let focus = seeds
            .exact_focus
            .filter(|_| seeds.pruned_hypernyms.len() == 1 && seeds.differentia_confidence > 0.0);
        let mut focus_targets = HashSet::new();
        let mut topology = TopologyEdges::new();
        let mut edge_mass = 0.0;
        let mut active = HashMap::new();
        let mut ordinary = HashMap::<NodeId, f64>::new();
        let mut undirected_mass = HashMap::<(NodeId, NodeId), f64>::new();
        for (i, edge) in sense.observed_edges.iter().enumerate() {
            run.check_cancelled()?;
            let source = concepts.node(edge.from)?;
            let target = concepts.node(edge.to)?;
            let conductance = graph.edge_weight(source, target);
            if conductance <= 0.0 {
                continue;
            }
            let flow = if edge.normalized_flow > 0.0 {
                edge.normalized_flow
            } else {
                edge.flow
            }
            .clamp(0.0, 1.0);
            if flow >= 0.015 {
                topology
                    .entry(source)
                    .or_default()
                    .push((i, source, target, flow));
                active.insert((source, target), flow);
                let key = (source.min(target), source.max(target));
                let prior = undirected_mass.entry(key).or_default();
                edge_mass += (flow - *prior).max(0.0);
                *prior = prior.max(flow);
            }
        }
        // Keep the observed first-hop kind even if the same edge later
        // carries a wormhole or return event.
        for &(from, to) in &obs.ordinary_first_hops {
            run.check_cancelled()?;
            let source = concepts.node(from)?;
            let target = concepts.node(to)?;
            if !explicit.contains(&source) || hops.get(&target) != Some(&1) {
                continue;
            }
            let conductance = graph.edge_weight(source, target);
            if conductance == 0.0 {
                continue;
            }
            if Some(from) == focus && active.contains_key(&(source, target)) {
                focus_targets.insert(target);
            }
            let (sum, highest) = graph.row_statistics(source);
            if sum > 0.0 && conductance >= 0.5 * highest {
                let value = ordinary.entry(target).or_default();
                *value = value.max(conductance / sum);
            }
        }
        let solved = &obs.fields().values;
        let fields = [&solved.local, &solved.transfer].map(|solution| {
            let peak = solution
                .values
                .iter()
                .map(|&(_, value)| value)
                .fold(0.0, f64::max);
            solution
                .values
                .iter()
                .map(|&(_, value)| if peak > 0.0 { value / peak } else { 0.0 })
                .collect()
        });
        let domains = [&solved.local, &solved.transfer].map(|solution| {
            solution
                .domain
                .iter()
                .map(|&id| concepts.node(id))
                .collect::<Result<HashSet<_>>>()
        });
        let [local, transfer] = domains;
        let coverage = obs
            .analysis()
            .pyramid
            .as_ref()
            .map_or(1.0, |pyramid| pyramid.coverage);
        let omega = network::omega(sense, coverage);
        let morphology = network::morphology(sense);
        Ok(Self {
            nodes,
            peak_energy,
            hops,
            max_hop: max_hop as f64,
            seeds: seed_energy,
            focus_targets,
            topology,
            edge_mass,
            active,
            ordinary,
            fields,
            domains: [local?, transfer?],
            field_mix: [RecallKind::LocalField, RecallKind::TransferField]
                .map(|kind| obs.vector(kind).is_some()),
            morphology,
            omega,
        })
    }
}

impl Evaluation<'_> {
    pub fn river(&self, options: &RiverOptions) -> Result<Scores> {
        self.analyze_river(&options.config.analysis, &options.run)?
            .read(&options.config.scoring, &options.run)
    }

    pub fn river_evidence(&self, options: &RiverAnalysisOptions) -> Result<RiverEvidence> {
        self.analyze_river(&options.config, &options.run)
    }

    fn analyze_river(&self, cfg: &RiverAnalysisConfig, run: &RunOptions) -> Result<RiverEvidence> {
        run.check_cancelled()?;
        for (field, value) in [
            ("semantic_field_inherit_cos", cfg.semantic_field_inherit_cos),
            ("tag_align_min_cos", cfg.tag_align_min_cos),
            ("weak_match_pivot", cfg.weak_match_pivot),
            ("genus_gate_floor", cfg.genus_gate_floor),
        ] {
            parameter("river", field, value, -1.0, 1.0)?;
        }
        if !cfg.genus_gate_span.is_finite() || cfg.genus_gate_span <= 0.0 {
            return Err(config("genus_gate_span", "must be positive and finite"));
        }
        let graph = self.observation.graph();
        let network = Network::new(self.observation, run)?;
        let cross = cfg
            .cross_source
            .as_ref()
            .map(|cfg| collect_routes(self.observation, cfg, run))
            .transpose()?;
        let workspace = path::Workspace {
            eval: self,
            network: &network,
            config: cfg,
            graph,
            run,
            pairs: HashMap::new(),
            contacts: HashMap::new(),
        };
        let analyze = |workspace: &mut path::Workspace<'_, '_>, i| {
            run.check_cancelled()?;
            let info = self.info(i);
            let mut results = Vec::new();
            for path in &info.paths {
                let mut item = workspace.score(i, Some(path))?;
                if let (Some(cfg), Some(routes)) = (&cfg.cross_source, &cross) {
                    let (strength, witnesses) =
                        wormhole::path_support(self, i, path, routes, cfg, run)?;
                    item.cross_source_transport_factor = strength;
                    item.witnesses = witnesses;
                }
                results.push(item);
            }
            if results.is_empty() {
                results.push(workspace.score(i, None)?);
            }
            Ok((self.candidates.ids()[i], info.document_id, results))
        };
        let paths = if run.parallelism.is_parallel() && self.candidates.len() >= 128 {
            (0..self.candidates.len())
                .into_par_iter()
                .with_min_len(128)
                .map_init(|| workspace.clone(), analyze)
                .collect::<Result<_>>()?
        } else {
            let mut workspace = workspace;
            (0..self.candidates.len())
                .map(|i| analyze(&mut workspace, i))
                .collect::<Result<_>>()?
        };
        run.check_cancelled()?;
        Ok(RiverEvidence {
            evaluation: self.identity().clone(),
            paths,
            omega: network.omega,
            morphology: network.morphology,
        })
    }
}

pub fn river_evidence(
    evaluation: &Evaluation<'_>,
    options: &RiverAnalysisOptions,
) -> Result<RiverEvidence> {
    evaluation.river_evidence(options)
}

pub fn bridge_witnesses(
    observation: &Observation,
    options: &CrossSourceOptions,
) -> Result<Vec<BridgeWitness>> {
    Ok(collect_routes(observation, &options.config, &options.run)?
        .into_iter()
        .flat_map(|route: CrossSourceRoute| route.witnesses)
        .collect())
}

#[cfg(test)]
mod reference_tests {
    use super::{PathEvidence, network::omega};
    use crate::{EnergyNode, EvidenceSource, ObservedEdge, SenseResult, SenseStats};
    use serde_json::Value;

    #[test]
    fn frozen_omega_and_continuous_budget_replay() {
        let data: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/vcp_5adb12b.json")).unwrap();
        let get = |x: &Value| x.as_f64().unwrap();
        for section in ["river", "wormhole"] {
            for case in data[section].as_array().unwrap() {
                let state = &case["input"]["queryState"];
                let nodes = state["riverNodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        let hop = row["hop"].as_u64().unwrap() as usize;
                        EnergyNode {
                            concept: row["id"].as_u64().unwrap(),
                            energy: get(&row["energy"]),
                            hop,
                            seed: None,
                            kind: if hop == 0 {
                                EvidenceSource::Core
                            } else {
                                EvidenceSource::Emergent
                            },
                        }
                    })
                    .collect();
                let observed_edges = state["riverEdges"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| ObservedEdge {
                        from: row["sourceId"].as_u64().unwrap(),
                        to: row["targetId"].as_u64().unwrap(),
                        flow: get(&row["flow"]),
                        normalized_flow: get(&row["normalizedFlow"]),
                        max_flow: get(&row["flow"]),
                        conductance: 0.8,
                        min_hop: 1,
                        wormhole: row["wormhole"].as_bool().unwrap(),
                        immediate_return: false,
                    })
                    .collect();
                let sense = SenseResult {
                    nodes,
                    observed_edges,
                    transitions: Vec::new(),
                    stats: SenseStats::default(),
                };
                let gate =
                    omega(&sense, get(&case["input"]["calibration"]["explainedRatio"])).omega;
                assert!(
                    (gate - get(&case["omega"])).abs() < 1e-12,
                    "{}",
                    case["case"]
                );
                let cap = 0.08;
                for row in case["results"].as_array().unwrap() {
                    let trace = &row["wormhole_unified"];
                    let transport = get(&trace["transportRequested"]) / (cap * gate);
                    let evidence = PathEvidence {
                        selected_path: None,
                        base_score: get(&row["base_score"]),
                        path_topology_factor: get(&row["path_bonus"]) / (cap * gate),
                        graph_request_factor: get(&trace["previousGraphBonus"]) / (cap * gate),
                        ordinary_transport_factor: if section == "river" { transport } else { 0.0 },
                        cross_source_transport_factor: if section == "wormhole" {
                            transport
                        } else {
                            0.0
                        },
                        field_support: 0.0,
                        node_alignment: 0.0,
                        edge_coverage: 0.0,
                        genus_gate: 1.0,
                        witnesses: vec![],
                    };
                    let (score, path, graph) = evidence.bonuses(cap, gate);
                    assert!((path - get(&row["path_bonus"])).abs() < 1e-12);
                    assert!((graph - get(&row["graph_bonus"])).abs() < 1e-12);
                    assert!((score - get(&row["score"])).abs() < 1e-12);
                }
            }
        }
    }
}
