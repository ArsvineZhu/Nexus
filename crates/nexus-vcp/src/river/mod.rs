// SPDX-License-Identifier: CC-BY-NC-SA-4.0

mod path;
mod pool;
mod ranking;
mod wormhole;

use pool::RiverFeatures;
pub use ranking::*;
pub use wormhole::*;

use crate::observation::{finite_vector, invalid};
use crate::{
    FieldValues, ObservationState, ProjectionLimits, PropagationCompletion, SeedSourceKind,
    SenseOutput,
};
use nexus_core::{
    CandidateId, ExecutionContext, PathId, ProvenanceView, QueryCosines, RootFactView,
    SupportGroupView, TransportGraphView, VcpResult, VectorMatrixView,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Combined topology budget used by the VCP reference implementation.
pub const VCP_TOPOLOGY_BUDGET: f64 = 0.08;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverAnalysisConfig {
    pub semantic_field_inherit_cos: f64,
    pub tag_align_min_cos: f64,
    pub weak_match_pivot: f64,
    pub genus_gate_floor: f64,
    pub genus_gate_span: f64,
    pub evidence: EvidencePolicy,
    pub cross_file: Option<CrossFileConfig>,
    /// Maximum distinct cached pair cosines in each analysis workspace.
    pub max_pair_cosines: Option<usize>,
}

impl Default for RiverAnalysisConfig {
    fn default() -> Self {
        Self {
            semantic_field_inherit_cos: 0.82,
            tag_align_min_cos: 0.48,
            weak_match_pivot: 0.65,
            genus_gate_floor: 0.50,
            genus_gate_span: 0.22,
            evidence: EvidencePolicy::default(),
            cross_file: Some(CrossFileConfig::default()),
            max_pair_cosines: Some(2_000_000),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RiverAnalysisInput<'call> {
    pub observation: &'call ObservationState,
    pub query: &'call QueryCosines<'call, 'call>,
    pub fields: &'call FieldValues,
    pub root_facts: Option<RootFactView<'call>>,
    pub graph: TransportGraphView<'call>,
    pub provenance: ProvenanceView<'call>,
    pub support_groups: SupportGroupView<'call>,
    pub tag_vectors: VectorMatrixView<'call, f32>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidencePolicy {
    #[default]
    FullWeightWhenUnattributed,
    DiscountWhenUnattributed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    #[default]
    Atomic,
    Propositional,
    Narrative,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryMorphology {
    pub atomic_weight: f64,
    pub propositional_weight: f64,
    pub narrative_weight: f64,
    pub confidence: f64,
    pub dominant_mode: QueryMode,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OmegaOutput {
    pub omega: f64,
    pub omega_edge: f64,
    pub omega_emerge: f64,
    pub omega_flow: f64,
    pub regime: String,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverScoringConfig {
    /// Maximum combined path and graph bonus. The default is 0.02.
    pub unified_budget_ct: f64,
    pub omega_gamma: f64,
}

impl Default for RiverScoringConfig {
    fn default() -> Self {
        Self {
            unified_budget_ct: 0.02,
            omega_gamma: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReadoutItem {
    pub candidate: CandidateId,
    pub chosen_path: Option<PathId>,
    pub scores: RiverScores,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RiverScores {
    pub base_score: f64,
    pub path_bonus: f64,
    pub graph_bonus: f64,
    pub score: f64,
    pub topology_bonus: f64,
    pub genus_gate: f64,
    pub direct_anchor_contacts: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathEvidence {
    pub path: Option<PathId>,
    pub base_score: f64,
    pub path_topology_factor: f64,
    pub graph_request_factor: f64,
    pub ordinary_transport_factor: f64,
    pub cross_file_transport_factor: f64,
    pub path_quality: f64,
    pub field_support: f64,
    pub node_alignment: f64,
    pub edge_coverage: f64,
    pub genus_gate: f64,
    pub anchor_contacts: u8,
}

impl PathEvidence {
    fn item(self, candidate: CandidateId, cap: f64, gate: f64) -> ReadoutItem {
        let path_bonus = cap * self.path_topology_factor * gate;
        let remaining = cap - path_bonus;
        let transport = cap
            * gate
            * self
                .ordinary_transport_factor
                .max(self.cross_file_transport_factor);
        let graph = (cap * self.graph_request_factor * gate).min(remaining);
        let merged = if transport == 0.0 {
            graph
        } else if graph == 0.0 {
            transport
        } else {
            cap * (1.0 - (1.0 - graph / cap) * (1.0 - transport / cap))
        };
        let mut bonus = merged.min(remaining);
        if path_bonus + bonus > cap {
            bonus = bonus.next_down();
        }
        ReadoutItem {
            candidate,
            chosen_path: self.path,
            scores: RiverScores {
                base_score: self.base_score,
                path_bonus,
                graph_bonus: bonus,
                score: (self.base_score + path_bonus + bonus).clamp(0.0, 1.0),
                topology_bonus: path_bonus + bonus,
                genus_gate: self.genus_gate,
                direct_anchor_contacts: self.anchor_contacts,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct RiverReadoutOutput {
    pub items: Vec<ReadoutItem>,
    pub omega: OmegaOutput,
    pub morphology: QueryMorphology,
    pub propagation: PropagationCompletion,
    pub projection: ProjectionLimits,
    pub elapsed_ms: f64,
}

fn explicit(kind: SeedSourceKind) -> bool {
    matches!(kind, SeedSourceKind::Core | SeedSourceKind::LexicalSeed)
}

fn omega(sense: &SenseOutput, coverage: f64, complete: bool) -> OmegaOutput {
    let nodes = &sense.nodes;
    let edges = &sense.edges;
    let seeds = nodes.iter().filter(|node| node.hop == 0).count();
    let seed_count = seeds.max(1) as f64;
    let omega_edge = (edges.len() as f64 / (0.5 * seed_count)).min(1.0);
    let omega_emerge = (nodes.len().saturating_sub(seeds) as f64 / (0.3 * seed_count)).min(1.0);
    let positive = || {
        edges
            .iter()
            .filter(|edge| edge.flow.is_finite() && edge.flow > 0.0)
    };
    let count = positive().count();
    let omega_flow = match count {
        0 => 0.0,
        1 => 0.5,
        _ => {
            let scale = positive().map(|edge| edge.flow).fold(0.0, f64::max);
            let total = positive().map(|edge| edge.flow / scale).sum::<f64>();
            let entropy = positive()
                .map(|edge| (edge.flow / scale) / total)
                .filter(|p| *p > 0.0)
                .map(|p| -p * p.ln())
                .sum::<f64>();
            (entropy / (count as f64).ln()).clamp(0.0, 1.0)
        }
    };
    let factor = if complete { 1.0 } else { 0.5 };
    let value = (omega_edge.max(0.02) * omega_emerge.max(0.02) * omega_flow.max(0.02)).cbrt()
        * factor
        * coverage.clamp(0.0, 1.0);
    let regime = if value < 0.12 {
        "collapsed"
    } else if value < 0.45 {
        "sparse"
    } else {
        "dense"
    };
    OmegaOutput {
        omega: value,
        omega_edge,
        omega_emerge,
        omega_flow,
        regime: regime.into(),
    }
}

fn morphology(sense: &SenseOutput, complete: bool, slots: usize) -> QueryMorphology {
    let nodes = &sense.nodes;
    let edges = &sense.edges;
    let mut hops = vec![0; slots];
    let mut degrees = vec![[0_usize; 2]; slots];
    let peak = nodes.iter().map(|node| node.energy).fold(0.0, f64::max);
    let energies = || {
        nodes
            .iter()
            .map(|node| (node, if peak > 0.0 { node.energy / peak } else { 0.0 }))
    };
    let total = energies().map(|(_, energy)| energy).sum::<f64>();
    let denom = total.max(1e-12);
    let weighted_hop = energies()
        .map(|(node, energy)| node.hop as f64 * energy)
        .sum::<f64>()
        / denom;
    let variance = energies()
        .map(|(node, energy)| energy * (node.hop as f64 - weighted_hop).powi(2))
        .sum::<f64>()
        / denom;
    let depth = 1.0 - (-weighted_hop / 1.75).exp();
    let depth_variance = 1.0 - (-variance.sqrt() / 1.5).exp();
    let shallow = if total > 1e-12 {
        energies()
            .filter(|(node, _)| node.hop <= 1)
            .map(|(_, energy)| energy)
            .sum::<f64>()
            / denom
    } else {
        1.0
    };
    let hhi = energies()
        .map(|(_, energy)| (energy / denom).powi(2))
        .sum::<f64>();
    let uniform = 1.0 / nodes.len().max(1) as f64;
    let concentration = if nodes.len() > 1 {
        ((hhi - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let mut occupied = HashSet::new();
    for node in nodes {
        hops[node.node.index()] = node.hop;
        occupied.insert(node.hop);
    }
    let mut forward = 0.0;
    let mut lateral = 0.0;
    let mut flow = 0.0;
    for edge in edges.iter().filter(|edge| edge.normalized_flow > 0.0) {
        let source = edge.source.index();
        let target = edge.target.index();
        degrees[source][0] += 1;
        degrees[target][1] += 1;
        flow += edge.normalized_flow;
        if hops[source] < hops[target] {
            forward += edge.normalized_flow;
        }
        if hops[source] == hops[target] {
            lateral += edge.normalized_flow;
        }
    }
    forward /= flow.max(1e-12);
    lateral /= flow.max(1e-12);
    let mut counts = [0; 4];
    for node in nodes.iter().filter(|node| node.hop > 0) {
        let [outbound, inbound] = degrees[node.node.index()];
        counts[0] += 1;
        counts[1] += usize::from(outbound <= 1 && inbound <= 1);
        counts[2] += outbound.saturating_sub(1);
        counts[3] += inbound.saturating_sub(1);
    }
    let [_, chain_fit, branching, merging] =
        counts.map(|count| (count as f64 / counts[0].max(1) as f64).min(1.0));
    let chain = chain_fit * forward.sqrt() * (0.35 + 0.65 * depth) * (1.0 - 0.5 * branching);
    let max_hop = hops.into_iter().max().unwrap_or(0);
    let persistence =
        (occupied.len().saturating_sub(1) as f64 / max_hop.max(1) as f64 * depth.sqrt()).min(1.0);
    let middle = 1.0 - (2.0 * depth - 1.0).abs();
    let complexity = 0.35 * lateral + 0.35 * branching + 0.3 * merging;
    let confidence = ((1.0 - (-(nodes.len() as f64) / 8.0).exp())
        * (1.0 - (-(edges.len() as f64) / 8.0).exp()))
    .sqrt()
        * if complete { 1.0 } else { 0.5 };
    let weights = [
        1.45 * shallow + 0.9 * concentration - 1.25 * depth - 0.65 * persistence - 0.45 * chain,
        1.25 * complexity + 0.7 * middle + 0.35 * depth_variance - 0.25 * chain,
        1.4 * depth + 1.15 * chain + 0.8 * forward + 0.65 * persistence
            - 0.65 * branching
            - 0.3 * concentration,
    ]
    .map(f64::exp);
    let sum = weights.iter().sum::<f64>();
    let [atomic, propositional, narrative] =
        weights.map(|value| confidence * value / sum + (1.0 - confidence) / 3.0);
    let mode = if atomic >= propositional && atomic >= narrative {
        QueryMode::Atomic
    } else if narrative >= propositional {
        QueryMode::Narrative
    } else {
        QueryMode::Propositional
    };
    QueryMorphology {
        atomic_weight: atomic,
        propositional_weight: propositional,
        narrative_weight: narrative,
        confidence,
        dominant_mode: mode,
    }
}

/// Query-level evidence, computed independently of candidate paths.
pub fn analyze_river_network(
    sense: &SenseOutput,
    coverage: f64,
    slots: usize,
) -> VcpResult<(OmegaOutput, QueryMorphology)> {
    if !coverage.is_finite()
        || sense
            .nodes
            .iter()
            .any(|node| node.node.index() >= slots || !node.energy.is_finite() || node.energy < 0.0)
        || sense
            .edges
            .iter()
            .any(|edge| edge.source.index() >= slots || edge.target.index() >= slots)
    {
        return Err(invalid(
            "river.network",
            "coverage and river network must match the graph",
        ));
    }
    let complete = sense.propagation == PropagationCompletion::Complete;
    Ok((
        omega(sense, coverage, complete),
        morphology(sense, complete, slots),
    ))
}

/// All legal candidate paths, independent of the final topology budget and exponent.
pub struct RiverEvidence {
    paths: Vec<Vec<PathEvidence>>,
    omega: OmegaOutput,
    morphology: QueryMorphology,
    propagation: PropagationCompletion,
    projection: ProjectionLimits,
}

impl RiverEvidence {
    pub fn paths(&self) -> &[Vec<PathEvidence>] {
        &self.paths
    }

    pub fn readout(
        &self,
        config: &RiverScoringConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<RiverReadoutOutput> {
        const STAGE: &str = "river.readout";
        let start = std::time::Instant::now();
        ctx.check_cancelled(STAGE)?;
        if !(0.0..=1.0).contains(&config.unified_budget_ct)
            || !config.omega_gamma.is_finite()
            || config.omega_gamma < 0.0
        {
            return Err(invalid(
                STAGE,
                "invalid River scoring budget or omega exponent",
            ));
        }
        let gate = if self.omega.omega > 0.0 {
            self.omega.omega.powf(config.omega_gamma)
        } else {
            0.0
        };
        let mut items = Vec::with_capacity(self.paths.len());
        for (index, paths) in self.paths.iter().enumerate() {
            ctx.check_cancelled(STAGE)?;
            items.push(select_path(
                CandidateId::from_index(index, STAGE)?,
                paths,
                config.unified_budget_ct,
                gate,
            ));
        }
        sort_items(&mut items);
        ctx.check_cancelled(STAGE)?;
        Ok(RiverReadoutOutput {
            items,
            omega: self.omega.clone(),
            morphology: self.morphology.clone(),
            propagation: self.propagation,
            projection: self.projection,
            elapsed_ms: start.elapsed().as_secs_f64() * 1_000.0,
        })
    }
}

pub fn prepare_river_evidence(
    input: RiverAnalysisInput<'_>,
    config: &RiverAnalysisConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<RiverEvidence> {
    let stage = "river.paths";
    ctx.check_cancelled(stage)?;
    input.tag_vectors.validate(stage)?;
    input.provenance.validate(input.graph.nnz(), stage)?;
    input.support_groups.validate(stage)?;
    if config.max_pair_cosines == Some(0)
        || ![
            config.semantic_field_inherit_cos,
            config.tag_align_min_cos,
            config.weak_match_pivot,
            config.genus_gate_floor,
        ]
        .into_iter()
        .all(|v| (0.0..1.0).contains(&v))
        || !config.genus_gate_span.is_finite()
        || config.genus_gate_span <= 0.0
    {
        return Err(invalid(stage, "invalid RiverMemo configuration"));
    }

    let n = input.graph.node_count;
    let vectors = input.tag_vectors;
    let state = input.observation;
    let fields = input.fields;
    let candidates = input.query.rows();
    let count = candidates.candidate_count;
    if vectors.rows != n
        || vectors.dim != candidates.candidate_vectors.dim
        || state.original_query != input.query.original_query()
        || !fields.dual_fields.local_converged
        || !fields.dual_fields.transfer_converged
        || state.sense.nodes.iter().any(|node| node.node.index() >= n)
        || state
            .sense
            .edges
            .iter()
            .any(|edge| edge.source.index() >= n || edge.target.index() >= n)
        || [
            &fields.dual_fields.local_field,
            &fields.dual_fields.transfer_field,
        ]
        .into_iter()
        .any(|field| field.len() != n)
        || [
            &fields.dual_fields.local_domain,
            &fields.dual_fields.transfer_domain,
            &state.disambiguated.pruned_hypernyms,
        ]
        .into_iter()
        .any(|set| set.iter().any(|node| node.index() >= n))
    {
        return Err(invalid(
            stage,
            "candidate pool, graph and observation dimensions must agree",
        ));
    }
    finite_vector(stage, &state.original_query, vectors.dim)?;
    let mut topology_edges = Vec::new();
    let mut active = HashMap::new();
    for edge in &state.sense.edges {
        let flow = observed_flow(edge);
        if flow >= 0.015
            && let Some(index) = input.graph.find_edge(edge.source, edge.target)
            && input.graph.effective_weight(index) > 0.0
        {
            active.insert((edge.source, edge.target), flow);
            if topology_edges.len() < 96 {
                topology_edges.push((*edge, index, flow));
            }
        }
    }
    let explicit_seeds: HashSet<_> = state
        .sense
        .nodes
        .iter()
        .filter(|node| node.hop == 0 && explicit(node.source_kind))
        .map(|node| node.node)
        .collect();
    let hops: HashMap<_, _> = state
        .sense
        .nodes
        .iter()
        .map(|node| (node.node, node.hop))
        .collect();
    let mut ordinary = HashMap::<_, f64>::new();
    let mut row_statistics = HashMap::new();
    if state.sense.propagation == PropagationCompletion::Complete {
        for edge in &state.sense.edges {
            if edge.wormhole
                || edge.immediate_return
                || edge.flow <= 0.0
                || !explicit_seeds.contains(&edge.source)
                || hops.get(&edge.target) != Some(&1)
            {
                continue;
            }
            let (sum, max) = *row_statistics.entry(edge.source).or_insert_with(|| {
                input
                    .graph
                    .row_range(edge.source)
                    .map(|e| input.graph.effective_weight(e))
                    .fold((0.0, 0.0_f64), |(sum, max), w| (sum + w, max.max(w)))
            });
            let conductance = input.graph.edge_weight(edge.source, edge.target);
            if sum > 0.0 && conductance > 0.0 && conductance >= 0.5 * max {
                let retained = ordinary.entry(edge.target).or_default();
                *retained = retained.max(conductance / sum);
            }
        }
    }
    let focus = state
        .gated_seeds
        .exact_focus
        .filter(|node| explicit_seeds.contains(node));
    let focus_targets: HashSet<_> = if state.disambiguated.pruned_hypernyms.len() == 1
        && state.disambiguated.differentia_confidence > 0.0
    {
        state
            .sense
            .edges
            .iter()
            .filter(|edge| {
                Some(edge.source) == focus
                    && !edge.wormhole
                    && !edge.immediate_return
                    && hops.get(&edge.target) == Some(&1)
                    && active.contains_key(&(edge.source, edge.target))
            })
            .map(|edge| edge.target)
            .collect()
    } else {
        HashSet::new()
    };
    let (omega, morphology) = analyze_river_network(&state.sense, state.pyramid.coverage, n)?;
    let field_mix = [
        fields.basins.local_direction.is_some(),
        fields.basins.transfer_direction.is_some(),
    ];
    let features = RiverFeatures::prepare(ctx, input.query, input.fields)?;
    let peak_energy = state
        .sense
        .nodes
        .iter()
        .map(|node| node.energy)
        .fold(0.0, f64::max);
    let fields = [
        &fields.dual_fields.local_field,
        &fields.dual_fields.transfer_field,
    ]
    .map(|field| {
        let peak = field.iter().copied().fold(0.0, f64::max);
        if peak > 0.0 {
            field.iter().map(|&value| value / peak).collect::<Vec<_>>()
        } else {
            vec![0.0; field.len()]
        }
    });
    let mut workspace = path::ReadoutWorkspace {
        input,
        config,
        active: &active,
        topology_edges: &topology_edges,
        ordinary: &ordinary,
        max_hop: hops.values().copied().max().unwrap_or(0).max(1) as f64,
        explicit_seeds: &explicit_seeds,
        hops: &hops,
        focus_targets: &focus_targets,
        morphology: &morphology,
        features: &features,
        peak_energy,
        field_mix,
        fields: fields.each_ref().map(Vec::as_slice),
        tag_norms: HashMap::new(),
        ctx,
        pair_cosines: HashMap::new(),
        contacts: HashMap::new(),
    };
    let score = |workspace: &mut path::ReadoutWorkspace<'_, '_, '_>, index| -> VcpResult<_> {
        ctx.check_cancelled(stage)?;
        let candidate = CandidateId::from_index(index, stage)?;
        let paths = candidates.candidate_paths(candidate);
        let mut variants = Vec::with_capacity(paths.len().max(1));
        for path in paths {
            variants.push(workspace.score(candidate, Some(PathId::from_index(path, stage)?))?);
        }
        if variants.is_empty() {
            variants.push(workspace.score(candidate, None)?);
        }
        Ok(variants)
    };
    let mut paths = if ctx.parallel.is_parallel() && count >= 128 {
        ctx.parallel.install(|| {
            (0..count)
                .into_par_iter()
                .with_min_len(128)
                .map_init(|| workspace.clone(), score)
                .collect::<VcpResult<Vec<_>>>()
        })?
    } else {
        (0..count)
            .map(|index| score(&mut workspace, index))
            .collect::<VcpResult<Vec<_>>>()?
    };
    if let (Some(config), Some(root_facts)) = (&config.cross_file, input.root_facts) {
        wormhole::add_cross_file(
            WormholeRouteInput {
                observation: state,
                query: input.query,
                graph: input.graph,
                root_facts,
                tag_vectors: input.tag_vectors,
            },
            &omega,
            &mut paths,
            config,
            ctx,
        )?;
    }
    Ok(RiverEvidence {
        paths,
        omega,
        morphology,
        propagation: state.sense.propagation,
        projection: input.fields.basins.limits,
    })
}

fn observed_flow(edge: &crate::ObservationEdge) -> f64 {
    if edge.normalized_flow > 0.0 {
        edge.normalized_flow.clamp(0.0, 1.0)
    } else {
        edge.flow.clamp(0.0, 1.0)
    }
}

fn select_path(
    candidate: CandidateId,
    variants: &[PathEvidence],
    cap: f64,
    gate: f64,
) -> ReadoutItem {
    variants
        .iter()
        .map(|&path| path.item(candidate, cap, gate))
        .max_by(|a, b| {
            a.scores
                .score
                .total_cmp(&b.scores.score)
                .then_with(|| a.scores.topology_bonus.total_cmp(&b.scores.topology_bonus))
                .then_with(|| b.chosen_path.cmp(&a.chosen_path))
        })
        .expect("analysis supplies at least one path variant per candidate")
}

fn sort_items(items: &mut [ReadoutItem]) {
    items.sort_unstable_by(|a, b| {
        b.scores
            .score
            .total_cmp(&a.scores.score)
            .then(a.candidate.cmp(&b.candidate))
    });
}
