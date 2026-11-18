// SPDX-License-Identifier: CC-BY-NC-SA-4.0

mod path;
mod pool;
mod ranking;
mod wormhole;

pub use pool::*;
pub use ranking::*;
pub use wormhole::*;

use crate::observation::{finite_vector, invalid};
use crate::{ObservationState, SeedSourceKind};
use nexus_core::kernel::norm_sq_f32;
use nexus_core::{
    CandidateId, CandidatePoolView, CsrGraphView, Diagnostics, ExecutionContext, Operator, PathId,
    ProvenanceView, ScratchPad, SupportGroupView, VcpResult, VectorMatrixView,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const TOPOLOGY_BUDGET: f64 = 0.08;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverReadoutConfig {
    pub unified_budget_ct: f64,
    pub semantic_field_inherit_cos: f64,
    pub tag_align_min_cos: f64,
    pub weak_match_pivot: f64,
    pub genus_gate_floor: f64,
    pub genus_gate_span: f64,
    pub omega_gamma: f64,
}

impl Default for RiverReadoutConfig {
    fn default() -> Self {
        Self {
            unified_budget_ct: TOPOLOGY_BUDGET,
            semantic_field_inherit_cos: 0.82,
            tag_align_min_cos: 0.48,
            weak_match_pivot: 0.65,
            genus_gate_floor: 0.50,
            genus_gate_span: 0.22,
            omega_gamma: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RiverReadoutInput<'asset, 'query> {
    pub observation: &'query ObservationState,
    pub pool: &'query PreparedCandidatePool,
    pub candidates: CandidatePoolView<'query>,
    pub graph: CsrGraphView<'asset>,
    pub provenance: ProvenanceView<'asset>,
    pub support_groups: SupportGroupView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryMorphology {
    pub atomic_weight: f64,
    pub propositional_weight: f64,
    pub narrative_weight: f64,
    pub confidence: f64,
    pub dominant_mode: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OmegaOutput {
    pub omega: f64,
    pub omega_edge: f64,
    pub omega_emerge: f64,
    pub omega_flow: f64,
    pub regime: String,
    pub complete_observation: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReadoutItem {
    pub candidate: CandidateId,
    pub chosen_path: Option<PathId>,
    pub baseline_score: f64,
    pub path_bonus: f64,
    pub wormhole_bonus: f64,
    pub anchor_bonus: f64,
    pub quality_score: f64,
    pub ranking_score: f64,
    pub omega_contribution: f64,
    pub genus_gate: f64,
    pub direct_anchor_contacts: u8,
}

#[derive(Clone, Copy, Debug)]
struct PathEvidence {
    path: Option<PathId>,
    base: f64,
    path_bonus: f64,
    graph_bonus: f64,
    ordinary_transport: f64,
    genus_gate: f64,
    anchor_contacts: u8,
}

impl PathEvidence {
    fn item(self, candidate: CandidateId, cap: f64, cross_file: f64) -> ReadoutItem {
        let transport = cap * self.ordinary_transport.max(cross_file);
        let graph = self.graph_bonus.min((cap - self.path_bonus).max(0.0));
        let merged = if transport <= 0.0 {
            graph
        } else if graph <= 0.0 {
            transport
        } else {
            cap * (1.0 - (1.0 - graph / cap) * (1.0 - transport / cap))
        };
        let bonus = merged.min((cap - self.path_bonus).max(0.0));
        let score = (self.base + self.path_bonus + bonus).clamp(0.0, 1.0);
        ReadoutItem {
            candidate,
            chosen_path: self.path,
            baseline_score: self.base,
            path_bonus: self.path_bonus,
            wormhole_bonus: bonus,
            anchor_bonus: 0.0,
            quality_score: score,
            ranking_score: score,
            omega_contribution: self.path_bonus + bonus,
            genus_gate: self.genus_gate,
            direct_anchor_contacts: self.anchor_contacts,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RiverReadoutOutput {
    pub items: Vec<ReadoutItem>,
    pub omega: OmegaOutput,
    pub morphology: QueryMorphology,
    pub diagnostics: Diagnostics,
    // Per-path raw evidence allows cross-file transport to reselect the winning
    // complete path, and recompute the bounded composition without compounding it.
    path_evidence: Vec<Vec<PathEvidence>>,
    budget: f64,
    gamma: f64,
}

fn explicit(kind: SeedSourceKind) -> bool {
    matches!(kind, SeedSourceKind::Core | SeedSourceKind::LexicalSeed)
}

fn omega(observation: &ObservationState, complete: bool) -> OmegaOutput {
    let nodes = &observation.sense.nodes;
    let edges = &observation.sense.edges;
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
        * observation.pyramid.coverage.clamp(0.0, 1.0);
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
        complete_observation: complete,
    }
}

fn morphology(observation: &ObservationState, complete: bool, slots: usize) -> QueryMorphology {
    let nodes = &observation.sense.nodes;
    let edges = &observation.sense.edges;
    let mut hops = vec![0; slots];
    let mut degrees = vec![[0_usize; 2]; slots];
    let energies = || nodes.iter().map(|node| (node, node.normalized_energy));
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
        "atomic"
    } else if narrative >= propositional {
        "narrative"
    } else {
        "propositional"
    };
    QueryMorphology {
        atomic_weight: atomic,
        propositional_weight: propositional,
        narrative_weight: narrative,
        confidence,
        dominant_mode: mode.into(),
    }
}

pub struct RiverReadoutOperator;

impl Operator for RiverReadoutOperator {
    type Config = RiverReadoutConfig;
    type Input<'asset, 'query> = RiverReadoutInput<'asset, 'query>;
    type Output = RiverReadoutOutput;

    fn name(&self) -> &'static str {
        "river.readout"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        _: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        input.candidates.validate(stage)?;
        input.graph.validate(stage)?;
        input.tag_vectors.validate(stage)?;
        input.provenance.validate(input.graph.nnz(), stage)?;
        input.support_groups.validate(stage)?;
        if !(0.0..=TOPOLOGY_BUDGET).contains(&config.unified_budget_ct)
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
            || !config.omega_gamma.is_finite()
            || config.omega_gamma < 0.0
        {
            return Err(invalid(stage, "invalid RiverMemo configuration"));
        }

        let n = input.graph.node_count;
        let vectors = input.tag_vectors;
        let state = input.observation;
        let count = input.candidates.candidate_count;
        let pool = input.pool;
        if vectors.rows != n
            || vectors.dim != input.candidates.candidate_vectors.dim
            || pool.candidate_count != count
            || [
                pool.query_cosines.as_slice(),
                &pool.denoised_cosines,
                &pool.local_cosines,
                &pool.transfer_cosines,
            ]
            .into_iter()
            .any(|column| column.len() != count || column.iter().any(|v| !v.is_finite()))
            || input
                .candidates
                .path_tags
                .iter()
                .any(|node| node.index() >= n)
            || state.sense.nodes.iter().any(|node| node.node.index() >= n)
            || state
                .sense
                .edges
                .iter()
                .any(|edge| edge.source.index() >= n || edge.target.index() >= n)
            || [
                &state.dual_fields.local_field,
                &state.dual_fields.transfer_field,
            ]
            .into_iter()
            .any(|field| field.len() != n || field.iter().any(|v| !v.is_finite() || *v < 0.0))
            || [
                &state.dual_fields.local_domain,
                &state.dual_fields.transfer_domain,
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
        let active: HashMap<_, _> = state
            .sense
            .edges
            .iter()
            .filter_map(|edge| {
                let flow = observed_flow(edge);
                (flow >= 0.015 && input.graph.edge_weight(edge.source, edge.target) > 0.0)
                    .then_some(((edge.source, edge.target), flow))
            })
            .collect();
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
        let complete = !state.sense.source_field.is_empty() && state.sense.state_truncations == 0;
        let omega = omega(state, complete);
        let morphology = morphology(state, complete, n);
        let path_gate = if omega.omega > 0.0 {
            omega.omega.powf(config.omega_gamma)
        } else {
            0.0
        };
        let field_mix = [&state.basins.local_vector, &state.basins.transfer_vector]
            .map(|vector| norm_sq_f32(vector, ctx.precision) > 1e-12);
        let tag_norms: Vec<_> = (0..vectors.rows)
            .map(|index| {
                vectors
                    .sq_norm_f64(index)
                    .unwrap_or_else(|| norm_sq_f32(vectors.row(index).unwrap(), ctx.precision))
            })
            .collect();
        let fields = [
            &state.dual_fields.local_field,
            &state.dual_fields.transfer_field,
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
            explicit_seeds: &explicit_seeds,
            hops: &hops,
            focus_targets: &focus_targets,
            morphology: &morphology,
            path_gate,
            field_mix,
            fields: fields.each_ref().map(Vec::as_slice),
            tag_norms: &tag_norms,
            ctx,
            pair_cosines: HashMap::new(),
            contacts: HashMap::new(),
        };
        let score = |workspace: &mut path::ReadoutWorkspace<'_, '_, '_, '_>,
                     index|
         -> VcpResult<_> {
            ctx.check_cancelled(stage)?;
            let candidate = CandidateId::from_index(index, stage)?;
            let paths = input.candidates.candidate_paths(candidate);
            let mut variants = Vec::with_capacity(paths.len().max(1));
            for path in paths {
                variants.push(workspace.score(candidate, Some(PathId::from_index(path, stage)?))?);
            }
            if variants.is_empty() {
                variants.push(workspace.score(candidate, None)?);
            }
            let item = select_path(candidate, &variants, config.unified_budget_ct, |_| 0.0);
            Ok((item, variants))
        };
        let results = if ctx.parallel.is_parallel() && count >= 128 {
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
        let (mut items, cache): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        sort_items(&mut items, input.candidates);
        Ok(RiverReadoutOutput {
            items,
            omega,
            morphology,
            diagnostics: state.sense.diagnostics.clone(),
            path_evidence: cache,
            budget: config.unified_budget_ct,
            gamma: config.omega_gamma,
        })
    }
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
    evidence: impl Fn(Option<PathId>) -> f64,
) -> ReadoutItem {
    variants
        .iter()
        .map(|&path| path.item(candidate, cap, evidence(path.path)))
        .max_by(|a, b| {
            a.omega_contribution
                .total_cmp(&b.omega_contribution)
                .then_with(|| b.chosen_path.cmp(&a.chosen_path))
        })
        .expect("a candidate has at least its pathless fallback")
}

fn sort_items(items: &mut [ReadoutItem], pool: CandidatePoolView<'_>) {
    items.sort_unstable_by(|a, b| {
        let ai = a.candidate.index();
        let bi = b.candidate.index();
        b.ranking_score
            .total_cmp(&a.ranking_score)
            .then_with(|| pool.union_scores[bi].total_cmp(&pool.union_scores[ai]))
            .then_with(|| pool.union_ranks[ai].cmp(&pool.union_ranks[bi]))
            .then(a.candidate.cmp(&b.candidate))
    });
}
