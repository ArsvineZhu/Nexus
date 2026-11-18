// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{FusedRanking, RiverReadoutOutput, TOPOLOGY_BUDGET, explicit, select_path, sort_items};
use crate::ObservationState;
use crate::observation::invalid;
use nexus_core::kernel::{cosine_f32, norm_sq_f32};
use nexus_core::{
    CandidateId, CandidatePoolView, CsrGraphView, ExecutionContext, NodeId, Operator, PathId,
    RootFactView, RootId, ScratchPad, VcpResult, VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CrossFileWormholeConfig {
    pub unified_budget_ct: f64,
    pub min_relative_flow: f64,
    pub semantic_node_threshold: f64,
    pub maximum_river_edges: usize,
    pub omega_gamma: f64,
}

impl Default for CrossFileWormholeConfig {
    fn default() -> Self {
        Self {
            unified_budget_ct: TOPOLOGY_BUDGET,
            min_relative_flow: 0.5,
            semantic_node_threshold: 0.48,
            maximum_river_edges: 96,
            omega_gamma: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CrossFileWormholeInput<'asset, 'query> {
    pub observation: &'query ObservationState,
    pub readout: &'query RiverReadoutOutput,
    pub candidates: CandidatePoolView<'query>,
    pub graph: CsrGraphView<'asset>,
    pub root_facts: RootFactView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
}

struct Route<'a> {
    vector: &'a [f32],
    target: NodeId,
    flow: f64,
    relative_flow: f64,
    witnesses: Vec<[RootId; 2]>,
}

fn root_is_disjoint(root: RootId, excluded: &[RootId], facts: RootFactView<'_>) -> bool {
    !excluded.contains(&root)
        && !facts
            .required_roots(root)
            .iter()
            .any(|id| excluded.contains(id))
}

fn witnesses(
    ctx: &ExecutionContext<'_>,
    input: CrossFileWormholeInput<'_, '_>,
    source: NodeId,
    target: NodeId,
    focus: &HashSet<NodeId>,
    allow_calibrated_focus: bool,
) -> VcpResult<Vec<[RootId; 2]>> {
    const STAGE: &str = "wormhole.witnesses";
    let facts = input.root_facts;
    let mut matching = Vec::new();
    for index in 0..facts.root_count {
        ctx.check_cancelled(STAGE)?;
        let id = RootId::from_index(index, STAGE)?;
        if facts.has_tag(id, source) && facts.has_tag(id, target) {
            matching.push(id);
            if matching.len() > 128 {
                return Ok(Vec::new());
            }
        }
    }
    let specificity = if input.graph.max_inbound > 0.0 {
        1.0 - (input.graph.inbound[source.index()] / input.graph.max_inbound)
            .clamp(0.0, 1.0)
            .sqrt()
    } else {
        0.0
    };
    let calibrated = input.observation.disambiguated.differentia_confidence > 0.0
        && !input.observation.disambiguated.pruned_hypernyms.is_empty();
    let single_focus = allow_calibrated_focus
        && focus.len() == 1
        && focus.contains(&source)
        && (calibrated || specificity >= 0.65);
    let mut pairs = Vec::new();
    for (index, &left) in matching.iter().enumerate() {
        let Some(checksum) = facts.checksum(left) else {
            continue;
        };
        let left_tags = facts.tags(left);
        if left_tags.len() > 128 {
            continue;
        }
        for &right in &matching[index + 1..] {
            ctx.check_cancelled(STAGE)?;
            let Some(other_checksum) = facts.checksum(right) else {
                continue;
            };
            let right_tags = facts.tags(right);
            if checksum == other_checksum
                || right_tags.len() > 128
                || !root_is_disjoint(left, facts.required_roots(right), facts)
                || facts.required_roots(left).contains(&right)
            {
                continue;
            }
            let other_focus = focus.iter().any(|&node| {
                node != source
                    && node != target
                    && (facts.has_tag(left, node) || facts.has_tag(right, node))
            });
            if !single_focus && !other_focus {
                continue;
            }
            let bridge = left_tags.iter().any(|&node| {
                node != source
                    && node != target
                    && !focus.contains(&node)
                    && right_tags.binary_search(&node).is_ok()
            });
            if bridge {
                pairs.push([left, right]);
            }
        }
    }
    Ok(pairs)
}

fn build_routes<'asset>(
    ctx: &ExecutionContext<'_>,
    input: CrossFileWormholeInput<'asset, '_>,
    limit: usize,
    relative: f64,
    allow_calibrated_focus: bool,
) -> VcpResult<Vec<Route<'asset>>> {
    const STAGE: &str = "wormhole.routes";
    ctx.check_cancelled(STAGE)?;
    input.candidates.validate(STAGE)?;
    input.graph.validate(STAGE)?;
    input.root_facts.validate(STAGE)?;
    input.tag_vectors.validate(STAGE)?;
    let n = input.graph.node_count;
    if input.tag_vectors.rows != n
        || input.tag_vectors.dim != input.candidates.candidate_vectors.dim
        || input.readout.path_evidence.len() != input.candidates.candidate_count
        || input
            .candidates
            .path_tags
            .iter()
            .any(|node| node.index() >= n)
        || input
            .readout
            .items
            .iter()
            .any(|item| item.candidate.index() >= input.candidates.candidate_count)
    {
        return Err(invalid(
            STAGE,
            "graph, candidate and readout shapes must agree",
        ));
    }
    if !input.readout.omega.complete_observation || input.readout.omega.omega <= 0.0 {
        return Ok(Vec::new());
    }
    let edges = &input.observation.sense.edges;
    let reference = edges.iter().map(|e| e.flow).fold(0.0, f64::max);
    if !reference.is_finite() || reference <= 0.0 {
        return Ok(Vec::new());
    }
    let focus: HashSet<_> = input
        .observation
        .sense
        .nodes
        .iter()
        .filter(|node| node.hop == 0 && explicit(node.source_kind))
        .map(|node| node.node)
        .collect();
    let mut distinct = BTreeMap::<_, f64>::new();
    for edge in edges {
        if edge.wormhole
            && !edge.immediate_return
            && edge.source != edge.target
            && edge.flow > 0.0
            && edge.flow / reference >= relative
            && focus.contains(&edge.source)
            && edge.source.index() < n
            && edge.target.index() < n
            && input.graph.edge_weight(edge.source, edge.target) > 0.0
        {
            distinct
                .entry((edge.source, edge.target))
                .and_modify(|flow| *flow = flow.max(edge.flow))
                .or_insert(edge.flow);
        }
    }
    let mut ranked: Vec<_> = distinct.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut routes = Vec::new();
    for ((source, target), flow) in ranked.into_iter().take(limit) {
        let vector = input.tag_vectors.row(target.index()).unwrap();
        if norm_sq_f32(vector, ctx.precision) <= 1e-15 || vector.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let witnesses = witnesses(ctx, input, source, target, &focus, allow_calibrated_focus)?;
        if !witnesses.is_empty() {
            routes.push(Route {
                vector,
                target,
                flow,
                relative_flow: flow / reference,
                witnesses,
            });
        }
    }
    Ok(routes)
}

fn route_score(
    ctx: &ExecutionContext<'_>,
    input: CrossFileWormholeInput<'_, '_>,
    route: &Route<'_>,
    candidate: CandidateId,
    path: Option<PathId>,
    threshold: f64,
) -> f64 {
    let Some(path) = path else { return 0.0 };
    let excluded = input.candidates.candidate_roots(candidate);
    if !route.witnesses.iter().any(|roots| {
        roots
            .iter()
            .all(|&root| root_is_disjoint(root, excluded, input.root_facts))
    }) {
        return 0.0;
    }
    let body = input
        .candidates
        .candidate_vectors
        .row(candidate.index())
        .unwrap();
    let cosine = cosine_f32(body, route.vector, ctx.precision).clamp(0.0, 1.0);
    if cosine == 0.0 {
        return 0.0;
    }
    let mut supported = false;
    for &node in input.candidates.path_steps(path).0 {
        let vector = input.tag_vectors.row(node.index()).unwrap();
        let similarity = if node == route.target {
            1.0
        } else {
            cosine_f32(vector, route.vector, ctx.precision)
        };
        if similarity > threshold && cosine_f32(vector, body, ctx.precision) > 0.0 {
            supported = true;
            break;
        }
    }
    if supported {
        cosine * route.relative_flow
    } else {
        0.0
    }
}

pub struct CrossFileWormholeOperator;

impl Operator for CrossFileWormholeOperator {
    type Config = CrossFileWormholeConfig;
    type Input<'asset, 'query> = CrossFileWormholeInput<'asset, 'query>;
    type Output = RiverReadoutOutput;

    fn name(&self) -> &'static str {
        "river.cross_file_wormhole"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        _: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        if config.unified_budget_ct != input.readout.budget
            || config.omega_gamma != input.readout.gamma
            || !(0.0..=1.0).contains(&config.min_relative_flow)
            || !(0.0..1.0).contains(&config.semantic_node_threshold)
        {
            return Err(invalid(
                stage,
                "cross-file evidence must use the readout's topology budget and omega exponent",
            ));
        }
        let routes = build_routes(
            ctx,
            input,
            config.maximum_river_edges,
            config.min_relative_flow,
            true,
        )?;
        let mut output = input.readout.clone();
        if routes.is_empty() {
            return Ok(output);
        }
        let gate = input.readout.omega.omega.powf(config.omega_gamma);
        for item in &mut output.items {
            ctx.check_cancelled(stage)?;
            let candidate = item.candidate;
            *item = select_path(
                candidate,
                &output.path_evidence[candidate.index()],
                output.budget,
                |path| {
                    gate * routes
                        .iter()
                        .map(|route| {
                            route_score(
                                ctx,
                                input,
                                route,
                                candidate,
                                path,
                                config.semantic_node_threshold,
                            )
                        })
                        .fold(0.0, f64::max)
                },
            );
        }
        sort_items(&mut output.items, input.candidates);
        Ok(output)
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WormholeRrfFuseConfig {
    pub rrf_k: f64,
    pub omega_gamma: f64,
}

impl Default for WormholeRrfFuseConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            omega_gamma: 1.0,
        }
    }
}

pub struct WormholeRrfFuseOperator;

impl Operator for WormholeRrfFuseOperator {
    type Config = WormholeRrfFuseConfig;
    type Input<'asset, 'query> = CrossFileWormholeInput<'asset, 'query>;
    type Output = FusedRanking;

    fn name(&self) -> &'static str {
        "ranking.wormhole_rrf"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        _: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        if !config.rrf_k.is_finite()
            || config.rrf_k <= 0.0
            || !config.omega_gamma.is_finite()
            || config.omega_gamma < 0.0
        {
            return Err(invalid(stage, "invalid route RRF configuration"));
        }
        let routes = build_routes(ctx, input, 96, 0.5, false)?;
        let readout = input.readout;
        let mut output = FusedRanking {
            items: readout.items.clone(),
            rrf_active: false,
            diagnostics: readout.diagnostics.clone(),
        };
        let edges = &input.observation.sense.edges;
        let ordinary = edges
            .iter()
            .filter(|edge| !edge.wormhole && !edge.immediate_return && edge.source != edge.target)
            .map(|edge| edge.flow)
            .fold(0.0, f64::max);
        let reference = if ordinary > 0.0 {
            ordinary
        } else {
            edges.iter().map(|edge| edge.flow).fold(0.0, f64::max)
        };
        if routes.is_empty() || reference <= 0.0 {
            return Ok(output);
        }
        let omega = readout.omega.omega.powf(config.omega_gamma);
        let mut contributions = vec![0.0_f64; output.items.len()];
        for route in &routes {
            let mut ranked = Vec::new();
            for (index, item) in output.items.iter().enumerate() {
                ctx.check_cancelled(stage)?;
                let score = route_score(ctx, input, route, item.candidate, item.chosen_path, 0.48);
                if score > 0.0 {
                    ranked.push((index, score));
                }
            }
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let weight = route.flow / reference * omega;
            for (rank, (index, _)) in ranked.into_iter().enumerate() {
                contributions[index] =
                    contributions[index].max(weight / (config.rrf_k + (rank + 1) as f64));
            }
        }
        if contributions.iter().any(|&score| score > 0.0) {
            for (index, (item, contribution)) in
                output.items.iter_mut().zip(contributions).enumerate()
            {
                item.ranking_score = 1.0 / (config.rrf_k + (index + 1) as f64) + contribution;
            }
            output
                .items
                .sort_by(|a, b| b.ranking_score.total_cmp(&a.ranking_score));
            output.rrf_active = true;
        }
        Ok(output)
    }
}
