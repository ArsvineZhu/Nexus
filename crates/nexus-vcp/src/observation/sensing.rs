// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{GatedSeeds, finite_vector, invalid, numeric_failure};
use nexus_core::kernel::cosine_f32;
use nexus_core::{
    EpochTable, ExecutionContext, NodeId, PrecisionMode, TransportGraphView, VcpResult,
    VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SenseConfig {
    pub max_safe_hops: usize,
    pub base_momentum: f64,
    pub firing_threshold: f64,
    pub base_decay: f64,
    pub wormhole_decay: f64,
    pub tension_threshold: f64,
    pub max_neighbors_per_node: usize,
    pub return_flow_factor: f64,
    pub fir_gamma: f64,
    /// Maximum number of active propagation states per hop.
    pub max_propagation_states: usize,
    pub minimum_injected_current: f64,
    /// Maximum number of saved transitions. None disables trace recording.
    pub max_transition_records: Option<usize>,
    pub dynamic_wormholes: Option<DynamicWormholeConfig>,
}

impl Default for SenseConfig {
    fn default() -> Self {
        Self {
            max_safe_hops: 4,
            base_momentum: 2.0,
            firing_threshold: 0.1,
            base_decay: 0.25,
            wormhole_decay: 0.70,
            tension_threshold: 1.0,
            max_neighbors_per_node: 20,
            return_flow_factor: 0.15,
            fir_gamma: 0.6,
            max_propagation_states: 2_000,
            minimum_injected_current: 0.01,
            max_transition_records: None,
            dynamic_wormholes: Some(DynamicWormholeConfig::default()),
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DynamicWormholeConfig {
    pub threshold: f64,
    pub slots: usize,
    pub max_targets: usize,
}

impl Default for DynamicWormholeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.2,
            slots: 2,
            max_targets: 4096,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SenseInput<'call> {
    pub graph: TransportGraphView<'call>,
    pub tag_vectors: VectorMatrixView<'call, f32>,
    pub seeds: &'call GatedSeeds,
    pub residuals: Option<&'call [Vec<f32>]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedSourceKind {
    Seed,
    Core,
    LexicalSeed,
    Emergent,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationNode {
    pub node: NodeId,
    pub energy: f64,
    pub source_kind: SeedSourceKind,
    pub hop: usize,
    pub seed_node: Option<NodeId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub flow: f64,
    pub max_flow: f64,
    pub normalized_flow: f64,
    pub conductance: f64,
    pub min_hop: usize,
    pub wormhole: bool,
    pub immediate_return: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationTransition {
    pub source: NodeId,
    pub target: NodeId,
    pub previous: Option<NodeId>,
    pub hop: usize,
    pub flow: f64,
    pub wormhole: bool,
    pub immediate_return: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PropagationCompletion {
    #[default]
    Complete,
    Limited {
        dropped_states: usize,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Trace {
    #[default]
    Disabled,
    Complete(Vec<ObservationTransition>),
    Limited(Vec<ObservationTransition>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SenseOutput {
    pub nodes: Vec<ObservationNode>,
    pub edges: Vec<ObservationEdge>,
    pub propagation: PropagationCompletion,
    pub trace: Trace,
    pub elapsed_ms: f64,
}

impl SenseOutput {
    fn reset(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.propagation = PropagationCompletion::Complete;
        self.elapsed_ms = 0.0;
    }
}

#[derive(Clone, Copy)]
struct Spike {
    node: NodeId,
    previous: Option<NodeId>,
    energy: f64,
    momentum: f64,
    seed: NodeId,
    source_kind: SeedSourceKind,
    layer: usize,
}

type StateKey = (Option<NodeId>, NodeId);

impl Spike {
    fn key(self) -> StateKey {
        (self.previous, self.node)
    }

    fn can_fire(self, config: &SenseConfig) -> bool {
        self.energy >= config.firing_threshold && self.momentum >= 0.0
    }
}

#[derive(Default)]
struct RoutingScores {
    evaluated: EpochTable,
    priorities: EpochTable,
    offsets: Vec<usize>,
    flux: Vec<f64>,
    targets: Vec<NodeId>,
    scores: Vec<f64>,
    ranked_edges: Vec<(usize, f64)>,
    selected_edges: Vec<(usize, bool)>,
}

impl RoutingScores {
    fn begin(&mut self, graph: TransportGraphView<'_>) {
        self.evaluated.next_epoch(graph.node_count);
        self.offsets.resize(graph.node_count, 0);
        self.flux.resize(graph.node_count, 0.0);
        self.scores.clear();
    }

    fn evaluate(
        &mut self,
        ctx: &ExecutionContext<'_>,
        input: SenseInput<'_>,
        residuals: &[Vec<f32>],
        active: &[Spike],
        config: &SenseConfig,
        routing: &DynamicWormholeConfig,
    ) -> VcpResult<()> {
        const STAGE: &str = "spike.sense";
        let graph = input.graph;
        let evaluated = self.scores.len() / residuals.len();
        let remaining = routing.max_targets.saturating_sub(evaluated);
        if remaining == 0 {
            return Ok(());
        }
        self.targets.clear();
        let (stamps, epoch) = self.priorities.next_epoch(graph.node_count);
        for spike in active.iter().copied().filter(|spike| {
            spike.can_fire(config) && spike.layer.saturating_add(1) < residuals.len()
        }) {
            ctx.check_cancelled(STAGE)?;
            let row = spike.node.index();
            for edge in graph.row_offsets[row]..graph.row_offsets[row + 1] {
                let conductance = graph.effective_weight(edge);
                let target = graph.targets[edge];
                let index = target.index();
                if conductance <= 0.0 || target == spike.node || self.evaluated.is_stamped(index) {
                    continue;
                }
                let flux = spike.energy * conductance;
                if stamps[index] == epoch {
                    self.flux[index] = self.flux[index].max(flux);
                } else {
                    self.targets.push(target);
                    stamps[index] = epoch;
                    self.flux[index] = flux;
                }
            }
        }
        self.targets.sort_unstable_by(|a, b| {
            self.flux[b.index()]
                .total_cmp(&self.flux[a.index()])
                .then(a.cmp(b))
        });
        for &target in self.targets.iter().take(remaining) {
            ctx.check_cancelled(STAGE)?;
            let vector = input
                .tag_vectors
                .row(target.index())
                .expect("graph and matrix were validated");
            let offset = self.scores.len();
            self.offsets[target.index()] = offset;
            self.evaluated.mark(target.index());
            // Entry L stores the strongest resonance from strictly later residuals.
            self.scores.resize(offset + residuals.len(), 0.0);
            let mut max = 0.0_f64;
            for level in (1..residuals.len()).rev() {
                max = max.max(cosine_f32(
                    &residuals[level],
                    vector,
                    PrecisionMode::FastSimd,
                ));
                self.scores[offset + level - 1] = max;
            }
        }
        Ok(())
    }

    fn select(&mut self, input: SenseInput<'_>, config: &SenseConfig, spike: Spike, layers: usize) {
        let graph = input.graph;
        let source = spike.node.index();
        let start = graph.row_offsets[source];
        let end = graph.row_offsets[source + 1];
        let limit = config.max_neighbors_per_node;
        self.selected_edges.clear();
        if let Some(routing) = &config.dynamic_wormholes
            && spike.layer.saturating_add(1) < layers
        {
            self.ranked_edges.clear();
            for edge in start..end {
                let target = graph.targets[edge];
                let index = target.index();
                if target == spike.node || !self.evaluated.is_stamped(index) {
                    continue;
                }
                let conductance = graph.effective_weight(edge);
                let score = self.scores[self.offsets[index] + spike.layer];
                let return_factor = if spike.previous == Some(target) {
                    config.return_flow_factor
                } else {
                    1.0
                };
                if conductance > 0.0
                    && score > 0.0
                    && score >= routing.threshold
                    && spike.energy * conductance * config.wormhole_decay * return_factor
                        >= config.minimum_injected_current
                {
                    self.ranked_edges.push((edge, conductance * score.powi(4)));
                }
            }
            self.ranked_edges.sort_unstable_by(|&(a, av), &(b, bv)| {
                bv.total_cmp(&av)
                    .then_with(|| {
                        graph
                            .effective_weight(b)
                            .total_cmp(&graph.effective_weight(a))
                    })
                    .then_with(|| graph.targets[a].cmp(&graph.targets[b]))
                    .then(a.cmp(&b))
            });
            self.selected_edges.extend(
                self.ranked_edges
                    .iter()
                    .take(routing.slots.min(limit))
                    .map(|&(edge, _)| (edge, true)),
            );
        }
        for edge in start..end {
            if self.selected_edges.len() == limit {
                break;
            }
            if graph.effective_weight(edge) > 0.0
                && !self
                    .selected_edges
                    .iter()
                    .any(|&(selected, _)| edge == selected)
            {
                self.selected_edges.push((edge, false));
            }
        }
    }
}

/// Per-caller propagation storage. Each request resets the logical state and retains capacity.
#[derive(Default)]
pub struct SenseWorkspace {
    active: Vec<Spike>,
    next: Vec<Spike>,
    state_slots: HashMap<StateKey, usize>,
    edge_flows: HashMap<(NodeId, NodeId), ObservationEdge>,
    node_slots: Vec<Option<usize>>,
    hop_energy: Vec<f64>,
    weights: Vec<f64>,
    hop_transitions: Vec<ObservationTransition>,
    trace_buffer: Vec<ObservationTransition>,
    routing: RoutingScores,
}

impl SenseWorkspace {
    fn clear(&mut self, n: usize) {
        self.active.clear();
        self.next.clear();
        self.state_slots.clear();
        self.edge_flows.clear();
        self.node_slots.resize(n, None);
        self.node_slots.fill(None);
        self.hop_energy.resize(n, 0.0);
        self.weights.clear();
        self.hop_transitions.clear();
    }

    fn accumulate(&mut self, output: &mut SenseOutput, spike: Spike, energy: f64, hop: usize) {
        let slot = &mut self.node_slots[spike.node.index()];
        if let Some(index) = *slot {
            output.nodes[index].energy += energy;
        } else {
            *slot = Some(output.nodes.len());
            output.nodes.push(ObservationNode {
                node: spike.node,
                energy,
                hop,
                source_kind: if hop == 0 {
                    spike.source_kind
                } else {
                    SeedSourceKind::Emergent
                },
                seed_node: Some(spike.seed),
            });
        }
    }
}

fn add_spike(spike: Spike, states: &mut Vec<Spike>, slots: &mut HashMap<StateKey, usize>) {
    match slots.entry(spike.key()) {
        std::collections::hash_map::Entry::Occupied(entry) => {
            let prior = &mut states[*entry.get()];
            prior.energy += spike.energy;
            prior.momentum = prior.momentum.max(spike.momentum);
            prior.layer = prior.layer.min(spike.layer);
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(states.len());
            states.push(spike);
        }
    }
}

fn cap_states(states: &mut Vec<Spike>, limit: usize) -> usize {
    let discarded = states.len().saturating_sub(limit);
    if discarded > 0 {
        states.sort_unstable_by(|a, b| b.energy.total_cmp(&a.energy).then(a.key().cmp(&b.key())));
        states.truncate(limit);
    }
    states.sort_unstable_by_key(|spike| spike.key());
    discarded
}

/// Run in caller-owned storage. For a warmed workspace and output with sufficient capacity,
/// the successful propagation path performs no heap allocations.
pub fn sense_into(
    input: SenseInput<'_>,
    config: &SenseConfig,
    ctx: &ExecutionContext<'_>,
    work: &mut SenseWorkspace,
    output: &mut SenseOutput,
) -> VcpResult<()> {
    output.reset();
    let mut trace = match std::mem::take(&mut output.trace) {
        Trace::Complete(trace) | Trace::Limited(trace) => trace,
        Trace::Disabled => std::mem::take(&mut work.trace_buffer),
    };
    trace.clear();
    let start = std::time::Instant::now();
    match propagate(ctx, config, input, work, output, &mut trace) {
        Ok(trace_limited) => {
            output.trace = if config.max_transition_records.is_none() {
                work.trace_buffer = trace;
                Trace::Disabled
            } else if trace_limited {
                Trace::Limited(trace)
            } else {
                Trace::Complete(trace)
            };
            output.elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
            Ok(())
        }
        Err(error) => {
            output.reset();
            trace.clear();
            work.trace_buffer = trace;
            Err(error)
        }
    }
}

fn propagate(
    ctx: &ExecutionContext<'_>,
    config: &SenseConfig,
    input: SenseInput<'_>,
    work: &mut SenseWorkspace,
    output: &mut SenseOutput,
    trace: &mut Vec<ObservationTransition>,
) -> VcpResult<bool> {
    let stage = "spike.sense";
    ctx.check_cancelled(stage)?;
    let graph = input.graph;
    input.tag_vectors.validate(stage)?;
    if input.tag_vectors.rows != graph.node_count {
        return Err(invalid(stage, "tag matrix must cover the graph"));
    }
    if config.max_safe_hops > 128
        || ![
            config.base_momentum,
            config.firing_threshold,
            config.tension_threshold,
            config.minimum_injected_current,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v >= 0.0)
        || ![
            config.base_decay,
            config.wormhole_decay,
            config.return_flow_factor,
            config.fir_gamma,
        ]
        .into_iter()
        .all(|v| (0.0..=1.0).contains(&v))
    {
        return Err(invalid(stage, "invalid propagation configuration"));
    }
    if config.max_propagation_states == 0 || config.max_transition_records == Some(0) {
        return Err(invalid(
            stage,
            "propagation and enabled trace budgets must be positive",
        ));
    }
    let routing = config.dynamic_wormholes.as_ref();
    if let Some(routing) = routing
        && (routing.slots == 0
            || routing.max_targets == 0
            || !(0.0..=1.0).contains(&routing.threshold))
    {
        return Err(invalid(stage, "invalid dynamic wormhole configuration"));
    }
    let residuals = input
        .residuals
        .filter(|r| routing.is_some() && r.len() >= 2);
    if let Some(residuals) = residuals {
        for vector in residuals {
            finite_vector(stage, vector, input.tag_vectors.dim)?;
        }
        work.routing.begin(graph);
    }
    work.clear(graph.node_count);
    let gamma = config.fir_gamma;
    work.weights
        .extend((0..=config.max_safe_hops).map(|hop| gamma.powi(hop as i32)));
    let total: f64 = work.weights.iter().sum();
    for weight in &mut work.weights {
        *weight /= total;
    }
    let state_limit = config.max_propagation_states;
    let transition_limit = config.max_transition_records.unwrap_or(0);
    let mut trace_limited = false;
    for seed in &input.seeds.seeds {
        ctx.check_cancelled(stage)?;
        if seed.node.index() >= graph.node_count || !seed.weight.is_finite() || seed.weight < 0.0 {
            return Err(invalid(stage, "seed slot or weight is invalid"));
        }
        if seed.weight > 0.0 {
            let source_kind = if seed.is_core {
                SeedSourceKind::Core
            } else if seed.lexical {
                SeedSourceKind::LexicalSeed
            } else {
                SeedSourceKind::Seed
            };
            add_spike(
                Spike {
                    node: seed.node,
                    previous: None,
                    energy: seed.weight,
                    momentum: config.base_momentum,
                    seed: seed.node,
                    source_kind,
                    layer: seed.residual_level,
                },
                &mut work.active,
                &mut work.state_slots,
            );
        }
    }
    let mut dropped_states = cap_states(&mut work.active, state_limit);
    for index in 0..work.active.len() {
        let spike = work.active[index];
        work.accumulate(output, spike, spike.energy * work.weights[0], 0);
    }
    for hop in 1..=config.max_safe_hops {
        if work.active.is_empty() {
            break;
        }
        ctx.check_cancelled(stage)?;
        work.state_slots.clear();
        work.hop_transitions.clear();
        if let (Some(residuals), Some(routing)) = (residuals, routing) {
            work.routing
                .evaluate(ctx, input, residuals, &work.active, config, routing)?;
        }

        for spike in work
            .active
            .iter()
            .copied()
            .filter(|spike| spike.can_fire(config))
        {
            ctx.check_cancelled(stage)?;
            work.routing
                .select(input, config, spike, residuals.map_or(0, <[_]>::len));
            for &(cursor, dynamic) in &work.routing.selected_edges {
                let target = graph.targets[cursor];
                let conductance = graph.effective_weight(cursor);
                let wormhole = dynamic
                    || graph.is_static_wormhole[cursor]
                    || conductance * graph.anchor_gains[target.index()] >= config.tension_threshold;
                let immediate_return = spike.previous == Some(target);
                let decay = if wormhole {
                    config.wormhole_decay
                } else {
                    config.base_decay
                };
                let return_factor = if immediate_return {
                    config.return_flow_factor
                } else {
                    1.0
                };
                let injected = spike.energy * conductance * decay * return_factor;
                if injected <= 0.0 || injected < config.minimum_injected_current {
                    continue;
                }
                let edge = work
                    .edge_flows
                    .entry((spike.node, target))
                    .or_insert(ObservationEdge {
                        source: spike.node,
                        target,
                        flow: 0.0,
                        max_flow: 0.0,
                        normalized_flow: 0.0,
                        conductance,
                        min_hop: hop,
                        wormhole: false,
                        immediate_return: false,
                    });
                edge.flow += injected;
                edge.max_flow = edge.max_flow.max(injected);
                edge.wormhole |= wormhole;
                edge.immediate_return |= immediate_return;
                let momentum = spike.momentum - if wormhole { 0.0 } else { 1.0 };
                if momentum < 0.0 {
                    continue;
                }
                if transition_limit > 0 {
                    if work.hop_transitions.len() < transition_limit {
                        work.hop_transitions.push(ObservationTransition {
                            source: spike.node,
                            target,
                            previous: spike.previous,
                            hop,
                            flow: injected,
                            wormhole,
                            immediate_return,
                        });
                    } else {
                        trace_limited = true;
                    }
                }
                add_spike(
                    Spike {
                        node: target,
                        previous: Some(spike.node),
                        energy: injected,
                        momentum,
                        ..spike
                    },
                    &mut work.next,
                    &mut work.state_slots,
                );
            }
        }
        dropped_states += cap_states(&mut work.next, state_limit);
        work.hop_transitions.retain(|event| {
            work.next
                .binary_search_by_key(&(Some(event.source), event.target), |state| state.key())
                .is_ok()
        });
        work.hop_transitions.sort_unstable_by(|a, b| {
            b.flow
                .total_cmp(&a.flow)
                .then(a.source.cmp(&b.source))
                .then(a.target.cmp(&b.target))
                .then(a.previous.cmp(&b.previous))
        });
        let remaining = transition_limit.saturating_sub(trace.len());
        trace_limited |= work.hop_transitions.len() > remaining;
        trace.extend(work.hop_transitions.iter().take(remaining).copied());

        work.hop_energy.fill(0.0);
        for index in 0..work.next.len() {
            let spike = work.next[index];
            work.hop_energy[spike.node.index()] += spike.energy;
            work.accumulate(output, spike, 0.0, hop);
        }
        let weight = work.weights[hop];
        let mut has_current = false;
        for (node, &current) in work.hop_energy.iter().enumerate() {
            if let Some(slot) = work.node_slots[node] {
                output.nodes[slot].energy += current * weight;
            }
            has_current |= current > config.minimum_injected_current;
        }
        std::mem::swap(&mut work.active, &mut work.next);
        work.next.clear();
        if !has_current {
            break;
        }
    }

    let total: f64 = output.nodes.iter().map(|node| node.energy).sum();
    let max_edge = work
        .edge_flows
        .values()
        .map(|edge| edge.flow)
        .fold(0.0, f64::max);
    if !total.is_finite() || !max_edge.is_finite() {
        return Err(numeric_failure(
            stage,
            "propagation produced non-finite energy",
        ));
    }
    output.nodes.retain(|node| node.energy > 0.0);
    output
        .nodes
        .sort_unstable_by(|a, b| b.energy.total_cmp(&a.energy).then(a.node.cmp(&b.node)));
    for edge in work.edge_flows.values() {
        output.edges.push(ObservationEdge {
            normalized_flow: edge.flow / max_edge,
            ..*edge
        });
    }
    output.edges.sort_unstable_by(|a, b| {
        b.flow
            .total_cmp(&a.flow)
            .then(a.source.cmp(&b.source))
            .then(a.target.cmp(&b.target))
    });
    if dropped_states > 0 {
        output.propagation = PropagationCompletion::Limited { dropped_states };
    }
    Ok(trace_limited)
}

pub fn sense(
    input: SenseInput<'_>,
    config: &SenseConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<SenseOutput> {
    let mut output = SenseOutput::default();
    sense_into(
        input,
        config,
        ctx,
        &mut SenseWorkspace::default(),
        &mut output,
    )?;
    Ok(output)
}
