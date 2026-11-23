// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::epoch::EpochTable;
use super::{ConceptGraph, Residual, Seeds, SenseOptions};
use crate::{
    ConceptId, ConceptView, Error, Result, RunOptions, config as bad_config, input as bad_input,
    invalid,
};
use nexus_math::kernel::{PrecisionMode, cosine_f32};
use nexus_math::{EnergyNode, EvidenceSource, NodeId, TransportGraph, TransportGraphView};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const RESONANCE_POWER: i32 = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SenseConfig {
    pub max_hops: usize,
    pub base_momentum: f64,
    pub firing_threshold: f64,
    pub ordinary_decay: f64,
    pub wormhole_decay: f64,
    pub tension_threshold: f64,
    pub neighbor_width: usize,
    pub return_flow_factor: f64,
    pub fir_gamma: f64,
    /// Maximum number of active propagation states per hop.
    pub beam_width: usize,
    pub minimum_injected_current: f64,
    pub dynamic_wormholes: Option<DynamicWormholeConfig>,
}

impl Default for SenseConfig {
    fn default() -> Self {
        Self {
            max_hops: 4,
            base_momentum: 2.0,
            firing_threshold: 0.1,
            ordinary_decay: 0.25,
            wormhole_decay: 0.70,
            tension_threshold: 1.0,
            neighbor_width: 20,
            return_flow_factor: 0.15,
            fir_gamma: 0.6,
            beam_width: 2_000,
            minimum_injected_current: 0.01,
            dynamic_wormholes: Some(DynamicWormholeConfig::default()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DynamicWormholeConfig {
    pub threshold: f64,
    pub slots: usize,
    pub target_budget: usize,
}

impl Default for DynamicWormholeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.2,
            slots: 2,
            target_budget: 4096,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SenseInput<'a> {
    pub graph: &'a TransportGraph,
    pub concepts: ConceptView<'a>,
    pub seeds: &'a Seeds,
    pub residuals: &'a [Residual],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedEdge {
    pub from: ConceptId,
    pub to: ConceptId,
    pub flow: f64,
    pub max_flow: f64,
    pub normalized_flow: f64,
    pub conductance: f64,
    pub min_hop: usize,
    pub wormhole: bool,
    pub immediate_return: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition {
    pub previous: Option<ConceptId>,
    pub from: ConceptId,
    pub to: ConceptId,
    pub hop: usize,
    pub flow: f64,
    pub wormhole: bool,
    pub immediate_return: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SenseStats {
    pub hops: usize,
    pub retained_states: usize,
    pub pruned_states: usize,
    pub evaluated_targets: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SenseResult {
    pub nodes: Vec<EnergyNode>,
    pub observed_edges: Vec<ObservedEdge>,
    pub transitions: Vec<Transition>,
    pub stats: SenseStats,
}

#[derive(Clone, Copy)]
struct Spike {
    node: NodeId,
    previous: Option<NodeId>,
    energy: f64,
    momentum: f64,
    seed: NodeId,
    source_kind: EvidenceSource,
    layer: usize,
}

type StateKey = (Option<NodeId>, NodeId);

impl Spike {
    fn key(self) -> StateKey {
        (self.previous, self.node)
    }

    fn stable_key(self, input: ConceptGraph<'_>) -> (Option<ConceptId>, ConceptId) {
        (
            self.previous.map(|node| input.id(node)),
            input.id(self.node),
        )
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
        run: &RunOptions,
        input: ConceptGraph<'_>,
        residuals: &[&Residual],
        active: &[Spike],
        config: &SenseConfig,
        routing: &DynamicWormholeConfig,
    ) -> Result<()> {
        let graph = input.view;
        let evaluated = self.scores.len() / residuals.len();
        let remaining = routing.target_budget.saturating_sub(evaluated);
        if remaining == 0 {
            return Ok(());
        }
        self.targets.clear();
        let (stamps, epoch) = self.priorities.next_epoch(graph.node_count);
        for spike in active.iter().copied().filter(|spike| {
            spike.can_fire(config)
                && residuals
                    .last()
                    .is_some_and(|residual| residual.level > spike.layer)
        }) {
            run.check_cancelled()?;
            let row = spike.node.index();
            let start = graph.row_offsets[row];
            for edge in start..graph.row_offsets[row + 1] {
                if (edge - start).is_multiple_of(256) {
                    run.check_cancelled()?;
                }
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
        let compare = |a: &NodeId, b: &NodeId| {
            self.flux[b.index()]
                .total_cmp(&self.flux[a.index()])
                .then(input.id(*a).cmp(&input.id(*b)))
        };
        if remaining < self.targets.len() {
            self.targets.select_nth_unstable_by(remaining, compare);
            self.targets.truncate(remaining);
        }
        self.targets.sort_unstable_by(compare);
        for &target in &self.targets {
            run.check_cancelled()?;
            let vector = input.row(target);
            let offset = self.scores.len();
            self.offsets[target.index()] = offset;
            self.evaluated.mark(target.index());
            // Each entry is a suffix maximum, so a spike can access only strictly deeper residuals.
            let end = offset
                .checked_add(residuals.len())
                .ok_or_else(|| invalid("sense.residuals", "target and residual sizes overflow"))?;
            self.scores.resize(end, 0.0);
            let mut max = 0.0_f64;
            for level in (0..residuals.len()).rev() {
                max = max.max(cosine_f32(
                    &residuals[level].vector,
                    vector,
                    PrecisionMode::FastSimd,
                ));
                self.scores[offset + level] = max;
            }
        }
        Ok(())
    }

    fn select(
        &mut self,
        input: ConceptGraph<'_>,
        config: &SenseConfig,
        spike: Spike,
        residuals: &[&Residual],
        run: &RunOptions,
    ) -> Result<()> {
        let graph = input.view;
        let source = spike.node.index();
        let start = graph.row_offsets[source];
        let end = graph.row_offsets[source + 1];
        let limit = config.neighbor_width;
        self.selected_edges.clear();
        let deeper = residuals.partition_point(|residual| residual.level <= spike.layer);
        if let Some(routing) = &config.dynamic_wormholes
            && deeper < residuals.len()
        {
            self.ranked_edges.clear();
            for edge in start..end {
                if (edge - start).is_multiple_of(256) {
                    run.check_cancelled()?;
                }
                let target = graph.targets[edge];
                let index = target.index();
                if target == spike.node || !self.evaluated.is_stamped(index) {
                    continue;
                }
                let conductance = graph.effective_weight(edge);
                let score = self.scores[self.offsets[index] + deeper];
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
                    self.ranked_edges
                        .push((edge, conductance * score.powi(RESONANCE_POWER)));
                }
            }
            let compare = |&(a, av): &(usize, f64), &(b, bv): &(usize, f64)| {
                bv.total_cmp(&av)
                    .then_with(|| {
                        graph
                            .effective_weight(b)
                            .total_cmp(&graph.effective_weight(a))
                    })
                    .then_with(|| input.id(graph.targets[a]).cmp(&input.id(graph.targets[b])))
                    .then(a.cmp(&b))
            };
            if routing.slots < self.ranked_edges.len() {
                self.ranked_edges
                    .select_nth_unstable_by(routing.slots, compare);
                self.ranked_edges.truncate(routing.slots);
            }
            self.ranked_edges.sort_unstable_by(compare);
            self.selected_edges
                .extend(self.ranked_edges.iter().map(|&(edge, _)| (edge, true)));
        }
        if self.selected_edges.len() == limit {
            return Ok(());
        }
        self.ranked_edges.clear();
        for edge in start..end {
            if (edge - start).is_multiple_of(256) {
                run.check_cancelled()?;
            }
            let weight = graph.effective_weight(edge);
            if weight > 0.0
                && !self
                    .selected_edges
                    .iter()
                    .any(|&(selected, _)| edge == selected)
            {
                self.ranked_edges.push((edge, weight));
            }
        }
        let ordinary = limit
            .saturating_sub(self.selected_edges.len())
            .min(self.ranked_edges.len());
        let compare = |&(a, av): &(usize, f64), &(b, bv): &(usize, f64)| {
            bv.total_cmp(&av)
                .then(input.id(graph.targets[a]).cmp(&input.id(graph.targets[b])))
        };
        if ordinary < self.ranked_edges.len() {
            self.ranked_edges.select_nth_unstable_by(ordinary, compare);
        }
        self.ranked_edges[..ordinary].sort_unstable_by(compare);
        self.selected_edges.extend(
            self.ranked_edges[..ordinary]
                .iter()
                .map(|&(edge, _)| (edge, false)),
        );
        Ok(())
    }
}

fn add_spike(
    spike: Spike,
    states: &mut Vec<Spike>,
    slots: &mut HashMap<StateKey, usize>,
) -> Result<()> {
    match slots.entry(spike.key()) {
        std::collections::hash_map::Entry::Occupied(entry) => {
            let prior = &mut states[*entry.get()];
            prior.energy += spike.energy;
            if !prior.energy.is_finite() {
                return Err(Error::numeric("sense", "state energy overflowed"));
            }
            prior.momentum = prior.momentum.max(spike.momentum);
            prior.layer = prior.layer.min(spike.layer);
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(states.len());
            states.push(spike);
        }
    }
    Ok(())
}

fn cap_states(states: &mut Vec<Spike>, limit: usize, input: ConceptGraph<'_>) -> usize {
    let discarded = states.len().saturating_sub(limit);
    let stable = |state: &Spike| state.stable_key(input);
    if discarded > 0 {
        states.select_nth_unstable_by(limit, |a, b| {
            b.energy
                .total_cmp(&a.energy)
                .then(stable(a).cmp(&stable(b)))
        });
        states.truncate(limit);
    }
    states.sort_unstable_by_key(stable);
    discarded
}

fn accumulate(
    nodes: &mut [Option<EnergyNode>],
    input: ConceptGraph<'_>,
    spike: Spike,
    factor: f64,
    hop: usize,
) {
    let energy = spike.energy * factor;
    if energy == 0.0 {
        return;
    }
    let node = &mut nodes[spike.node.index()];
    if let Some(node) = node {
        node.energy += energy;
    } else {
        *node = Some(EnergyNode {
            concept: input.id(spike.node),
            energy,
            hop,
            kind: if hop == 0 {
                spike.source_kind
            } else {
                EvidenceSource::Emergent
            },
            seed: Some(input.id(spike.seed)),
        });
    }
}

fn validate(config: &SenseConfig) -> Result<usize> {
    let hops = config
        .max_hops
        .checked_add(1)
        .ok_or_else(|| bad_config("sense.max_hops", "hop count overflows"))?;
    if ![
        config.base_momentum,
        config.firing_threshold,
        config.tension_threshold,
        config.minimum_injected_current,
    ]
    .into_iter()
    .all(|v| v.is_finite() && v >= 0.0)
        || ![
            config.ordinary_decay,
            config.wormhole_decay,
            config.return_flow_factor,
            config.fir_gamma,
        ]
        .into_iter()
        .all(|v| (0.0..=1.0).contains(&v))
        || config.neighbor_width == 0
        || config.beam_width == 0
    {
        return Err(bad_config("sense", "invalid propagation configuration"));
    }
    if let Some(routing) = &config.dynamic_wormholes
        && (routing.slots > config.neighbor_width
            || routing.target_budget == 0
            || !(0.0..=1.0).contains(&routing.threshold))
    {
        return Err(bad_config(
            "sense.dynamic_wormholes",
            "invalid dynamic wormhole configuration",
        ));
    }
    Ok(hops)
}

pub fn sense(input: SenseInput<'_>, options: &SenseOptions) -> Result<SenseResult> {
    options.run.check_cancelled()?;
    let graph = ConceptGraph::new(input.graph, input.concepts, &options.run)?;
    let propagation = propagate(
        graph,
        input.seeds,
        input.residuals,
        &options.config,
        &options.run,
    )?;
    Ok(propagation.sense)
}

#[derive(Clone, Copy)]
pub(crate) struct WormholeFlow {
    pub(crate) from: ConceptId,
    pub(crate) to: ConceptId,
    pub(crate) flow: f64,
}

struct EdgeFlow {
    summary: ObservedEdge,
    forward_wormhole: f64,
    ordinary_first_hop: bool,
}

pub(super) struct Propagation {
    pub(super) sense: SenseResult,
    pub(super) wormholes: Vec<WormholeFlow>,
    pub(super) ordinary_first_hops: Vec<(ConceptId, ConceptId)>,
}

pub(super) fn propagate(
    input: ConceptGraph<'_>,
    seeds: &Seeds,
    residuals: &[Residual],
    config: &SenseConfig,
    run: &RunOptions,
) -> Result<Propagation> {
    run.check_cancelled()?;
    let graph = input.view;
    let hops = validate(config)?;
    let dim = input.concepts.dimension();
    let mut levels = Vec::new();
    for residual in residuals {
        run.check_cancelled()?;
        if residual.vector.len() != dim || !residual.vector.iter().all(|value| value.is_finite()) {
            return Err(invalid(
                "sense.residuals",
                "residuals must be finite and match the concept dimension",
            ));
        }
        levels.push(residual);
    }
    levels.sort_unstable_by_key(|residual| residual.level);
    let dynamic = config.dynamic_wormholes.as_ref().filter(|routing| {
        routing.slots > 0 && levels.last().is_some_and(|residual| residual.level > 0)
    });
    let mut routing = RoutingScores::default();
    if dynamic.is_some() {
        routing.begin(graph);
    } else {
        levels.clear();
    }

    let gamma = config.fir_gamma;
    let total = if gamma == 1.0 {
        hops as f64
    } else {
        -((hops as f64) * gamma.ln()).exp_m1() / (1.0 - gamma)
    };
    let mut weight = 1.0 / total;
    let mut state_slots = HashMap::new();
    let mut active = Vec::new();
    let mut next = Vec::new();
    let mut node_mass = vec![None; graph.node_count];
    let mut hop_activity = HashMap::<NodeId, Spike>::new();
    let mut flows = HashMap::<(NodeId, NodeId), EdgeFlow>::new();
    let mut events = Vec::new();
    let mut transitions = Vec::new();
    let mut stats = SenseStats::default();
    for seed in &seeds.items {
        run.check_cancelled()?;
        let node = input.slot(seed.concept)?;
        if !seed.weight.is_finite() || seed.weight < 0.0 || seed.kind == EvidenceSource::Emergent {
            return Err(bad_input(
                "sense.seeds",
                Some(seed.concept),
                "invalid seed weight or kind",
            ));
        }
        if seed.weight > 0.0 {
            add_spike(
                Spike {
                    node,
                    previous: None,
                    energy: seed.weight,
                    momentum: config.base_momentum,
                    seed: node,
                    source_kind: seed.kind,
                    layer: if seed.kind == EvidenceSource::Core {
                        0
                    } else {
                        seed.residual_level
                    },
                },
                &mut active,
                &mut state_slots,
            )?;
        }
    }
    active.sort_unstable_by_key(|state| state.stable_key(input));
    stats.retained_states = active.len();
    for &spike in &active {
        accumulate(&mut node_mass, input, spike, weight, 0);
    }

    for hop in 1..=config.max_hops {
        if !active.iter().any(|spike| spike.can_fire(config)) {
            break;
        }
        run.check_cancelled()?;
        stats.hops = hop;
        state_slots.clear();
        events.clear();
        if let Some(dynamic) = dynamic {
            routing.evaluate(run, input, &levels, &active, config, dynamic)?;
        }
        for spike in active
            .iter()
            .copied()
            .filter(|spike| spike.can_fire(config))
        {
            run.check_cancelled()?;
            routing.select(input, config, spike, &levels, run)?;
            for &(edge, dynamic) in &routing.selected_edges {
                let target = graph.targets[edge];
                let conductance = graph.effective_weight(edge);
                let wormhole = dynamic
                    || graph.is_static_wormhole[edge]
                    || conductance * graph.anchor_gains[target.index()] >= config.tension_threshold;
                let immediate_return = spike.previous == Some(target);
                let decay = if wormhole {
                    config.wormhole_decay
                } else {
                    config.ordinary_decay
                };
                let return_factor = if immediate_return {
                    config.return_flow_factor
                } else {
                    1.0
                };
                let injected = spike.energy * conductance * decay * return_factor;
                if injected == 0.0 || injected < config.minimum_injected_current {
                    continue;
                }
                let from = input.id(spike.node);
                let to = input.id(target);
                let observed = flows.entry((spike.node, target)).or_insert(EdgeFlow {
                    summary: ObservedEdge {
                        from,
                        to,
                        flow: 0.0,
                        max_flow: 0.0,
                        normalized_flow: 0.0,
                        conductance,
                        min_hop: hop,
                        wormhole: false,
                        immediate_return: false,
                    },
                    forward_wormhole: 0.0,
                    ordinary_first_hop: false,
                });
                observed.summary.flow += injected;
                observed.summary.max_flow = observed.summary.max_flow.max(injected);
                observed.summary.wormhole |= wormhole;
                observed.summary.immediate_return |= immediate_return;
                if wormhole && !immediate_return {
                    observed.forward_wormhole += injected;
                }
                observed.ordinary_first_hop |= hop == 1 && !wormhole && !immediate_return;
                let momentum = spike.momentum - if wormhole { 0.0 } else { 1.0 };
                if momentum < 0.0 {
                    continue;
                }
                let state = Spike {
                    node: target,
                    previous: Some(spike.node),
                    energy: injected,
                    momentum,
                    ..spike
                };
                events.push((
                    state.key(),
                    Transition {
                        from,
                        to,
                        previous: spike.previous.map(|node| input.id(node)),
                        hop,
                        flow: injected,
                        wormhole,
                        immediate_return,
                    },
                ));
                add_spike(state, &mut next, &mut state_slots)?;
            }
        }
        stats.pruned_states = stats
            .pruned_states
            .checked_add(cap_states(&mut next, config.beam_width, input))
            .ok_or_else(|| invalid("sense", "pruned state count overflows"))?;
        stats.retained_states = stats
            .retained_states
            .checked_add(next.len())
            .ok_or_else(|| invalid("sense", "retained state count overflows"))?;
        let retained: HashSet<_> = next.iter().map(|spike| spike.key()).collect();
        events.retain(|(key, _)| retained.contains(key));
        events.sort_unstable_by(|(_, a), (_, b)| {
            b.flow
                .total_cmp(&a.flow)
                .then(a.from.cmp(&b.from))
                .then(a.to.cmp(&b.to))
                .then(a.previous.cmp(&b.previous))
        });
        transitions.extend(events.iter().map(|(_, event)| *event));
        weight *= gamma;
        hop_activity.clear();
        for &spike in &next {
            hop_activity
                .entry(spike.node)
                .and_modify(|first| first.energy += spike.energy)
                .or_insert(spike);
        }
        let mut propagated = false;
        for &spike in hop_activity.values() {
            if !spike.energy.is_finite() {
                return Err(Error::numeric("sense", "hop activity overflowed"));
            }
            accumulate(&mut node_mass, input, spike, weight, hop);
            propagated |= spike.energy > config.minimum_injected_current;
        }
        if !propagated {
            break;
        }
        std::mem::swap(&mut active, &mut next);
        next.clear();
    }

    if dynamic.is_some() {
        stats.evaluated_targets = routing.scores.len() / levels.len();
    }
    let mut nodes: Vec<_> = node_mass.into_iter().flatten().collect();
    let mut observed_edges = Vec::with_capacity(flows.len());
    let mut wormholes = Vec::new();
    let mut ordinary_first_hops = Vec::new();
    for EdgeFlow {
        summary,
        forward_wormhole,
        ordinary_first_hop,
    } in flows.into_values()
    {
        if forward_wormhole > 0.0 {
            wormholes.push(WormholeFlow {
                from: summary.from,
                to: summary.to,
                flow: forward_wormhole,
            });
        }
        if ordinary_first_hop {
            ordinary_first_hops.push((summary.from, summary.to));
        }
        observed_edges.push(summary);
    }
    let max_edge = observed_edges
        .iter()
        .map(|edge| edge.flow)
        .fold(0.0, f64::max);
    if !nodes
        .iter()
        .map(|node| node.energy)
        .sum::<f64>()
        .is_finite()
        || !max_edge.is_finite()
    {
        return Err(Error::numeric(
            "sense",
            "propagation produced non-finite energy",
        ));
    }
    nodes.sort_unstable_by(|a, b| {
        b.energy
            .total_cmp(&a.energy)
            .then(a.concept.cmp(&b.concept))
    });
    for edge in &mut observed_edges {
        edge.normalized_flow = edge.flow / max_edge;
    }
    observed_edges.sort_unstable_by(|a, b| {
        b.flow
            .total_cmp(&a.flow)
            .then(a.from.cmp(&b.from))
            .then(a.to.cmp(&b.to))
    });
    Ok(Propagation {
        sense: SenseResult {
            nodes,
            observed_edges,
            transitions,
            stats,
        },
        wormholes,
        ordinary_first_hops,
    })
}
