// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use super::{OmegaOutput, QueryMode, QueryMorphology};
use crate::SenseResult;
use std::collections::{HashMap, HashSet};
pub(super) fn omega(sense: &SenseResult, coverage: f64) -> OmegaOutput {
    let nodes = &sense.nodes;
    let edges = &sense.observed_edges;
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
    let value = (omega_edge.max(0.02) * omega_emerge.max(0.02) * omega_flow.max(0.02)).cbrt()
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

pub(super) fn morphology(sense: &SenseResult) -> QueryMorphology {
    let nodes = &sense.nodes;
    let edges = &sense.observed_edges;
    let mut hops = HashMap::new();
    let mut degrees = HashMap::<u64, [usize; 2]>::new();
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
        hops.insert(node.concept, node.hop);
        occupied.insert(node.hop);
    }
    let mut forward = 0.0;
    let mut lateral = 0.0;
    let mut flow = 0.0;
    for edge in edges.iter().filter(|edge| edge.normalized_flow > 0.0) {
        let source = edge.from;
        let target = edge.to;
        degrees.entry(source).or_default()[0] += 1;
        degrees.entry(target).or_default()[1] += 1;
        flow += edge.normalized_flow;
        if hops.get(&source).copied().unwrap_or_default()
            < hops.get(&target).copied().unwrap_or_default()
        {
            forward += edge.normalized_flow;
        }
        if hops.get(&source).copied().unwrap_or_default()
            == hops.get(&target).copied().unwrap_or_default()
        {
            lateral += edge.normalized_flow;
        }
    }
    forward /= flow.max(1e-12);
    lateral /= flow.max(1e-12);
    let mut counts = [0; 4];
    for node in nodes.iter().filter(|node| node.hop > 0) {
        let [outbound, inbound] = degrees.get(&node.concept).copied().unwrap_or_default();
        counts[0] += 1;
        counts[1] += usize::from(outbound <= 1 && inbound <= 1);
        counts[2] += outbound.saturating_sub(1);
        counts[3] += inbound.saturating_sub(1);
    }
    let [_, chain_fit, branching, merging] =
        counts.map(|count| (count as f64 / counts[0].max(1) as f64).min(1.0));
    let chain = chain_fit * forward.sqrt() * (0.35 + 0.65 * depth) * (1.0 - 0.5 * branching);
    let max_hop = hops.into_values().max().unwrap_or(0);
    let persistence =
        (occupied.len().saturating_sub(1) as f64 / max_hop.max(1) as f64 * depth.sqrt()).min(1.0);
    let middle = 1.0 - (2.0 * depth - 1.0).abs();
    let complexity = 0.35 * lateral + 0.35 * branching + 0.3 * merging;
    let confidence = ((1.0 - (-(nodes.len() as f64) / 8.0).exp())
        * (1.0 - (-(edges.len() as f64) / 8.0).exp()))
    .sqrt();
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
