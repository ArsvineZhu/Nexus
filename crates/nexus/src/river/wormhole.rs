// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::BTreeSet;

use nexus_math::NodeId;
use nexus_math::kernel::{PrecisionMode, cosine_f32};

use super::CrossSourceConfig;
use crate::candidates::BoundPath;
use crate::{
    BridgeWitness, Evaluation, EvidenceSource, Observation, Result, RunOptions, config, parameter,
};

pub(super) struct CrossSourceRoute {
    target: NodeId,
    relative_flow: f64,
    pub(super) witnesses: Vec<BridgeWitness>,
}

pub(super) fn collect_routes(
    obs: &Observation,
    cfg: &CrossSourceConfig,
    run: &RunOptions,
) -> Result<Vec<CrossSourceRoute>> {
    run.check_cancelled()?;
    parameter(
        "cross_source",
        "min_relative_flow",
        cfg.min_relative_flow,
        0.0,
        1.0,
    )?;
    parameter(
        "cross_source",
        "semantic_node_threshold",
        cfg.semantic_node_threshold,
        -1.0,
        1.0,
    )?;
    if cfg.max_river_edges == 0 {
        return Err(config("max_river_edges", "must be positive"));
    }
    let graph = obs.graph();
    let concepts = obs.model.concepts();
    let sense = &obs.analysis().sense;
    let foci = obs
        .analysis()
        .seeds
        .items
        .iter()
        .filter(|seed| {
            seed.weight > 0.0 && matches!(seed.kind, EvidenceSource::Core | EvidenceSource::Lexical)
        })
        .map(|seed| concepts.node(seed.concept))
        .collect::<Result<BTreeSet<_>>>()?;
    let max_flow = sense.observed_edges.first().map_or(0.0, |edge| edge.flow);
    if max_flow == 0.0 || foci.is_empty() {
        return Ok(Vec::new());
    }
    let mut ranked = Vec::new();
    for edge in &obs.wormholes {
        run.check_cancelled()?;
        if edge.from == edge.to || edge.flow / max_flow < cfg.min_relative_flow {
            continue;
        }
        let from = concepts.node(edge.from)?;
        let to = concepts.node(edge.to)?;
        if foci.contains(&from) && graph.edge_weight(from, to) > 0.0 {
            ranked.push(((from, to), edge.flow));
        }
    }
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let book = &obs.model.graph_assets().sources;
    let mut routes = Vec::new();
    for ((from, to), flow) in ranked.into_iter().take(cfg.max_river_edges) {
        run.check_cancelled()?;
        let specificity = if graph.max_inbound > 0.0 {
            1.0 - (graph.inbound[from.index()] / graph.max_inbound)
                .clamp(0.0, 1.0)
                .sqrt()
        } else {
            1.0
        };
        let single = foci.len() == 1
            && (obs.analysis().seeds.differentia_confidence > 0.0 || specificity >= 0.65);
        let left = &book.by_node[from.index()];
        let right = &book.by_node[to.index()];
        let smaller = if left.len() <= right.len() {
            left
        } else {
            right
        };
        let eligible: Vec<_> = smaller
            .iter()
            .map(|&i| &book.records[i])
            .filter(|source| {
                source.roots.is_some() && source.nodes.contains(&from) && source.nodes.contains(&to)
            })
            .collect();
        let relative_flow = flow / max_flow;
        let mut witnesses = Vec::new();
        for (index, a) in eligible.iter().enumerate() {
            for b in &eligible[index + 1..] {
                run.check_cancelled()?;
                if !a
                    .roots
                    .as_ref()
                    .unwrap()
                    .is_disjoint(b.roots.as_ref().unwrap())
                {
                    continue;
                }
                let other_foci: Vec<_> = foci
                    .iter()
                    .filter(|&&node| {
                        node != from
                            && node != to
                            && (a.nodes.contains(&node) || b.nodes.contains(&node))
                    })
                    .map(|node| concepts.ids()[node.index()])
                    .collect();
                if !single && other_foci.is_empty() {
                    continue;
                }
                if let Some(third) = a
                    .nodes
                    .intersection(&b.nodes)
                    .filter(|&&node| node != from && node != to && !foci.contains(&node))
                    .min_by_key(|node| concepts.ids()[node.index()])
                {
                    witnesses.push(BridgeWitness {
                        from: concepts.ids()[from.index()],
                        to: concepts.ids()[to.index()],
                        source_a: a.id,
                        source_b: b.id,
                        third: concepts.ids()[third.index()],
                        other_foci,
                        support: relative_flow,
                    });
                }
            }
        }
        if !witnesses.is_empty() {
            routes.push(CrossSourceRoute {
                target: to,
                relative_flow,
                witnesses,
            });
        }
    }
    Ok(routes)
}

pub(super) fn path_support(
    eval: &Evaluation<'_>,
    candidate: usize,
    path: &BoundPath,
    routes: &[CrossSourceRoute],
    cfg: &CrossSourceConfig,
    run: &RunOptions,
) -> Result<(f64, Vec<BridgeWitness>)> {
    if eval.info(candidate).roots.is_none() || path.nodes.is_empty() {
        return Ok((0.0, Vec::new()));
    }
    let concepts = eval.observation.model.concepts();
    let mut best = 0.0;
    let mut witnesses = Vec::new();
    for route in routes {
        run.check_cancelled()?;
        let closure = eval.closure(candidate, route.target);
        if closure <= 0.0 {
            continue;
        }
        let supports = path.nodes.iter().any(|&node| {
            if eval.closure(candidate, node) <= 0.0 {
                return false;
            }
            node == route.target
                || cosine_f32(
                    concepts.row(node),
                    concepts.row(route.target),
                    PrecisionMode::StrictLeftFold,
                ) > cfg.semantic_node_threshold
        });
        if supports {
            best = f64::max(best, closure * route.relative_flow);
            witnesses.extend(route.witnesses.iter().cloned());
        }
    }
    Ok((best, witnesses))
}
