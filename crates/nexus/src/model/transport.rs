// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::BTreeMap;

use nexus_math::kernel::{PrecisionMode, cosine_f32};
use nexus_math::{GraphInputOptions, NodeId, TransportGraph, WeightedEdge};
use serde::{Deserialize, Serialize};

use super::provenance::{Edge, EdgeSupport, SourceBook};
use super::{ConceptView, LocalGeometry, Relation, Source};
use crate::{Result, RunOptions, config, numeric, parameter};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FactsConfig {
    pub forward_gain: f64,
    pub reverse_gain: f64,
    pub min_reverse_gain: f64,
    pub max_reverse_gain: f64,
    pub distance_decay: f64,
    pub reverse_anchor_boost: bool,
    pub reverse_anchor_max: f64,
    pub reverse_inversion_guard: f64,
    pub semantic_gain_enabled: bool,
    pub semantic_gain_peak: f64,
    pub semantic_gain_sigma: f64,
}

impl Default for FactsConfig {
    fn default() -> Self {
        Self {
            forward_gain: 1.0,
            reverse_gain: 0.35,
            min_reverse_gain: 0.25,
            max_reverse_gain: 0.6,
            distance_decay: 0.08,
            reverse_anchor_boost: true,
            reverse_anchor_max: 1.35,
            reverse_inversion_guard: 0.9,
            semantic_gain_enabled: true,
            semantic_gain_peak: 0.65,
            semantic_gain_sigma: 0.25,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransportConfig {
    pub evidence_compression: f64,
    pub wormhole_gain: f64,
    pub wormhole_evidence_threshold: f64,
    pub hub_penalty_exponent: f64,
    pub hub_penalty_min: f64,
    pub hub_penalty_max: f64,
    pub hub_smoothing_ratio: f64,
    pub outbound_mass: f64,
    pub association_reserve_mass: f64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            evidence_compression: 1.0,
            wormhole_gain: 1.35,
            wormhole_evidence_threshold: 1.0,
            hub_penalty_exponent: 0.3,
            hub_penalty_min: 0.55,
            hub_penalty_max: 1.8,
            hub_smoothing_ratio: 0.1,
            outbound_mass: 0.95,
            association_reserve_mass: 0.05,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphConfig {
    pub facts: FactsConfig,
    pub transport: TransportConfig,
}

#[derive(Clone, Default)]
pub struct GraphOptions {
    pub config: GraphConfig,
    pub run: RunOptions,
}

pub struct GraphAssets {
    transport: TransportGraph,
    pub(crate) sources: SourceBook,
    pub(crate) support: BTreeMap<Edge, EdgeSupport>,
}

impl GraphAssets {
    pub fn transport(&self) -> &TransportGraph {
        &self.transport
    }
}

pub fn compile_graph(
    concepts: ConceptView<'_>,
    sources: &[Source],
    relations: &[Relation],
    local: &LocalGeometry,
    options: &GraphOptions,
) -> Result<GraphAssets> {
    let sources = SourceBook::new(concepts, sources, relations, &options.run)?;
    build_graph(concepts, sources, local, &options.config, &options.run)
}

pub(super) fn build_graph(
    concepts: ConceptView<'_>,
    sources: SourceBook,
    local: &LocalGeometry,
    cfg: &GraphConfig,
    run: &RunOptions,
) -> Result<GraphAssets> {
    local.check(concepts)?;
    let (raw, support) = accumulate(concepts, &sources, &local.anchor_gains, &cfg.facts, run)?;
    let mut transport = calibrate(concepts, raw, &local.anchor_gains, &cfg.transport, run)?;
    transport.set_anchor_gains(&local.anchor_gains)?;
    Ok(GraphAssets {
        transport,
        sources,
        support,
    })
}

type RawFacts = BTreeMap<Edge, f64>;
type SupportLedger = BTreeMap<Edge, EdgeSupport>;

fn accumulate(
    concepts: ConceptView<'_>,
    sources: &SourceBook,
    anchors: &[f64],
    cfg: &FactsConfig,
    run: &RunOptions,
) -> Result<(RawFacts, SupportLedger)> {
    for (field, value, lower, upper) in [
        ("forward_gain", cfg.forward_gain, 0.0, f64::INFINITY),
        (
            "min_reverse_gain",
            cfg.min_reverse_gain,
            0.0,
            cfg.reverse_gain,
        ),
        (
            "max_reverse_gain",
            cfg.max_reverse_gain,
            cfg.reverse_gain,
            f64::INFINITY,
        ),
        ("reverse_gain", cfg.reverse_gain, 0.0, f64::INFINITY),
        ("distance_decay", cfg.distance_decay, 0.0, f64::INFINITY),
        (
            "reverse_anchor_max",
            cfg.reverse_anchor_max,
            0.0,
            f64::INFINITY,
        ),
        (
            "reverse_inversion_guard",
            cfg.reverse_inversion_guard,
            0.0,
            1.0,
        ),
        ("semantic_gain_peak", cfg.semantic_gain_peak, -1.0, 1.0),
    ] {
        parameter("facts", field, value, lower, upper)?;
    }
    if !cfg.semantic_gain_sigma.is_finite() || cfg.semantic_gain_sigma <= 0.0 {
        return Err(config("semantic_gain_sigma", "must be positive and finite"));
    }
    let mut raw = RawFacts::new();
    let mut ledger = SupportLedger::new();
    let mut unknown = RawFacts::new();
    for source in &sources.records {
        let mut facts: BTreeMap<Edge, (f64, f64)> = source
            .relations
            .iter()
            .map(|(&edge, &m)| (edge, (m, m)))
            .collect();
        let mut record = |a, b, graph: f64, attributed: f64| -> Result<()> {
            if !graph.is_finite() || !attributed.is_finite() {
                return Err(numeric("facts", "source contribution overflowed"));
            }
            let values = facts.entry((a, b)).or_default();
            values.0 = values.0.max(graph);
            values.1 = values.1.max(attributed);
            Ok(())
        };
        for path in &source.paths {
            let denominator = path.unique.len().saturating_sub(1).max(1) as f64;
            let phi = |position: usize| 0.9 - 0.4 * position as f64 / denominator;
            for (i, &(_, a)) in path.unique.iter().enumerate() {
                run.check_cancelled()?;
                let anchor = if cfg.reverse_anchor_boost {
                    anchors[a.index()].min(cfg.reverse_anchor_max)
                } else {
                    1.0
                };
                let reverse_gain =
                    (cfg.reverse_gain * anchor).clamp(cfg.min_reverse_gain, cfg.max_reverse_gain);
                for (offset, &(_, b)) in path.unique[i + 1..].iter().enumerate() {
                    if offset.is_multiple_of(256) {
                        run.check_cancelled()?;
                    }
                    let decay = (-cfg.distance_decay * offset as f64).exp();
                    let semantic = if cfg.semantic_gain_enabled {
                        let z = cosine_f32(
                            concepts.row(a),
                            concepts.row(b),
                            PrecisionMode::StrictLeftFold,
                        );
                        if z < 0.15 {
                            (0.4 + z).max(0.0)
                        } else {
                            0.5 + 0.8
                                * (-0.5
                                    * ((z - cfg.semantic_gain_peak) / cfg.semantic_gain_sigma)
                                        .powi(2))
                                .exp()
                        }
                    } else {
                        1.0
                    };
                    let joint = phi(i) * phi(i + offset + 1) * decay * semantic;
                    let forward = joint * cfg.forward_gain;
                    let reverse = (joint * reverse_gain).min(forward * cfg.reverse_inversion_guard);
                    record(a, b, forward, cfg.forward_gain * decay)?;
                    record(b, a, reverse, cfg.reverse_gain * decay)?;
                }
            }
        }
        for (edge, (mass, attribution)) in facts {
            *raw.entry(edge).or_default() += mass;
            if attribution <= 0.0 {
                continue;
            }
            if let Some(roots) = &source.roots {
                let share = attribution / roots.len() as f64;
                let entry = ledger.entry(edge).or_default();
                for &root in roots {
                    let mass = entry.roots.entry(root).or_default();
                    *mass = mass.max(share);
                }
            } else {
                let mass = unknown.entry(edge).or_default();
                *mass = mass.max(attribution);
                ledger.entry(edge).or_default();
            }
        }
    }
    for (edge, value) in &mut ledger {
        value.known = value.roots.values().sum();
        value.total = value.known + unknown.get(edge).copied().unwrap_or(0.0);
        if !value.total.is_finite() {
            return Err(numeric("facts", "source support overflowed"));
        }
    }
    Ok((raw, ledger))
}

fn calibrate(
    concepts: ConceptView<'_>,
    raw: RawFacts,
    anchors: &[f64],
    cfg: &TransportConfig,
    run: &RunOptions,
) -> Result<TransportGraph> {
    for (field, value, min, max) in [
        ("wormhole_gain", cfg.wormhole_gain, 0.0, f64::INFINITY),
        (
            "wormhole_evidence_threshold",
            cfg.wormhole_evidence_threshold,
            0.0,
            f64::INFINITY,
        ),
        (
            "hub_penalty_exponent",
            cfg.hub_penalty_exponent,
            0.0,
            f64::INFINITY,
        ),
        (
            "hub_penalty_max",
            cfg.hub_penalty_max,
            cfg.hub_penalty_min,
            f64::INFINITY,
        ),
        (
            "hub_smoothing_ratio",
            cfg.hub_smoothing_ratio,
            0.0,
            f64::INFINITY,
        ),
        ("outbound_mass", cfg.outbound_mass, 0.0, 1.0),
        (
            "association_reserve_mass",
            cfg.association_reserve_mass,
            0.0,
            cfg.outbound_mass,
        ),
    ] {
        parameter("transport", field, value, min, max)?;
    }
    for (field, value) in [
        ("evidence_compression", cfg.evidence_compression),
        ("hub_penalty_min", cfg.hub_penalty_min),
    ] {
        if value <= 0.0 || !value.is_finite() {
            return Err(config(field, "must be positive and finite"));
        }
    }
    if cfg.outbound_mass == 1.0 {
        return Err(config("outbound_mass", "must be less than 1"));
    }
    let ids = concepts.ids();
    let mut rows: Vec<Vec<(NodeId, f64, bool)>> = vec![Vec::new(); ids.len()];
    let mut inbound = vec![0.0; ids.len()];
    for ((a, b), mass) in raw {
        run.check_cancelled()?;
        let evidence = (cfg.evidence_compression * mass).ln_1p();
        let tension = evidence * anchors[b.index()];
        let wormhole = tension >= cfg.wormhole_evidence_threshold;
        let value = evidence * if wormhole { cfg.wormhole_gain } else { 1.0 };
        inbound[b.index()] += value;
        if !value.is_finite() || !tension.is_finite() || !inbound[b.index()].is_finite() {
            return Err(numeric("transport", "conductance overflowed"));
        }
        if value > 0.0 {
            rows[a.index()].push((b, value, wormhole));
        }
    }
    let mut positive: Vec<_> = inbound
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .collect();
    let middle = positive.len() / 2;
    let median = if positive.is_empty() {
        1.0
    } else {
        *positive.select_nth_unstable_by(middle, f64::total_cmp).1
    };
    let denominator = median + (median * cfg.hub_smoothing_ratio).max(1e-9);
    if !denominator.is_finite() {
        return Err(numeric("transport", "hub smoothing overflowed"));
    }
    let penalties: Vec<_> = inbound
        .iter()
        .map(|&value| {
            (value / denominator)
                .max(1e-9)
                .powf(-cfg.hub_penalty_exponent)
                .clamp(cfg.hub_penalty_min, cfg.hub_penalty_max)
        })
        .collect();
    let mut edges = Vec::new();
    for (&from, mut row) in ids.iter().zip(rows) {
        run.check_cancelled()?;
        let mut total = 0.0;
        let mut jump = 0.0;
        for (to, value, wormhole) in &mut row {
            *value *= penalties[to.index()];
            total += *value;
            if *wormhole {
                jump += *value;
            }
        }
        if !total.is_finite() {
            return Err(numeric("transport", "row mass overflowed"));
        }
        if total == 0.0 {
            continue;
        }
        let reserve = if jump > 0.0 {
            cfg.association_reserve_mass
        } else {
            0.0
        };
        for (to, value, wormhole) in row {
            let weight = (cfg.outbound_mass - reserve) * (value / total)
                + if wormhole && reserve > 0.0 {
                    reserve * (value / jump)
                } else {
                    0.0
                };
            edges.push(WeightedEdge {
                from,
                to: ids[to.index()],
                weight,
                wormhole,
            });
        }
    }
    TransportGraph::from_edges(ids, &edges, &GraphInputOptions { run: run.clone() })
}

#[cfg(test)]
mod source_support_tests {
    use super::*;
    use crate::{Concept, ConceptData};
    use std::collections::BTreeSet;

    #[test]
    fn source_budget_uses_maximum_per_root_and_unknown_bucket() {
        let concepts = ConceptData::new(
            2,
            vec![
                Concept::new(101, vec![1.0, 0.0]),
                Concept::new(272, vec![0.0, 1.0]),
            ],
        )
        .unwrap();
        let concepts = concepts.view();
        let edge = (concepts.node(101).unwrap(), concepts.node(272).unwrap());
        let run = RunOptions::default();
        let support = |sources: &[Source], rows: &[(u64, f64)]| {
            let relations: Vec<_> = rows
                .iter()
                .map(|&(source, support)| Relation {
                    from: 101,
                    to: 272,
                    support,
                    source,
                })
                .collect();
            let book = SourceBook::new(concepts, sources, &relations, &run).unwrap();
            let (_, mut ledger) =
                accumulate(concepts, &book, &[1.0, 1.0], &FactsConfig::default(), &run).unwrap();
            ledger.remove(&edge).unwrap()
        };
        let sources = [
            Source::independent(7, vec![]),
            Source::independent(9, vec![]),
            Source::derived(11, vec![7, 7, 9], vec![]),
            Source::derived(13, vec![9, 7], vec![]),
        ];
        let roots = BTreeSet::from([7]);
        for rows in [
            &[(7, 2.0), (9, 1.0), (11, 4.0)][..],
            &[(7, 2.0), (9, 1.0), (11, 4.0), (13, 4.0)],
        ] {
            let ledger = support(&sources, rows);
            assert_eq!(ledger.roots, BTreeMap::from([(7, 2.0), (9, 2.0)]));
            assert_eq!(ledger.factor(Some(&roots)), 0.5);
        }
        assert_eq!(support(&sources, &[(7, 2.0)]).factor(Some(&roots)), 0.15);

        let sources = [
            Source::independent(7, vec![]),
            Source::independent(9, vec![]),
            Source::unknown_origin(23, vec![]),
            Source::derived(29, vec![7, 23], vec![]),
        ];
        let ledger = support(&sources, &[(9, 1.0), (23, 3.0), (29, 2.0)]);
        assert_eq!(ledger.roots, BTreeMap::from([(9, 1.0)]));
        assert_eq!(ledger.total, 4.0);
        assert_eq!(ledger.factor(Some(&roots)), 0.25);
        assert_eq!(ledger.factor(None), 0.15);
    }

    #[test]
    fn default_transport_uses_weighted_order_and_reserves_wormhole_mass() {
        let concepts = ConceptData::new(
            2,
            vec![
                Concept::new(0, vec![1.0, 0.0]),
                Concept::new(1, vec![0.0, 1.0]),
                Concept::new(2, vec![0.8, 0.6]),
                Concept::new(3, vec![0.0, -1.0]),
            ],
        )
        .unwrap();
        let view = concepts.view();
        let anchors = [1.5, 1.0, 1.0, 1.0];
        let source = Source::independent(50, vec![vec![0, 1, 2], vec![0, 1, 2]]);
        let run = RunOptions::default();
        let book = SourceBook::new(view, &[source], &[], &run).unwrap();
        let (facts, ledger) =
            accumulate(view, &book, &anchors, &FactsConfig::default(), &run).unwrap();
        let key = |a, b| (view.node(a).unwrap(), view.node(b).unwrap());
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        close(facts[&key(0, 1)], 0.9 * 0.7 * 0.4);
        close(facts[&key(1, 0)], 0.9 * 0.7 * 0.4 * 0.35 * 1.35);
        let distant = 0.9 * 0.5 * (-0.08_f64).exp() * (0.5 + 0.8 * (-0.18_f64).exp());
        close(facts[&key(0, 2)], distant);
        close(ledger[&key(2, 0)].total, 0.35 * (-0.08_f64).exp());

        let graph = calibrate(
            view,
            RawFacts::from([(key(0, 1), 3.0), (key(0, 2), 0.5)]),
            &anchors,
            &TransportConfig::default(),
            &run,
        )
        .unwrap();
        let graph = graph.view();
        let (a, b) = key(0, 1);
        let (_, c) = key(0, 2);
        let wormhole = 4.0_f64.ln() * 1.35;
        let ordinary = 1.5_f64.ln();
        let penalty = |value: f64| (value / (wormhole * 1.1)).powf(-0.3).clamp(0.55, 1.8);
        let w = wormhole * penalty(wormhole);
        let o = ordinary * penalty(ordinary);
        close(graph.edge_weight(a, b), 0.9 * w / (w + o) + 0.05);
        close(graph.edge_weight(a, c), 0.9 * o / (w + o));
        close(graph.row_statistics(a).0, 0.95);
        assert_eq!(graph.row_statistics(view.node(3).unwrap()), (0.0, 0.0));
    }
}
