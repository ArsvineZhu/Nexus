// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{Network, PathEvidence, RiverAnalysisConfig};
use crate::candidates::BoundPath;
use crate::{Evaluation, Result, RunOptions};
use nexus_math::kernel::{PrecisionMode, cosine_f32};
use nexus_math::{NodeId, TransportGraphView};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy)]
struct Step {
    node: NodeId,
    position: usize,
    closure: f64,
}

#[derive(Clone, Copy, Default)]
pub(super) struct FieldContact {
    potential: [f64; 2],
    coverage: [f64; 2],
}

#[derive(Clone, Copy)]
struct Alignment {
    step: Step,
    quality: f64,
}

#[derive(Default)]
struct GraphEvidence {
    node_coverage: f64,
    edge_coverage: f64,
    node_alignment: f64,
    direction: f64,
    closure: f64,
    strongest: f64,
    has_exact: bool,
    has_edges: bool,
    node_score: f64,
    edge_score: f64,
}

#[derive(Clone)]
pub(super) struct Workspace<'a, 'data> {
    pub(super) eval: &'a Evaluation<'data>,
    pub(super) network: &'a Network,
    pub(super) config: &'a RiverAnalysisConfig,
    pub(super) graph: TransportGraphView<'a>,
    pub(super) run: &'a RunOptions,
    pub(super) pairs: HashMap<(NodeId, NodeId), f64>,
    pub(super) contacts: HashMap<NodeId, FieldContact>,
}
impl Workspace<'_, '_> {
    fn pair(&mut self, a: NodeId, b: NodeId) -> f64 {
        if a == b {
            return 1.0;
        }
        let key = (a.min(b), a.max(b));
        if let Some(&value) = self.pairs.get(&key) {
            return value;
        }
        let concepts = self.eval.observation.model.concepts();
        let value = cosine_f32(
            concepts.row(a),
            concepts.row(b),
            PrecisionMode::StrictLeftFold,
        )
        .clamp(-1.0, 1.0);
        if self.pairs.len() < 65_536 {
            self.pairs.insert(key, value);
        }
        value
    }

    fn contact(&mut self, tag: NodeId) -> Result<FieldContact> {
        if let Some(value) = self.contacts.get(&tag) {
            return Ok(*value);
        }
        let fields = &self.network.fields;
        let domains = &self.network.domains;
        let mut result = FieldContact {
            potential: std::array::from_fn(|lane| fields[lane][tag.index()]),
            coverage: domains
                .each_ref()
                .map(|domain| f64::from(domain.contains(&tag))),
        };
        if result.potential.contains(&0.0) {
            let mut best = None;
            let floor = self.config.semantic_field_inherit_cos;
            for &(node, _) in &self.network.nodes {
                self.run.check_cancelled()?;
                let similarity = self.pair(node, tag);
                if similarity > floor
                    && best.is_none_or(|(id, score)| {
                        similarity > score || (similarity == score && node < id)
                    })
                {
                    best = Some((node, similarity));
                }
            }
            if let Some((node, similarity)) = best {
                let discount = 0.75 * ((similarity - floor) / (1.0 - floor)).clamp(0.0, 1.0);
                for lane in 0..2 {
                    if result.potential[lane] == 0.0 {
                        result.potential[lane] = discount * fields[lane][node.index()];
                        result.coverage[lane] = discount * f64::from(domains[lane].contains(&node));
                    }
                }
            }
        }
        self.contacts.insert(tag, result);
        Ok(result)
    }
    fn path_quality(&mut self, steps: &[Step]) -> Result<f64> {
        let graph = self.graph;
        let fields = self.network.fields.each_ref();
        let domains = &self.network.domains;
        let mut used = HashSet::new();
        let mut quality = 0.0;
        let mut support = 0;
        for (index, step) in steps.iter().enumerate().take(steps.len().saturating_sub(1)) {
            let mut best = None;
            for (span, next) in steps[index + 1..].iter().take(2).enumerate() {
                let a = step.node;
                let b = next.node;
                let relation = (a.min(b), a.max(b));
                if a == b || used.contains(&relation) {
                    continue;
                }
                let forward = graph.edge_weight(a, b);
                let reverse = graph.edge_weight(b, a);
                if forward + reverse <= 0.0
                    || !domains.iter().any(|d| d.contains(&a) && d.contains(&b))
                {
                    continue;
                }
                let [local, transfer] =
                    fields.map(|field| (field[a.index()] * field[b.index()]).sqrt());
                let active_forward = self.network.active.get(&(a, b)).copied().unwrap_or(0.0);
                let active_reverse = self.network.active.get(&(b, a)).copied().unwrap_or(0.0);
                let activity = active_forward.max(active_reverse).clamp(0.0, 1.0);
                let direction = if activity > 0.0 {
                    active_forward / (active_forward + active_reverse).max(1e-12)
                } else {
                    forward / (forward + reverse)
                };
                let semantic = ((self.pair(a, b) + 1.0) * 0.5).clamp(0.0, 1.0);
                let field =
                    (local.max(transfer) * fields[0][b.index()].max(fields[1][b.index()])).sqrt();
                let continuity = (0.5 * semantic + 0.5 * field).clamp(0.0, 1.0);
                let result = (0.6 * local + 0.4 * transfer)
                    * direction.max(0.05).sqrt()
                    * continuity.sqrt()
                    * (0.5 + 0.5 * activity.sqrt())
                    * if span == 0 { 1.0 } else { 0.7 };
                if result > 0.0 && best.is_none_or(|(score, _)| result > score) {
                    best = Some((result, relation));
                }
            }
            if let Some((score, relation)) = best {
                used.insert(relation);
                quality += score;
                support += 1;
            }
        }
        let segments = steps.len().saturating_sub(1);
        if segments > 0 {
            quality /= segments as f64;
        } else if let Some(step) = steps.first() {
            quality = fields[0][step.node.index()].max(fields[1][step.node.index()]) * 0.5;
        }
        let closure =
            steps.iter().map(|step| step.closure).sum::<f64>() / steps.len().max(1) as f64;
        Ok(
            (quality * (0.5 + 0.25 * support as f64 / segments.max(1) as f64 + 0.25 * closure))
                .clamp(0.0, 1.0),
        )
    }

    fn topology(&mut self, candidate: usize, steps: &[Step]) -> Result<GraphEvidence> {
        if steps.is_empty() {
            return Ok(GraphEvidence::default());
        }
        let mut bindings = Vec::new();
        let mut total_node = 0.0;
        let threshold = self.config.tag_align_min_cos;
        for &(node, energy) in &self.network.nodes {
            self.run.check_cancelled()?;
            let weight = (energy / self.network.peak_energy).max(1e-9);
            total_node += weight;
            for (index, &step) in steps.iter().enumerate() {
                let exact = node == step.node;
                let similarity = if exact {
                    1.0
                } else {
                    self.pair(node, step.node)
                };
                if !exact && similarity <= threshold {
                    continue;
                }
                let normalized = if exact {
                    1.0
                } else {
                    ((similarity - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
                };
                let weak = if !exact
                    && self.config.weak_match_pivot > 0.0
                    && similarity < self.config.weak_match_pivot
                {
                    (similarity / self.config.weak_match_pivot).powi(2)
                } else {
                    1.0
                };
                let quality = (normalized * step.closure).sqrt() * weak;
                bindings.push((exact, node, index, weight, Alignment { step, quality }));
            }
        }
        bindings.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| (b.3 * b.4.quality).total_cmp(&(a.3 * a.4.quality)))
                .then(a.1.cmp(&b.1))
                .then(a.4.step.node.cmp(&b.4.step.node))
                .then(a.2.cmp(&b.2))
        });
        let mut chosen = BTreeMap::new();
        let mut used_tags = HashSet::new();
        let mut matched_node = 0.0;
        let mut node_quality = 0.0;
        let mut graph = GraphEvidence::default();
        for (exact, node, _, weight, alignment) in bindings {
            if chosen.contains_key(&node) || !used_tags.insert(alignment.step.node) {
                continue;
            }
            matched_node += weight * alignment.quality;
            node_quality += weight * alignment.quality.powi(2);
            graph.strongest = graph.strongest.max(alignment.quality);
            graph.has_exact |= exact;
            chosen.insert(node, alignment);
        }
        let density = 1.0
            / (1.0
                + (steps.len() as f64 - chosen.values().map(|a| a.quality).sum::<f64>() - 6.0)
                    .max(0.0)
                    / 16.0);
        graph.node_coverage = if total_node > 0.0 {
            (matched_node * density / total_node).clamp(0.0, 1.0)
        } else {
            0.0
        };
        graph.node_alignment = if matched_node > 0.0 {
            (node_quality / matched_node).clamp(0.0, 1.0)
        } else {
            0.0
        };
        graph.closure =
            chosen.values().map(|a| a.step.closure).sum::<f64>() / chosen.len().max(1) as f64;

        let max_hop = self.network.max_hop;
        let span = steps
            .first()
            .zip(steps.last())
            .map_or(1, |(a, b)| a.position.abs_diff(b.position).max(1)) as f64;
        let roots = self.eval.info(candidate).roots.as_ref();
        let graph_assets = self.eval.observation.model.graph_assets();
        let [
            mut matched_edge,
            mut distance_mass,
            mut direction_mass,
            mut topology_mass,
        ] = [0.0; 4];
        let total_edge = self.network.edge_mass;
        let mut ordered = chosen
            .keys()
            .filter_map(|source| self.network.topology.get(source))
            .flatten()
            .collect::<Vec<_>>();
        // Use the observation order for stable floating-point reduction.
        ordered.sort_unstable_by_key(|&&(order, _, _, _)| order);
        let mut used = HashSet::new();
        for &&(_, source, target, weight) in &ordered {
            let (Some(a), Some(b)) = (chosen.get(&source), chosen.get(&target)) else {
                continue;
            };
            let undirected = (a.step.node.min(b.step.node), a.step.node.max(b.step.node));
            if !used.insert(undirected) {
                continue;
            }
            let source_hop = self.network.hops.get(&source).copied().unwrap_or_default();
            let target_hop = self.network.hops.get(&target).copied().unwrap_or_default();
            let query_distance = source_hop.abs_diff(target_hop).max(1) as f64 / max_hop;
            let candidate_distance = a.step.position.abs_diff(b.step.position) as f64 / span;
            let distance = (-(query_distance - candidate_distance).abs() / 0.35).exp();
            let direction = if b.step.position > a.step.position {
                1.0
            } else {
                0.25
            };
            let independent = graph_assets
                .support
                .get(&(source, target))
                .map_or(0.15, |support| support.factor(roots));
            let endpoint = (a.quality * b.quality).sqrt();
            matched_edge += weight * endpoint;
            distance_mass += weight * endpoint * distance;
            direction_mass += weight * endpoint * direction;
            topology_mass += weight * endpoint.powi(2) * distance * direction * independent;
            graph.has_edges = true;
        }
        graph.edge_coverage = if total_edge > 0.0 {
            (matched_edge / total_edge).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (distance, topology) = if matched_edge > 0.0 {
            graph.direction = (direction_mass / matched_edge).clamp(0.0, 1.0);
            (distance_mass / matched_edge, topology_mass / matched_edge)
        } else {
            (0.0, 0.0)
        };
        graph.edge_score = (0.18 * graph.node_alignment
            + 0.22 * distance
            + 0.18 * graph.direction
            + 0.42 * topology)
            .clamp(0.0, 1.0);
        graph.node_score = (graph.node_alignment * graph.closure).sqrt();
        Ok(graph)
    }

    fn intent(&mut self, steps: &[Step]) -> Result<f64> {
        let mut total = 0.0;
        let mut max = 0.0_f64;
        let mut contacts = Vec::new();
        for &(seed, energy) in &self.network.seeds {
            self.run.check_cancelled()?;
            total += energy;
            max = max.max(energy);
            for step in steps {
                let exact = step.node == seed;
                let similarity = self.pair(seed, step.node);
                if similarity >= 0.72 {
                    contacts.push((exact, similarity, seed, step.node, energy));
                }
            }
        }
        if total <= 1e-12 {
            return Ok(1.0);
        }
        contacts.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then(a.2.cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        let mut used_seeds = HashSet::new();
        let mut used_tags = HashSet::new();
        let mut hit = 0.0;
        for (_, _, seed, tag, energy) in contacts {
            if !used_seeds.contains(&seed) && used_tags.insert(tag) {
                used_seeds.insert(seed);
                hit += energy;
            }
        }
        Ok((hit / (0.55 * total).max(max)).sqrt().clamp(0.35, 1.0))
    }

    pub(super) fn score(
        &mut self,
        candidate: usize,
        path: Option<&BoundPath>,
    ) -> Result<PathEvidence> {
        let cfg = self.config;
        let state = &self.eval.observation.analysis().seeds;
        let raw = self.eval.lane(candidate, 0).max(0.0);
        let steps: Vec<_> = path
            .into_iter()
            .flat_map(|path| &path.nodes)
            .enumerate()
            .map(|(index, &node)| Step {
                node,
                position: index + 1,
                closure: self.eval.closure(candidate, node).max(0.0),
            })
            .collect();
        let differentia = state.differentia.as_ref();
        let model = &self.eval.observation.model;
        let mut genus = 1.0_f64;
        if differentia.is_some()
            && state.differentia_confidence > 0.0
            && !steps
                .iter()
                .any(|step| self.network.focus_targets.contains(&step.node))
        {
            for &parent in &state.pruned_hypernyms {
                let parent = model.concepts().node(parent)?;
                let mut affinity = self.eval.closure(candidate, parent);
                for step in &steps {
                    affinity = affinity.max(if step.node == parent {
                        1.0
                    } else {
                        self.pair(parent, step.node)
                    });
                }
                genus = genus
                    .min(((affinity - cfg.genus_gate_floor) / cfg.genus_gate_span).clamp(0.0, 1.0));
            }
        }
        let mut field = FieldContact::default();
        for step in &steps {
            let contact = self.contact(step.node)?;
            for lane in 0..2 {
                field.potential[lane] += contact.potential[lane];
                field.coverage[lane] += contact.coverage[lane];
            }
        }
        let count = steps.len().max(1) as f64;
        let [local, transfer] = field.potential.map(|v| (v / count).clamp(0.0, 1.0));
        let [local_coverage, transfer_coverage] =
            field.coverage.map(|v| (v / count).clamp(0.0, 1.0));
        let field_support = |coverage: f64, potential: f64| {
            let domain =
                ((coverage / 0.5).clamp(0.0, 1.0) * (potential / 0.02).clamp(0.0, 1.0)).sqrt();
            let tail = 0.1 * (potential / (potential + 0.02)).sqrt();
            domain.max(tail)
        };
        let support = field_support(local_coverage, local)
            .max(field_support(transfer_coverage, transfer))
            * self.intent(&steps)?;
        let query = if differentia.is_some() {
            let mix = 0.5 * state.differentia_confidence.clamp(0.0, 1.0);
            (1.0 - mix) * raw + mix * genus * self.eval.differentia(candidate).max(0.0)
        } else {
            raw
        };
        let local_weight = if self.network.field_mix[0] { 0.2 } else { 0.0 };
        let transfer_weight = if self.network.field_mix[1] { 0.1 } else { 0.0 };
        let mass: f64 = local_weight + transfer_weight;
        let resonance = if mass > 0.0 {
            (local_weight * self.eval.lane(candidate, 2).max(0.0)
                + transfer_weight * self.eval.lane(candidate, 3).max(0.0))
                / mass
        } else {
            query
        };
        let base = query + (mass / (0.7 + mass) * (resonance - query).max(0.0)).min(0.03) * support;
        let geometry = self.path_quality(&steps)?;
        let topology_raw = (0.625 * geometry
            + 0.375
                * (0.35 * local_coverage
                    + 0.25 * transfer_coverage
                    + 0.25 * local
                    + 0.15 * transfer))
            .clamp(0.0, 1.0);
        let path_topology_factor = topology_raw * ((geometry / 0.15).min(1.0) * raw).sqrt();

        let evidence = self.topology(candidate, &steps)?;
        let node_branch = (evidence.node_coverage / 0.35).min(1.0);
        let edge_branch = (evidence.edge_coverage / 0.20).min(1.0);
        let morphology = &self.network.morphology;
        let non_atomic = if evidence.has_edges {
            0.70 * edge_branch + 0.30 * node_branch
        } else {
            0.35 * node_branch
        };
        let branch =
            morphology.atomic_weight * node_branch + (1.0 - morphology.atomic_weight) * non_atomic;
        let confidence = (branch * evidence.closure * evidence.node_alignment)
            .clamp(0.0, 1.0)
            .sqrt();
        let direction = 0.25 + 0.75 * evidence.direction;
        let grounding = if !evidence.has_exact && evidence.strongest < 0.65 {
            0.65
        } else {
            1.0
        };
        let node_weight = 0.75 * morphology.atomic_weight
            + 0.25 * morphology.propositional_weight
            + 0.15 * morphology.narrative_weight;
        let graph_request_factor = (node_weight * evidence.node_score
            + (1.0 - node_weight) * evidence.edge_score)
            * confidence
            * direction
            * grounding;
        let ordinary_transport_factor = if self.network.ordinary.is_empty() {
            0.0
        } else {
            steps
                .iter()
                .filter_map(|step| {
                    self.network
                        .ordinary
                        .get(&step.node)
                        .map(|conductance| conductance * step.closure)
                })
                .fold(0.0_f64, f64::max)
        };
        Ok(PathEvidence {
            selected_path: path
                .filter(|path| !path.nodes.is_empty())
                .map(|path| path.selected(model)),
            base_score: base,
            path_topology_factor,
            graph_request_factor,
            field_support: support,
            node_alignment: evidence.node_alignment,
            edge_coverage: evidence.edge_coverage,
            ordinary_transport_factor,
            cross_source_transport_factor: 0.0,
            witnesses: Vec::new(),
            genus_gate: genus,
        })
    }
}
