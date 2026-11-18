// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{PathEvidence, QueryMorphology, RiverReadoutConfig, RiverReadoutInput};
use nexus_core::kernel::{cosine_f32, cosine_f32_with_norms, norm_sq_f32};
use nexus_core::{CandidateId, ExecutionContext, NodeId, PathId, VcpError, VcpResult};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy)]
struct Step {
    node: NodeId,
    position: u32,
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
pub(super) struct ReadoutWorkspace<'r, 'asset, 'query, 'ctx> {
    pub(super) input: RiverReadoutInput<'asset, 'query>,
    pub(super) config: &'r RiverReadoutConfig,
    pub(super) active: &'r HashMap<(NodeId, NodeId), f64>,
    pub(super) explicit_seeds: &'r HashSet<NodeId>,
    pub(super) hops: &'r HashMap<NodeId, usize>,
    pub(super) focus_targets: &'r HashSet<NodeId>,
    pub(super) morphology: &'r QueryMorphology,
    pub(super) path_gate: f64,
    pub(super) field_mix: [bool; 2],
    pub(super) fields: [&'r [f64]; 2],
    pub(super) tag_norms: &'r [f64],
    pub(super) ctx: &'r ExecutionContext<'ctx>,
    pub(super) pair_cosines: HashMap<(NodeId, NodeId), f64>,
    pub(super) contacts: HashMap<NodeId, FieldContact>,
}

impl ReadoutWorkspace<'_, '_, '_, '_> {
    fn pair(&mut self, a: NodeId, b: NodeId) -> VcpResult<f64> {
        const STAGE: &str = "river.readout";
        let key = (a.min(b), a.max(b));
        let count = self.pair_cosines.len();
        let limit = self.ctx.limits.pair_work_limit;
        Ok(match self.pair_cosines.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                if count >= limit {
                    return Err(VcpError::ResourceLimitExceeded {
                        stage: STAGE,
                        limit_name: "pair_work_limit",
                        actual: count.saturating_add(1),
                        limit,
                    });
                }
                let matrix = self.input.tag_vectors;
                *entry.insert(cosine_f32_with_norms(
                    matrix.row(a.index()).unwrap(),
                    matrix.row(b.index()).unwrap(),
                    self.tag_norms[a.index()],
                    self.tag_norms[b.index()],
                    self.ctx.precision,
                    1e-12,
                ))
            }
        })
    }

    fn contact(&mut self, tag: NodeId) -> VcpResult<FieldContact> {
        if let Some(value) = self.contacts.get(&tag) {
            return Ok(*value);
        }
        let state = self.input.observation;
        let fields = self.fields;
        let domains = [
            &state.dual_fields.local_domain,
            &state.dual_fields.transfer_domain,
        ];
        let mut result = FieldContact {
            potential: fields.map(|field| field[tag.index()]),
            coverage: domains.map(|domain| f64::from(domain.contains(&tag))),
        };
        if result.potential.contains(&0.0) {
            let mut best = None;
            for node in &state.sense.nodes {
                let similarity = self.pair(node.node, tag)?;
                if similarity > self.config.semantic_field_inherit_cos
                    && best.is_none_or(|(id, score)| {
                        similarity > score || (similarity == score && node.node < id)
                    })
                {
                    best = Some((node.node, similarity));
                }
            }
            if let Some((node, similarity)) = best {
                let floor = self.config.semantic_field_inherit_cos;
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

    fn path_quality(&mut self, steps: &[Step]) -> VcpResult<f64> {
        let state = self.input.observation;
        let graph = self.input.graph;
        let fields = self.fields;
        let domains = [
            &state.dual_fields.local_domain,
            &state.dual_fields.transfer_domain,
        ];
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
                let active_forward = self.active.get(&(a, b)).copied().unwrap_or(0.0);
                let active_reverse = self.active.get(&(b, a)).copied().unwrap_or(0.0);
                let activity = active_forward.max(active_reverse).clamp(0.0, 1.0);
                let direction = if activity > 0.0 {
                    active_forward / (active_forward + active_reverse).max(1e-12)
                } else {
                    forward / (forward + reverse)
                };
                let semantic = ((self.pair(a, b)? + 1.0) * 0.5).clamp(0.0, 1.0);
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

    fn topology(&mut self, candidate: CandidateId, steps: &[Step]) -> VcpResult<GraphEvidence> {
        if steps.is_empty() {
            return Ok(GraphEvidence::default());
        }
        let state = self.input.observation;
        let mut bindings = Vec::new();
        let mut total_node = 0.0;
        let threshold = self.config.tag_align_min_cos;
        for node in &state.sense.nodes {
            self.ctx.check_cancelled("river.readout")?;
            let weight = node.normalized_energy.max(1e-9);
            total_node += weight;
            for (index, &step) in steps.iter().enumerate() {
                let exact = node.node == step.node;
                let similarity = if exact {
                    1.0
                } else {
                    self.pair(node.node, step.node)?
                };
                if similarity <= threshold {
                    continue;
                }
                let normalized = ((similarity - threshold) / (1.0 - threshold)).clamp(0.0, 1.0);
                let weak = if !exact && similarity < self.config.weak_match_pivot {
                    (similarity / self.config.weak_match_pivot).powi(2)
                } else {
                    1.0
                };
                let quality = (normalized * step.closure).sqrt() * weak;
                bindings.push((exact, node.node, index, weight, Alignment { step, quality }));
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

        let max_hop = self.hops.values().copied().max().unwrap_or(0).max(1) as f64;
        let span = steps
            .first()
            .zip(steps.last())
            .map_or(1, |(a, b)| a.position.abs_diff(b.position).max(1)) as f64;
        let roots = self.input.candidates.candidate_roots(candidate);
        let [
            mut total_edge,
            mut matched_edge,
            mut distance_mass,
            mut direction_mass,
            mut topology_mass,
        ] = [0.0; 5];
        for edge in state
            .sense
            .edges
            .iter()
            .filter(|edge| self.active.contains_key(&(edge.source, edge.target)))
            .take(96)
        {
            let weight = self.active[&(edge.source, edge.target)];
            total_edge += weight;
            let (Some(a), Some(b)) = (chosen.get(&edge.source), chosen.get(&edge.target)) else {
                continue;
            };
            let source_hop = self.hops.get(&edge.source).copied().unwrap_or_default();
            let target_hop = self.hops.get(&edge.target).copied().unwrap_or_default();
            let query_distance = source_hop.abs_diff(target_hop).max(1) as f64 / max_hop;
            let candidate_distance = a.step.position.abs_diff(b.step.position) as f64 / span;
            let distance = (-(query_distance - candidate_distance).abs() / 0.35).exp();
            let direction = if b.step.position > a.step.position {
                1.0
            } else {
                0.25
            };
            let index = self
                .input
                .graph
                .find_edge(edge.source, edge.target)
                .unwrap();
            let independent = self.input.provenance.independent_fraction_excluding(
                index,
                roots,
                &self.input.support_groups,
            );
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

    fn intent(&mut self, steps: &[Step]) -> VcpResult<f64> {
        let mut total = 0.0;
        let mut max = 0.0_f64;
        let mut contacts = Vec::new();
        for seed in self
            .input
            .observation
            .sense
            .nodes
            .iter()
            .filter(|node| self.explicit_seeds.contains(&node.node))
        {
            total += seed.energy;
            max = max.max(seed.energy);
            for step in steps {
                let exact = step.node == seed.node;
                let similarity = if exact {
                    1.0
                } else {
                    self.pair(seed.node, step.node)?
                };
                if similarity >= 0.72 {
                    contacts.push((exact, similarity, seed.node, step.node, seed.energy));
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
        candidate: CandidateId,
        path: Option<PathId>,
    ) -> VcpResult<PathEvidence> {
        let input = self.input;
        let cfg = self.config;
        let state = input.observation;
        let vector = input
            .candidates
            .candidate_vectors
            .row(candidate.index())
            .unwrap();
        let precision = self.ctx.precision;
        let (tags, positions) =
            path.map_or((&[][..], &[][..]), |path| input.candidates.path_steps(path));
        let differentia = &state.disambiguated.differentia_q_perp;
        let raw = input.pool.query_cosines[candidate.index()].clamp(0.0, 1.0);
        if tags.is_empty() && differentia.is_none() {
            return Ok(PathEvidence {
                path,
                base: raw,
                path_bonus: 0.0,
                graph_bonus: 0.0,
                ordinary_transport: 0.0,
                genus_gate: 1.0,
                anchor_contacts: 0,
            });
        }
        let candidate_norm = input
            .candidates
            .candidate_vectors
            .sq_norm_f64(candidate.index())
            .unwrap_or_else(|| norm_sq_f32(vector, precision));
        let closure = |node: NodeId| {
            cosine_f32_with_norms(
                input.tag_vectors.row(node.index()).unwrap(),
                vector,
                self.tag_norms[node.index()],
                candidate_norm,
                precision,
                1e-12,
            )
        };
        let steps: Vec<_> = tags
            .iter()
            .zip(positions)
            .map(|(&node, &position)| Step {
                node,
                position,
                closure: closure(node).clamp(0.0, 1.0),
            })
            .collect();
        let mut genus = 1.0_f64;
        if differentia.is_some()
            && state.disambiguated.differentia_confidence > 0.0
            && !steps
                .iter()
                .any(|step| self.focus_targets.contains(&step.node))
        {
            for &parent in &state.disambiguated.pruned_hypernyms {
                let mut affinity = closure(parent);
                for step in &steps {
                    affinity = affinity.max(if step.node == parent {
                        1.0
                    } else {
                        self.pair(parent, step.node)?
                    });
                }
                genus = genus
                    .min(((affinity - cfg.genus_gate_floor) / cfg.genus_gate_span).clamp(0.0, 1.0));
            }
        }
        let mut field = FieldContact::default();
        let mut seen = HashSet::new();
        let mut anchors = 0_u8;
        for step in &steps {
            if seen.insert(step.node) {
                let contact = self.contact(step.node)?;
                for lane in 0..2 {
                    field.potential[lane] += contact.potential[lane];
                    field.coverage[lane] += contact.coverage[lane];
                }
                anchors =
                    anchors.saturating_add(u8::from(self.explicit_seeds.contains(&step.node)));
            }
        }
        let count = seen.len().max(1) as f64;
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
        let query = if let Some(direction) = differentia {
            let mix = 0.5 * state.disambiguated.differentia_confidence.clamp(0.0, 1.0);
            (1.0 - mix) * raw
                + mix * genus * cosine_f32(direction, vector, precision).clamp(0.0, 1.0)
        } else {
            raw
        };
        let local_weight = if self.field_mix[0] { 0.2 } else { 0.0 };
        let transfer_weight = if self.field_mix[1] { 0.1 } else { 0.0 };
        let mass: f64 = local_weight + transfer_weight;
        let resonance = if mass > 0.0 {
            (local_weight * input.pool.local_cosines[candidate.index()].clamp(0.0, 1.0)
                + transfer_weight * input.pool.transfer_cosines[candidate.index()].clamp(0.0, 1.0))
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
        let path_bonus = cfg.unified_budget_ct
            * topology_raw
            * ((geometry / 0.15).min(1.0) * raw).sqrt()
            * self.path_gate;

        let evidence = self.topology(candidate, &steps)?;
        let node_branch = (evidence.node_coverage / 0.35).min(1.0);
        let edge_branch = (evidence.edge_coverage / 0.20).min(1.0);
        let mode = self.morphology.dominant_mode.as_str();
        let branch = if mode == "atomic" {
            node_branch
        } else if evidence.has_edges {
            0.70 * edge_branch + 0.30 * node_branch
        } else {
            0.35 * node_branch
        };
        let confidence = (branch * evidence.closure * evidence.node_alignment)
            .clamp(0.0, 1.0)
            .sqrt();
        let direction = 0.25 + 0.75 * evidence.direction;
        let grounding = if !evidence.has_exact && evidence.strongest < 0.65 {
            0.65
        } else {
            1.0
        };
        let node_weight = 0.75 * self.morphology.atomic_weight
            + 0.25 * self.morphology.propositional_weight
            + 0.15 * self.morphology.narrative_weight;
        let graph_bonus = cfg.unified_budget_ct
            * (node_weight * evidence.node_score + (1.0 - node_weight) * evidence.edge_score)
            * confidence
            * direction
            * grounding
            * self.path_gate;
        let mut ordinary_transport = 0.0_f64;
        for edge in &state.sense.edges {
            if edge.wormhole
                || edge.immediate_return
                || !self.explicit_seeds.contains(&edge.source)
                || self.hops.get(&edge.target) != Some(&1)
                || edge.flow <= 0.0
                || state.sense.state_truncations > 0
            {
                continue;
            }
            let Some(closure) = steps
                .iter()
                .filter(|step| step.node == edge.target)
                .map(|step| step.closure)
                .max_by(f64::total_cmp)
            else {
                continue;
            };
            let row = input.graph.row_range(edge.source);
            let mut sum = 0.0;
            let mut max = 0.0_f64;
            for edge in row {
                let weight = input.graph.effective_weight(edge);
                sum += weight;
                max = max.max(weight);
            }
            let conductance = input.graph.edge_weight(edge.source, edge.target);
            if sum > 0.0 && conductance > 0.0 && conductance >= 0.5 * max {
                ordinary_transport =
                    ordinary_transport.max(conductance / sum * closure * self.path_gate);
            }
        }
        Ok(PathEvidence {
            path,
            base,
            path_bonus,
            graph_bonus,
            ordinary_transport,
            genus_gate: genus,
            anchor_contacts: anchors,
        })
    }
}
