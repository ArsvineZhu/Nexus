// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::observation::invalid;
use crate::{ObservationTransition, SenseOutput};
use nexus_core::kernel::cosine_f32;
use nexus_core::{
    CandidateId, ExecutionContext, PathId, PrecisionMode, ProvenanceView, QueryCosines,
    SupportGroupView, ThinkingStageInputView, TransportGraphView, VcpError, VcpResult,
    VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThinkingConfig {
    pub min_closure: f64,
    pub relevance_weight: f64,
    pub route_weight: f64,
    pub novelty_repeat_factor: f64,
    pub root_no_parent_penalty: f64,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            min_closure: 0.2,
            relevance_weight: 0.35,
            route_weight: 0.65,
            novelty_repeat_factor: 0.2,
            root_no_parent_penalty: 0.8,
        }
    }
}

/// Records all retained transitions from a completed propagation.
#[derive(Clone, Copy, Debug)]
pub struct CompleteTrace<'a> {
    transitions: &'a [ObservationTransition],
}

impl SenseOutput {
    pub fn complete_trace(&self) -> VcpResult<CompleteTrace<'_>> {
        let crate::Trace::Complete(transitions) = &self.trace else {
            return Err(VcpError::ThinkingUnavailable {
                reason: "a complete recorded trace is required",
            });
        };
        if self.propagation != crate::PropagationCompletion::Complete {
            return Err(VcpError::ThinkingUnavailable {
                reason: "propagation trace unavailable or incomplete",
            });
        }
        Ok(CompleteTrace { transitions })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ThinkingInput<'call> {
    pub trace: CompleteTrace<'call>,
    pub query: &'call QueryCosines<'call, 'call>,
    pub stages: &'call [ThinkingStageInputView<'call>],
    pub graph: TransportGraphView<'call>,
    pub provenance: ProvenanceView<'call>,
    pub support_groups: SupportGroupView<'call>,
    pub tag_vectors: VectorMatrixView<'call>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingRelation {
    Analogy,
    Continuation,
    Root,
    Supplement,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThinkingSelection {
    pub candidate: CandidateId,
    pub chosen_path: Option<PathId>,
    pub score: f64,
    pub parents: Vec<CandidateId>,
    pub relation: ThinkingRelation,
    pub route: Option<ObservationTransition>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThinkingStageOutput {
    pub stage_label: String,
    pub k: usize,
    pub results: Vec<ThinkingSelection>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThinkingPlan {
    pub stages: Vec<ThinkingStageOutput>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Copy)]
struct Route {
    path: PathId,
    edge: ObservationTransition,
    strength: f64,
}

struct Module {
    candidate: CandidateId,
    relevance: f64,
    routes: Vec<Route>,
}

fn route_key(
    edge: ObservationTransition,
) -> (
    usize,
    Option<nexus_core::NodeId>,
    nexus_core::NodeId,
    nexus_core::NodeId,
) {
    (edge.hop, edge.previous, edge.source, edge.target)
}

fn follows(left: ObservationTransition, right: ObservationTransition) -> bool {
    !right.immediate_return
        && left.hop.checked_add(1) == Some(right.hop)
        && left.target == right.source
        && right.previous == Some(left.source)
}

fn build_module(
    ctx: &ExecutionContext<'_>,
    input: ThinkingInput<'_>,
    config: &ThinkingConfig,
    candidate: CandidateId,
    maximum: f64,
    edges: &HashMap<nexus_core::NodeId, Vec<(usize, ObservationTransition, usize)>>,
) -> VcpResult<Module> {
    const STAGE: &str = "thinking.plan";
    let candidates = input.query.rows();
    let vector = candidates
        .candidate_vectors
        .row(candidate.index())
        .ok_or_else(|| invalid(STAGE, "stage candidate is outside the candidate set"))?;
    let relevance = input.query.original_cosines()[candidate.index()];
    let mut routes = Vec::new();
    let roots = candidates.candidate_roots(candidate);
    for index in candidates.candidate_paths(candidate) {
        ctx.check_cancelled(STAGE)?;
        let path = PathId::from_index(index, STAGE)?;
        let (nodes, positions) = candidates.path_steps(path);
        let mut stamps = HashMap::<_, Vec<_>>::new();
        for (&node, &position) in nodes.iter().zip(positions) {
            let tag = input
                .tag_vectors
                .row(node.index())
                .ok_or_else(|| invalid(STAGE, "path node is outside the graph"))?;
            let closure = cosine_f32(vector, tag, PrecisionMode::FastSimd);
            if closure >= config.min_closure {
                stamps.entry(node).or_default().push((position, closure));
            }
        }
        let mut matching: Vec<_> = stamps
            .keys()
            .filter_map(|node| edges.get(node))
            .flatten()
            .filter(|(_, edge, _)| stamps.contains_key(&edge.target))
            .copied()
            .collect();
        matching.sort_unstable_by_key(|&(order, _, _)| order);
        for (_, edge, index) in matching {
            let sources = &stamps[&edge.source];
            let targets = &stamps[&edge.target];
            let closure = sources
                .iter()
                .flat_map(|&(a, ac)| {
                    targets
                        .iter()
                        .filter(move |&&(b, _)| a < b)
                        .map(move |&(_, bc)| (ac * bc).sqrt())
                })
                .fold(0.0, f64::max);
            if closure > 0.0 {
                let independent = input.provenance.independent_fraction_excluding(
                    index,
                    roots,
                    &input.support_groups,
                );
                let strength = edge.flow / maximum * closure * independent;
                routes.push(Route {
                    path,
                    edge,
                    strength,
                });
            }
        }
    }
    Ok(Module {
        candidate,
        relevance,
        routes,
    })
}

pub fn plan_thinking(
    input: ThinkingInput<'_>,
    config: &ThinkingConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<ThinkingPlan> {
    let stage = "thinking.plan";
    ctx.check_cancelled(stage)?;
    let start = std::time::Instant::now();
    input.provenance.validate(input.graph.nnz(), stage)?;
    input.support_groups.validate(stage)?;
    input.tag_vectors.validate(stage)?;
    let rows = input.query.rows();
    if input.graph.node_count != input.tag_vectors.rows
        || input.tag_vectors.dim != rows.candidate_vectors.dim
        || ![
            config.min_closure,
            config.relevance_weight,
            config.route_weight,
            config.novelty_repeat_factor,
            config.root_no_parent_penalty,
        ]
        .into_iter()
        .all(|v| (0.0..=1.0).contains(&v))
        || input.stages.is_empty()
        || input.stages.len() > 32
        || input.stages.iter().any(|stage| {
            stage.k > 32 || stage.candidates.len() > 256 || stage.stage_label.trim().is_empty()
        })
    {
        return Err(invalid(stage, "invalid Thinking inputs or configuration"));
    }
    let maximum = input
        .trace
        .transitions
        .iter()
        .map(|edge| edge.flow)
        .fold(1e-12, f64::max);
    let mut edges = HashMap::<_, Vec<_>>::new();
    for (order, &edge) in input.trace.transitions.iter().enumerate() {
        if !edge.immediate_return
            && edge.flow > 0.0
            && edge.source != edge.target
            && let Some(index) = input.graph.find_edge(edge.source, edge.target)
            && input.graph.effective_weight(index) > 0.0
        {
            edges
                .entry(edge.source)
                .or_default()
                .push((order, edge, index));
        }
    }
    let mut chosen = Vec::<ThinkingSelection>::new();
    let mut used_documents = HashSet::new();
    let mut used_edges = HashSet::new();
    let mut stages = Vec::new();
    let mut modules = HashMap::new();
    for request in input.stages {
        let mut candidates = HashSet::new();
        for &candidate in request.candidates {
            if candidates.insert(candidate)
                && let std::collections::hash_map::Entry::Vacant(entry) = modules.entry(candidate)
            {
                entry.insert(build_module(
                    ctx, input, config, candidate, maximum, &edges,
                )?);
            }
        }

        let mut results = Vec::new();
        for _ in 0..request.k {
            ctx.check_cancelled(stage)?;
            let mut best = None::<ThinkingSelection>;
            for &candidate in &candidates {
                if used_documents.contains(&rows.document_ids[candidate.index()]) {
                    continue;
                }
                let module = &modules[&candidate];
                let mut parents = Vec::new();
                let mut best_route = None;
                let mut route_score = 0.0;
                for route in &module.routes {
                    let potential_parents = || {
                        chosen
                            .iter()
                            .filter(|prior| {
                                prior.route.is_some_and(|edge| follows(edge, route.edge))
                            })
                            .map(|prior| prior.candidate)
                    };
                    let has_parent = potential_parents().next().is_some();
                    if route.edge.hop != 1 && !has_parent {
                        continue;
                    }
                    let novelty = if used_edges.contains(&route_key(route.edge)) {
                        config.novelty_repeat_factor
                    } else {
                        1.0
                    };
                    let parent_gain = if !has_parent {
                        config.root_no_parent_penalty
                    } else {
                        1.0
                    };
                    let score = route.strength * novelty * parent_gain;
                    if score > route_score {
                        best_route = Some(*route);
                        route_score = score;
                        parents.clear();
                        parents.extend(potential_parents());
                    }
                }
                let relation = match best_route {
                    Some(route) if route.edge.wormhole => ThinkingRelation::Analogy,
                    Some(_) if !parents.is_empty() => ThinkingRelation::Continuation,
                    Some(_) => ThinkingRelation::Root,
                    None => ThinkingRelation::Supplement,
                };
                let offer = ThinkingSelection {
                    candidate: module.candidate,
                    chosen_path: best_route.map(|route| route.path),
                    score: config.relevance_weight * module.relevance
                        + config.route_weight * route_score,
                    parents,
                    relation,
                    route: best_route.map(|route| route.edge),
                };
                if best.as_ref().is_none_or(|prior| {
                    offer
                        .route
                        .is_some()
                        .cmp(&prior.route.is_some())
                        .then(offer.score.total_cmp(&prior.score))
                        .then(prior.candidate.cmp(&offer.candidate))
                        .is_gt()
                }) {
                    best = Some(offer);
                }
            }
            let Some(choice) = best else { break };
            used_documents.insert(rows.document_ids[choice.candidate.index()]);
            if let Some(edge) = choice.route {
                used_edges.insert(route_key(edge));
            }
            chosen.push(choice.clone());
            results.push(choice);
        }
        stages.push(ThinkingStageOutput {
            stage_label: request.stage_label.into(),
            k: request.k,
            results,
        });
    }
    Ok(ThinkingPlan {
        stages,
        elapsed_ms: start.elapsed().as_secs_f64() * 1_000.0,
    })
}
