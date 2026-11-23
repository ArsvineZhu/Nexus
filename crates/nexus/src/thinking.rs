// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::{HashMap, HashSet};

use crate::{
    CandidateId, Evaluation, Result, RunOptions, SelectedPath, Transition, config, input, numeric,
    parameter,
};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Default)]
pub struct ThinkingOptions {
    pub config: ThinkingConfig,
    pub run: RunOptions,
}

pub struct ThinkingStage<'a> {
    pub label: &'a str,
    pub candidates: &'a [CandidateId],
    pub limit: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingResult {
    pub stages: Vec<ThinkingStageResult>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingStageResult {
    pub label: String,
    pub selections: Vec<ThinkingSelection>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingSelection {
    pub candidate: CandidateId,
    pub value: f64,
    pub selected_path: Option<SelectedPath>,
    pub parents: Vec<CandidateId>,
    pub relation: ThinkingRelation,
    pub route: Option<Transition>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingRelation {
    Analogy,
    Continuation,
    Root,
    Supplement,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Entity {
    Document(u64),
    Candidate(CandidateId),
}

#[derive(Clone, Copy)]
struct Route {
    path: usize,
    transition: usize,
    strength: f64,
}

struct Choice {
    slot: usize,
    entity: Entity,
    value: f64,
    route: Option<Route>,
    parents: Vec<CandidateId>,
}

type RouteKey = (usize, Option<u64>, u64, u64);

fn key(edge: &Transition) -> RouteKey {
    (edge.hop, edge.previous, edge.from, edge.to)
}

fn follows(left: &Transition, right: &Transition) -> bool {
    left.hop.checked_add(1) == Some(right.hop)
        && left.to == right.from
        && right.previous == Some(left.from)
}

fn routes(eval: &Evaluation<'_>, slot: usize, options: &ThinkingOptions) -> Result<Vec<Route>> {
    let observation = eval.observation;
    let concepts = observation.model.concepts();
    let graph = observation.graph();
    let graph_assets = observation.model.graph_assets();
    let info = eval.info(slot);
    let mut out = Vec::new();
    let peak = observation.peak_transition_flow;
    if peak <= 0.0 {
        return Ok(out);
    }
    for (path, bound) in info.paths.iter().enumerate() {
        options.run.check_cancelled()?;
        let mut steps = HashMap::new();
        for (position, &node) in bound.nodes.iter().enumerate() {
            let closure = eval.closure(slot, node);
            if closure > 0.0 && closure >= options.config.min_closure {
                steps.insert(concepts.ids()[node.index()], (node, position, closure));
            }
        }
        for &from in steps.keys() {
            for &transition in observation
                .transition_index
                .get(&from)
                .into_iter()
                .flatten()
            {
                options.run.check_cancelled()?;
                let edge = &observation.transitions()[transition];
                if edge.immediate_return || edge.flow <= 0.0 {
                    continue;
                }
                let (a, i, left) = steps[&from];
                let Some(&(b, j, right)) = steps.get(&edge.to) else {
                    continue;
                };
                if i >= j || graph.edge_weight(a, b) == 0.0 {
                    continue;
                }
                let support = graph_assets
                    .support
                    .get(&(a, b))
                    .map_or(0.15, |value| value.factor(info.roots.as_ref()));
                out.push(Route {
                    path,
                    transition,
                    strength: edge.flow / peak * (left * right).sqrt() * support,
                });
            }
        }
    }
    out.sort_unstable_by_key(|route| (route.path, route.transition));
    Ok(out)
}

impl Evaluation<'_> {
    pub fn thinking(
        &self,
        stages: &[ThinkingStage<'_>],
        options: &ThinkingOptions,
    ) -> Result<ThinkingResult> {
        plan_thinking(self, stages, options)
    }
}

/// Selects sequential routes through the observation's accepted transitions.
pub fn plan_thinking(
    eval: &Evaluation<'_>,
    stages: &[ThinkingStage<'_>],
    options: &ThinkingOptions,
) -> Result<ThinkingResult> {
    let cfg = &options.config;
    let run = &options.run;
    run.check_cancelled()?;
    for (field, value, min, max) in [
        ("min_closure", cfg.min_closure, -1.0, 1.0),
        ("relevance_weight", cfg.relevance_weight, 0.0, f64::INFINITY),
        ("route_weight", cfg.route_weight, 0.0, f64::INFINITY),
        ("novelty_repeat_factor", cfg.novelty_repeat_factor, 0.0, 1.0),
        (
            "root_no_parent_penalty",
            cfg.root_no_parent_penalty,
            0.0,
            1.0,
        ),
    ] {
        parameter("thinking", field, value, min, max)?;
    }
    if cfg.relevance_weight == 0.0 && cfg.route_weight == 0.0 {
        return Err(config(
            "thinking.weights",
            "at least one weight must be positive",
        ));
    }
    let transitions = eval.observation.transitions();
    let mut selected_entities = HashSet::new();
    let mut used_routes = HashSet::new();
    let mut selected_routes: Vec<(CandidateId, usize)> = Vec::new();
    let mut analyzed = HashMap::new();
    let mut results = Vec::with_capacity(stages.len());
    for stage in stages {
        run.check_cancelled()?;
        let mut seen = HashSet::new();
        let slots = stage
            .candidates
            .iter()
            .map(|&id| {
                if !seen.insert(id) {
                    return Err(input(
                        "thinking.stage",
                        Some(id),
                        "duplicate candidate in stage",
                    ));
                }
                eval.slot(id, "thinking.stage")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut selections = Vec::new();
        for _ in 0..stage.limit.min(slots.len()) {
            let mut best: Option<Choice> = None;
            for &slot in &slots {
                run.check_cancelled()?;
                let id = eval.candidates.ids()[slot];
                let entity = eval
                    .info(slot)
                    .document_id
                    .map_or(Entity::Candidate(id), Entity::Document);
                if selected_entities.contains(&entity) {
                    continue;
                }
                let available = match analyzed.entry(slot) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(routes(eval, slot, options)?)
                    }
                };
                let mut route_score = 0.0;
                let mut choice = None;
                let mut parents = Vec::new();
                for route in available {
                    let edge = &transitions[route.transition];
                    let parent_ids: Vec<_> = selected_routes
                        .iter()
                        .filter(|&&(_, index)| follows(&transitions[index], edge))
                        .map(|&(id, _)| id)
                        .collect();
                    if edge.hop != 1 && parent_ids.is_empty() {
                        continue;
                    }
                    let novelty = if used_routes.contains(&key(edge)) {
                        cfg.novelty_repeat_factor
                    } else {
                        1.0
                    };
                    let root = if parent_ids.is_empty() {
                        cfg.root_no_parent_penalty
                    } else {
                        1.0
                    };
                    let score = route.strength * novelty * root;
                    if score > route_score {
                        route_score = score;
                        choice = Some(*route);
                        parents = parent_ids;
                    }
                }
                let value =
                    cfg.relevance_weight * eval.lane(slot, 0) + cfg.route_weight * route_score;
                if !value.is_finite() {
                    return Err(numeric("thinking", "route score overflowed"));
                }
                if best.as_ref().is_none_or(|prior| {
                    choice
                        .is_some()
                        .cmp(&prior.route.is_some())
                        .then(value.total_cmp(&prior.value))
                        .then(eval.candidates.ids()[prior.slot].cmp(&id))
                        .is_gt()
                }) {
                    best = Some(Choice {
                        slot,
                        entity,
                        value,
                        route: choice,
                        parents,
                    });
                }
            }
            let Some(Choice {
                slot,
                entity,
                value,
                route,
                parents,
            }) = best
            else {
                break;
            };
            let candidate = eval.candidates.ids()[slot];
            selected_entities.insert(entity);
            let (selected_path, edge, relation) = if let Some(route) = route {
                let edge = &transitions[route.transition];
                used_routes.insert(key(edge));
                selected_routes.push((candidate, route.transition));
                let relation = if edge.wormhole {
                    ThinkingRelation::Analogy
                } else if parents.is_empty() {
                    ThinkingRelation::Root
                } else {
                    ThinkingRelation::Continuation
                };
                (
                    Some(eval.info(slot).paths[route.path].selected(&eval.observation.model)),
                    Some(*edge),
                    relation,
                )
            } else {
                (None, None, ThinkingRelation::Supplement)
            };
            selections.push(ThinkingSelection {
                candidate,
                value,
                selected_path,
                parents,
                relation,
                route: edge,
            });
        }
        results.push(ThinkingStageResult {
            label: stage.label.into(),
            selections,
        });
    }
    Ok(ThinkingResult { stages: results })
}
