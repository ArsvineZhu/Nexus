// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{ConceptGraph, DualFields, FieldProjectionOptions, FieldSolution, SenseResult};
use crate::{
    ConceptId, ConceptView, Error, Result, RunOptions, config, input as bad_input, invalid,
};
use nexus_math::kernel::{PrecisionMode, normalize_f64};
use nexus_math::{NodeId, TransportGraph, UnitVector};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectionConfig {
    pub max_outer_nodes: Option<usize>,
    pub reserved_outer_per_basin: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            max_outer_nodes: Some(256),
            reserved_outer_per_basin: 2,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FieldProjectionInput<'a> {
    pub graph: &'a TransportGraph,
    pub concepts: ConceptView<'a>,
    pub sense: &'a SenseResult,
    pub fields: &'a DualFields,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Basin {
    pub root: ConceptId,
    pub inner: Vec<ConceptId>,
    pub importance: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectedField {
    pub support: Vec<ConceptId>,
    pub direction: Option<UnitVector>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldProjection {
    pub local: ProjectedField,
    pub transfer: ProjectedField,
    pub basins: Vec<Basin>,
}

struct InternalBasin {
    root: NodeId,
    inner: Vec<NodeId>,
    importance: f64,
}

fn seed_basins(
    run: &RunOptions,
    space: ConceptGraph<'_>,
    sense: &SenseResult,
) -> Result<Vec<InternalBasin>> {
    let graph = space.view;
    let mut nodes = vec![None; graph.node_count];
    let mut roots = Vec::new();
    for node in &sense.nodes {
        run.check_cancelled()?;
        let slot = space.slot(node.concept)?;
        if !node.energy.is_finite() || node.energy < 0.0 || nodes[slot.index()].is_some() {
            return Err(bad_input(
                "fields.projection.sense.nodes",
                Some(node.concept),
                "node energy must be unique, finite and non-negative",
            ));
        }
        nodes[slot.index()] = Some(node.energy);
        if node.hop == 0 {
            roots.push(slot);
        }
        if let Some(seed) = node.seed {
            roots.push(space.slot(seed)?);
        }
    }
    roots.retain(|node| nodes.get(node.index()).is_some_and(Option::is_some));
    roots.sort_unstable_by_key(|node| space.ids()[node.index()]);
    roots.dedup();
    let mut adjacency = vec![Vec::new(); graph.node_count];
    for edge in &sense.observed_edges {
        run.check_cancelled()?;
        let from = space.slot(edge.from)?;
        let to = space.slot(edge.to)?;
        if !edge.flow.is_finite() || edge.flow < 0.0 {
            return Err(bad_input(
                "fields.projection.sense.edges",
                Some(edge.from),
                "flow must be finite and non-negative",
            ));
        }
        if edge.flow > 0.0
            && nodes[from.index()].is_some()
            && nodes[to.index()].is_some()
            && graph.edge_weight(from, to) > 0.0
        {
            adjacency[from.index()].push(to);
        }
    }
    for row in &mut adjacency {
        row.sort_unstable();
        row.dedup();
    }
    let mut owners = vec![None; graph.node_count];
    let mut queue = VecDeque::new();
    let mut basins = Vec::new();
    for root in roots {
        owners[root.index()] = Some(basins.len());
        queue.push_back((root, basins.len()));
        basins.push(InternalBasin {
            root,
            inner: vec![root],
            importance: nodes[root.index()].unwrap(),
        });
    }
    while let Some((source, basin)) = queue.pop_front() {
        run.check_cancelled()?;
        for &target in &adjacency[source.index()] {
            if owners[target.index()].is_none() {
                owners[target.index()] = Some(basin);
                basins[basin].inner.push(target);
                basins[basin].importance += nodes[target.index()].unwrap();
                queue.push_back((target, basin));
            }
        }
    }
    for node in &sense.nodes {
        run.check_cancelled()?;
        let slot = space.slot(node.concept)?;
        if owners[slot.index()].is_none() {
            basins.push(InternalBasin {
                root: slot,
                inner: vec![slot],
                importance: node.energy,
            });
        }
    }
    for basin in &mut basins {
        basin.inner.sort_unstable();
    }
    if basins.iter().any(|basin| !basin.importance.is_finite()) {
        return Err(Error::numeric(
            "project_fields",
            "basin importance overflowed",
        ));
    }
    basins.sort_unstable_by_key(|basin| space.ids()[basin.root.index()]);
    Ok(basins)
}

#[derive(Clone, Copy)]
struct Arrival {
    node: NodeId,
    distance: f64,
    id: ConceptId,
}

impl PartialEq for Arrival {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Arrival {}
impl PartialOrd for Arrival {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Arrival {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then(other.id.cmp(&self.id))
    }
}

struct Lane<'a> {
    field: usize,
    basin: &'a InternalBasin,
    best: HashMap<NodeId, f64>,
    frontier: BinaryHeap<Arrival>,
    outer: usize,
    gain: f64,
}

impl<'a> Lane<'a> {
    fn new(field: usize, basin: &'a InternalBasin, ids: &[ConceptId]) -> Self {
        Self {
            field,
            basin,
            best: basin.inner.iter().map(|&node| (node, 0.0)).collect(),
            frontier: basin
                .inner
                .iter()
                .map(|&node| Arrival {
                    node,
                    distance: 0.0,
                    id: ids[node.index()],
                })
                .collect(),
            outer: 0,
            gain: 1.0,
        }
    }

    fn priority(&self, reserved: usize) -> f64 {
        self.basin.importance.sqrt() * self.gain.max(1e-12)
            / (1.0 + self.outer as f64 / reserved.max(1) as f64)
    }
}

pub fn project_fields(
    input: FieldProjectionInput<'_>,
    options: &FieldProjectionOptions,
) -> Result<FieldProjection> {
    options.run.check_cancelled()?;
    let space = ConceptGraph::new(input.graph, input.concepts, &options.run)?;
    project(
        space,
        input.sense,
        input.fields,
        &options.config,
        &options.run,
    )
}

struct FieldMass {
    values: Vec<f64>,
    target: f64,
    total: f64,
}

impl FieldMass {
    fn from_solution(
        field: &FieldSolution,
        space: ConceptGraph<'_>,
        run: &RunOptions,
    ) -> Result<Self> {
        const FIELD: &str = "fields.projection.values";
        if field.values.len() != space.ids().len() {
            return Err(invalid(FIELD, "must contain one mass for every concept"));
        }
        let mut masses = vec![None; field.values.len()];
        for &(concept, mass) in &field.values {
            run.check_cancelled()?;
            let slot = space.slot(concept)?;
            if !mass.is_finite() || mass < 0.0 || masses[slot.index()].is_some() {
                return Err(bad_input(
                    FIELD,
                    Some(concept),
                    "mass must be unique, finite and non-negative",
                ));
            }
            masses[slot.index()] = Some(mass);
        }
        let values: Vec<_> = masses.into_iter().flatten().collect();
        let total = values.iter().sum::<f64>();
        if !total.is_finite() {
            return Err(Error::numeric(
                "project_fields",
                "scalar field mass overflowed",
            ));
        }
        let mut seen = HashSet::new();
        let mut target = 0.0;
        for &concept in &field.domain {
            let slot = space.slot(concept)?;
            if !seen.insert(slot) {
                return Err(bad_input(
                    "fields.projection.domain",
                    Some(concept),
                    "repeated domain concept",
                ));
            }
            target += values[slot.index()];
        }
        Ok(Self {
            values,
            target,
            total,
        })
    }
}

pub(super) fn project(
    space: ConceptGraph<'_>,
    sense: &SenseResult,
    dual_fields: &DualFields,
    cfg: &ProjectionConfig,
    run: &RunOptions,
) -> Result<FieldProjection> {
    run.check_cancelled()?;
    if cfg.reserved_outer_per_basin == 0 {
        return Err(config(
            "fields.projection.reserved_outer_per_basin",
            "must be positive",
        ));
    }
    let graph = space.view;
    let ids = space.ids();
    let fields = [
        FieldMass::from_solution(&dual_fields.local, space, run)?,
        FieldMass::from_solution(&dual_fields.transfer, space, run)?,
    ];
    let basins = seed_basins(run, space, sense)?;
    let mut core: Vec<_> = basins
        .iter()
        .flat_map(|basin| basin.inner.iter().copied())
        .collect();
    core.sort_unstable();
    let mut supported = [core.clone(), core.clone()];
    let mut selected: [HashSet<_>; 2] = std::array::from_fn(|_| core.iter().copied().collect());
    let mut union = selected[0].clone();
    let mut outer_count = 0;
    let limit = cfg.max_outer_nodes.unwrap_or(graph.node_count);
    let mut mass: [f64; 2] = std::array::from_fn(|field| {
        core.iter()
            .map(|node| fields[field].values[node.index()])
            .sum()
    });
    let mut lanes: Vec<_> = if limit == 0 {
        Vec::new()
    } else {
        basins
            .len()
            .checked_mul(2)
            .ok_or_else(|| invalid("fields.projection", "projection lane count overflows"))?;
        basins
            .iter()
            .flat_map(|basin| [Lane::new(0, basin, ids), Lane::new(1, basin, ids)])
            .collect()
    };
    let mut cursor = 0;
    while (0..2).any(|field| mass[field] < fields[field].target) {
        run.check_cancelled()?;
        let active = |lane: &Lane<'_>| {
            mass[lane.field] < fields[lane.field].target && !lane.frontier.is_empty()
        };
        let reserved = (0..lanes.len())
            .map(|offset| (cursor + offset) % lanes.len())
            .find(|&lane| active(&lanes[lane]) && lanes[lane].outer < cfg.reserved_outer_per_basin);
        let choice = reserved.or_else(|| {
            lanes
                .iter()
                .enumerate()
                .filter(|(_, lane)| active(lane))
                .max_by(|(i, a), (j, b)| {
                    a.priority(cfg.reserved_outer_per_basin)
                        .total_cmp(&b.priority(cfg.reserved_outer_per_basin))
                        .then(j.cmp(i))
                })
                .map(|(index, _)| index)
        });
        let Some(index) = choice else {
            break;
        };
        cursor = (index + 1) % lanes.len();
        let lane = &mut lanes[index];
        let field = lane.field;
        while let Some(arrival) = lane.frontier.pop() {
            run.check_cancelled()?;
            if arrival.distance > lane.best[&arrival.node] {
                continue;
            }
            if !union.contains(&arrival.node) {
                if outer_count == limit {
                    continue;
                }
                union.insert(arrival.node);
                outer_count += 1;
            }
            let row = arrival.node.index();
            let mut gain = 0.0;
            if selected[field].insert(arrival.node) {
                supported[field].push(arrival.node);
                let amount = fields[field].values[row];
                mass[field] += amount;
                gain = amount / fields[field].total.max(1e-30);
            }
            if mass[field] < fields[field].target {
                let start = graph.row_offsets[row];
                for edge in start..graph.row_offsets[row + 1] {
                    if (edge - start).is_multiple_of(256) {
                        run.check_cancelled()?;
                    }
                    let weight = graph.effective_weight(edge);
                    let node = graph.targets[edge];
                    if weight <= 0.0 {
                        continue;
                    }
                    let distance = arrival.distance - weight.ln();
                    let best = lane.best.entry(node).or_insert(f64::INFINITY);
                    if distance < *best {
                        *best = distance;
                        lane.frontier.push(Arrival {
                            node,
                            distance,
                            id: ids[node.index()],
                        });
                    }
                }
            }
            if core.binary_search(&arrival.node).is_err() {
                lane.outer += 1;
                lane.gain = 0.5 * lane.gain + 0.5 * gain;
                break;
            }
        }
    }
    let direction = |field: usize| -> Result<ProjectedField> {
        let mut vector = vec![0.0; space.concepts.dimension()];
        let scale = supported[field]
            .iter()
            .map(|node| fields[field].values[node.index()])
            .fold(0.0, f64::max);
        for &node in &supported[field] {
            if scale == 0.0 {
                break;
            }
            run.check_cancelled()?;
            let values = space.row(node);
            let hub = if graph.max_inbound > 0.0 && core.binary_search(&node).is_err() {
                (graph.inbound[node.index()] / graph.max_inbound).sqrt()
            } else {
                0.0
            };
            let weight = (fields[field].values[node.index()] / scale) / (1.0 + hub);
            for (out, &value) in vector.iter_mut().zip(values) {
                *out += f64::from(value) * weight;
            }
        }
        let direction = if normalize_f64(&mut vector, 0.0, PrecisionMode::FastSimd) {
            let values: Vec<_> = vector.into_iter().map(|value| value as f32).collect();
            Some(nexus_math::normalize(&values)?)
        } else {
            None
        };
        Ok(ProjectedField {
            support: supported[field]
                .iter()
                .map(|node| ids[node.index()])
                .collect(),
            direction,
        })
    };

    Ok(FieldProjection {
        local: direction(0)?,
        transfer: direction(1)?,
        basins: basins
            .into_iter()
            .map(|basin| Basin {
                root: ids[basin.root.index()],
                inner: basin
                    .inner
                    .into_iter()
                    .map(|node| ids[node.index()])
                    .collect(),
                importance: basin.importance,
            })
            .collect(),
    })
}
