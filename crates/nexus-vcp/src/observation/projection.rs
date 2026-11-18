// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{DualFields, SenseOutput, finite_vector, invalid};
use nexus_core::kernel::normalize_f64;
use nexus_core::{
    CsrGraphView, Diagnostics, ExecutionContext, NodeId, Operator, ScratchPad, VcpError, VcpResult,
    VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BasinProjectionConfig {
    pub max_field_projection_nodes: usize,
    pub reserved_outer_per_basin: usize,
    pub state_limit: usize,
    pub edge_limit: usize,
}

impl Default for BasinProjectionConfig {
    fn default() -> Self {
        Self {
            max_field_projection_nodes: 256,
            reserved_outer_per_basin: 2,
            state_limit: 8192,
            edge_limit: 65_536,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BasinProjectionInput<'asset, 'query> {
    pub graph: CsrGraphView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub sense: &'query SenseOutput,
    pub dual_fields: &'query DualFields,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Basin {
    pub root: NodeId,
    pub inner: Vec<NodeId>,
    pub importance: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BasinSupports {
    pub basins: Vec<Basin>,
    pub local_support_nodes: Vec<NodeId>,
    pub transfer_support_nodes: Vec<NodeId>,
    pub local_vector: Vec<f32>,
    pub transfer_vector: Vec<f32>,
    pub diagnostics: Diagnostics,
}

fn seed_basins(
    ctx: &ExecutionContext<'_>,
    graph: CsrGraphView<'_>,
    sense: &SenseOutput,
    stage: &'static str,
) -> VcpResult<Vec<Basin>> {
    let mut nodes = vec![None; graph.node_count];
    let mut roots = Vec::new();
    for node in &sense.nodes {
        if node.node.index() >= nodes.len() || !node.energy.is_finite() || node.energy < 0.0 {
            return Err(invalid(stage, "invalid observed node"));
        }
        nodes[node.node.index()] = Some(node.energy);
        if node.hop == 0 {
            roots.push(node.node);
        }
        if let Some(seed) = node.seed_node {
            roots.push(seed);
        }
    }
    roots.retain(|node| nodes.get(node.index()).is_some_and(Option::is_some));
    roots.sort_unstable();
    roots.dedup();
    let mut adjacency = vec![Vec::new(); graph.node_count];
    for edge in &sense.edges {
        if edge.source.index() >= nodes.len() || edge.target.index() >= nodes.len() {
            return Err(invalid(stage, "observed edge is outside the graph"));
        }
        if edge.flow > 0.0
            && nodes[edge.source.index()].is_some()
            && nodes[edge.target.index()].is_some()
            && graph.edge_weight(edge.source, edge.target) > 0.0
        {
            adjacency[edge.source.index()].push(edge.target);
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
        basins.push(Basin {
            root,
            inner: vec![root],
            importance: nodes[root.index()].unwrap(),
        });
    }
    while let Some((source, basin)) = queue.pop_front() {
        ctx.check_cancelled(stage)?;
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
        if owners[node.node.index()].is_none() {
            owners[node.node.index()] = Some(basins.len());
            basins.push(Basin {
                root: node.node,
                inner: vec![node.node],
                importance: node.energy,
            });
        }
    }
    for basin in &mut basins {
        basin.inner.sort_unstable();
    }
    basins.sort_unstable_by_key(|basin| basin.root);
    Ok(basins)
}

#[derive(Clone, Copy)]
struct Arrival {
    node: NodeId,
    distance: f64,
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
            .then(other.node.cmp(&self.node))
    }
}

struct Lane<'a> {
    field: usize,
    basin: &'a Basin,
    best: HashMap<NodeId, f64>,
    frontier: BinaryHeap<Arrival>,
    outer: usize,
    gain: f64,
}

impl<'a> Lane<'a> {
    fn new(field: usize, basin: &'a Basin) -> Self {
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

pub struct BasinProjectionOperator;

impl Operator for BasinProjectionOperator {
    type Config = BasinProjectionConfig;
    type Input<'asset, 'query> = BasinProjectionInput<'asset, 'query>;
    type Output = BasinSupports;

    fn name(&self) -> &'static str {
        "basin.project"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        scratch: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        let graph = input.graph;
        graph.validate(stage)?;
        input.tag_vectors.validate(stage)?;
        let fields = [
            input.dual_fields.local_field.as_slice(),
            input.dual_fields.transfer_field.as_slice(),
        ];
        let domains = [
            &input.dual_fields.local_domain,
            &input.dual_fields.transfer_domain,
        ];
        if input.tag_vectors.rows != graph.node_count
            || fields.iter().any(|field| {
                field.len() != graph.node_count || field.iter().any(|v| !v.is_finite() || *v < 0.0)
            })
            || domains
                .iter()
                .any(|domain| domain.iter().any(|node| node.index() >= graph.node_count))
        {
            return Err(invalid(
                stage,
                "fields, domains and tag matrix must match the graph",
            ));
        }
        let basins = seed_basins(ctx, graph, input.sense, stage)?;
        let mut core: Vec<_> = basins
            .iter()
            .flat_map(|basin| basin.inner.iter().copied())
            .collect();
        core.sort_unstable();
        let limit = if config.max_field_projection_nodes == 0 {
            graph.node_count
        } else {
            config.max_field_projection_nodes
        };
        if core.len() > limit {
            return Err(VcpError::ResourceLimitExceeded {
                stage,
                limit_name: "max_field_projection_nodes",
                actual: core.len(),
                limit,
            });
        }

        let mut ids = [core.clone(), core.clone()];
        let mut selected: [HashSet<_>; 2] = std::array::from_fn(|_| core.iter().copied().collect());
        let mut union = selected[0].clone();
        let target: [f64; 2] = std::array::from_fn(|field| {
            domains[field]
                .iter()
                .map(|node| fields[field][node.index()])
                .sum()
        });
        let mut mass: [f64; 2] =
            std::array::from_fn(|field| core.iter().map(|node| fields[field][node.index()]).sum());
        let total = fields.map(|field| field.iter().sum::<f64>());
        let cutoffs = fields.map(|field| field.iter().copied().fold(0.0, f64::max) * 1e-12);
        let mut lanes: Vec<_> = basins
            .iter()
            .flat_map(|basin| [Lane::new(0, basin), Lane::new(1, basin)])
            .collect();
        let mut discovered = core.len().saturating_mul(2);
        let mut settled = 0;
        let mut scanned = 0;
        let mut cursor = 0;
        let mut diagnostics = Diagnostics::default();
        let edge_limit = config.edge_limit.min(ctx.limits.pair_work_limit);

        while (0..2).any(|field| mass[field] < target[field]) {
            ctx.check_cancelled(stage)?;
            let active = |lane: &Lane<'_>| {
                mass[lane.field] < target[lane.field] && !lane.frontier.is_empty()
            };
            let reserved = (0..lanes.len())
                .map(|offset| (cursor + offset) % lanes.len())
                .find(|&index| {
                    active(&lanes[index]) && lanes[index].outer < config.reserved_outer_per_basin
                });
            let choice = reserved.or_else(|| {
                lanes
                    .iter()
                    .enumerate()
                    .filter(|(_, lane)| active(lane))
                    .max_by(|(ai, a), (bi, b)| {
                        a.priority(config.reserved_outer_per_basin)
                            .total_cmp(&b.priority(config.reserved_outer_per_basin))
                            .then(bi.cmp(ai))
                    })
                    .map(|(index, _)| index)
            });
            let Some(index) = choice else { break };
            if settled >= config.state_limit || scanned >= edge_limit {
                diagnostics.mark_truncated("basin projection work limit");
                break;
            }
            cursor = (index + 1) % lanes.len();
            let lane = &mut lanes[index];
            let field = lane.field;
            let mut outer = 0;
            let mut gain = 0.0;
            while outer < if reserved.is_some() { 1 } else { 8 }
                && settled < config.state_limit
                && scanned < edge_limit
            {
                ctx.check_cancelled(stage)?;
                let Some(arrival) = lane.frontier.pop() else {
                    break;
                };
                if arrival.distance > lane.best[&arrival.node] {
                    continue;
                }
                settled += 1;
                if !union.contains(&arrival.node) && union.len() == limit {
                    diagnostics.mark_truncated("basin projection node limit");
                    continue;
                }
                union.insert(arrival.node);
                if lane.basin.inner.binary_search(&arrival.node).is_err() {
                    outer += 1;
                    lane.outer += 1;
                }
                let row = arrival.node.index();
                if selected[field].insert(arrival.node) {
                    ids[field].push(arrival.node);
                    let amount = fields[field][row];
                    mass[field] += amount;
                    gain += amount / total[field].max(1e-30);
                }
                if mass[field] >= target[field] {
                    break;
                }
                for edge in graph.row_offsets[row]..graph.row_offsets[row + 1] {
                    if scanned == edge_limit {
                        diagnostics.mark_truncated("basin projection edge limit");
                        break;
                    }
                    scanned += 1;
                    let weight = graph.effective_weight(edge);
                    let node = graph.targets[edge];
                    let value = fields[field][node.index()];
                    if !(0.0..=1.0).contains(&weight)
                        || weight == 0.0
                        || value <= 0.0
                        || value < cutoffs[field]
                    {
                        continue;
                    }
                    let distance = arrival.distance - weight.ln();
                    match lane.best.get_mut(&node) {
                        Some(best) if distance < *best => *best = distance,
                        Some(_) => continue,
                        None if discovered >= config.state_limit => {
                            diagnostics.mark_truncated("basin projection state limit");
                            continue;
                        }
                        None => {
                            discovered += 1;
                            lane.best.insert(node, distance);
                        }
                    }
                    lane.frontier.push(Arrival { node, distance });
                }
            }
            if outer > 0 {
                lane.gain = 0.5 * lane.gain + 0.5 * gain / outer as f64;
            }
        }
        let mut project = |field: usize| -> VcpResult<Vec<f32>> {
            let dim = input.tag_vectors.dim;
            let vector = ScratchPad::prepare_f64(&mut scratch.f64_buf_a, dim, 0.0);
            for &node in &ids[field] {
                let values = input.tag_vectors.row(node.index()).unwrap();
                finite_vector(stage, values, dim)?;
                let hub = if graph.max_inbound > 0.0 && core.binary_search(&node).is_err() {
                    (graph.inbound[node.index()] / graph.max_inbound).sqrt()
                } else {
                    0.0
                };
                let weight = fields[field][node.index()] / (1.0 + hub);
                for (out, &value) in vector.iter_mut().zip(values) {
                    *out += f64::from(value) * weight;
                }
            }
            normalize_f64(vector, 1e-12, ctx.precision);
            Ok(vector.iter().map(|&v| v as f32).collect())
        };
        let local_vector = project(0)?;
        let transfer_vector = project(1)?;
        let [local_support_nodes, transfer_support_nodes] = ids;
        Ok(BasinSupports {
            basins,
            local_support_nodes,
            transfer_support_nodes,
            local_vector,
            transfer_vector,
            diagnostics,
        })
    }
}
