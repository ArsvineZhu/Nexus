// SPDX-License-Identifier: Apache-2.0

use crate::RunOptions;
use crate::{Error, NodeId};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct WeightedEdge {
    pub from: u64,
    pub to: u64,
    pub weight: f64,
    pub wormhole: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GraphInputOptions {
    pub run: RunOptions,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportGraph {
    graph: CsrStorage,
    nodes: Vec<u64>,
    slots: HashMap<u64, NodeId>,
}

impl TransportGraph {
    pub fn from_edges(
        nodes: &[u64],
        edges: &[WeightedEdge],
        options: &GraphInputOptions,
    ) -> crate::Result<Self> {
        let run = &options.run;
        run.check_cancelled()?;
        let mut slots = HashMap::with_capacity(nodes.len());
        for (index, &node) in nodes.iter().enumerate() {
            if index.is_multiple_of(256) {
                run.check_cancelled()?;
            }
            if slots
                .insert(node, NodeId::from_index(index, "nodes")?)
                .is_some()
            {
                return Err(Error::Input {
                    field: "nodes",
                    id: Some(node),
                    message: "duplicate node".into(),
                });
            }
        }
        let mut rows = vec![HashMap::<NodeId, (f64, bool)>::new(); nodes.len()];
        for (index, edge) in edges.iter().enumerate() {
            if index.is_multiple_of(256) {
                run.check_cancelled()?;
            }
            let endpoint = |node, field| {
                slots.get(&node).copied().ok_or_else(|| Error::Input {
                    field,
                    id: Some(node),
                    message: "unknown graph node".into(),
                })
            };
            let from = endpoint(edge.from, "edges.from")?;
            let to = endpoint(edge.to, "edges.to")?;
            if !edge.weight.is_finite() || edge.weight < 0.0 {
                return Err(Error::Input {
                    field: "edges.weight",
                    id: Some(edge.from),
                    message: "weight must be finite and nonnegative".into(),
                });
            }
            let entry = rows[from.index()].entry(to).or_insert((0.0, edge.wormhole));
            if entry.1 != edge.wormhole {
                return Err(Error::Input {
                    field: "edges.wormhole",
                    id: Some(edge.from),
                    message: format!("conflicting flags on edge to {}", edge.to),
                });
            }
            entry.0 += edge.weight;
        }

        let mut row_offsets = vec![0];
        let mut row_statistics = Vec::with_capacity(nodes.len());
        let mut targets: Vec<NodeId> = Vec::with_capacity(edges.len());
        let mut weights = Vec::with_capacity(edges.len());
        let mut is_static_wormhole = Vec::with_capacity(edges.len());
        let mut inbound = vec![0.0; nodes.len()];
        let mut target_sorted_indices: Vec<u32> = Vec::with_capacity(edges.len());
        for (index, row) in rows.into_iter().enumerate() {
            run.check_cancelled()?;
            let mut row: Vec<_> = row
                .into_iter()
                .filter(|(_, (weight, _))| *weight > 0.0)
                .collect();
            row.sort_unstable_by(|(a, (aw, _)), (b, (bw, _))| {
                bw.total_cmp(aw).then_with(|| a.cmp(b))
            });
            let row_mass: f64 = row.iter().map(|(_, (weight, _))| *weight).sum();
            if row_mass > 1.0 {
                return Err(Error::Input {
                    field: "edges.weight",
                    id: Some(nodes[index]),
                    message: format!("row mass {row_mass} exceeds 1"),
                });
            }

            let row_max = row.first().map_or(0.0, |(_, (weight, _))| *weight);
            row_statistics.push((row_mass, row_max));
            let row_start = targets.len();
            for (to, (weight, wormhole)) in row {
                targets.push(to);
                weights.push(weight);
                is_static_wormhole.push(wormhole);
                inbound[to.index()] += weight;
            }
            let target_start = target_sorted_indices.len();
            for slot in row_start..targets.len() {
                target_sorted_indices.push(
                    u32::try_from(slot)
                        .map_err(|_| Error::input("edges", "too many transport edges"))?,
                );
            }
            target_sorted_indices[target_start..]
                .sort_unstable_by_key(|&slot| targets[slot as usize]);
            row_offsets.push(targets.len());
        }
        let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);
        let graph = CsrStorage {
            row_offsets,
            row_statistics,
            targets,
            weights,
            is_static_wormhole,
            inbound,
            max_inbound,
            anchor_gains: vec![1.0; nodes.len()],
            target_sorted_indices,
        };
        run.check_cancelled()?;
        Ok(Self {
            graph,
            nodes: nodes.to_vec(),
            slots,
        })
    }

    pub fn nodes(&self) -> &[u64] {
        &self.nodes
    }

    #[doc(hidden)]
    pub fn view(&self) -> TransportGraphView<'_> {
        TransportGraphView {
            node_count: self.nodes.len(),
            row_offsets: &self.graph.row_offsets,
            row_statistics: &self.graph.row_statistics,
            targets: &self.graph.targets,
            weights: &self.graph.weights,
            is_static_wormhole: &self.graph.is_static_wormhole,
            inbound: &self.graph.inbound,
            max_inbound: self.graph.max_inbound,
            anchor_gains: &self.graph.anchor_gains,
            target_sorted_indices: &self.graph.target_sorted_indices,
        }
    }

    #[doc(hidden)]
    pub fn slot(&self, node: u64) -> Option<NodeId> {
        self.slots.get(&node).copied()
    }

    #[doc(hidden)]
    pub fn set_anchor_gains(&mut self, gains: &[f64]) -> crate::Result<()> {
        if gains.len() != self.nodes.len() {
            return Err(Error::dimension(
                "anchor_gains",
                self.nodes.len(),
                gains.len(),
            ));
        }
        if gains.iter().any(|gain| !gain.is_finite() || *gain < 0.0) {
            return Err(Error::input(
                "anchor_gains",
                "anchor gains must be finite and nonnegative",
            ));
        }
        self.graph.anchor_gains.copy_from_slice(gains);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CsrStorage {
    row_offsets: Vec<usize>,
    row_statistics: Vec<(f64, f64)>,
    targets: Vec<NodeId>,
    weights: Vec<f64>,
    is_static_wormhole: Vec<bool>,
    inbound: Vec<f64>,
    max_inbound: f64,
    anchor_gains: Vec<f64>,
    target_sorted_indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TransportGraphView<'a> {
    pub node_count: usize,
    pub row_offsets: &'a [usize],
    pub targets: &'a [NodeId],
    pub weights: &'a [f64],
    pub is_static_wormhole: &'a [bool],
    pub inbound: &'a [f64],
    pub max_inbound: f64,
    pub anchor_gains: &'a [f64],
    target_sorted_indices: &'a [u32],
    row_statistics: &'a [(f64, f64)],
}

impl TransportGraphView<'_> {
    pub fn nnz(&self) -> usize {
        self.targets.len()
    }

    pub fn row_range(&self, node: NodeId) -> std::ops::Range<usize> {
        if node.index() >= self.node_count {
            return 0..0;
        }
        self.row_offsets[node.index()]..self.row_offsets[node.index() + 1]
    }

    pub fn row_statistics(&self, node: NodeId) -> (f64, f64) {
        self.row_statistics
            .get(node.index())
            .copied()
            .unwrap_or_default()
    }

    pub fn find_edge(&self, from: NodeId, to: NodeId) -> Option<usize> {
        let row = &self.target_sorted_indices[self.row_range(from)];
        let index = row
            .binary_search_by_key(&to, |&index| self.targets[index as usize])
            .ok()?;
        Some(row[index] as usize)
    }

    pub fn edge_weight(&self, from: NodeId, to: NodeId) -> f64 {
        self.find_edge(from, to)
            .map_or(0.0, |index| self.weights[index])
    }

    pub fn effective_weight(&self, index: usize) -> f64 {
        self.weights[index]
    }

    pub fn is_wormhole_edge(&self, index: usize) -> bool {
        self.is_static_wormhole.get(index).copied().unwrap_or(false)
    }

    pub fn anchor_gain(&self, node: NodeId) -> f64 {
        self.anchor_gains.get(node.index()).copied().unwrap_or(1.0)
    }

    pub fn inbound_mass(&self, node: NodeId) -> f64 {
        self.inbound.get(node.index()).copied().unwrap_or(0.0)
    }

    pub fn attenuate(self, gates: &[f64], run: &RunOptions) -> crate::Result<EffectiveTransport> {
        run.check_cancelled()?;
        if gates.len() != self.nnz() {
            return Err(Error::dimension("edge_gates", self.nnz(), gates.len()));
        }
        let mut weights = Vec::with_capacity(self.nnz());
        let mut inbound = vec![0.0; self.node_count];
        let mut row_statistics = Vec::with_capacity(self.node_count);
        for slots in self.row_offsets.windows(2) {
            let mut mass = 0.0;
            let mut max = 0.0_f64;
            for index in slots[0]..slots[1] {
                if index.is_multiple_of(256) {
                    run.check_cancelled()?;
                }
                let gate = gates[index];
                if !(0.0..=1.0).contains(&gate) {
                    return Err(Error::input(
                        "edge_gates",
                        "edge gates must be finite and in [0, 1]",
                    ));
                }
                let weight = self.weights[index] * gate;
                weights.push(weight);
                inbound[self.targets[index].index()] += weight;
                mass += weight;
                max = max.max(weight);
            }
            row_statistics.push((mass, max));
        }
        let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);
        run.check_cancelled()?;
        Ok(EffectiveTransport {
            weights,
            inbound,
            max_inbound,
            row_statistics,
        })
    }
}

/// Request-local weights over a Model's immutable topology.
#[derive(Clone, Debug)]
pub struct EffectiveTransport {
    weights: Vec<f64>,
    inbound: Vec<f64>,
    max_inbound: f64,
    row_statistics: Vec<(f64, f64)>,
}

impl EffectiveTransport {
    pub fn view<'a>(&'a self, base: TransportGraphView<'a>) -> TransportGraphView<'a> {
        TransportGraphView {
            weights: &self.weights,
            inbound: &self.inbound,
            max_inbound: self.max_inbound,
            row_statistics: &self.row_statistics,
            ..base
        }
    }
}
