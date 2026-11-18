// SPDX-License-Identifier: Apache-2.0

use crate::ExecutionEnv;
use nexus_core::{ExecutionContext, ScratchPad, VcpError, VcpResult};
use std::collections::VecDeque;
#[cfg(any(feature = "math-nodes", feature = "vcp-nodes"))]
use std::sync::Arc;

macro_rules! ports {
    ($($(#[$enabled:meta])* $variant:ident($data:ty),)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum PortKind {
            $($(#[$enabled])* $variant,)*
        }

        #[derive(Clone, Debug)]
        pub enum PortPayload {
            $($(#[$enabled])* $variant(Arc<$data>),)*
        }

        impl PortPayload {
            pub fn kind(&self) -> PortKind {
                match *self {
                    $($(#[$enabled])* Self::$variant(_) => PortKind::$variant,)*
                }
            }
        }
    }
}

ports! {
    #[cfg(feature = "vcp-nodes")] EpaAnalysis(nexus_vcp::EpaAnalysis),
    #[cfg(feature = "vcp-nodes")] PyramidResidual(nexus_vcp::PyramidResidual),
    #[cfg(feature = "vcp-nodes")] DisambiguatedTags(nexus_vcp::DisambiguatedTags),
    #[cfg(feature = "vcp-nodes")] GatedSeeds(nexus_vcp::GatedSeeds),
    #[cfg(feature = "vcp-nodes")] SenseOutput(nexus_vcp::SenseOutput),
    #[cfg(feature = "vcp-nodes")] FusedObservation(nexus_vcp::FusedObservation),
    #[cfg(feature = "vcp-nodes")] DualFields(nexus_vcp::DualFields),
    #[cfg(feature = "vcp-nodes")] BasinSupports(nexus_vcp::BasinSupports),
    #[cfg(feature = "vcp-nodes")] ObservationState(nexus_vcp::ObservationState),
    #[cfg(feature = "vcp-nodes")] PreparedCandidatePool(nexus_vcp::PreparedCandidatePool),
    #[cfg(feature = "vcp-nodes")] RiverReadoutOutput(nexus_vcp::RiverReadoutOutput),
    #[cfg(feature = "vcp-nodes")] FusedRanking(nexus_vcp::FusedRanking),
    #[cfg(feature = "vcp-nodes")] ThinkingPlan(nexus_vcp::ThinkingPlan),
    #[cfg(feature = "vcp-nodes")] DedupResult(nexus_vcp::DedupResult),
    #[cfg(feature = "vcp-nodes")] ContextAggregation(nexus_vcp::ContextAggregation),
    #[cfg(feature = "vcp-nodes")] ContextSegments(nexus_vcp::ContextSegments),
    #[cfg(feature = "vcp-nodes")] GeometryState(nexus_vcp::GeometryState),
    #[cfg(feature = "vcp-nodes")] EpaBasis(nexus_vcp::EpaBasis),
    #[cfg(feature = "vcp-nodes")] FactMatrixOutput(nexus_vcp::FactMatrixOutput),
    #[cfg(feature = "vcp-nodes")] TransportGraph(nexus_vcp::TransportGraph),
    #[cfg(feature = "vcp-nodes")] ProvenanceData(nexus_vcp::ProvenanceData),
    #[cfg(feature = "math-nodes")] DtscField(nexus_math::DtscField),
    #[cfg(feature = "math-nodes")] DtscSamples(Vec<nexus_math::DtscSample>),
    #[cfg(feature = "math-nodes")] DtscOutput(nexus_math::DtscOutput),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphNodeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortRef {
    pub node: GraphNodeId,
    pub port: u16,
}

impl GraphNodeId {
    pub const fn port(self, port: u16) -> PortRef {
        PortRef { node: self, port }
    }
}

pub trait ExecutableNode: Send + Sync {
    fn op_name(&self) -> &'static str;
    fn input_ports(&self) -> &'static [PortKind];
    fn output_ports(&self) -> &'static [PortKind];
    fn requires_full_transitions(&self, _port: u16) -> bool {
        false
    }
    fn provides_full_transitions(&self, _port: u16) -> bool {
        false
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>>;
}

#[derive(Clone, Copy, Debug)]
pub struct GraphEdge {
    pub from: PortRef,
    pub to: PortRef,
}

#[derive(Default)]
pub struct ComputeGraph {
    pub nodes: Vec<Box<dyn ExecutableNode>>,
    pub edges: Vec<GraphEdge>,
    pub outputs: Vec<PortRef>,
}

impl ComputeGraph {
    pub fn add_node(&mut self, node: impl ExecutableNode + 'static) -> GraphNodeId {
        let id = GraphNodeId(self.nodes.len());
        self.nodes.push(Box::new(node));
        id
    }

    pub fn connect(&mut self, from: PortRef, to: PortRef) {
        self.edges.push(GraphEdge { from, to });
    }
}

pub struct ExecutionPlan<'graph> {
    graph: &'graph ComputeGraph,
    order: Vec<usize>,
    input_slots: Vec<Vec<usize>>,
    output_slots: Vec<Vec<usize>>,
    consumer_counts: Vec<usize>,
    requested: Vec<usize>,
}

pub struct GraphOutputs {
    /// Payloads follow the graph's requested output order.
    pub slots: Vec<PortPayload>,
}

pub(crate) fn graph_error(detail: impl Into<String>) -> VcpError {
    VcpError::GraphValidation {
        detail: detail.into(),
    }
}

pub struct GraphValidator;

impl GraphValidator {
    pub fn validate(graph: &ComputeGraph) -> VcpResult<ExecutionPlan<'_>> {
        let length = graph.nodes.len();
        let mut output_slots = Vec::with_capacity(length);
        let mut slot_count = 0;
        for node in &graph.nodes {
            let end = slot_count + node.output_ports().len();
            output_slots.push((slot_count..end).collect::<Vec<_>>());
            slot_count = end;
        }
        let mut counts = vec![0; slot_count];
        let mut input_slots: Vec<Vec<Option<usize>>> = graph
            .nodes
            .iter()
            .map(|node| vec![None; node.input_ports().len()])
            .collect();
        let mut outgoing = vec![Vec::new(); length];
        let mut in_degree = vec![0_usize; length];
        let output = |reference: PortRef| -> VcpResult<(usize, PortKind)> {
            let index = usize::from(reference.port);
            let source = graph
                .nodes
                .get(reference.node.0)
                .ok_or_else(|| graph_error(format!("unknown output node {:?}", reference.node)))?;
            let kind = *source
                .output_ports()
                .get(index)
                .ok_or_else(|| graph_error(format!("unknown output port {reference:?}")))?;
            Ok((output_slots[reference.node.0][index], kind))
        };
        for edge in &graph.edges {
            let (slot, kind) = output(edge.from)?;
            let target = graph
                .nodes
                .get(edge.to.node.0)
                .ok_or_else(|| graph_error(format!("unknown input node {:?}", edge.to.node)))?;
            let index = usize::from(edge.to.port);
            if target.input_ports().get(index) != Some(&kind) {
                return Err(graph_error(format!(
                    "invalid or incompatible port {:?} for {kind:?}",
                    edge.to
                )));
            }
            if target.requires_full_transitions(edge.to.port)
                && !graph.nodes[edge.from.node.0].provides_full_transitions(edge.from.port)
            {
                return Err(graph_error(
                    "Thinking requires an observation producer with transition recording enabled",
                ));
            }
            if input_slots[edge.to.node.0][index].replace(slot).is_some() {
                return Err(graph_error(format!(
                    "port {:?} has multiple producers",
                    edge.to
                )));
            }
            counts[slot] += 1;
            in_degree[edge.to.node.0] += 1;
            outgoing[edge.from.node.0].push(edge.to.node.0);
        }

        let input_slots = input_slots
            .into_iter()
            .enumerate()
            .map(|(node, slots)| {
                slots
                    .into_iter()
                    .enumerate()
                    .map(|(port, slot)| {
                        slot.ok_or_else(|| {
                            graph_error(format!("node {node} input port {port} has no producer"))
                        })
                    })
                    .collect::<VcpResult<Vec<_>>>()
            })
            .collect::<VcpResult<Vec<_>>>()?;
        let mut requested = Vec::with_capacity(graph.outputs.len());
        for &reference in &graph.outputs {
            let (slot, _) = output(reference)?;
            requested.push(slot);
            counts[slot] += 1;
        }
        let mut ready: VecDeque<_> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(node, &degree)| (degree == 0).then_some(node))
            .collect();
        let mut order = Vec::with_capacity(length);
        while let Some(node) = ready.pop_front() {
            order.push(node);
            for &consumer in &outgoing[node] {
                in_degree[consumer] -= 1;
                if in_degree[consumer] == 0 {
                    ready.push_back(consumer);
                }
            }
        }
        if order.len() != length {
            return Err(graph_error("graph contains a dependency cycle"));
        }
        Ok(ExecutionPlan {
            graph,
            order,
            input_slots,
            output_slots,
            consumer_counts: counts,
            requested,
        })
    }
}

struct SlotStore {
    values: Vec<Option<PortPayload>>,
    consumers: Vec<usize>,
}

impl SlotStore {
    fn get(&self, slot: usize) -> VcpResult<&PortPayload> {
        self.values[slot]
            .as_ref()
            .ok_or_else(|| graph_error(format!("slot {slot} has no value")))
    }

    fn release(&mut self, slot: usize) {
        self.consumers[slot] -= 1;
        if self.consumers[slot] == 0 {
            self.values[slot] = None;
        }
    }
}

pub struct GraphExecutor;

impl GraphExecutor {
    /// Slots own shared results within one invocation. Fan-out borrows each
    /// result and drops it as soon as its final consumer completes.
    pub fn execute(
        plan: &ExecutionPlan<'_>,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        scratch: &mut ScratchPad,
    ) -> VcpResult<GraphOutputs> {
        ctx.check_cancelled("graph.execute")?;
        let mut slots = SlotStore {
            values: vec![None; plan.consumer_counts.len()],
            consumers: plan.consumer_counts.clone(),
        };
        let mut inputs = Vec::new();
        for &index in &plan.order {
            let node = &plan.graph.nodes[index];
            ctx.check_cancelled(node.op_name())?;
            // Borrow for this node only, before the slot store is updated.
            inputs.clear();
            inputs.extend(plan.input_slots[index].iter().copied());
            let outputs = {
                let values = inputs
                    .iter()
                    .map(|&slot| slots.get(slot))
                    .collect::<VcpResult<Vec<_>>>()?;
                node.run(ctx, env, &values, scratch)?
            };
            let kinds = node.output_ports();
            if kinds.len() != outputs.len()
                || kinds
                    .iter()
                    .zip(&outputs)
                    .any(|(kind, value)| *kind != value.kind())
            {
                return Err(graph_error(format!(
                    "node {} returned incompatible port values",
                    node.op_name()
                )));
            }
            for &slot in &inputs {
                slots.release(slot);
            }
            for (&slot, output) in plan.output_slots[index].iter().zip(outputs) {
                if slots.consumers[slot] > 0 {
                    slots.values[slot] = Some(output);
                }
            }
        }
        let outputs = plan
            .requested
            .iter()
            .map(|&slot| {
                let value = if slots.consumers[slot] == 1 {
                    slots.values[slot].take()
                } else {
                    slots.values[slot].clone()
                };
                slots.release(slot);
                value.ok_or_else(|| graph_error(format!("slot {slot} has no value")))
            })
            .collect::<VcpResult<_>>()?;
        Ok(GraphOutputs { slots: outputs })
    }
}
