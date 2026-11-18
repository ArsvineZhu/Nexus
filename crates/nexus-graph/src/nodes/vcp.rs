// SPDX-License-Identifier: Apache-2.0

use super::required;
use crate::{ExecutableNode, ExecutionEnv, PortKind, PortPayload, dag::graph_error};
use nexus_core::{ExecutionContext, Operator, ScratchPad, VcpResult};
use nexus_vcp::*;
use std::sync::Arc;

pub struct ObserveNode(pub ObservationConfig);

impl ExecutableNode for ObserveNode {
    fn op_name(&self) -> &'static str {
        "observe"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::ObservationState]
    }
    fn provides_full_transitions(&self, port: u16) -> bool {
        port == 0 && self.0.sense.max_transition_records > 0
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        _: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let output = observe(
            ctx,
            &self.0,
            ObservationInput {
                query: env.query_vec,
                query_text: env.query_text,
                candidates: required(env.pyramid_candidates, "a pyramid candidate provider")?,
                graph: required(env.assets.transport, "a transport graph")?,
                tag_vectors: env.assets.tag_vectors,
                lexical_meta: required(env.assets.lexical_meta, "tag lexical metadata")?,
                epa_basis: env.assets.epa_basis,
                core_tags: env.core_tags,
                ghost_vectors: env.ghost_vectors,
                ghost_is_core: env.ghost_is_core,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::ObservationState(Arc::new(output))])
    }
}

pub struct PoolCalibrateNode(pub PoolCalibrateConfig);

impl ExecutableNode for PoolCalibrateNode {
    fn op_name(&self) -> &'static str {
        PoolCalibrateOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::ObservationState]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::PreparedCandidatePool]
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let [PortPayload::ObservationState(observation)] = inputs else {
            return Err(graph_error("pool calibration requires one observation"));
        };
        let output = PoolCalibrateOperator.execute(
            ctx,
            &self.0,
            PoolCalibrateInput {
                observation,
                candidates: required(env.candidates, "candidate pool")?,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::PreparedCandidatePool(Arc::new(output))])
    }
}

pub struct RiverReadoutNode(pub RiverReadoutConfig);

impl ExecutableNode for RiverReadoutNode {
    fn op_name(&self) -> &'static str {
        RiverReadoutOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::ObservationState, PortKind::PreparedCandidatePool]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::RiverReadoutOutput]
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let [
            PortPayload::ObservationState(observation),
            PortPayload::PreparedCandidatePool(pool),
        ] = inputs
        else {
            return Err(graph_error(
                "River readout requires an observation and calibrated pool",
            ));
        };
        let output = RiverReadoutOperator.execute(
            ctx,
            &self.0,
            RiverReadoutInput {
                observation,
                pool,
                candidates: required(env.candidates, "candidate pool")?,
                graph: required(env.assets.transport, "a transport graph")?,
                provenance: required(env.assets.provenance, "edge provenance")?,
                support_groups: required(env.assets.support_groups, "support groups")?,
                tag_vectors: env.assets.tag_vectors,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::RiverReadoutOutput(Arc::new(output))])
    }
}

pub struct CrossFileWormholeNode(pub CrossFileWormholeConfig);

impl ExecutableNode for CrossFileWormholeNode {
    fn op_name(&self) -> &'static str {
        CrossFileWormholeOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::ObservationState, PortKind::RiverReadoutOutput]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::RiverReadoutOutput]
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let [
            PortPayload::ObservationState(observation),
            PortPayload::RiverReadoutOutput(readout),
        ] = inputs
        else {
            return Err(graph_error(
                "cross-file wormhole requires an observation and readout",
            ));
        };
        let output = CrossFileWormholeOperator.execute(
            ctx,
            &self.0,
            CrossFileWormholeInput {
                observation,
                readout,
                candidates: required(env.candidates, "candidate pool")?,
                graph: required(env.assets.transport, "a transport graph")?,
                root_facts: required(env.assets.root_facts, "independent root facts")?,
                tag_vectors: env.assets.tag_vectors,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::RiverReadoutOutput(Arc::new(output))])
    }
}

pub struct ThinkingNode(pub ThinkingConfig);

impl ExecutableNode for ThinkingNode {
    fn op_name(&self) -> &'static str {
        ThinkingPlanOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::ObservationState, PortKind::PreparedCandidatePool]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::ThinkingPlan]
    }
    fn requires_full_transitions(&self, port: u16) -> bool {
        port == 0
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let [
            PortPayload::ObservationState(observation),
            PortPayload::PreparedCandidatePool(pool),
        ] = inputs
        else {
            return Err(graph_error(
                "Thinking requires an observation and calibrated pool",
            ));
        };
        let output = ThinkingPlanOperator.execute(
            ctx,
            &self.0,
            ThinkingInput {
                observation,
                candidates: required(env.candidates, "candidate pool")?,
                query_cosines: Some(&pool.query_cosines),
                stages: env.thinking_stages,
                graph: required(env.assets.transport, "a transport graph")?,
                provenance: required(env.assets.provenance, "edge provenance")?,
                support_groups: required(env.assets.support_groups, "support groups")?,
                tag_vectors: env.assets.tag_vectors,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::ThinkingPlan(Arc::new(output))])
    }
}
