// SPDX-License-Identifier: Apache-2.0

use super::required;
use crate::{ExecutableNode, ExecutionEnv, PortKind, PortPayload, dag::graph_error};
use nexus_core::{ExecutionContext, Operator, ScratchPad, VcpResult};
use nexus_math::*;
use std::sync::Arc;

pub struct DtscFieldBuildNode(pub DtscFieldConfig);

impl ExecutableNode for DtscFieldBuildNode {
    fn op_name(&self) -> &'static str {
        DtscFieldBuildOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::DtscField]
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        _: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let output = DtscFieldBuildOperator.execute(
            ctx,
            &self.0,
            DtscFieldBuildInput {
                nodes: env.dtsc_nodes,
                tag_vectors: env.assets.tag_vectors,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::DtscField(Arc::new(output))])
    }
}

pub struct DtscFieldSampleNode(pub DtscFieldConfig);

impl ExecutableNode for DtscFieldSampleNode {
    fn op_name(&self) -> &'static str {
        DtscFieldSampleOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::DtscField]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::DtscSamples]
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let [PortPayload::DtscField(field)] = inputs else {
            return Err(graph_error("DTSC sampling requires a field"));
        };
        let graph = required(
            env.assets.transport,
            "a transport graph with inbound masses",
        )?;
        let output = DtscFieldSampleOperator.execute(
            ctx,
            &self.0,
            DtscFieldSampleInput {
                field,
                query_ids: env.dtsc_query_ids,
                query_vectors: required(env.dtsc_query_vectors, "DTSC sample vectors")?,
                inbound: graph.inbound,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::DtscSamples(Arc::new(output))])
    }
}

pub struct DtscReadoutNode(pub DtscConfig);

impl ExecutableNode for DtscReadoutNode {
    fn op_name(&self) -> &'static str {
        DtscReadoutOperator.name()
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::DtscField]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::DtscOutput]
    }
    fn run(
        &self,
        ctx: &ExecutionContext<'_>,
        env: &ExecutionEnv<'_, '_>,
        inputs: &[&PortPayload],
        scratch: &mut ScratchPad,
    ) -> VcpResult<Vec<PortPayload>> {
        let [PortPayload::DtscField(field)] = inputs else {
            return Err(graph_error("DTSC readout requires a field"));
        };
        let output = DtscReadoutOperator.execute(
            ctx,
            &self.0,
            DtscReadoutInput {
                field,
                candidates: required(env.candidates, "candidate pool")?,
                graph: required(env.assets.transport, "a transport graph")?,
                tag_vectors: env.assets.tag_vectors,
                original_query: env.query_vec,
                enhanced_query: env.dtsc_enhanced_query.unwrap_or(env.query_vec),
                geometry: env.dtsc_geometry,
            },
            scratch,
        )?;
        Ok(vec![PortPayload::DtscOutput(Arc::new(output))])
    }
}

/// Consume the shared observation directly and pass its vectors by reference
/// to DTSC, which remains independently usable through its own nodes.
#[cfg(feature = "vcp-nodes")]
pub struct ObservedDtscReadoutNode(pub DtscConfig);

#[cfg(feature = "vcp-nodes")]
impl ExecutableNode for ObservedDtscReadoutNode {
    fn op_name(&self) -> &'static str {
        "observed.dtsc_readout"
    }
    fn input_ports(&self) -> &'static [PortKind] {
        &[PortKind::ObservationState, PortKind::PreparedCandidatePool]
    }
    fn output_ports(&self) -> &'static [PortKind] {
        &[PortKind::DtscOutput]
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
                "observed DTSC readout requires an observation and calibrated pool",
            ));
        };
        let nodes: Vec<_> = observation
            .sense
            .nodes
            .iter()
            .map(|node| DtscNodeInput {
                node: node.node,
                energy: node.energy,
                normalized_energy: node.normalized_energy,
                source_type: match node.source_kind {
                    nexus_vcp::SeedSourceKind::Seed => "seed",
                    nexus_vcp::SeedSourceKind::Core => "core",
                    nexus_vcp::SeedSourceKind::LexicalSeed => "lexical_seed",
                    nexus_vcp::SeedSourceKind::Emergent => "emergent",
                }
                .into(),
            })
            .collect();
        let field = DtscFieldBuildOperator.execute(
            ctx,
            &self.0.field,
            DtscFieldBuildInput {
                nodes: &nodes,
                tag_vectors: env.assets.tag_vectors,
            },
            scratch,
        )?;
        let epa = &observation.epa;
        let pyramid = &observation.pyramid;
        let input = DtscReadoutInput {
            field: &field,
            candidates: required(env.candidates, "candidate pool")?,
            graph: required(env.assets.transport, "a transport graph")?,
            tag_vectors: env.assets.tag_vectors,
            original_query: &observation.original_query,
            enhanced_query: &observation.fused.enhanced_query,
            geometry: DtscGeometryState {
                epa_logic_depth: epa.logic_depth,
                epa_entropy: epa.entropy,
                epa_resonance: epa.resonance,
                pyramid_coverage: pyramid.coverage,
                pyramid_novelty: pyramid.novelty,
                pyramid_depth: pyramid.levels.len() as f64,
            },
        };
        let output = if pool.precision == nexus_core::PrecisionMode::StrictLeftFold {
            DtscReadoutOperator.execute_with_query_cosines(
                ctx,
                &self.0,
                input,
                [&pool.query_cosines, &pool.denoised_cosines],
                scratch,
            )
        } else {
            DtscReadoutOperator.execute(ctx, &self.0, input, scratch)
        }?;
        Ok(vec![PortPayload::DtscOutput(Arc::new(output))])
    }
}
