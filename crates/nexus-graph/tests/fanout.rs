// SPDX-License-Identifier: Apache-2.0
#![cfg(all(feature = "vcp-nodes", feature = "math-nodes"))]

use nexus_core::*;
use nexus_graph::*;
use nexus_vcp::*;
use std::cell::Cell;

#[test]
fn one_observation_drives_river_dtsc_and_two_thinking_stages() {
    let mut graph = ComputeGraph::default();
    let observe = graph.add_node(ObserveNode(ObservationConfig {
        sense: SenseConfig {
            max_safe_hops: 2,
            max_transition_records: 32,
            minimum_injected_current: 0.001,
            dynamic_wormholes: false,
            ..Default::default()
        },
        ..Default::default()
    }));
    let prepare = graph.add_node(PoolCalibrateNode(PoolCalibrateConfig {
        precision: Some(PrecisionMode::StrictLeftFold),
        ..PoolCalibrateConfig::default()
    }));
    let river = graph.add_node(RiverReadoutNode(RiverReadoutConfig::default()));
    let dtsc = graph.add_node(ObservedDtscReadoutNode(nexus_math::DtscConfig::default()));
    let thinking = graph.add_node(ThinkingNode(ThinkingConfig::default()));
    for consumer in [prepare, river, dtsc, thinking] {
        graph.connect(observe.port(0), consumer.port(0));
    }
    graph.connect(prepare.port(0), river.port(1));
    graph.connect(prepare.port(0), dtsc.port(1));
    graph.connect(prepare.port(0), thinking.port(1));
    graph.outputs.extend([
        river.port(0),
        dtsc.port(0),
        thinking.port(0),
        observe.port(0),
    ]);
    let plan = GraphValidator::validate(&graph).unwrap();

    let tags = VectorMatrixView::new(&[1.0, 0.0, 0.8, 0.6, 0.6, 0.8], 3, 2).unwrap();
    let mut assets = AssetBundle::new(tags);
    assets.transport = Some(CsrGraphView {
        node_count: 3,
        row_offsets: &[0, 1, 2, 2],
        targets: &[NodeId(1), NodeId(2)],
        weights: &[0.8, 0.8],
        is_static_wormhole: &[false; 2],
        inbound: &[0.0, 0.8, 0.8],
        max_inbound: 0.8,
        anchor_gains: &[0.0; 3],
        target_sorted_indices: None,
        edge_gates: None,
    });
    assets.lexical_meta = Some(TagLexicalMetaView {
        node_count: 3,
        names: &["root", "bridge", "answer"],
        alias_offsets: &[0; 4],
        aliases: &[],
    });
    assets.provenance = Some(ProvenanceView {
        edge_count: 2,
        edge_prov_offsets: &[0; 3],
        prov_groups: &[],
        prov_masses: &[],
    });
    assets.support_groups = Some(SupportGroupView::empty());
    let queries = Cell::new(0);
    let recall = |_, _: &[f32], _, buffer: &mut Vec<_>| {
        queries.set(queries.get() + 1);
        buffer.push((NodeId(0), 1.0));
        Ok(())
    };
    let mut env = ExecutionEnv::new(assets, &[1.0, 0.0]);
    env.core_tags = &[NodeId(0)];
    env.pyramid_candidates = Some(&recall);
    env.thinking_stages = &[
        ThinkingStageInputView {
            stage_label: "start",
            k: 1,
            candidates: &[CandidateId(0)],
        },
        ThinkingStageInputView {
            stage_label: "continue",
            k: 1,
            candidates: &[CandidateId(1)],
        },
    ];
    env.candidates = Some(CandidatePoolView {
        candidate_count: 2,
        baseline_scores: &[0.9, 0.8],
        union_scores: &[0.9, 0.8],
        union_ranks: &[1, 2],
        bm25_scores: &[0.0, 0.0],
        bm25_reference: 1.0,
        anchor_scores: &[0.0; 2],
        curve_document_ids: &[DocumentId(0), DocumentId(1)],
        candidate_vectors: VectorMatrixView::new(&[0.9, 0.4, 0.7, 0.7], 2, 2).unwrap(),
        candidate_root_offsets: &[0; 3],
        candidate_roots: &[],
        candidate_path_offsets: &[0, 2, 3],
        path_tag_offsets: &[0, 2, 3, 5],
        path_tags: &[NodeId(0), NodeId(1), NodeId(2), NodeId(1), NodeId(2)],
        path_positions: &[0, 1, 0, 0, 1],
    });

    let ctx = ExecutionContext::default();
    let outputs = GraphExecutor::execute(&plan, &ctx, &env, &mut ScratchPad::new()).unwrap();
    assert_eq!(queries.get(), 1);
    let [
        PortPayload::RiverReadoutOutput(river),
        PortPayload::DtscOutput(dtsc),
        PortPayload::ThinkingPlan(thinking),
        PortPayload::ObservationState(state),
    ] = outputs.slots.as_slice()
    else {
        panic!("unexpected readouts")
    };
    assert_eq!(river.items.len(), 2);
    assert!(river.items.iter().all(|item| {
        item.anchor_bonus.to_bits() == 0
            && item.path_bonus + item.wormhole_bonus <= 0.08
            && item.quality_score.to_bits() == item.ranking_score.to_bits()
    }));
    assert_eq!(dtsc.results.len(), 2);
    assert_eq!(thinking.stages[0].results[0].chosen_path, Some(PathId(0)));
    assert_eq!(thinking.stages[1].results[0].parents, [CandidateId(0)]);
    assert_eq!(
        thinking.stages[1].results[0].relation,
        ThinkingRelation::Continuation
    );
    assert_eq!(state.sense.transitions.len(), 2);

    let fused = OmegaRrfFuseOperator
        .execute(
            &ctx,
            &OmegaRrfFuseConfig::default(),
            OmegaRrfFuseInput {
                baseline_readout: river,
                external_ranked: &[(CandidateId(1), 2.0), (CandidateId(0), 1.0)],
            },
            &mut ScratchPad::new(),
        )
        .unwrap();
    assert!(fused.rrf_active);
    for item in fused.items {
        let prior = river
            .items
            .iter()
            .find(|row| row.candidate == item.candidate)
            .unwrap();
        assert_eq!(item.quality_score.to_bits(), prior.quality_score.to_bits());
        assert_ne!(item.ranking_score.to_bits(), prior.ranking_score.to_bits());
    }
}

#[test]
fn thinking_requires_a_producer_that_records_transitions() {
    let mut graph = ComputeGraph::default();
    let observe = graph.add_node(ObserveNode(ObservationConfig::default()));
    let thinking = graph.add_node(ThinkingNode(ThinkingConfig::default()));
    let prepare = graph.add_node(PoolCalibrateNode(PoolCalibrateConfig::default()));
    graph.connect(observe.port(0), thinking.port(0));
    graph.connect(observe.port(0), prepare.port(0));
    graph.connect(prepare.port(0), thinking.port(1));
    assert!(matches!(
        GraphValidator::validate(&graph),
        Err(VcpError::GraphValidation { .. })
    ));
}
