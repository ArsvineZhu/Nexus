// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::*;
use nexus_vcp::*;

#[test]
fn context_mass_is_invariant_under_twenty_disjoint_fragments() {
    let ctx = ExecutionContext::default();

    for count in [1, 20] {
        let keys: Vec<_> = (0..count).map(|row| row.to_string()).collect();
        let names: Vec<_> = keys.iter().map(String::as_str).collect();
        let ranges: Vec<_> = (0..count)
            .map(|row| Some((row as u64, row as u64 + 1)))
            .collect();
        let values = vec![1.0; count];
        let fragments = ContextFragmentsView {
            count,
            keys: &names,
            roles: &vec!["user"; count],
            stages: &vec!["retrieval"; count],
            parent_keys: &vec![Some("parent"); count],
            sequences: &vec![0; count],
            weights: &vec![1.0; count],
            relevances: &vec![0.8; count],
            validities: &vec![1.0; count],
            distances: &vec![0.0; count],
            half_lives: &vec![None; count],
            source_keys: &vec![Some("source"); count],
            source_ranges: &ranges,
        };
        let result = ContextAggregateOperator
            .execute(
                &ctx,
                &ContextConfig::default(),
                ContextAggregateInput {
                    raw_query: None,
                    fragments,
                    fragment_vectors: VectorMatrixView::new(&values, count, 1).unwrap(),
                },
            )
            .unwrap();
        let contribution = result
            .allocations
            .iter()
            .map(|row| row.contribution)
            .sum::<f64>();
        assert!((contribution - 0.08888888888888889).abs() < 1e-12);
        assert_eq!(result.effective_vector, [1.0]);
    }

    let matrix = VectorMatrixView::new(&[1.0, 0.0, 0.9, 0.1, 0.0, 1.0], 3, 2).unwrap();
    let segmented = ContextSegmentOperator
        .execute(
            &ctx,
            &ContextSegmentConfig::default(),
            ContextSegmentInput { vectors: matrix },
        )
        .unwrap();
    assert_eq!(
        segmented.ranges,
        [
            SegmentRange { start: 0, end: 2 },
            SegmentRange { start: 2, end: 3 }
        ]
    );
}

#[test]
fn suffix_pruning_matches_unbounded_greedy_dedup() {
    let dim = 600;
    let values: Vec<_> = (0..25)
        .flat_map(|row| (0..dim).map(move |coordinate| ((row * coordinate % 101) as f32).sin()))
        .collect();
    let ranks: Vec<_> = (0..25).rev().map(CandidateId).collect();
    let input = CosineDedupInput {
        ranked: &ranks,
        candidate_vectors: VectorMatrixView::new(&values, 25, dim).unwrap(),
    };
    let ctx = ExecutionContext::default();

    for threshold in [0.1, 0.5, 0.92, 1.0] {
        let config = CosineDedupConfig {
            threshold,
            ..Default::default()
        };
        let bounded = CosineDedupOperator.execute(&ctx, &config, input).unwrap();
        let exact = CosineDedupOperator
            .execute(
                &ctx,
                &CosineDedupConfig {
                    bound_slow_budget: 0,
                    ..config
                },
                input,
            )
            .unwrap();
        assert_eq!(bounded, exact);
    }
}

#[test]
fn cross_file_evidence_requires_independent_witnesses_and_rrf_keeps_quality() {
    let ctx = ExecutionContext::default();

    let tag_vectors = VectorMatrixView::new(&[1., 0., 1., 0., 0., 1., 0., 1.], 4, 2).unwrap();
    let graph = CsrGraphView::new(CsrParts {
        node_count: 4,
        row_offsets: &[0, 1, 1, 1, 1],
        targets: &[NodeId(2)],
        weights: &[0.8],
        is_static_wormhole: &[true],
        inbound: &[0., 0., 0.8, 0.],
        max_inbound: 0.8,
        anchor_gains: &[0.0; 4],
        target_sorted_indices: None,
        edge_gates: None,
    })
    .unwrap();
    let recall = |_, _: &[f32], _, out: &mut Vec<_>, _: Option<&EpaAnalysis>, _: &[NodeId]| {
        out.push((NodeId(0), 1.0));
        Ok(())
    };
    let observation = Observe
        .execute(
            &ctx,
            &ObservationConfig::default(),
            ObservationInput {
                graph,
                tag_vectors,
                query: &[1., 0.],
                candidates: &recall,
                query_text: "source",
                lexical_meta: TagLexicalMetaView {
                    names: TextRows::Borrowed(&["source", "focus", "target", "bridge"]),
                    node_count: 4,
                    alias_offsets: &[0; 5],
                    aliases: TextRows::Borrowed(&[]),
                },
                epa_basis: None,
                core_tags: &[NodeId(0), NodeId(1)],
                ghost_vectors: None,
                ghost_is_core: &[],
            },
        )
        .unwrap();
    let fields = ComputeFields
        .execute(
            &ctx,
            &FieldConfig::default(),
            FieldInput {
                graph,
                tag_vectors,
                observation: &observation,
            },
        )
        .unwrap();
    let candidates = CandidatePoolView {
        candidate_count: 2,
        baseline_scores: &[0.0; 2],
        union_scores: &[0.0; 2],
        union_ranks: &[1, 2],
        bm25_scores: &[0.0; 2],
        bm25_reference: 0.0,
        anchor_scores: &[0.0; 2],
        curve_document_ids: &[DocumentId(0), DocumentId(1)],
        candidate_vectors: VectorMatrixView::new(&[0.0, 1.0, 0.0, 1.0], 2, 2).unwrap(),
        candidate_root_offsets: &[0, 1, 1],
        candidate_roots: &[RootId(0)],
        candidate_path_offsets: &[0, 1, 2],
        path_tag_offsets: &[0, 1, 2],
        path_tags: &[NodeId(2), NodeId(2)],
        path_positions: &[0, 0],
    };
    let pool = PoolCalibrateOperator
        .execute(
            &ctx,
            &PoolCalibrateConfig::default(),
            PoolCalibrateInput {
                fields: &fields,
                candidates,
            },
        )
        .unwrap();
    let river = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                fields: &fields,
                evidence: EvidencePolicy::Nous,
                pool: &pool,
                candidates,
                graph,
                tag_vectors,
                provenance: ProvenanceView {
                    edge_count: 1,
                    edge_prov_offsets: &[0, 0],
                    prov_groups: &[],
                    prov_masses: &[],
                },
                support_groups: SupportGroupView::empty(),
            },
        )
        .unwrap();
    let input = CrossFileWormholeInput {
        observation: &observation,
        readout: &river,
        candidates,
        graph,
        tag_vectors,
        root_facts: RootFactView {
            root_count: 2,
            checksum_hashes: &[1, 2],
            root_tag_offsets: &[0, 4, 7],
            root_tags: &[
                NodeId(0),
                NodeId(1),
                NodeId(2),
                NodeId(3),
                NodeId(0),
                NodeId(2),
                NodeId(3),
            ],
            root_req_offsets: &[0, 1, 2],
            root_required_roots: &[RootId(0), RootId(1)],
        },
    };
    let augmented = CrossFileWormholeOperator
        .execute(&ctx, &CrossFileWormholeConfig::default(), input)
        .unwrap();
    for item in &augmented.items {
        let baseline = river
            .items
            .iter()
            .find(|row| row.candidate == item.candidate)
            .unwrap();
        if item.candidate == CandidateId(0) {
            assert_eq!(baseline.graph_bonus.to_bits(), item.graph_bonus.to_bits());
        } else {
            assert!(item.graph_bonus > baseline.graph_bonus);
        }
        assert!(item.path_bonus + item.graph_bonus <= 0.08);
    }
    let reranked = WormholeRrfFuseOperator
        .execute(&ctx, &WormholeRrfFuseConfig::default(), input)
        .unwrap();
    assert!(reranked.rrf_active);
    for item in &reranked.items {
        let index = river
            .items
            .iter()
            .position(|row| row.candidate == item.candidate)
            .unwrap();
        let prior = &river.items[index];
        let route = if item.candidate == CandidateId(1) {
            river.omega.omega / 61.0
        } else {
            0.0
        };
        assert!((item.ranking_score - (1.0 / (61.0 + index as f64) + route)).abs() <= 1e-15);
        assert_eq!(item.quality_score.to_bits(), prior.quality_score.to_bits());
    }
}

// The only path to node 2 is a verified ordinary hop from the exact query focus.
#[test]
fn single_parent_genus_rescue_requires_observed_exact_focus_contact() {
    let ctx = ExecutionContext::default();

    let tag_vectors = VectorMatrixView::new(&[1., 0., 0.8, 0.6, 0., 1., 0., 1.], 4, 2).unwrap();
    let graph = CsrGraphView::new(CsrParts {
        node_count: 4,
        row_offsets: &[0, 0, 1, 1, 1],
        targets: &[NodeId(2)],
        weights: &[0.8],
        is_static_wormhole: &[false],
        inbound: &[0., 0., 0.8, 0.],
        max_inbound: 0.8,
        anchor_gains: &[0.0; 4],
        target_sorted_indices: None,
        edge_gates: None,
    })
    .unwrap();
    let recall = |_, _: &[f32], _, out: &mut Vec<_>, _: Option<&EpaAnalysis>, _: &[NodeId]| {
        out.extend([(NodeId(1), 1.0), (NodeId(0), 0.8)]);
        Ok(())
    };
    let observation = Observe
        .execute(
            &ctx,
            &ObservationConfig::default(),
            ObservationInput {
                graph,
                tag_vectors,
                query: &[0.8, 0.6],
                candidates: &recall,
                query_text: "car engine",
                lexical_meta: TagLexicalMetaView {
                    names: TextRows::Borrowed(&["car", "car engine", "repair", "manual"]),
                    node_count: 4,
                    alias_offsets: &[0; 5],
                    aliases: TextRows::Borrowed(&[]),
                },
                epa_basis: None,
                core_tags: &[],
                ghost_vectors: None,
                ghost_is_core: &[],
            },
        )
        .unwrap();
    let fields = ComputeFields
        .execute(
            &ctx,
            &FieldConfig::default(),
            FieldInput {
                graph,
                tag_vectors,
                observation: &observation,
            },
        )
        .unwrap();
    assert_eq!(observation.disambiguated.pruned_hypernyms, [NodeId(0)]);
    assert_eq!(observation.gated_seeds.exact_focus, Some(NodeId(1)));
    let candidates = CandidatePoolView {
        candidate_count: 2,
        baseline_scores: &[0.0; 2],
        union_scores: &[0.0; 2],
        union_ranks: &[1, 2],
        bm25_scores: &[0.0; 2],
        bm25_reference: 0.0,
        anchor_scores: &[0.0; 2],
        curve_document_ids: &[DocumentId(0), DocumentId(1)],
        candidate_vectors: VectorMatrixView::new(&[0.0, 1.0, 0.0, 1.0], 2, 2).unwrap(),
        candidate_root_offsets: &[0; 3],
        candidate_roots: &[],
        candidate_path_offsets: &[0, 1, 2],
        path_tag_offsets: &[0, 1, 2],
        path_tags: &[NodeId(2), NodeId(3)],
        path_positions: &[0, 0],
    };
    let pool = PoolCalibrateOperator
        .execute(
            &ctx,
            &PoolCalibrateConfig::default(),
            PoolCalibrateInput {
                fields: &fields,
                candidates,
            },
        )
        .unwrap();
    let result = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                fields: &fields,
                evidence: EvidencePolicy::Nous,
                pool: &pool,
                candidates,
                graph,
                tag_vectors,
                provenance: ProvenanceView {
                    edge_count: 1,
                    edge_prov_offsets: &[0, 0],
                    prov_groups: &[],
                    prov_masses: &[],
                },
                support_groups: SupportGroupView::empty(),
            },
        )
        .unwrap();
    for item in result.items {
        assert_eq!(item.genus_gate, f64::from(item.candidate == CandidateId(0)));
    }
}
