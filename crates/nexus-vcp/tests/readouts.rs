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
    let graph = TransportGraphView::new(
        CsrGraphView::new(CsrParts {
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
        .unwrap(),
    )
    .unwrap();
    let recall = |_, _: &[f32], _, out: &mut Vec<_>, _: Option<&EpaAnalysis>, _: &[NodeId]| {
        out.push(NodeId(0));
        Ok(())
    };
    let observation_input = ObservationInput {
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
    };
    let observation = Observe
        .execute(&ctx, &ObservationConfig::default(), observation_input)
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
    let candidates = CandidateRowsView {
        candidate_count: 2,
        document_ids: &[DocumentId(0), DocumentId(1)],
        candidate_vectors: VectorMatrixView::new(&[0.0, 1.0, 0.0, 1.0], 2, 2).unwrap(),
        candidate_root_offsets: &[0, 1, 1],
        candidate_roots: &[RootId(0)],
        candidate_path_offsets: &[0, 1, 2],
        path_tag_offsets: &[0, 1, 2],
        path_tags: &[NodeId(2), NodeId(2)],
        path_positions: &[0, 0],
    };
    let query = QueryCosines::prepare(
        &ctx,
        candidates,
        &observation.original_query,
        &observation.fused.enhanced_query,
    )
    .unwrap();
    let features = RiverFeatures::prepare(&ctx, &query, &fields).unwrap();
    let river_input = RiverReadoutInput {
        observation: &observation,
        evidence: EvidencePolicy::DiscountWhenUnattributed,
        features: &features,
        graph,
        tag_vectors,
        provenance: ProvenanceView {
            edge_count: 1,
            edge_prov_offsets: &[0, 0],
            prov_groups: &[],
            prov_masses: &[],
        },
        support_groups: SupportGroupView::empty(),
    };
    let river = RiverReadoutOperator
        .execute(&ctx, &RiverReadoutConfig::default(), river_input)
        .unwrap();
    let input = CrossFileWormholeInput {
        observation: &observation,
        readout: &river,
        features: &features,
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
        assert_eq!(
            item.score,
            (item.base_score + item.path_bonus + item.graph_bonus).clamp(0.0, 1.0)
        );
    }
    let repeated = CrossFileWormholeOperator
        .execute(
            &ctx,
            &CrossFileWormholeConfig::default(),
            CrossFileWormholeInput {
                readout: &augmented,
                ..input
            },
        )
        .unwrap();
    assert_eq!(repeated.items, augmented.items);

    let zero = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig {
                unified_budget_ct: 0.0,
                ..Default::default()
            },
            river_input,
        )
        .unwrap();
    let zero = CrossFileWormholeOperator
        .execute(
            &ctx,
            &CrossFileWormholeConfig::default(),
            CrossFileWormholeInput {
                readout: &zero,
                ..input
            },
        )
        .unwrap();
    for item in zero.items {
        assert_eq!(item.path_bonus, 0.0);
        assert_eq!(item.graph_bonus, 0.0);
        assert_eq!(item.score, item.base_score.clamp(0.0, 1.0));
    }

    let mut forged = river.clone();
    forged.path_evidence[0][0].path_bonus = f64::NAN;
    assert!(matches!(
        CrossFileWormholeOperator.execute(
            &ctx,
            &CrossFileWormholeConfig::default(),
            CrossFileWormholeInput {
                readout: &forged,
                ..input
            }
        ),
        Err(VcpError::InvalidInput { .. })
    ));

    let reranked = WormholeRrfFuseOperator
        .execute(&ctx, &WormholeRrfFuseConfig::default(), input)
        .unwrap();
    assert!(reranked.active);
    for item in &reranked.items {
        let index = river
            .items
            .iter()
            .position(|row| row.candidate == item.candidate)
            .unwrap();
        let route = if item.candidate == CandidateId(1) {
            river.omega.omega / 61.0
        } else {
            0.0
        };
        assert!((item.score - (1.0 / (61.0 + index as f64) + route)).abs() <= 1e-15);
    }
    let inactive = WormholeRrfFuseOperator
        .execute(
            &ctx,
            &WormholeRrfFuseConfig::default(),
            CrossFileWormholeInput {
                root_facts: RootFactView::empty(),
                ..input
            },
        )
        .unwrap();
    assert!(!inactive.active);
    for (item, prior) in inactive.items.iter().zip(&river.items) {
        assert_eq!(item.candidate, prior.candidate);
        assert_eq!(item.score, prior.score);
    }

    let two_paths = CandidateRowsView {
        candidate_vectors: VectorMatrixView::new(&[0.0, 1.0, 0.6, 0.8], 2, 2).unwrap(),
        candidate_path_offsets: &[0, 1, 3],
        path_tag_offsets: &[0, 1, 2, 3],
        path_tags: &[NodeId(2), NodeId(0), NodeId(2)],
        path_positions: &[0; 3],
        ..candidates
    };
    let two_paths = QueryCosines::prepare(
        &ctx,
        two_paths,
        &observation.original_query,
        &observation.fused.enhanced_query,
    )
    .unwrap();
    let two_paths = RiverFeatures::prepare(&ctx, &two_paths, &fields).unwrap();
    let baseline = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                features: &two_paths,
                ..river_input
            },
        )
        .unwrap();
    let recomposed = CrossFileWormholeOperator
        .execute(
            &ctx,
            &CrossFileWormholeConfig::default(),
            CrossFileWormholeInput {
                features: &two_paths,
                readout: &baseline,
                ..input
            },
        )
        .unwrap();
    let before = baseline
        .items
        .iter()
        .find(|item| item.candidate == CandidateId(1))
        .unwrap();
    let after = recomposed
        .items
        .iter()
        .find(|item| item.candidate == CandidateId(1))
        .unwrap();
    assert_eq!(before.chosen_path, Some(PathId(1)));
    assert_eq!(after.chosen_path, Some(PathId(2)));
    let supported = baseline.path_evidence[1][1];
    let remaining = TOPOLOGY_BUDGET - supported.path_bonus;
    let graph_request = supported.graph_bonus.min(remaining);
    let [x, y] = [0.6_f32, 0.8_f32].map(f64::from);
    let transport = baseline.omega.omega * y / x.hypot(y);
    let graph_bonus = (TOPOLOGY_BUDGET
        * (1.0 - (1.0 - graph_request / TOPOLOGY_BUDGET) * (1.0 - transport)))
        .min(remaining);
    assert!((after.score - (supported.base + supported.path_bonus + graph_bonus)).abs() < 1e-14);
    assert_eq!(after.direct_anchor_contacts, supported.anchor_contacts);

    let graph = TransportGraphView::new(
        CsrGraphView::new(CsrParts {
            row_offsets: &[0, 1, 2, 2, 2],
            targets: &[NodeId(2), NodeId(2)],
            weights: &[0.8, 0.8],
            is_static_wormhole: &[true; 2],
            inbound: &[0.0, 0.0, 1.6, 0.0],
            max_inbound: 1.6,
            ..*graph.as_csr()
        })
        .unwrap(),
    )
    .unwrap();
    let observation = Observe
        .execute(
            &ctx,
            &ObservationConfig::default(),
            ObservationInput {
                graph,
                ..observation_input
            },
        )
        .unwrap();
    let [first, second] = observation.sense.edges.as_slice() else {
        panic!(
            "expected two witnessed routes: {:?}",
            observation.sense.edges
        );
    };
    assert!(first.wormhole && second.wormhole);
    assert_ne!(first.source, second.source);
    assert!(first.flow.min(second.flow) >= 0.5 * first.flow.max(second.flow));
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
    let features = RiverFeatures::prepare(&ctx, &query, &fields).unwrap();
    let river = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                graph,
                observation: &observation,
                features: &features,
                provenance: ProvenanceView {
                    edge_count: 2,
                    edge_prov_offsets: &[0, 0, 0],
                    ..river_input.provenance
                },
                ..river_input
            },
        )
        .unwrap();
    let reranked = WormholeRrfFuseOperator
        .execute(
            &ctx,
            &WormholeRrfFuseConfig::default(),
            CrossFileWormholeInput {
                observation: &observation,
                features: &features,
                graph,
                readout: &river,
                root_facts: RootFactView {
                    root_tags: &[
                        NodeId(0),
                        NodeId(1),
                        NodeId(2),
                        NodeId(3),
                        NodeId(0),
                        NodeId(1),
                        NodeId(2),
                        NodeId(3),
                    ],
                    root_tag_offsets: &[0, 4, 8],
                    ..input.root_facts
                },
                ..input
            },
        )
        .unwrap();
    let baseline_rank = river
        .items
        .iter()
        .position(|item| item.candidate == CandidateId(1))
        .unwrap()
        + 1;
    let score = reranked
        .items
        .iter()
        .find(|item| item.candidate == CandidateId(1))
        .unwrap()
        .score;
    assert!(reranked.active);
    assert!((score - 1.0 / (60 + baseline_rank) as f64 - river.omega.omega / 61.0).abs() < 1e-15);
}

#[test]
fn river_selects_the_complete_path_score_before_topology() {
    let ctx = ExecutionContext::default();
    let graph = TransportGraphView::new(
        CsrGraphView::new(CsrParts {
            node_count: 2,
            row_offsets: &[0; 3],
            targets: &[],
            weights: &[],
            is_static_wormhole: &[],
            inbound: &[0.0; 2],
            max_inbound: 0.0,
            anchor_gains: &[0.0; 2],
            target_sorted_indices: None,
            edge_gates: None,
        })
        .unwrap(),
    )
    .unwrap();
    let tag_values = [1.0, 0.0, 0.66, (1.0 - 0.66_f64.powi(2)).sqrt() as f32];
    let tag_vectors = VectorMatrixView::new(&tag_values, 2, 2).unwrap();
    let recall = |layer, _: &[f32], _, out: &mut Vec<_>, _: Option<&EpaAnalysis>, _: &[NodeId]| {
        if layer == 0 {
            out.extend([NodeId(0), NodeId(1)]);
        }
        Ok(())
    };
    let observation = Observe
        .execute(
            &ctx,
            &ObservationConfig::default(),
            ObservationInput {
                query: &[1.0, 0.0],
                query_text: "narrow concept",
                candidates: &recall,
                graph,
                tag_vectors,
                lexical_meta: TagLexicalMetaView {
                    names: TextRows::Borrowed(&["narrow concept", "concept"]),
                    node_count: 2,
                    alias_offsets: &[0; 3],
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
    let candidates = CandidateRowsView {
        candidate_count: 2,
        document_ids: &[DocumentId(0), DocumentId(1)],
        candidate_vectors: VectorMatrixView::new(&[1.0, 0.0, 1.0, 0.0], 2, 2).unwrap(),
        candidate_root_offsets: &[0; 3],
        candidate_roots: &[],
        candidate_path_offsets: &[0, 3, 4],
        path_tag_offsets: &[0, 1, 2, 3, 4],
        path_tags: &[NodeId(1), NodeId(0), NodeId(1), NodeId(1)],
        path_positions: &[0; 4],
    };
    let query = QueryCosines::prepare(
        &ctx,
        candidates,
        &observation.original_query,
        &observation.fused.enhanced_query,
    )
    .unwrap();
    let features = RiverFeatures::prepare(&ctx, &query, &fields).unwrap();
    let result = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                observation: &observation,
                features: &features,
                evidence: EvidencePolicy::default(),
                graph,
                provenance: ProvenanceView::empty(),
                support_groups: SupportGroupView::empty(),
                tag_vectors,
            },
        )
        .unwrap();
    let item = &result.items[0];
    let paths = &result.path_evidence[0];
    let score = |evidence: &PathEvidence| {
        let remaining = TOPOLOGY_BUDGET - evidence.path_bonus;
        (evidence.base + evidence.path_bonus + evidence.graph_bonus.min(remaining)).clamp(0.0, 1.0)
    };
    assert!(paths[1].path_bonus + paths[1].graph_bonus > item.topology_bonus);
    assert!(score(&paths[1]) < score(&paths[0]));
    assert_eq!(score(&paths[0]), score(&paths[2]));
    assert_eq!(item.candidate, CandidateId(0));
    assert_eq!(result.items[1].candidate, CandidateId(1));
    assert_eq!(item.chosen_path, Some(PathId(0)));
    assert_eq!(item.base_score, paths[0].base);
    assert_eq!(item.genus_gate, paths[0].genus_gate);
    assert_eq!(item.direct_anchor_contacts, paths[0].anchor_contacts);
    assert_eq!(item.score, score(&paths[0]));
    assert_eq!(item.score, result.items[1].score);
}

// The only path to node 2 is a verified ordinary hop from the exact query focus.
#[test]
fn single_parent_genus_rescue_requires_observed_exact_focus_contact() {
    let ctx = ExecutionContext::default();

    let tag_vectors = VectorMatrixView::new(&[1., 0., 0.8, 0.6, 0., 1., 0., 1.], 4, 2).unwrap();
    let graph = TransportGraphView::new(
        CsrGraphView::new(CsrParts {
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
        .unwrap(),
    )
    .unwrap();
    let recall = |_, _: &[f32], _, out: &mut Vec<_>, _: Option<&EpaAnalysis>, _: &[NodeId]| {
        out.extend([NodeId(1), NodeId(0)]);
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
    let candidates = CandidateRowsView {
        candidate_count: 2,
        document_ids: &[DocumentId(0), DocumentId(1)],
        candidate_vectors: VectorMatrixView::new(&[0.0, 1.0, 0.0, 1.0], 2, 2).unwrap(),
        candidate_root_offsets: &[0; 3],
        candidate_roots: &[],
        candidate_path_offsets: &[0, 1, 2],
        path_tag_offsets: &[0, 1, 2],
        path_tags: &[NodeId(2), NodeId(3)],
        path_positions: &[0, 0],
    };
    let query = QueryCosines::prepare(
        &ctx,
        candidates,
        &observation.original_query,
        &observation.fused.enhanced_query,
    )
    .unwrap();
    let features = RiverFeatures::prepare(&ctx, &query, &fields).unwrap();
    let result = RiverReadoutOperator
        .execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                observation: &observation,
                evidence: EvidencePolicy::DiscountWhenUnattributed,
                features: &features,
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
