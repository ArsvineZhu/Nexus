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
        let result = aggregate(
            ContextAggregateInput {
                raw_query: None,
                fragments,
                fragment_vectors: VectorMatrixView::new(&values, count, 1).unwrap(),
            },
            &ContextConfig::default(),
            &ctx,
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
    let segmented = segment(
        ContextSegmentInput { vectors: matrix },
        &ContextSegmentConfig::default(),
        &ctx,
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
        let bounded = cosine_dedup(input, &config, &ctx).unwrap();
        let exact = cosine_dedup(
            input,
            &CosineDedupConfig {
                bound_slow_budget: 0,
                ..config
            },
            &ctx,
        )
        .unwrap();
        assert_eq!(bounded, exact);
    }
}

#[test]
fn cross_file_evidence_requires_independent_witnesses_and_rrf_uses_rank_scores() {
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
        candidates: Some(&recall),
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
    let observation = observe(observation_input, &ObservationConfig::default(), &ctx).unwrap();
    let fields = compute_fields(
        FieldInput {
            graph,
            tag_vectors,
            observation: &observation,
        },
        &FieldConfig::default(),
        &ctx,
    )
    .unwrap();
    let candidates = CandidateRowsView {
        candidate_count: 2,
        document_ids: &[0, 1],
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
    let river_input = RiverAnalysisInput {
        observation: &observation,
        query: &query,
        fields: &fields,
        root_facts: None,
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
    let analysis = RiverAnalysisConfig {
        evidence: EvidencePolicy::DiscountWhenUnattributed,
        ..Default::default()
    };
    let scoring = RiverScoringConfig {
        unified_budget_ct: VCP_TOPOLOGY_BUDGET,
        ..Default::default()
    };
    let river = prepare_river_evidence(river_input, &analysis, &ctx)
        .unwrap()
        .readout(&scoring, &ctx)
        .unwrap();
    let input = WormholeRouteInput {
        observation: &observation,
        query: &query,
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
    let evidence = prepare_river_evidence(
        RiverAnalysisInput {
            root_facts: Some(input.root_facts),
            ..river_input
        },
        &analysis,
        &ctx,
    )
    .unwrap();
    let augmented = evidence.readout(&scoring, &ctx).unwrap();
    for item in &augmented.items {
        let baseline = river
            .items
            .iter()
            .find(|row| row.candidate == item.candidate)
            .unwrap();
        if item.candidate == CandidateId(0) {
            assert_eq!(
                baseline.scores.graph_bonus.to_bits(),
                item.scores.graph_bonus.to_bits()
            );
        } else {
            assert!(item.scores.graph_bonus > baseline.scores.graph_bonus);
        }
        assert!(item.scores.path_bonus + item.scores.graph_bonus <= scoring.unified_budget_ct);
        assert_eq!(
            item.scores.score,
            (item.scores.base_score + item.scores.path_bonus + item.scores.graph_bonus)
                .clamp(0.0, 1.0)
        );
    }
    let zero = evidence
        .readout(
            &RiverScoringConfig {
                unified_budget_ct: 0.0,
                ..scoring.clone()
            },
            &ctx,
        )
        .unwrap();
    for item in zero.items {
        assert_eq!(item.scores.path_bonus, 0.0);
        assert_eq!(item.scores.graph_bonus, 0.0);
        assert_eq!(item.scores.score, item.scores.base_score.clamp(0.0, 1.0));
    }

    let reranked = wormhole_rrf(
        WormholeRrfInput {
            routes: input,
            baseline: &river,
        },
        &WormholeRrfFuseConfig::default(),
        &ctx,
    )
    .unwrap();
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
    let inactive = wormhole_rrf(
        WormholeRrfInput {
            routes: WormholeRouteInput {
                root_facts: RootFactView::empty(),
                ..input
            },
            baseline: &river,
        },
        &WormholeRrfFuseConfig::default(),
        &ctx,
    )
    .unwrap();
    for (rank, (item, prior)) in inactive.items.iter().zip(&river.items).enumerate() {
        assert_eq!(item.candidate, prior.candidate);
        assert_eq!(item.score, 1.0 / (61.0 + rank as f64));
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
    let baseline = prepare_river_evidence(
        RiverAnalysisInput {
            query: &two_paths,
            ..river_input
        },
        &analysis,
        &ctx,
    )
    .unwrap()
    .readout(&scoring, &ctx)
    .unwrap();
    let evidence = prepare_river_evidence(
        RiverAnalysisInput {
            query: &two_paths,
            root_facts: Some(input.root_facts),
            ..river_input
        },
        &analysis,
        &ctx,
    )
    .unwrap();
    let recomposed = evidence.readout(&scoring, &ctx).unwrap();
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
    let supported = &evidence.paths()[1][1];
    assert!(supported.cross_file_transport_factor > 0.0);
    assert_eq!(
        after.scores.direct_anchor_contacts,
        supported.anchor_contacts
    );
    assert_eq!(after.scores.base_score, supported.base_score);
    let zero_budget = evidence
        .readout(
            &RiverScoringConfig {
                unified_budget_ct: 0.0,
                ..scoring.clone()
            },
            &ctx,
        )
        .unwrap();
    let zero_winner = zero_budget
        .items
        .iter()
        .find(|item| item.candidate == CandidateId(1))
        .unwrap();
    assert_eq!(zero_winner.chosen_path, before.chosen_path);

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
    let observation = observe(
        ObservationInput {
            graph,
            ..observation_input
        },
        &ObservationConfig::default(),
        &ctx,
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
    let fields = compute_fields(
        FieldInput {
            graph,
            tag_vectors,
            observation: &observation,
        },
        &FieldConfig::default(),
        &ctx,
    )
    .unwrap();
    let river = prepare_river_evidence(
        RiverAnalysisInput {
            graph,
            observation: &observation,
            fields: &fields,
            provenance: ProvenanceView {
                edge_count: 2,
                edge_prov_offsets: &[0, 0, 0],
                ..river_input.provenance
            },
            ..river_input
        },
        &analysis,
        &ctx,
    )
    .unwrap()
    .readout(&scoring, &ctx)
    .unwrap();
    let reranked = wormhole_rrf(
        WormholeRrfInput {
            baseline: &river,
            routes: WormholeRouteInput {
                observation: &observation,
                graph,
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
        },
        &WormholeRrfFuseConfig::default(),
        &ctx,
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
    let observation = observe(
        ObservationInput {
            query: &[1.0, 0.0],
            query_text: "narrow concept",
            candidates: Some(&recall),
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
        &ObservationConfig::default(),
        &ctx,
    )
    .unwrap();
    let fields = compute_fields(
        FieldInput {
            graph,
            tag_vectors,
            observation: &observation,
        },
        &FieldConfig::default(),
        &ctx,
    )
    .unwrap();
    let candidates = CandidateRowsView {
        candidate_count: 2,
        document_ids: &[0, 1],
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
    let evidence = prepare_river_evidence(
        RiverAnalysisInput {
            observation: &observation,
            query: &query,
            fields: &fields,
            root_facts: None,
            graph,
            provenance: ProvenanceView::empty(),
            support_groups: SupportGroupView::empty(),
            tag_vectors,
        },
        &RiverAnalysisConfig::default(),
        &ctx,
    )
    .unwrap();
    let result = evidence
        .readout(&RiverScoringConfig::default(), &ctx)
        .unwrap();
    let item = &result.items[0];
    let paths = &evidence.paths()[0];
    assert!(
        paths[1].path_topology_factor + paths[1].graph_request_factor
            > paths[0].path_topology_factor + paths[0].graph_request_factor
    );
    assert!(paths[1].base_score < paths[0].base_score);
    assert_eq!(paths[0].base_score, paths[2].base_score);
    assert_eq!(item.candidate, CandidateId(0));
    assert_eq!(result.items[1].candidate, CandidateId(1));
    assert_eq!(item.chosen_path, Some(PathId(0)));
    assert_eq!(item.scores.base_score, paths[0].base_score);
    assert_eq!(item.scores.genus_gate, paths[0].genus_gate);
    assert_eq!(item.scores.direct_anchor_contacts, paths[0].anchor_contacts);
    assert_eq!(item.scores.score, result.items[1].scores.score);
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
    let observation = observe(
        ObservationInput {
            graph,
            tag_vectors,
            query: &[0.8, 0.6],
            candidates: Some(&recall),
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
        &ObservationConfig::default(),
        &ctx,
    )
    .unwrap();
    let fields = compute_fields(
        FieldInput {
            graph,
            tag_vectors,
            observation: &observation,
        },
        &FieldConfig::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(observation.disambiguated.pruned_hypernyms, [NodeId(0)]);
    assert_eq!(observation.gated_seeds.exact_focus, Some(NodeId(1)));
    let candidates = CandidateRowsView {
        candidate_count: 2,
        document_ids: &[0, 1],
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
    let result = prepare_river_evidence(
        RiverAnalysisInput {
            observation: &observation,
            fields: &fields,
            query: &query,
            root_facts: None,
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
        &RiverAnalysisConfig {
            evidence: EvidencePolicy::DiscountWhenUnattributed,
            ..Default::default()
        },
        &ctx,
    )
    .unwrap()
    .readout(&RiverScoringConfig::default(), &ctx)
    .unwrap();
    for item in result.items {
        assert_eq!(
            item.scores.genus_gate,
            f64::from(item.candidate == CandidateId(0))
        );
    }
}
