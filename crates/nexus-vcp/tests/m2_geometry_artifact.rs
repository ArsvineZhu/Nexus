// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::*;
use nexus_vcp::*;

fn make_seq_view<'a>(
    seq_offsets: &'a [u32],
    seq_documents: &'a [DocumentId],
    seq_source_roots: &'a [RootId],
    tag_ids: &'a [NodeId],
    positions: &'a [u32],
) -> SequenceSliceView<'a> {
    SequenceSliceView {
        sequence_count: seq_documents.len(),
        seq_offsets,
        seq_documents,
        seq_source_roots,
        tag_ids,
        positions,
    }
}

#[test]
fn provenance_contract_1_multi_root_exclusion_and_dedup_floor() {
    let ctx = ExecutionContext::default();

    // 3 roots contributing to directed edge (0 -> 1):
    // root 101: 3.0, root 202: 1.0, root 303: 2.0 (total = 6.0).
    let vecs = [1.0_f32, 0.0, 0.0, 1.0];
    let tag_vectors = VectorMatrixView::new(&vecs, 2, 2).unwrap();
    let seqs = SequenceSliceView::empty();
    let anchor_gains = [1.0, 1.0];

    let sources = [NodeId(0), NodeId(0), NodeId(0)];
    let targets = [NodeId(1), NodeId(1), NodeId(1)];
    let roots = [RootId(101), RootId(202), RootId(303)];
    let groups = [GroupId(101), GroupId(202), GroupId(303)];
    let masses = [3.0, 1.0, 2.0];
    let pos_rel = PositiveRelationView {
        sources: &sources,
        targets: &targets,
        roots: &roots,
        groups: &groups,
        masses: &masses,
    };

    let fm_out = FactMatrixOperator
        .execute(
            &ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: seqs,
                anchor_gains: &anchor_gains,
                positive_relations: Some(pos_rel),
                negative_relations: None,
            },
        )
        .unwrap();

    let transport = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &fm_out.fact_matrix,
                anchor_gains: &anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();

    let prov = ProvenanceBuildOperator
        .execute(
            &ctx,
            &ProvenanceConfig::default(),
            ProvenanceBuildInput {
                transport_row_offsets: transport.graph.view().row_offsets,
                transport_targets: transport.graph.view().targets,
                sequences: seqs,
                positive_relations: Some(pos_rel),
                support_groups: None,
            },
        )
        .unwrap();

    let csr = transport.graph.view().as_csr();
    let edge_idx = csr.find_edge(NodeId(0), NodeId(1)).unwrap();
    let prov_view = prov.as_view();
    let empty_sg = SupportGroupView::empty();

    let frac_ab =
        prov_view.independent_fraction_excluding(edge_idx, &[RootId(101), RootId(202)], &empty_sg);
    assert!((frac_ab - 2.0 / 6.0).abs() < 1e-15);

    // Duplicate root 101 must not be double-counted.
    let frac_dup = prov_view.independent_fraction_excluding(
        edge_idx,
        &[RootId(101), RootId(101), RootId(202)],
        &empty_sg,
    );
    assert!((frac_dup - 2.0 / 6.0).abs() < 1e-15);

    // All roots excluded -> attenuation floor 0.15.
    assert_eq!(
        prov_view.independent_fraction_excluding(
            edge_idx,
            &[RootId(101), RootId(202), RootId(303)],
            &empty_sg,
        ),
        0.15
    );

    // Excluding only root 101 leaves 3.0 / 6.0 = 0.5.
    assert_eq!(
        prov_view.independent_support_fraction(edge_idx, &[RootId(101)], &empty_sg),
        Some(0.5)
    );
}

#[test]
fn provenance_contract_2_joint_support_group_and_max_dedup() {
    let ctx = ExecutionContext::default();

    // Group 0: joint roots {101, 202} (Interpretation), mass 2.0 (with duplicate 1.0 that must dedup via max).
    // Group 1: root {303} (Source), mass 3.0. Total = 5.0.
    let vecs = [1.0_f32, 0.0, 0.0, 1.0, 0.7, 0.7];
    let tag_vectors = VectorMatrixView::new(&vecs, 3, 2).unwrap();
    let anchor_gains = [1.0, 1.0, 1.0];

    let group_root_offsets = [0_u32, 2, 3];
    let group_roots = [RootId(101), RootId(202), RootId(303)];
    let group_kinds = [SupportKind::Interpretation, SupportKind::Source];
    let sg = SupportGroupView {
        group_count: 2,
        group_root_offsets: &group_root_offsets,
        group_roots: &group_roots,
        group_kinds: &group_kinds,
    };

    let sources = [NodeId(0), NodeId(0), NodeId(0)];
    let targets = [NodeId(1), NodeId(1), NodeId(1)];
    let roots = [RootId(101), RootId(101), RootId(303)];
    let groups = [GroupId(0), GroupId(0), GroupId(1)];
    let masses = [1.0, 2.0, 3.0];
    let pos_rel = PositiveRelationView {
        sources: &sources,
        targets: &targets,
        roots: &roots,
        groups: &groups,
        masses: &masses,
    };

    let fm_out = FactMatrixOperator
        .execute(
            &ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: SequenceSliceView::empty(),
                anchor_gains: &anchor_gains,
                positive_relations: Some(pos_rel),
                negative_relations: None,
            },
        )
        .unwrap();
    assert_eq!(fm_out.fact_matrix.raw_weights, vec![5.0]);

    let transport = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &fm_out.fact_matrix,
                anchor_gains: &anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();

    let prov = ProvenanceBuildOperator
        .execute(
            &ctx,
            &ProvenanceConfig::default(),
            ProvenanceBuildInput {
                transport_row_offsets: transport.graph.view().row_offsets,
                transport_targets: transport.graph.view().targets,
                sequences: SequenceSliceView::empty(),
                positive_relations: Some(pos_rel),
                support_groups: Some(sg),
            },
        )
        .unwrap();

    let csr = transport.graph.view().as_csr();
    let edge_idx = csr.find_edge(NodeId(0), NodeId(1)).unwrap();
    let prov_view = prov.as_view();

    assert_eq!(
        prov_view.independent_support_fraction(edge_idx, &[RootId(101)], &sg),
        Some(0.6)
    );
    assert_eq!(
        prov_view.independent_support_fraction(edge_idx, &[RootId(101), RootId(303)], &sg),
        Some(0.0)
    );
    assert_eq!(
        prov_view.independent_fraction_excluding(edge_idx, &[RootId(101), RootId(303)], &sg),
        0.15
    );
    assert_eq!(csr.find_edge(NodeId(1), NodeId(2)), None);
    assert_eq!(
        prov_view.independent_support_fraction(usize::MAX, &[], &sg),
        None
    );
}

#[test]
fn provenance_contract_3_long_101_occurrence_sequence_no_truncation() {
    let ctx = ExecutionContext::default();

    let mut vecs = Vec::with_capacity(5 * 3);
    for id in 1..=5 {
        vecs.push(1.0_f32);
        vecs.push(id as f32 * 0.1);
        vecs.push(0.3_f32);
    }
    let tag_vectors = VectorMatrixView::new(&vecs, 5, 3).unwrap();

    let mut tag_ids = Vec::with_capacity(101);
    let mut positions = Vec::with_capacity(101);
    for i in 0..101_u32 {
        tag_ids.push(NodeId(i % 5));
        positions.push(i + 1);
    }
    let seq_offsets = [0_u32, 101];
    let seq_docs = [DocumentId(0)];
    let seq_roots = [RootId(500)];
    let seqs = make_seq_view(&seq_offsets, &seq_docs, &seq_roots, &tag_ids, &positions);

    let geom = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: seqs,
                explicit_edges: None,
            },
        )
        .unwrap();

    assert_eq!(geom.node_count, 5);
    assert_eq!(geom.computed_nodes.iter().filter(|&&c| c).count(), 5);
    assert!(geom.intrinsics.iter().all(|v| v.neighbor_count == 4));

    // Also verify FactMatrix + ProvenanceBuild process all 101 occurrences without truncation,
    // and pair_work_limit triggers explicit ResourceLimitExceeded when set below 101*100/2 = 5050.
    let mut limited_ctx = ExecutionContext::default();
    limited_ctx.limits.pair_work_limit = 100;
    let err = LocalConeAgsOperator
        .execute(
            &limited_ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: seqs,
                explicit_edges: None,
            },
        )
        .unwrap_err();
    assert!(matches!(err, VcpError::ResourceLimitExceeded { .. }));

    let mut unlimited_ctx = ExecutionContext::default();
    unlimited_ctx.limits.pair_work_limit = 0;
    let fm_out = FactMatrixOperator
        .execute(
            &unlimited_ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: seqs,
                anchor_gains: &geom.anchor_gains,
                positive_relations: None,
                negative_relations: None,
            },
        )
        .unwrap();
    assert_eq!(fm_out.fact_matrix.targets.len(), 20);
}

#[test]
fn counterevidence_zero_bit_isolation() {
    let ctx = ExecutionContext::default();

    let vecs = [
        1.0_f32, 0.0, 0.0, 0.0, // 0
        0.8, 0.6, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, 0.0, // 2
        0.0, 0.0, 1.0, 0.0, // 3
    ];
    let tag_vectors = VectorMatrixView::new(&vecs, 4, 4).unwrap();
    let seq_offsets = [0_u32, 4];
    let seq_docs = [DocumentId(0)];
    let seq_roots = [RootId(10)];
    let tag_ids = [NodeId(0), NodeId(1), NodeId(2), NodeId(3)];
    let positions = [1_u32, 2, 3, 4];
    let seqs = make_seq_view(&seq_offsets, &seq_docs, &seq_roots, &tag_ids, &positions);
    let anchor_gains = [1.2, 0.9, 1.5, 1.0];

    let neg_sources = [NodeId(0), NodeId(2)];
    let neg_targets = [NodeId(3), NodeId(1)];
    let neg_roots = [RootId(77), RootId(88)];
    let neg_kinds = [1_u16, 2];
    let neg_strengths = [0.95, 0.4];
    let neg_confidences = [0.9, 0.8];
    let neg_view = NegativeRelationView {
        sources: &neg_sources,
        targets: &neg_targets,
        roots: &neg_roots,
        relation_kinds: &neg_kinds,
        strengths: &neg_strengths,
        confidences: &neg_confidences,
    };

    let out_clean = FactMatrixOperator
        .execute(
            &ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: seqs,
                anchor_gains: &anchor_gains,
                positive_relations: None,
                negative_relations: None,
            },
        )
        .unwrap();
    assert_eq!(out_clean.counterevidence.sources.len(), 0);

    let out_with_neg = FactMatrixOperator
        .execute(
            &ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: seqs,
                anchor_gains: &anchor_gains,
                positive_relations: None,
                negative_relations: Some(neg_view),
            },
        )
        .unwrap();

    // 0-bit identical FactMatrix!
    assert_eq!(
        out_clean.fact_matrix.row_offsets,
        out_with_neg.fact_matrix.row_offsets
    );
    assert_eq!(
        out_clean.fact_matrix.targets,
        out_with_neg.fact_matrix.targets
    );
    for (a, b) in out_clean
        .fact_matrix
        .raw_weights
        .iter()
        .zip(&out_with_neg.fact_matrix.raw_weights)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // All negative relations preserved in CounterevidenceAdj.
    assert_eq!(out_with_neg.counterevidence.sources, neg_sources);
    assert_eq!(out_with_neg.counterevidence.targets, neg_targets);
    assert_eq!(out_with_neg.counterevidence.roots, neg_roots);
    assert_eq!(out_with_neg.counterevidence.relation_kinds, neg_kinds);
    assert_eq!(out_with_neg.counterevidence.strengths, neg_strengths);
    assert_eq!(out_with_neg.counterevidence.confidences, neg_confidences);
    out_with_neg
        .counterevidence
        .as_view()
        .validate(4, "test")
        .unwrap();

    let tr_clean = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &out_clean.fact_matrix,
                anchor_gains: &anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();
    let tr_neg = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &out_with_neg.fact_matrix,
                anchor_gains: &anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();

    assert_eq!(
        tr_clean.graph.view().row_offsets,
        tr_neg.graph.view().row_offsets
    );
    assert_eq!(tr_clean.graph.view().targets, tr_neg.graph.view().targets);
    assert_eq!(
        tr_clean.graph.view().is_static_wormhole,
        tr_neg.graph.view().is_static_wormhole
    );
    assert_eq!(
        tr_clean.graph.view().target_sorted_indices,
        tr_neg.graph.view().target_sorted_indices
    );
    for (a, b) in tr_clean
        .graph
        .view()
        .weights
        .iter()
        .zip(tr_neg.graph.view().weights)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in tr_clean
        .graph
        .view()
        .inbound
        .iter()
        .zip(tr_neg.graph.view().inbound)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(
        tr_clean.graph.view().max_inbound.to_bits(),
        tr_neg.graph.view().max_inbound.to_bits()
    );
}

#[test]
fn local_cone_ags_24_basis_and_global_epa_orthogonality_and_monotonic_energy() {
    let ctx = ExecutionContext::default();

    // 32 nodes in 32-dim space: node 0 is a dense target, nodes 1..=30 are tilted orthonormal vectors.
    let dim = 32;
    let node_count = 31;
    let mut data = vec![0.0_f32; node_count * dim];
    for cell in data.iter_mut().take(dim) {
        *cell = 1.0_f32 / (dim as f32).sqrt();
    }
    for n in 1..node_count {
        data[n * dim + (n - 1)] = 1.0_f32;
        data[n * dim + 31] = 0.25_f32;
    }
    let tag_vectors = VectorMatrixView::new(&data, node_count, dim).unwrap();

    // Star/clique edges so node 0 sees 30 neighbors and every node 1..=30 has >= 3 support.
    let mut explicit_edges = Vec::new();
    for i in 0..node_count {
        for j in 0..node_count {
            if i != j {
                explicit_edges.push((NodeId(i as u32), NodeId(j as u32), 2.0));
            }
        }
    }

    let geom = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: SequenceSliceView::empty(),
                explicit_edges: Some(&explicit_edges),
            },
        )
        .unwrap();

    assert_eq!(geom.intrinsics[0].basis_used, 24);
    assert!(geom.local_residuals[0] >= 0.0 && geom.local_residuals[0] <= 1.0);
    let expected_gain = (0.75 + 1.25 * geom.local_residuals[0]).clamp(0.5, 2.0);
    assert!((geom.anchor_gains[0] - expected_gain).abs() < 1e-12);

    // GlobalEpa with k_max = 24, epsilon = 0.0, knee_delta = 0.0 forces 24 orthonormal basis axes.
    let epa = GlobalEpaOperator
        .execute(
            &ctx,
            &GlobalEpaConfig {
                k_max: 24,
                k_sem: 24,
                min_gain: 0.001,
                epsilon: 0.0,
                knee_delta: 0.0,
                ..Default::default()
            },
            GlobalEpaInput {
                tag_vectors,
                geometry: &geom,
            },
        )
        .unwrap();

    assert_eq!(epa.basis_count, 24);
    epa.as_view().validate("test").unwrap();

    // Orthonormality check: |<b_i, b_j>| < 1e-10 for i != j, |<b_i, b_i> - 1| < 1e-10.
    let view = epa.as_view();
    for i in 0..epa.basis_count {
        let bi = view.basis_row(i).unwrap();
        let norm_sq: f64 = bi.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "axis {i} norm_sq = {norm_sq}"
        );
        for j in (i + 1)..epa.basis_count {
            let bj = view.basis_row(j).unwrap();
            let dot: f64 = bi.iter().zip(bj).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-10, "<b_{i}, b_{j}> = {dot}");
        }
    }

    // Monotonic captured energy coverage.
    assert_eq!(epa.coverage.len(), 24);
    for w in epa.coverage.windows(2) {
        assert!(w[1] >= w[0], "coverage not monotonic: {} -> {}", w[0], w[1]);
    }
    assert!(epa.coverage.last().copied().unwrap() <= 1.0);
}

#[test]
fn local_cone_ags_mgs2_fallback_and_zero_norm_insufficient_neighbors() {
    let ctx = ExecutionContext::default();

    // Node 0: target = normalize([0.1, 0.7, 0.7, 0.0])
    // Node 1: [1.0, 0.0, 0.0, 0.0]
    // Node 2: [1.0, 1e-3, 0.0, 0.0] (nearly collinear with node 1 -> triggers MGS2 fallback!)
    // Node 3: [1.0, 0.0, 1e-3, 0.0] (nearly collinear with node 1 -> triggers MGS2 fallback!)
    // Node 4: [0.0, 0.0, 0.0, 0.0] (zero norm -> safe fallback, anchor_gain == 1.0)
    // Node 5: [0.0, 0.0, 0.0, 1.0] (only 1 neighbor < min_neighbors=3 -> anchor_gain == 1.0)
    let vecs = [
        0.1_f32, 0.7, 0.7, 0.0, // 0
        1.0, 0.0, 0.0, 0.0, // 1
        1.0, 1e-3, 0.0, 0.0, // 2
        1.0, 0.0, 1e-3, 0.0, // 3
        0.0, 0.0, 0.0, 0.0, // 4
        0.0, 0.0, 0.0, 1.0, // 5
    ];
    let tag_vectors = VectorMatrixView::new(&vecs, 6, 4).unwrap();
    let explicit_edges = [
        (NodeId(0), NodeId(1), 5.0),
        (NodeId(0), NodeId(2), 4.0),
        (NodeId(0), NodeId(3), 3.0),
        (NodeId(0), NodeId(4), 10.0), // invalid vector neighbor should be filtered out
        (NodeId(5), NodeId(0), 2.0),
    ];

    let geom = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig {
                semantic_gate_enabled: false,
                ..Default::default()
            },
            LocalConeAgsInput {
                tag_vectors,
                sequences: SequenceSliceView::empty(),
                explicit_edges: Some(&explicit_edges),
            },
        )
        .unwrap();

    assert!(geom.intrinsics[0].mgs2_fallback);
    assert!(geom.intrinsics[0].basis_used >= 2);
    assert!(geom.computed_nodes[0]);

    // Zero-norm node 4: safe fallback without NNLS error, anchor_gain == 1.0.
    assert_eq!(geom.intrinsics[4].residual_ratio, 1.0);
    assert_eq!(geom.intrinsics[4].basis_used, 0);
    assert!(!geom.intrinsics[4].mgs2_fallback);
    assert_eq!(geom.anchor_gains[4], 1.0);
    assert!(!geom.computed_nodes[4]);

    // Insufficient-neighbors node 5 (< 3): safe fallback, anchor_gain == 1.0.
    assert_eq!(geom.intrinsics[5].residual_ratio, 1.0);
    assert_eq!(geom.intrinsics[5].basis_used, 0);
    assert_eq!(geom.intrinsics[5].neighbor_count, 1);
    assert_eq!(geom.anchor_gains[5], 1.0);
    assert!(!geom.computed_nodes[5]);
}

#[test]
fn local_cone_ags_5adb12b_explain_mode_conditional_gain_recovery() {
    let ctx = ExecutionContext::default();

    // Node 0 target: [0.005, 0.9999875, 0.0, 0.0]
    // Node 1: [1.0, 0.0, 0.0, 0.0] (initial dot = 0.005 < min_gain=0.015)
    // Node 2: [0.999, 0.04471018, 0.0, 0.0] (initial dot = 0.0497 > 0.015, selected first!)
    // Node 3: [0.0, 0.0, 1.0, 0.0] (orthogonal dummy to satisfy min_neighbors=3)
    // On round 1, Node 1 has signed.abs() = 0.005 < 0.015, so pre-filter skips it for round 1
    // WITHOUT permanently removing it from `active`. After Node 2 is selected, Node 1's orthogonal
    // residual aligns along -Y with |signed| > 0.015, so Node 1 is recovered in round 2!
    let vecs = [
        0.005_f32, 0.9999875, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, 0.0, // 1
        0.999, 0.04471018, 0.0, 0.0, // 2
        0.0, 0.0, 1.0, 0.0, // 3
    ];
    let tag_vectors = VectorMatrixView::new(&vecs, 4, 4).unwrap();
    let explicit_edges = [
        (NodeId(0), NodeId(1), 5.0),
        (NodeId(0), NodeId(2), 4.0),
        (NodeId(0), NodeId(3), 3.0),
    ];

    let geom = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig {
                semantic_gate_enabled: false,
                min_gain: 0.015,
                ..Default::default()
            },
            LocalConeAgsInput {
                tag_vectors,
                sequences: SequenceSliceView::empty(),
                explicit_edges: Some(&explicit_edges),
            },
        )
        .unwrap();

    // Both Node 2 and recovered Node 1 should be picked (2 basis vectors).
    assert_eq!(geom.intrinsics[0].basis_used, 2);
}

#[test]
fn global_epa_empty_and_zero_norm_safe_degradation() {
    let ctx = ExecutionContext::default();

    // Completely empty corpus.
    let empty_vecs = VectorMatrixView::<f32>::empty(8);
    let empty_geom = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors: empty_vecs,
                sequences: SequenceSliceView::empty(),
                explicit_edges: None,
            },
        )
        .unwrap();
    let epa_empty = GlobalEpaOperator
        .execute(
            &ctx,
            &GlobalEpaConfig::default(),
            GlobalEpaInput {
                tag_vectors: empty_vecs,
                geometry: &empty_geom,
            },
        )
        .unwrap();
    assert_eq!(epa_empty.dim, 8);
    assert_eq!(epa_empty.basis_count, 0);
    assert_eq!(epa_empty.mean, vec![0.0; 8]);
    assert!(epa_empty.basis.is_empty());
    assert!(epa_empty.coverage.is_empty());
    epa_empty.as_view().validate("test").unwrap();

    // All-zero-norm corpus.
    let zeros = [0.0_f32; 12];
    let zero_vecs = VectorMatrixView::new(&zeros, 3, 4).unwrap();
    let zero_geom = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors: zero_vecs,
                sequences: SequenceSliceView::empty(),
                explicit_edges: None,
            },
        )
        .unwrap();
    let epa_zero = GlobalEpaOperator
        .execute(
            &ctx,
            &GlobalEpaConfig::default(),
            GlobalEpaInput {
                tag_vectors: zero_vecs,
                geometry: &zero_geom,
            },
        )
        .unwrap();
    assert_eq!(epa_zero.basis_count, 0);
    assert_eq!(epa_zero.mean, vec![0.0; 4]);
}

#[test]
fn transport_calibrate_binary_search_edge_gates_mass_conservation_and_hub_sign() {
    let ctx = ExecutionContext::default();

    // Build a 4-node FactMatrix directly:
    // Node 0 -> 1 (w=0.5), 2 (w=4.0, wormhole), 3 (w=1.0)
    // Node 1 -> 2 (w=3.0, wormhole)
    // Node 2 -> 0 (w=2.0)
    // Node 3 -> 1 (w=0.8), 2 (w=2.5, wormhole)
    let fm = FactMatrix {
        node_count: 4,
        row_offsets: vec![0, 3, 4, 5, 7],
        targets: vec![
            NodeId(1),
            NodeId(2),
            NodeId(3),
            NodeId(2),
            NodeId(0),
            NodeId(1),
            NodeId(2),
        ],
        raw_weights: vec![0.5, 4.0, 1.0, 3.0, 2.0, 0.8, 2.5],
    };
    let anchor_gains = [1.0, 1.0, 1.5, 1.0];

    let transport = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &fm,
                anchor_gains: &anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();
    let view_sorted = transport.graph.view().as_csr();
    let view_linear = CsrGraphView::new(CsrParts {
        target_sorted_indices: None,
        ..*view_sorted
    })
    .unwrap();

    // O(log deg) binary-search parity against linear scan.
    for u in 0..4_u32 {
        for v in 0..4_u32 {
            let src = NodeId(u);
            let dst = NodeId(v);
            assert_eq!(
                view_sorted.find_edge(src, dst),
                view_linear.find_edge(src, dst)
            );
            assert_eq!(
                view_sorted.edge_weight(src, dst).to_bits(),
                view_linear.edge_weight(src, dst).to_bits()
            );
        }
    }

    // Row mass conservation (0.95 outbound total on every non-empty row).
    for u in 0..4_u32 {
        let range = view_sorted.row_range(NodeId(u));
        let row_sum: f64 = transport.graph.view().weights[range].iter().sum();
        assert!((row_sum - 0.95).abs() < 1e-13, "row {u} sum = {row_sum}");
    }

    // Hub exponent sign safety: -0.3 and +0.3 produce 0-bit identical hub penalty.
    let tr_pos_exp = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig {
                hub_penalty_exponent: 0.3,
                ..Default::default()
            },
            TransportCalibrateInput {
                fact_matrix: &fm,
                anchor_gains: &anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();
    for (a, b) in transport
        .graph
        .view()
        .weights
        .iter()
        .zip(tr_pos_exp.graph.view().weights)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Post-normalization edge_gates without redistribution:
    // Gate edge 0 (0 -> 1) by 0.25, leave all others at 1.0.
    let gates = [0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let tr_gated = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &fm,
                anchor_gains: &anchor_gains,
                edge_gates: Some(&gates),
            },
        )
        .unwrap();
    let view_gated = tr_gated.graph.view().as_csr();
    view_gated.validate("test_gated").unwrap();

    let w01_base = view_sorted.edge_weight(NodeId(0), NodeId(1));
    let w01_gated = view_gated.edge_weight(NodeId(0), NodeId(1));
    assert!((w01_gated - 0.25 * w01_base).abs() < 1e-15);

    // Other edges in row 0 must NOT receive redistributed mass (0-bit identical).
    assert_eq!(
        view_gated.edge_weight(NodeId(0), NodeId(2)).to_bits(),
        view_sorted.edge_weight(NodeId(0), NodeId(2)).to_bits()
    );
    assert_eq!(
        view_gated.edge_weight(NodeId(0), NodeId(3)).to_bits(),
        view_sorted.edge_weight(NodeId(0), NodeId(3)).to_bits()
    );
}

#[test]
fn unsorted_sequence_positions_match_sorted_bit_for_bit() {
    let ctx = ExecutionContext::default();

    let vecs = [
        1.0_f32, 0.1, 0.2, 0.0, // 0
        0.2, 1.0, 0.1, 0.0, // 1
        0.1, 0.2, 1.0, 0.0, // 2
        0.0, 0.3, 0.4, 1.0, // 3
    ];
    let tag_vectors = VectorMatrixView::new(&vecs, 4, 4).unwrap();
    let seq_offsets = [0_u32, 4];
    let seq_docs = [DocumentId(0)];
    let seq_roots = [RootId(9)];

    let sorted_tags = [NodeId(0), NodeId(1), NodeId(2), NodeId(3)];
    let sorted_pos = [1_u32, 2, 4, 7];

    let unsorted_tags = [NodeId(3), NodeId(1), NodeId(0), NodeId(2)];
    let unsorted_pos = [7_u32, 2, 1, 4];

    let g_sorted = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: make_seq_view(
                    &seq_offsets,
                    &seq_docs,
                    &seq_roots,
                    &sorted_tags,
                    &sorted_pos,
                ),
                explicit_edges: None,
            },
        )
        .unwrap();
    let g_unsorted = LocalConeAgsOperator
        .execute(
            &ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: make_seq_view(
                    &seq_offsets,
                    &seq_docs,
                    &seq_roots,
                    &unsorted_tags,
                    &unsorted_pos,
                ),
                explicit_edges: None,
            },
        )
        .unwrap();

    for (a, b) in g_sorted.anchor_gains.iter().zip(&g_unsorted.anchor_gains) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let fm_sorted = FactMatrixOperator
        .execute(
            &ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: make_seq_view(
                    &seq_offsets,
                    &seq_docs,
                    &seq_roots,
                    &sorted_tags,
                    &sorted_pos,
                ),
                anchor_gains: &g_sorted.anchor_gains,
                positive_relations: None,
                negative_relations: None,
            },
        )
        .unwrap();
    let fm_unsorted = FactMatrixOperator
        .execute(
            &ctx,
            &FactMatrixConfig::default(),
            FactMatrixInput {
                tag_vectors,
                sequences: make_seq_view(
                    &seq_offsets,
                    &seq_docs,
                    &seq_roots,
                    &unsorted_tags,
                    &unsorted_pos,
                ),
                anchor_gains: &g_unsorted.anchor_gains,
                positive_relations: None,
                negative_relations: None,
            },
        )
        .unwrap();

    assert_eq!(
        fm_sorted.fact_matrix.targets,
        fm_unsorted.fact_matrix.targets
    );
    for (a, b) in fm_sorted
        .fact_matrix
        .raw_weights
        .iter()
        .zip(&fm_unsorted.fact_matrix.raw_weights)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn config_serde_defaults_and_deny_unknown_fields() {
    use serde::Deserialize;
    use serde::de::value::{Error as DeError, F64Deserializer, MapDeserializer};

    let entries = [
        ("evidence_compression", 1.5_f64),
        ("hub_smoothing_ratio", 0.2),
    ];
    let de = MapDeserializer::<_, DeError>::new(
        entries
            .into_iter()
            .map(|(k, v)| (k, F64Deserializer::<DeError>::new(v))),
    );
    let t_cfg = TransportConfig::deserialize(de).unwrap();
    assert_eq!(t_cfg.evidence_compression, 1.5);
    assert_eq!(t_cfg.hub_smoothing_ratio, 0.2);
    assert_eq!(t_cfg.outbound_mass, 0.95);

    let bad_de = || {
        MapDeserializer::<_, DeError>::new(
            [("unknown_field", F64Deserializer::<DeError>::new(1.0))].into_iter(),
        )
    };
    assert!(TransportConfig::deserialize(bad_de()).is_err());
    assert!(LocalConeAgsConfig::deserialize(bad_de()).is_err());
    assert!(GlobalEpaConfig::deserialize(bad_de()).is_err());
    assert!(FactMatrixConfig::deserialize(bad_de()).is_err());
    assert!(ProvenanceConfig::deserialize(bad_de()).is_err());
}

#[test]
fn parallel_policy_matches_sequential_bit_for_bit_and_semantic_fact_matrix() {
    let seq_ctx = ExecutionContext::default();
    let par_ctx = ExecutionContext {
        parallel: ParallelPolicy::GlobalRayon,
        ..Default::default()
    };

    let vecs = [
        1.0_f32, 0.2, 0.1, 0.0, // 0
        0.9, 0.4, 0.1, 0.0, // 1
        0.2, 0.9, 0.3, 0.0, // 2
        0.1, 0.3, 0.9, 0.2, // 3
        0.0, 0.1, 0.4, 0.9, // 4
    ];
    let tag_vectors = VectorMatrixView::new(&vecs, 5, 4).unwrap();
    let seq_offsets = [0_u32, 5];
    let seq_docs = [DocumentId(0)];
    let seq_roots = [RootId(50)];
    let tag_ids = [NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)];
    let positions = [1_u32, 2, 3, 4, 5];
    let seqs = make_seq_view(&seq_offsets, &seq_docs, &seq_roots, &tag_ids, &positions);

    let g_seq = LocalConeAgsOperator
        .execute(
            &seq_ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: seqs,
                explicit_edges: None,
            },
        )
        .unwrap();
    let g_par = LocalConeAgsOperator
        .execute(
            &par_ctx,
            &LocalConeAgsConfig::default(),
            LocalConeAgsInput {
                tag_vectors,
                sequences: seqs,
                explicit_edges: None,
            },
        )
        .unwrap();

    for (a, b) in g_seq.anchor_gains.iter().zip(&g_par.anchor_gains) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let epa_cfg = GlobalEpaConfig {
        k_max: 4,
        k_sem: 2,
        min_support: 1.0,
        ..Default::default()
    };
    let epa_seq = GlobalEpaOperator
        .execute(
            &seq_ctx,
            &epa_cfg,
            GlobalEpaInput {
                tag_vectors,
                geometry: &g_seq,
            },
        )
        .unwrap();
    let epa_par = GlobalEpaOperator
        .execute(
            &par_ctx,
            &epa_cfg,
            GlobalEpaInput {
                tag_vectors,
                geometry: &g_par,
            },
        )
        .unwrap();
    assert_eq!(epa_seq.basis_count, epa_par.basis_count);
    for (a, b) in epa_seq.basis.iter().zip(&epa_par.basis) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in epa_seq.coverage.iter().zip(&epa_par.coverage) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Exercise semantic_gain_enabled + reverse_anchor_boost + sequence provenance.
    let fm_out = FactMatrixOperator
        .execute(
            &seq_ctx,
            &FactMatrixConfig {
                semantic_gain_enabled: true,
                reverse_anchor_boost: true,
                distance_decay: 0.1,
                ..Default::default()
            },
            FactMatrixInput {
                tag_vectors,
                sequences: seqs,
                anchor_gains: &g_seq.anchor_gains,
                positive_relations: None,
                negative_relations: None,
            },
        )
        .unwrap();
    let tr = TransportCalibrateOperator
        .execute(
            &seq_ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &fm_out.fact_matrix,
                anchor_gains: &g_seq.anchor_gains,
                edge_gates: None,
            },
        )
        .unwrap();
    let prov = ProvenanceBuildOperator
        .execute(
            &seq_ctx,
            &ProvenanceConfig::default(),
            ProvenanceBuildInput {
                transport_row_offsets: tr.graph.view().row_offsets,
                transport_targets: tr.graph.view().targets,
                sequences: seqs,
                positive_relations: None,
                support_groups: None,
            },
        )
        .unwrap();
    assert_eq!(prov.edge_count, tr.graph.view().targets.len());
    assert_eq!(
        prov.as_view()
            .independent_support_fraction(0, &[RootId(50)], &SupportGroupView::empty()),
        Some(0.0)
    );
}

#[cfg(feature = "schemars")]
#[test]
fn schemars_schemas_generate_cleanly() {
    let _ = schemars::schema_for!(LocalConeAgsConfig);
    let _ = schemars::schema_for!(GlobalEpaConfig);
    let _ = schemars::schema_for!(FactMatrixConfig);
    let _ = schemars::schema_for!(TransportConfig);
    let _ = schemars::schema_for!(ProvenanceConfig);
    let _ = schemars::schema_for!(GeometryState);
    let _ = schemars::schema_for!(EpaBasis);
    let _ = schemars::schema_for!(FactMatrixOutput);
    let _ = schemars::schema_for!(ProvenanceData);
}
