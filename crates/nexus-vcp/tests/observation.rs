// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::kernel::{csr_spmv_dual, normalize_direction};
use nexus_core::*;
use nexus_vcp::*;
use std::cell::RefCell;

#[test]
fn observes_m2_transport_with_residual_search_and_temporal_gates() {
    let ctx = ExecutionContext::default();

    let graph = calibrate_transport(
        TransportCalibrateInput {
            fact_matrix: &FactMatrix {
                node_count: 4,
                row_offsets: vec![0, 0, 1, 1, 1],
                targets: vec![NodeId(3)],
                raw_weights: vec![1.0],
            },
            anchor_gains: &[1.0; 4],
        },
        &TransportConfig::default(),
        &ctx,
    )
    .unwrap();
    let graph = graph.graph.view();
    let matrix =
        VectorMatrixView::new(&[1., 0., 0., 0.8, 0.6, 0., 0., 0., 1., 0., 1., 0.], 4, 3).unwrap();
    let seen = RefCell::new(Vec::new());
    let search = |level,
                  residual: &[f32],
                  top_k,
                  output: &mut Vec<_>,
                  _: Option<&EpaAnalysis>,
                  _: &[NodeId]| {
        assert_eq!(top_k, 10);
        assert!(output.is_empty());
        seen.borrow_mut().push(residual.to_vec());
        match level {
            0 => output.extend([NodeId(1), NodeId(0)]),
            1 => output.push(NodeId(2)),
            _ => panic!("search continued after capturing the tail"),
        }
        Ok(())
    };
    let query = normalize_direction(&[0.8, 0.6, 0.5], 3, "test").unwrap();
    let input = ObservationInput {
        query: &query,
        query_text: "car engine",
        candidates: Some(&search),
        graph,
        tag_vectors: matrix,
        lexical_meta: TagLexicalMetaView {
            node_count: 4,
            names: TextRows::Borrowed(&["car", "car engine", "repair", "manual"]),
            alias_offsets: &[0; 5],
            aliases: TextRows::Borrowed(&[]),
        },
        epa_basis: Some(EpaBasisView::empty(3)),
        core_tags: &[],
        ghost_vectors: None,
        ghost_is_core: &[],
    };
    let cfg = ObservationConfig {
        sense: SenseConfig {
            max_transition_records: Some(128),
            ..Default::default()
        },
        ..Default::default()
    };
    let observed = observe(input, &cfg, &ctx).unwrap();
    assert!(!observed.epa.cache_available);
    assert!(observed.epa.in_domain_gate.is_finite());
    assert_eq!(seen.borrow().len(), 2);
    assert!((seen.borrow()[1][2] - query[2]).abs() < 1e-7);
    assert!(seen.borrow()[1][..2].iter().all(|value| value.abs() < 1e-7));
    assert_eq!(observed.disambiguated.pruned_hypernyms, [NodeId(0)]);
    assert!(observed.disambiguated.differentia_q_perp.as_ref().unwrap()[0].abs() < 1e-12);
    assert!(
        observed
            .sense
            .edges
            .iter()
            .any(|edge| edge.target == NodeId(3))
    );
    let cfg_fields = FieldConfig::default();
    let fields = compute_fields(
        FieldInput {
            graph,
            tag_vectors: matrix,
            observation: &observed,
        },
        &cfg_fields,
        &ctx,
    )
    .unwrap();
    assert!(fields.dual_fields.transfer_field[3] > 0.0);
    assert!(fields.basins.transfer_support_nodes.contains(&NodeId(3)));

    seen.borrow_mut().clear();
    let explicit = observe(
        ObservationInput {
            core_tags: &[NodeId(0)],
            graph: graph.with_edge_gates(&[0.0]).unwrap(),
            ..input
        },
        &cfg,
        &ctx,
    )
    .unwrap();
    assert!(explicit.disambiguated.pruned_hypernyms.is_empty());
    assert!(
        explicit
            .gated_seeds
            .seeds
            .iter()
            .any(|seed| seed.node == NodeId(0) && seed.is_core)
    );
    assert!(explicit.sense.edges.is_empty());
    let explicit_fields = compute_fields(
        FieldInput {
            graph: graph.with_edge_gates(&[0.0]).unwrap(),
            tag_vectors: matrix,
            observation: &explicit,
        },
        &cfg_fields,
        &ctx,
    )
    .unwrap();
    assert_eq!(explicit_fields.dual_fields.transfer_field[3], 0.0);

    let fields = &fields.dual_fields;
    let mut local = [0.0; 4];
    let mut transfer = [0.0; 4];
    csr_spmv_dual(
        &graph.as_csr(),
        &fields.local_field,
        &fields.transfer_field,
        &mut local,
        &mut transfer,
        ctx.parallel,
    )
    .unwrap();
    let mut source = [0.0; 4];
    let total: f64 = observed.sense.nodes.iter().map(|node| node.energy).sum();
    for node in &observed.sense.nodes {
        source[node.node.index()] = node.energy / total;
    }
    for (alpha, field, propagated) in [
        (cfg_fields.solve.local_alpha, &fields.local_field, local),
        (
            cfg_fields.solve.transfer_alpha,
            &fields.transfer_field,
            transfer,
        ),
    ] {
        let residual: f64 = source
            .iter()
            .zip(propagated)
            .zip(field)
            .map(|((&s, p), &x)| ((1.0 - alpha) * s + alpha * p - x).abs())
            .sum();
        assert!(residual < 1e-9, "fixed-point residual {residual}");
    }
}

#[test]
fn pyramid_propagates_host_failure() {
    let host = |_, _: &[f32], _, _: &mut Vec<_>, _: Option<&EpaAnalysis>, _: &[NodeId]| {
        Err(VcpError::Cancelled { stage: "host" })
    };
    let result = analyze_pyramid(
        PyramidAnalyzeInput {
            query: &[1.],
            tag_vectors: VectorMatrixView::new(&[1.], 1, 1).unwrap(),
            candidates: Some(&host),
            epa: None,
        },
        &PyramidConfig::default(),
        &ExecutionContext::default(),
    );
    assert!(matches!(result, Err(VcpError::Cancelled { stage: "host" })));
}

#[test]
fn prepared_query_contract_rejects_non_unit_inputs_and_fusion_rejects_cancellation() {
    let ctx = ExecutionContext::default();
    for query in [&[2.0_f32, 0.0], &[0.0, 0.0], &[f32::NAN, 0.0]] {
        let result = analyze_epa(
            EpaAnalyzeInput { query, basis: None },
            &EpaConfig::default(),
            &ctx,
        );
        assert!(matches!(result, Err(VcpError::InvalidInput { .. })));
    }

    let mut gated = GatedSeeds {
        effective_tag_boost: 0.5,
        ..Default::default()
    };
    let tags = VectorMatrixView::new(&[-1.0, 0.0], 1, 2).unwrap();
    let sense = SenseOutput::default();
    let fuse = |gated: &GatedSeeds| {
        fuse_observation(
            ObservationFuseInput {
                query: &[1.0, 0.0],
                tag_vectors: tags,
                sense: &sense,
                gated,
                core_tags: &[],
                ghost_vectors: None,
                ghost_is_core: &[],
            },
            &ObservationFuseConfig::default(),
            &ctx,
        )
    };
    assert_eq!(fuse(&gated).unwrap().enhanced_query, [1.0, 0.0]);
    gated.seeds.push(GatedSeed {
        node: NodeId(0),
        weight: 1.0,
        is_core: true,
        lexical: true,
        residual_level: 0,
    });
    assert!(matches!(
        fuse(&gated),
        Err(VcpError::InvalidInput {
            stage: "observation.fuse",
            ..
        })
    ));
}

#[test]
fn fixed_point_reports_the_returned_iterate_and_fields_require_convergence() {
    let ctx = ExecutionContext::default();
    let graph = TransportGraphView::new(
        CsrGraphView::new(CsrParts {
            node_count: 2,
            row_offsets: &[0, 1, 2],
            targets: &[NodeId(1), NodeId(0)],
            weights: &[0.9, 0.9],
            is_static_wormhole: &[false; 2],
            inbound: &[0.9, 0.9],
            max_inbound: 0.9,
            anchor_gains: &[0.0; 2],
            target_sorted_indices: None,
            edge_gates: None,
        })
        .unwrap(),
    )
    .unwrap();
    let source_field = [(NodeId(0), 0.25), (NodeId(0), 0.75)];
    let solve = |config: &DualFieldConfig| {
        solve_dual_fields(
            DualFieldSolveInput {
                graph,
                source_field: &source_field,
            },
            config,
            &ctx,
        )
    };
    let solution = solve(&DualFieldConfig::default()).unwrap();
    assert!(solution.local_converged && solution.transfer_converged);
    let fixed = 0.45_f64 / (1.0 - (0.55_f64 * 0.9).powi(2));
    assert!((solution.transfer_field[0] - fixed).abs() < 1e-9);
    assert!((solution.transfer_field[1] - fixed * 0.55 * 0.9).abs() < 1e-9);

    let limited = DualFieldConfig {
        local_alpha: 0.0,
        transfer_alpha: 0.99,
        max_iterations: 1,
        ..Default::default()
    };
    let solution = solve(&limited).unwrap();
    assert!(solution.local_converged && !solution.transfer_converged);
    let a = 0.99 * 0.9;
    let after = [1.0 - 0.99, a];
    assert_eq!(solution.transfer_field, after);
    let residual = (a * after[1]).abs() + (a * after[0] - after[1]).abs();
    assert!((solution.transfer_residual - residual).abs() < 1e-14);
    let observation = ObservationState {
        original_query: vec![1.0, 0.0],
        epa: EpaAnalysis::default(),
        pyramid: PyramidResidual::default(),
        disambiguated: DisambiguatedTags::default(),
        gated_seeds: GatedSeeds::default(),
        sense: SenseOutput {
            nodes: vec![ObservationNode {
                node: NodeId(0),
                energy: 1.,
                source_kind: SeedSourceKind::Core,
                hop: 0,
                seed_node: None,
            }],
            ..Default::default()
        },
        fused: FusedObservation::default(),
    };
    let result = compute_fields(
        FieldInput {
            graph,
            tag_vectors: VectorMatrixView::new(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap(),
            observation: &observation,
        },
        &FieldConfig {
            solve: limited.clone(),
            ..Default::default()
        },
        &ctx,
    );
    assert!(matches!(
        result,
        Err(VcpError::NotConverged { stage: "field.transfer", iterations: 1, residual: actual, tolerance })
            if (actual - residual).abs() < 1e-14 && tolerance == limited.transfer_tolerance
    ));

    for transfer_alpha in [1.0, -0.1, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            solve(&DualFieldConfig {
                transfer_alpha,
                ..limited
            }),
            Err(VcpError::InvalidInput { .. })
        ));
    }
}

#[test]
fn composes_epa_and_pyramid_with_borrowed_query_and_host_recall() -> VcpResult<()> {
    let query = vec![1.0_f32, 0.0];
    let tag = vec![1.0_f32, 0.0];
    let tag_vectors = VectorMatrixView::new(&tag, 1, 2)?;
    let hits = [NodeId(0)];
    let recall =
        |layer, _: &[f32], _, out: &mut Vec<_>, epa: Option<&EpaAnalysis>, _: &[NodeId]| {
            assert!(epa.is_some());
            if layer == 0 {
                out.extend_from_slice(&hits);
            }
            Ok(())
        };
    let ctx = ExecutionContext::default();
    let epa = analyze_epa(
        EpaAnalyzeInput {
            query: &query,
            basis: None,
        },
        &EpaConfig::default(),
        &ctx,
    )?;
    let pyramid = analyze_pyramid(
        PyramidAnalyzeInput {
            query: &query,
            candidates: Some(&recall),
            epa: Some(&epa),
            tag_vectors,
        },
        &PyramidConfig::default(),
        &ctx,
    )?;
    assert_eq!(pyramid.levels[0].tags[0].node, NodeId(0));
    assert!((pyramid.coverage - 1.0).abs() < 1e-12);
    Ok(())
}
