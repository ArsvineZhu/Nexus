// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::kernel::csr_spmv_dual;
use nexus_core::*;
use nexus_vcp::*;
use std::cell::RefCell;

#[test]
fn observes_m2_transport_with_residual_search_and_temporal_gates() {
    let ctx = ExecutionContext::default().with_precision(PrecisionMode::StrictLeftFold);

    let graph = TransportCalibrateOperator
        .execute(
            &ctx,
            &TransportConfig::default(),
            TransportCalibrateInput {
                fact_matrix: &FactMatrix {
                    node_count: 4,
                    row_offsets: vec![0, 0, 1, 1, 1],
                    targets: vec![NodeId(3)],
                    raw_weights: vec![1.0],
                },
                anchor_gains: &[1.0; 4],
                edge_gates: None,
            },
        )
        .unwrap();
    let graph = graph.as_csr_view();
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
            0 => output.extend([(NodeId(1), 0.9), (NodeId(0), 0.7)]),
            1 => output.push((NodeId(2), 1.0)),
            _ => panic!("search continued after capturing the tail"),
        }
        Ok(())
    };
    let input = ObservationInput {
        query: &[0.8, 0.6, 0.5],
        query_text: "car engine",
        candidates: &search,
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
            max_transition_records: 128,
            ..Default::default()
        },
        ..Default::default()
    };
    let observed = Observe.execute(&ctx, &cfg, input).unwrap();
    assert!(!observed.epa.cache_available);
    assert!(observed.epa.in_domain_gate.is_finite());
    assert_eq!(seen.borrow().len(), 2);
    assert!((seen.borrow()[1][2] - 0.5).abs() < 1e-7);
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
    let fields = ComputeFields
        .execute(
            &ctx,
            &cfg_fields,
            FieldInput {
                graph,
                tag_vectors: matrix,
                observation: &observed,
            },
        )
        .unwrap();
    assert!(fields.dual_fields.transfer_field[3] > 0.0);
    assert!(fields.basins.transfer_support_nodes.contains(&NodeId(3)));

    seen.borrow_mut().clear();
    let explicit = Observe
        .execute(
            &ctx,
            &cfg,
            ObservationInput {
                core_tags: &[NodeId(0)],
                graph: graph.with_edge_gates(&[0.0]).unwrap(),
                ..input
            },
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
    let explicit_fields = ComputeFields
        .execute(
            &ctx,
            &cfg_fields,
            FieldInput {
                graph: graph.with_edge_gates(&[0.0]).unwrap(),
                tag_vectors: matrix,
                observation: &explicit,
            },
        )
        .unwrap();
    assert_eq!(explicit_fields.dual_fields.transfer_field[3], 0.0);

    let fields = &fields.dual_fields;
    let mut local = [0.0; 4];
    let mut transfer = [0.0; 4];
    csr_spmv_dual(
        &graph,
        &fields.local_field,
        &fields.transfer_field,
        &mut local,
        &mut transfer,
        ctx.parallel,
    )
    .unwrap();
    let mut source = [0.0; 4];
    for &(node, mass) in &observed.fused.source_field {
        source[node.index()] += mass;
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
    let result = PyramidAnalyzeOperator.execute(
        &ExecutionContext::default(),
        &PyramidConfig::default(),
        PyramidAnalyzeInput {
            query: &[1.],
            tag_vectors: VectorMatrixView::new(&[1.], 1, 1).unwrap(),
            candidates: &host,
            epa: None,
        },
    );
    assert!(matches!(result, Err(VcpError::Cancelled { stage: "host" })));
}

#[test]
fn composes_epa_and_pyramid_with_borrowed_query_and_host_recall() -> VcpResult<()> {
    let query = vec![1.0_f32, 0.0];
    let tag = vec![1.0_f32, 0.0];
    let tag_vectors = VectorMatrixView::new(&tag, 1, 2)?;
    let hits = [(NodeId(0), 1.0)];
    let recall =
        |layer, _: &[f32], _, out: &mut Vec<_>, epa: Option<&EpaAnalysis>, _: &[NodeId]| {
            assert!(epa.is_some());
            if layer == 0 {
                out.extend_from_slice(&hits);
            }
            Ok(())
        };
    let mut plan = Plan::new();
    let epa = plan.add(EpaAnalyzeOperator, EpaConfig::default(), (), |_| {
        Ok(EpaAnalyzeInput {
            query: &query,
            basis: None,
        })
    })?;
    let pyramid = plan.add(
        PyramidAnalyzeOperator,
        PyramidConfig::default(),
        epa,
        |inputs| {
            Ok(PyramidAnalyzeInput {
                query: &query,
                candidates: &recall,
                epa: Some(inputs.get()),
                tag_vectors,
            })
        },
    )?;
    let result = plan.run(pyramid, &ExecutionContext::default())?;
    let pyramid = result.get(pyramid)?;
    assert_eq!(pyramid.levels[0].tags[0].node, NodeId(0));
    assert!((pyramid.coverage - 1.0).abs() < 1e-12);
    Ok(())
}
