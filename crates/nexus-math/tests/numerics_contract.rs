// SPDX-License-Identifier: Apache-2.0

use nexus_math::kernel::csr_spmv_dual;
use nexus_math::{
    CancelToken, CosineOptions, Error, GraphInputOptions, NumericColumns, Parallelism, RunOptions,
    Scores, TransportGraph, UnitVectors, VectorRows, WeightedEdge, cosine_scores, normalize,
    solve_nnls,
};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn validates_vector_contract_and_handles_all_finite_scales() {
    assert!(matches!(
        VectorRows::new(&[1, 2], &[], usize::MAX),
        Err(Error::Input {
            field: "values",
            ..
        })
    ));
    assert!(matches!(
        VectorRows::new(&[], &[], 0),
        Err(Error::Input {
            field: "dimension",
            ..
        })
    ));
    assert!(matches!(
        VectorRows::new(&[1], &[1.0], 2),
        Err(Error::Input {
            field: "values",
            ..
        })
    ));
    assert!(matches!(
        VectorRows::new(&[7, 7], &[1.0, 1.0], 1),
        Err(Error::Input { id: Some(7), .. })
    ));
    for vector in [&[0.0, 0.0], &[f32::INFINITY, 1.0], &[f32::NAN, 1.0]] {
        assert!(matches!(
            VectorRows::new(&[7], vector, 2),
            Err(Error::Input { id: Some(7), .. })
        ));
        assert!(matches!(
            normalize(vector),
            Err(Error::Input {
                field: "vector",
                ..
            })
        ));
    }
    assert!(UnitVectors::new(&[1], &[1.0 + f32::EPSILON, 0.0], 2).is_ok());
    assert!(UnitVectors::new(&[1], &[1.0 + 8.0 * f32::EPSILON, 0.0], 2).is_err());

    let tiny = f32::from_bits(1);
    let values = [f32::MAX, 0.0, 0.0, -2.0, tiny, 0.0];
    let rows = VectorRows::new(&[9, 18, 6], &values, 2).unwrap();
    let query = [tiny, 0.0];
    assert_eq!(normalize(&query).unwrap().as_slice(), &[1.0, 0.0]);
    let seq = cosine_scores(&query, rows, &Default::default()).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let parallel = pool
        .install(|| {
            cosine_scores(
                &query,
                rows,
                &CosineOptions {
                    run: RunOptions {
                        parallelism: Parallelism::Rayon,
                        ..Default::default()
                    },
                },
            )
        })
        .unwrap();
    assert_eq!(parallel, seq);
    assert_eq!(
        parallel.values().collect::<Vec<_>>(),
        [(9, 1.0), (18, 0.0), (6, 1.0)]
    );
    assert_eq!(parallel.top(Some(2)).ids(), &[9, 6]);
    assert!(parallel.top(Some(0)).rows().is_empty());

    let empty = VectorRows::new(&[], &[], 2).unwrap();
    assert!(
        cosine_scores(&query, empty, &Default::default())
            .unwrap()
            .top(None)
            .rows()
            .is_empty()
    );
    assert!(matches!(
        cosine_scores(&[0.0, 0.0], empty, &Default::default()),
        Err(Error::Input { field: "query", .. })
    ));

    let cancel = CancelToken::default();
    let options = CosineOptions {
        run: RunOptions {
            cancel: Some(cancel.clone()),
            ..Default::default()
        },
    };
    assert!(cosine_scores(&query, rows, &options).is_ok());
    cancel.cancel();
    assert!(matches!(
        cosine_scores(&query, rows, &options),
        Err(Error::Cancelled)
    ));
}

#[test]
fn nnls_returns_coefficients_in_original_matrix_scale_after_qr() {
    for scale in [1e-160, 1.0, 1e160] {
        let values = [
            2.0 * scale,
            scale,
            0.0,
            0.0,
            -3.0 * scale,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let target = [scale, 0.125 * scale, 0.75 * scale];
        let fit = solve_nnls(
            NumericColumns {
                dimension: 3,
                column_count: 3,
                values: &values,
            },
            &target,
            &Default::default(),
        )
        .unwrap();
        close(fit.coefficients[0], 0.5);
        close(fit.coefficients[1], 0.125);
        assert_eq!(fit.coefficients[2], 0.0);
        for (axis, expected) in [1.0, 0.125, 0.0].into_iter().enumerate() {
            close(fit.projection[axis] / scale, expected);
        }
        close(fit.residual[2] / scale, 0.75);
        close(
            fit.residual_ratio,
            0.75_f64.powi(2) / (1.0 + 0.125_f64.powi(2) + 0.75_f64.powi(2)),
        );
        assert!(fit.kkt_error < 2e-8);
    }

    let empty = NumericColumns {
        dimension: 2,
        column_count: 0,
        values: &[],
    };
    let residual = solve_nnls(empty, &[-2.0, 1.0], &Default::default()).unwrap();
    assert_eq!(residual.residual, [-2.0, 1.0]);
    assert_eq!(residual.residual_ratio, 1.0);
    let zero = solve_nnls(
        NumericColumns {
            dimension: 2,
            column_count: 1,
            values: &[-1.0, 5.0],
        },
        &[0.0, 0.0],
        &Default::default(),
    )
    .unwrap();
    assert_eq!(zero.residual_ratio, 0.0);
    assert_eq!(zero.coefficients, [0.0]);

    assert!(matches!(
        solve_nnls(
            NumericColumns {
                dimension: 2,
                column_count: 1,
                values: &[f64::NAN, 0.0]
            },
            &[1.0, 0.0],
            &Default::default()
        ),
        Err(Error::Input { .. })
    ));
    assert!(matches!(
        solve_nnls(
            NumericColumns {
                dimension: 2,
                column_count: usize::MAX,
                values: &[]
            },
            &[1.0, 0.0],
            &Default::default()
        ),
        Err(Error::Input {
            field: "columns.values",
            ..
        })
    ));
}

#[test]
fn transport_coalesces_edges_and_attenuates_without_renormalizing() {
    let edge = |from, to, weight, wormhole| WeightedEdge {
        from,
        to,
        weight,
        wormhole,
    };
    let nodes = [u64::MAX, 12, 19];
    let largest = nodes[0];
    let graph = TransportGraph::from_edges(
        &nodes,
        &[
            edge(largest, 12, 0.25, false),
            edge(largest, 12, 0.5, false),
            edge(largest, 19, 0.25, true),
            edge(12, largest, 0.0, false),
        ],
        &GraphInputOptions::default(),
    )
    .unwrap();
    assert_eq!(graph.nodes(), &nodes);
    assert!(matches!(
        TransportGraph::from_edges(
            &nodes,
            &[edge(largest, 12, 0.5, false), edge(largest, 12, 0.0, true)],
            &Default::default()
        ),
        Err(Error::Input {
            field: "edges.wormhole",
            ..
        })
    ));
    assert!(matches!(
        TransportGraph::from_edges(
            &nodes,
            &[
                edge(largest, 12, 0.75, false),
                edge(largest, 19, 0.5, false)
            ],
            &Default::default()
        ),
        Err(Error::Input {
            field: "edges.weight",
            ..
        })
    ));
    assert!(matches!(
        TransportGraph::from_edges(
            &nodes,
            &[edge(largest, 77, 0.0, false)],
            &Default::default()
        ),
        Err(Error::Input { id: Some(77), .. })
    ));
    let base = graph.view();
    let requested = base.attenuate(&[0.0, 0.5], &Default::default()).unwrap();
    let gated = requested.view(base);
    let start = graph.slot(largest).unwrap();
    assert_eq!(base.row_statistics(start), (1.0, 0.75));
    assert_eq!(gated.row_statistics(start), (0.125, 0.125));
    assert_eq!(gated.inbound, [0.0, 0.0, 0.125]);
    let mut local = [0.0; 3];
    let mut transfer = local;
    csr_spmv_dual(
        &gated,
        &[2.0, 0.0, 0.0],
        &[4.0, 0.0, 0.0],
        &mut local,
        &mut transfer,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(local, [0.0, 0.0, 0.25]);
    assert_eq!(transfer, [0.0, 0.0, 0.5]);
}

#[test]
fn score_inputs_reject_duplicates_and_nonfinite_values() {
    assert!(matches!(
        Scores::new([(5, 1.0), (5, 2.0)]),
        Err(Error::Input { id: Some(5), .. })
    ));
    assert!(matches!(
        Scores::new([(7, f64::NAN)]),
        Err(Error::Input { id: Some(7), .. })
    ));
}

#[test]
fn nnls_matches_vcp_reference_projection_residual_and_coefficients() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/vcp-nnls.json")).unwrap();
    assert_eq!(
        capture["revision"],
        "5adb12bb55e616e22bd0e0dbfdc13032c069941a"
    );
    let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    for case in capture["nnls"].as_array().unwrap() {
        let query: Vec<f64> = serde_json::from_value(case["query"].clone()).unwrap();
        let vectors: Vec<Vec<f64>> = serde_json::from_value(case["vectors"].clone()).unwrap();
        let ids: Vec<u64> = serde_json::from_value(case["ids"].clone()).unwrap();
        let weights: Vec<f64> = serde_json::from_value(case["weights"].clone()).unwrap();
        // The source uses ID ties to select among equivalent columns; public
        // NNLS breaks those same ties by the columns' input order.
        let mut order: Vec<_> = (0..ids.len()).collect();
        order.sort_unstable_by_key(|&index| ids[index]);
        let flat = order
            .iter()
            .flat_map(|&index| vectors[index].iter().copied())
            .collect::<Vec<_>>();
        let actual = solve_nnls(
            NumericColumns {
                dimension: query.len(),
                column_count: vectors.len(),
                values: &flat,
            },
            &query,
            &Default::default(),
        )
        .unwrap();
        for (field, values) in [
            ("projection", actual.projection),
            ("residual", actual.residual),
        ] {
            let expected: Vec<f64> = serde_json::from_value(case[field].clone()).unwrap();
            assert_eq!(values.len(), expected.len());
            for (value, expected) in values.into_iter().zip(expected) {
                assert!(
                    (value - expected).abs() <= 2e-7,
                    "{} {field}: {value} != {expected}",
                    case["case"]
                );
            }
        }
        for (value, index) in actual.coefficients.into_iter().zip(order) {
            let expected = if weights[index] == 0.0 {
                0.0
            } else {
                weights[index] * norm(&query) / norm(&vectors[index])
            };
            assert!(
                (value - expected).abs() <= 1e-10,
                "{} coefficient: {value} != {expected}",
                case["case"]
            );
        }
        assert!((actual.residual_ratio - case["residual_ratio"].as_f64().unwrap()).abs() <= 1e-10);
    }
}

#[test]
fn owned_rows_and_packed_cosines_preserve_selected_order() {
    let matrix = nexus_math::OwnedUnitVectors::new(
        [17, 28, 39].as_slice().into(),
        vec![2.0, 0.0, 0.0, f32::MAX, -2.0, 0.0],
        2,
        &Default::default(),
    )
    .unwrap();
    let rows = matrix.view().as_rows();
    let selection = rows.select_rows(&[39, 17], &[2, 0]).unwrap();
    let lanes = nexus_math::cosine_columns(
        [Some(&[1.0, 0.0]), None, Some(&[0.0, 1.0])],
        selection,
        &RunOptions {
            parallelism: Parallelism::Rayon,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(lanes, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    assert!(matches!(
        rows.select_rows(&[17, 17], &[0, 0]),
        Err(Error::Input { id: Some(17), .. })
    ));
    assert!(matches!(
        rows.select_rows(&[17], &[2]),
        Err(Error::Input { id: Some(17), .. })
    ));
    let invalid = nexus_math::OwnedUnitVectors::new(
        [7].as_slice().into(),
        vec![0.0, 0.0],
        2,
        &Default::default(),
    );
    assert!(matches!(invalid, Err(Error::Input { id: Some(7), .. })));
}
