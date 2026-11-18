// SPDX-License-Identifier: Apache-2.0

use nexus_math::nexus_core::{ExecutionContext, NodeId, Operator, ScratchPad, VectorMatrixView};
use nexus_math::{
    DtscField, DtscFieldBuildInput, DtscFieldBuildOperator, DtscFieldConfig, DtscFieldNode,
    DtscFieldSampleInput, DtscFieldSampleOperator, DtscFieldSampler, DtscNodeInput, DtscSample,
    dtsc_cosine,
};

fn unit(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn baseline_reference_cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut cross = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for (a, b) in a.iter().zip(b) {
        let a = f64::from(*a);
        let b = f64::from(*b);
        cross += a * b;
        aa += a * a;
        bb += b * b;
    }
    if aa > 1e-15 && bb > 1e-15 {
        cross / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

fn baseline_cosine(a: &[f32], b: &[f32]) -> f64 {
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
    if norm(a) > 1e-12 && norm(b) > 1e-12 {
        baseline_reference_cosine(a, b)
    } else {
        0.0
    }
}

fn baseline_sample(
    node: NodeId,
    vector: &[f32],
    field: &DtscField,
    inbound: &[f64],
    config: &DtscFieldConfig,
) -> DtscSample {
    let exact = field.exact_field.iter().find(|s| s.node == node);
    let maximum = inbound.iter().copied().fold(0.0, f64::max);
    let threshold = config.field_similarity_threshold.clamp(-1.0, 1.0);
    let mut neighbors = field
        .field_nodes
        .iter()
        .filter(|n| n.node != node)
        .filter_map(|n| {
            let similarity = baseline_cosine(vector, &n.vector);
            if similarity < threshold {
                return None;
            }
            let local = unit((similarity - threshold) / (1.0 - threshold).max(1e-6));
            let specific = if maximum <= 0.0 {
                1.0
            } else {
                let w = inbound.get(n.node.index()).copied().unwrap_or(0.0);
                config
                    .public_hub_floor
                    .clamp(0.05, 1.0)
                    .max(1.0 - unit(w / maximum).sqrt())
            };
            Some((
                n.potential * local.powf(config.field_kernel_exponent.max(0.25)) * specific,
                n,
            ))
        })
        .collect::<Vec<_>>();
    neighbors.sort_by(|a, b| b.0.total_cmp(&a.0));
    neighbors.truncate(config.max_field_neighbors.max(1));
    let interpolated = if neighbors.is_empty() {
        0.0
    } else {
        neighbors.iter().map(|(p, _)| *p).sum::<f64>() / (neighbors.len() as f64).sqrt()
    };
    DtscSample {
        node,
        potential: unit(exact.map_or(0.0, |s| s.potential).max(interpolated)),
        exact: exact.is_some(),
        source_type: exact
            .map(|s| s.source_type.clone())
            .or_else(|| neighbors.first().map(|(_, n)| n.source_type.clone()))
            .unwrap_or_else(|| "unknown".into()),
    }
}

fn vectors() -> Vec<Vec<f32>> {
    let mut result = vec![
        vec![],
        vec![0.0],
        vec![-0.0],
        vec![1e-6],
        vec![1e-7],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
        vec![f32::NAN],
        vec![f32::from_bits(0x7fa12345)],
        vec![f32::MAX],
        vec![1.0, f32::INFINITY],
        vec![1.0, f32::NAN],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
        vec![1e-6, 1e-6],
    ];
    let mut seed = 42_u64;
    for dim in [1, 2, 3, 7, 16, 31, 128, 512] {
        for _ in 0..8 {
            result.push(
                (0..dim)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 32) as u32 as f32 / u32::MAX as f32 - 0.5
                    })
                    .collect(),
            );
        }
    }
    result
}

fn dense_first_wins(len: usize, pairs: &[(usize, f64)]) -> Vec<f64> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; len];
    let mut seen = vec![false; len];
    for &(idx, w) in pairs {
        if idx < len && !seen[idx] {
            seen[idx] = true;
            out[idx] = w;
        }
    }
    out
}

#[test]
fn cosine_matches_reference_bits() {
    let vectors = vectors();
    assert_eq!(vectors.len(), 80);
    for (i, a) in vectors.iter().enumerate() {
        for (j, b) in vectors.iter().enumerate() {
            assert_eq!(
                dtsc_cosine(a, b).to_bits(),
                baseline_cosine(a, b).to_bits(),
                "vector pair {i}, {j}"
            );
        }
    }
}

#[test]
fn sampler_matches_reference_bits_and_order() {
    let vectors = vectors();
    let field = DtscField {
        exact_field: vec![
            DtscSample {
                node: NodeId(3),
                potential: 0.4,
                exact: true,
                source_type: "first".into(),
            },
            DtscSample {
                node: NodeId(3),
                potential: 0.9,
                exact: true,
                source_type: "duplicate".into(),
            },
        ],
        field_nodes: vectors
            .iter()
            .enumerate()
            .map(|(i, vector)| DtscFieldNode {
                node: NodeId((i % 13) as u32),
                potential: (i % 4) as f64 / 3.0,
                vector: vector.clone(),
                source_type: format!("node-{i}"),
            })
            .collect(),
        field_entropy: 0.0,
        field_trusted: true,
        total_energy: 0.0,
        max_energy: 0.0,
    };

    let inbound_cases = [
        dense_first_wins(17, &[]),
        dense_first_wins(17, &[(1, 1.0), (1, 0.2), (3, 0.5)]),
        dense_first_wins(17, &[(1, f64::NAN), (3, f64::INFINITY)]),
        dense_first_wins(17, &[(1, -1.0)]),
    ];

    for inbound in &inbound_cases {
        for threshold in [-1.0, 0.0, 0.5, 1.0, f64::NAN] {
            for neighbors in [0, 1, 4, 128] {
                let config = DtscFieldConfig {
                    field_similarity_threshold: threshold,
                    max_field_neighbors: neighbors,
                    ..Default::default()
                };
                let sampler = DtscFieldSampler::new(&field, inbound, &config);
                for (i, vector) in vectors.iter().enumerate() {
                    let node = NodeId((i % 17) as u32);
                    let actual = sampler.sample(node, vector);
                    let expected = baseline_sample(node, vector, &field, inbound, &config);
                    assert_eq!(
                        actual.potential.to_bits(),
                        expected.potential.to_bits(),
                        "vector {i}, threshold {threshold}, neighbors {neighbors}"
                    );
                    assert_eq!(actual.node, expected.node);
                    assert_eq!(actual.exact, expected.exact);
                    assert_eq!(actual.source_type, expected.source_type);
                }
            }
        }
    }
}

#[test]
fn field_build_retains_node_zero_and_sample_operator_matches_bits() {
    let vectors = vectors();
    let dim = 128;
    let dim128_vecs: Vec<&Vec<f32>> = vectors.iter().filter(|v| v.len() == dim).collect();
    assert_eq!(dim128_vecs.len(), 8);

    let mut flat_tags = Vec::with_capacity(dim128_vecs.len() * dim);
    for v in &dim128_vecs {
        flat_tags.extend_from_slice(v);
    }
    let tag_matrix = VectorMatrixView::new(&flat_tags, dim128_vecs.len(), dim).unwrap();

    // Include NodeId(0) and duplicate NodeId(3) ("first" then "duplicate").
    let nodes = vec![
        DtscNodeInput {
            node: NodeId(0),
            energy: 1.0,
            normalized_energy: 0.0,
            source_type: "seed".into(),
        },
        DtscNodeInput {
            node: NodeId(3),
            energy: 0.4,
            normalized_energy: 0.0,
            source_type: "first".into(),
        },
        DtscNodeInput {
            node: NodeId(3),
            energy: 0.9,
            normalized_energy: 0.0,
            source_type: "duplicate".into(),
        },
        DtscNodeInput {
            node: NodeId(1),
            energy: 0.8,
            normalized_energy: 0.0,
            source_type: "core".into(),
        },
        DtscNodeInput {
            node: NodeId(2),
            energy: 0.6,
            normalized_energy: 0.0,
            source_type: "seed".into(),
        },
    ];

    let ctx = ExecutionContext::default(); // FastSimd default; DTSC must still enforce StrictLeftFold.
    let mut scratch = ScratchPad::default();
    let cfg = DtscFieldConfig::default();

    let field = DtscFieldBuildOperator
        .execute(
            &ctx,
            &cfg,
            DtscFieldBuildInput {
                nodes: &nodes,
                tag_vectors: tag_matrix,
            },
            &mut scratch,
        )
        .unwrap();

    assert!(field.field_trusted);
    assert!(field.exact_field.iter().any(|s| s.node == NodeId(0)));
    assert!(field.field_nodes.iter().any(|n| n.node == NodeId(0)));
    // First-duplicate-wins preserved in exact_field order.
    let first_3 = field
        .exact_field
        .iter()
        .find(|s| s.node == NodeId(3))
        .unwrap();
    assert_eq!(first_3.source_type, "first");

    let query_ids: Vec<NodeId> = (0..dim128_vecs.len()).map(|i| NodeId(i as u32)).collect();
    let inbound = vec![0.2, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
    let samples = DtscFieldSampleOperator
        .execute(
            &ctx,
            &cfg,
            DtscFieldSampleInput {
                field: &field,
                query_ids: &query_ids,
                query_vectors: tag_matrix,
                inbound: &inbound,
            },
            &mut scratch,
        )
        .unwrap();

    assert_eq!(samples.len(), query_ids.len());
    let sampler = DtscFieldSampler::new(&field, &inbound, &cfg);
    for (i, actual) in samples.iter().enumerate() {
        let expected_sampler = sampler.sample(query_ids[i], dim128_vecs[i]);
        let expected_ref = baseline_sample(query_ids[i], dim128_vecs[i], &field, &inbound, &cfg);
        assert_eq!(
            actual.potential.to_bits(),
            expected_sampler.potential.to_bits()
        );
        assert_eq!(actual.potential.to_bits(), expected_ref.potential.to_bits());
        assert_eq!(actual.node, expected_ref.node);
        assert_eq!(actual.exact, expected_ref.exact);
        assert_eq!(actual.source_type, expected_ref.source_type);
    }
}
