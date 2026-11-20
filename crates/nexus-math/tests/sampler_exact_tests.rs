// SPDX-License-Identifier: Apache-2.0

use nexus_math::nexus_core::{ExecutionContext, NodeId, VectorMatrixView};
use nexus_math::{
    DtscFieldBuildInput, DtscFieldConfig, DtscFieldSampleInput, DtscFieldSampler, DtscFieldValues,
    DtscNodeInput, DtscSample, DtscSourceType, build_dtsc_field, dtsc_cosine, sample_dtsc_field,
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
    field: &DtscFieldValues,
    inbound: &[f64],
) -> DtscSample {
    let config = field.config();
    let exact = field.exact_field().iter().find(|s| s.node == node);
    let maximum = inbound.iter().copied().fold(0.0, f64::max);
    let threshold = config.field_similarity_threshold.clamp(-1.0, 1.0);
    let mut neighbors = field
        .field_nodes()
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
            .map(|s| s.source_type)
            .or_else(|| neighbors.first().map(|(_, n)| n.source_type))
            .unwrap_or(DtscSourceType::Unknown),
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
    let field_rows: Vec<_> = vectors
        .iter()
        .filter(|vector| vector.len() == 7)
        .cloned()
        .collect();
    let flat = field_rows.concat();
    let tags = VectorMatrixView::new(&flat, field_rows.len(), 7).unwrap();
    let nodes: Vec<_> = (0..field_rows.len())
        .map(|i| DtscNodeInput {
            node: NodeId(i as u32),
            mass: (i + 1) as f64,

            source_type: [
                DtscSourceType::Core,
                DtscSourceType::Seed,
                DtscSourceType::Emergent,
            ][i % 3],
        })
        .collect();
    let ctx = ExecutionContext::default();

    let inbound_cases = [
        dense_first_wins(17, &[]),
        dense_first_wins(17, &[(1, 1.0), (1, 0.2), (3, 0.5)]),
        dense_first_wins(17, &[(1, f64::NAN), (3, f64::INFINITY)]),
        dense_first_wins(17, &[(1, -1.0)]),
    ];

    for inbound in &inbound_cases {
        for threshold in [-1.0, 0.0, 0.5, 1.0] {
            for neighbors in [1, 4, 128] {
                let config = DtscFieldConfig {
                    min_field_tags: nodes.len(),
                    field_similarity_threshold: threshold,
                    max_field_neighbors: neighbors,
                    ..Default::default()
                };
                let field = build_dtsc_field(
                    DtscFieldBuildInput {
                        nodes: &nodes,
                        tag_vectors: tags,
                    },
                    &config,
                    &ctx,
                )
                .unwrap();
                let sampler = DtscFieldSampler::new(&field, inbound);
                for (i, vector) in vectors.iter().enumerate() {
                    let node = NodeId((i % 17) as u32);
                    let actual = sampler.sample(node, vector);
                    let expected = baseline_sample(node, vector, &field, inbound);
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
fn field_build_rejects_duplicate_ids_and_samples_unique_nodes_bit_exactly() {
    let vectors = vectors();
    let dim = 128;
    let dim128_vecs: Vec<&Vec<f32>> = vectors.iter().filter(|v| v.len() == dim).collect();
    assert_eq!(dim128_vecs.len(), 8);

    let mut flat_tags = Vec::with_capacity(dim128_vecs.len() * dim);
    for v in &dim128_vecs {
        flat_tags.extend_from_slice(v);
    }
    let tag_matrix = VectorMatrixView::new(&flat_tags, dim128_vecs.len(), dim).unwrap();

    let mut nodes = vec![
        DtscNodeInput {
            node: NodeId(0),
            mass: 1.0,

            source_type: DtscSourceType::Seed,
        },
        DtscNodeInput {
            node: NodeId(3),
            mass: 0.4,

            source_type: DtscSourceType::Core,
        },
        DtscNodeInput {
            node: NodeId(3),
            mass: 0.9,

            source_type: DtscSourceType::Emergent,
        },
        DtscNodeInput {
            node: NodeId(1),
            mass: 0.8,

            source_type: DtscSourceType::Core,
        },
        DtscNodeInput {
            node: NodeId(2),
            mass: 0.6,

            source_type: DtscSourceType::Seed,
        },
    ];

    let ctx = ExecutionContext::default();

    let cfg = DtscFieldConfig {
        field_energy_mass_ratio: 0.70,
        ..Default::default()
    };

    let build = |nodes| {
        build_dtsc_field(
            DtscFieldBuildInput {
                nodes,
                tag_vectors: tag_matrix,
            },
            &cfg,
            &ctx,
        )
    };
    assert!(matches!(
        build(&nodes),
        Err(nexus_core::VcpError::InvalidInput { .. })
    ));
    nodes[2].node = NodeId(4);
    let tiny: Vec<_> = nodes
        .iter()
        .cloned()
        .map(|mut node| {
            node.mass *= 1e-30;
            node
        })
        .collect();
    let field = build_dtsc_field(
        DtscFieldBuildInput {
            nodes: &nodes,
            tag_vectors: tag_matrix,
        },
        &cfg,
        &ctx,
    )
    .unwrap();

    assert!(field.exact_field().iter().any(|s| s.node == NodeId(0)));
    assert!(field.field_nodes().iter().any(|n| n.node == NodeId(0)));
    let scaled = build_dtsc_field(
        DtscFieldBuildInput {
            nodes: &tiny,
            tag_vectors: tag_matrix,
        },
        &cfg,
        &ctx,
    )
    .unwrap();
    assert_eq!(scaled.retained_node_count(), field.retained_node_count());
    assert!((scaled.field_entropy() - field.field_entropy()).abs() < 1e-14);

    let query_ids: Vec<NodeId> = (0..dim128_vecs.len()).map(|i| NodeId(i as u32)).collect();
    let inbound = vec![0.2, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
    let samples = sample_dtsc_field(
        DtscFieldSampleInput {
            field: &field,
            query_ids: &query_ids,
            query_vectors: tag_matrix,
            inbound: &inbound,
        },
        &ctx,
    )
    .unwrap();

    assert_eq!(samples.len(), query_ids.len());
    let sampler = DtscFieldSampler::new(&field, &inbound);
    for (i, actual) in samples.iter().enumerate() {
        let expected_sampler = sampler.sample(query_ids[i], dim128_vecs[i]);
        let expected_ref = baseline_sample(query_ids[i], dim128_vecs[i], &field, &inbound);
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
