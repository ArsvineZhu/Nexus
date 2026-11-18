// SPDX-License-Identifier: Apache-2.0

use nexus_math::nexus_core::{
    CancellationToken, CandidateId, CandidatePoolView, CsrGraphView, DocumentId, ExecutionContext,
    NodeId, Operator, RootId, ScratchPad, VcpError, VectorMatrixView,
};
use nexus_math::{
    DtscAuxConfig, DtscConfig, DtscField, DtscFieldBuildInput, DtscFieldBuildOperator,
    DtscFieldConfig, DtscGeometryState, DtscIdentityConfig, DtscNodeInput, DtscReadoutInput,
    DtscReadoutOperator,
};
use serde::Deserialize;
use serde::de::IntoDeserializer;
use serde::de::value::{MapDeserializer, StrDeserializer};
use std::sync::atomic::AtomicBool;

fn unit_vec(dim: usize, hot_idx: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[hot_idx] = 1.0;
    v
}

fn mix_vec(dim: usize, a: usize, wa: f32, b: usize, wb: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[a] = wa;
    v[b] = wb;
    v
}

fn build_test_field(
    nodes: &[DtscNodeInput],
    tag_vectors: VectorMatrixView<'_, f32>,
    cfg: &DtscFieldConfig,
) -> DtscField {
    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::default();
    DtscFieldBuildOperator
        .execute(
            &ctx,
            cfg,
            DtscFieldBuildInput { nodes, tag_vectors },
            &mut scratch,
        )
        .unwrap()
}

struct OwnedGraph {
    node_count: usize,
    row_offsets: Vec<usize>,
    targets: Vec<NodeId>,
    weights: Vec<f64>,
    is_static_wormhole: Vec<bool>,
    inbound: Vec<f64>,
    max_inbound: f64,
    anchor_gains: Vec<f64>,
    target_sorted_indices: Vec<u32>,
}

impl OwnedGraph {
    fn from_edges(
        node_count: usize,
        edges: &[(usize, usize, f64)],
        inbound: Vec<f64>,
        anchor_gains: Vec<f64>,
    ) -> Self {
        let mut adj: Vec<Vec<(NodeId, f64)>> = vec![Vec::new(); node_count];
        for &(u, v, w) in edges {
            adj[u].push((NodeId(v as u32), w));
        }
        let mut row_offsets = Vec::with_capacity(node_count + 1);
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut target_sorted_indices = Vec::new();
        row_offsets.push(0);
        for row in &mut adj {
            // Primary order descending by weight.
            row.sort_by(|a, b| b.1.total_cmp(&a.1));
            let start = targets.len();
            for &(dst, w) in row.iter() {
                targets.push(dst);
                weights.push(w);
            }
            let end = targets.len();
            let mut indices: Vec<u32> = (start..end).map(|i| i as u32).collect();
            indices.sort_by_key(|&e| (targets[e as usize], e));
            target_sorted_indices.extend(indices);
            row_offsets.push(end);
        }
        let nnz = targets.len();
        let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);
        Self {
            node_count,
            row_offsets,
            targets,
            weights,
            is_static_wormhole: vec![false; nnz],
            inbound,
            max_inbound,
            anchor_gains,
            target_sorted_indices,
        }
    }

    fn view(&self) -> CsrGraphView<'_> {
        CsrGraphView {
            node_count: self.node_count,
            row_offsets: &self.row_offsets,
            targets: &self.targets,
            weights: &self.weights,
            is_static_wormhole: &self.is_static_wormhole,
            inbound: &self.inbound,
            max_inbound: self.max_inbound,
            anchor_gains: &self.anchor_gains,
            target_sorted_indices: Some(&self.target_sorted_indices),
            edge_gates: None,
        }
    }
}

struct OwnedPool {
    candidate_count: usize,
    baseline_scores: Vec<f64>,
    union_scores: Vec<f64>,
    union_ranks: Vec<u32>,
    bm25_scores: Vec<f64>,
    anchor_scores: Vec<f64>,
    curve_document_ids: Vec<DocumentId>,
    cand_vec_data: Vec<f32>,
    dim: usize,
    candidate_root_offsets: Vec<u32>,
    candidate_roots: Vec<RootId>,
    candidate_path_offsets: Vec<u32>,
    path_tag_offsets: Vec<u32>,
    path_tags: Vec<NodeId>,
    path_positions: Vec<u32>,
}

type CandidateSpec = (f64, Vec<f32>, Vec<Vec<(NodeId, u32)>>);

impl OwnedPool {
    fn from_candidates(dim: usize, items: &[CandidateSpec]) -> Self {
        let candidate_count = items.len();
        let mut baseline_scores = Vec::with_capacity(candidate_count);
        let mut cand_vec_data = Vec::with_capacity(candidate_count * dim);
        let mut candidate_root_offsets = vec![0_u32];
        let mut candidate_path_offsets = vec![0_u32];
        let mut path_tag_offsets = vec![0_u32];
        let mut path_tags = Vec::new();
        let mut path_positions = Vec::new();

        for (score, vec, paths) in items {
            baseline_scores.push(*score);
            cand_vec_data.extend_from_slice(vec);
            candidate_root_offsets.push(0);
            for path in paths {
                for &(node, pos) in path {
                    path_tags.push(node);
                    path_positions.push(pos);
                }
                path_tag_offsets.push(path_tags.len() as u32);
            }
            candidate_path_offsets.push((path_tag_offsets.len() - 1) as u32);
        }

        Self {
            candidate_count,
            baseline_scores: baseline_scores.clone(),
            union_scores: baseline_scores,
            union_ranks: (0..candidate_count as u32).collect(),
            bm25_scores: vec![0.0; candidate_count],
            anchor_scores: vec![0.0; candidate_count],
            curve_document_ids: (0..candidate_count as u32).map(DocumentId).collect(),
            cand_vec_data,
            dim,
            candidate_root_offsets,
            candidate_roots: Vec::new(),
            candidate_path_offsets,
            path_tag_offsets,
            path_tags,
            path_positions,
        }
    }

    fn view(&self) -> CandidatePoolView<'_> {
        CandidatePoolView {
            candidate_count: self.candidate_count,
            baseline_scores: &self.baseline_scores,
            union_scores: &self.union_scores,
            union_ranks: &self.union_ranks,
            bm25_scores: &self.bm25_scores,
            bm25_reference: 1.0,
            anchor_scores: &self.anchor_scores,
            curve_document_ids: &self.curve_document_ids,
            candidate_vectors: VectorMatrixView::new(
                &self.cand_vec_data,
                self.candidate_count,
                self.dim,
            )
            .unwrap(),
            candidate_root_offsets: &self.candidate_root_offsets,
            candidate_roots: &self.candidate_roots,
            candidate_path_offsets: &self.candidate_path_offsets,
            path_tag_offsets: &self.path_tag_offsets,
            path_tags: &self.path_tags,
            path_positions: &self.path_positions,
        }
    }
}

#[test]
fn trusted_field_curve_scoring_and_aux_identity_boost() {
    let dim = 8;
    let mut tag_data = Vec::new();
    for i in 0..5 {
        tag_data.extend_from_slice(&unit_vec(dim, i));
    }
    let tag_vectors = VectorMatrixView::new(&tag_data, 5, dim).unwrap();
    let cfg_field = DtscFieldConfig::default();
    let field = build_test_field(
        &[
            DtscNodeInput {
                node: NodeId(0),
                energy: 1.0,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(1),
                energy: 0.9,
                normalized_energy: 0.0,
                source_type: "core".into(),
            },
            DtscNodeInput {
                node: NodeId(2),
                energy: 0.85,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(3),
                energy: 0.8,
                normalized_energy: 0.0,
                source_type: "emergent".into(),
            },
        ],
        tag_vectors,
        &cfg_field,
    );
    assert!(field.field_trusted);

    let graph = OwnedGraph::from_edges(
        5,
        &[(0, 1, 0.8), (1, 2, 0.7), (2, 3, 0.6)],
        vec![0.1, 0.1, 0.1, 0.2, 0.5],
        vec![1.2, 1.1, 1.0, 1.0, 1.0],
    );

    // Candidate 0 has lower knn score (0.50) but strong direct path matching query field.
    // Candidate 1 has higher knn score (0.54) but weak / irrelevant tag 4.
    let cand0_vec = mix_vec(dim, 0, 0.6, 1, 0.6);
    let cand1_vec = unit_vec(dim, 4);
    let pool = OwnedPool::from_candidates(
        dim,
        &[
            (
                0.50,
                cand0_vec,
                vec![vec![
                    (NodeId(0), 1),
                    (NodeId(1), 2),
                    (NodeId(2), 3),
                    (NodeId(3), 4),
                ]],
            ),
            (0.54, cand1_vec, vec![vec![(NodeId(4), 1)]]),
        ],
    );

    let query = mix_vec(dim, 0, 0.7, 1, 0.7);
    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::default();

    let cfg = DtscConfig {
        auxiliary: DtscAuxConfig {
            enabled: true,
            identity_anchor: DtscIdentityConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let out = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg,
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &query,
                enhanced_query: &query,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();

    assert!(out.field_trusted);
    assert!(!out.fallback_used);
    assert_eq!(out.fallback_reason, None);
    assert_eq!(out.results.len(), 2);
    // Candidate 0 boosted past Candidate 1.
    assert_eq!(out.results[0].candidate, CandidateId(0));
    assert_eq!(out.results[0].geo_effect, "boost");
    assert_eq!(out.results[0].geo_evidence_class, "direct");
    assert!(out.results[0].geo_reward_eligible);
    assert!(out.results[0].geo_bonus > 0.0);
    assert!(out.results[0].score > out.results[0].original_knn_score);
    assert!(out.results[0].score > out.results[1].score);
    assert_eq!(out.results[1].candidate, CandidateId(1));
    assert_eq!(out.results[1].geo_effect, "neutral");
}

#[test]
fn sparse_association_relief_and_edge_gates_effect() {
    let dim = 8;
    // Build vectors:
    // nodes 0, 2, 4 share a semantic direction (cosine >= 0.48) with chunk vector;
    // nodes 1, 3 are orthogonal cold gaps separating 0, 2, 4 so all 3 contacts are isolated.
    let v0 = mix_vec(dim, 0, 0.9, 5, 0.4);
    let v1 = unit_vec(dim, 6);
    let v2 = mix_vec(dim, 0, 0.85, 5, 0.5);
    let v3 = unit_vec(dim, 7);
    let v4 = mix_vec(dim, 0, 0.88, 5, 0.45);
    let mut tag_data = Vec::new();
    for v in [&v0, &v1, &v2, &v3, &v4] {
        tag_data.extend_from_slice(v);
    }
    let tag_vectors = VectorMatrixView::new(&tag_data, 5, dim).unwrap();
    let cfg_field = DtscFieldConfig::default();
    let field = build_test_field(
        &[
            DtscNodeInput {
                node: NodeId(0),
                energy: 1.0,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(2),
                energy: 0.9,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(4),
                energy: 0.85,
                normalized_energy: 0.0,
                source_type: "core".into(),
            },
        ],
        tag_vectors,
        &cfg_field,
    );
    assert!(field.field_trusted);

    // Non-adjacent edges (0->2, 2->4, 0->4) provide sparse-association conductance.
    let graph = OwnedGraph::from_edges(
        5,
        &[(0, 2, 0.8), (2, 4, 0.7), (0, 4, 0.6)],
        vec![0.1; 5],
        vec![1.0; 5],
    );
    let chunk = unit_vec(dim, 0);
    let pool = OwnedPool::from_candidates(
        dim,
        &[(
            0.5,
            chunk.clone(),
            vec![vec![
                (NodeId(0), 1),
                (NodeId(1), 2),
                (NodeId(2), 3),
                (NodeId(3), 4),
                (NodeId(4), 5),
            ]],
        )],
    );

    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::default();

    let cfg_sparse_on = DtscConfig::default();
    let mut cfg_sparse_off = DtscConfig::default();
    cfg_sparse_off.sparse.enabled = false;

    let out_on = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_sparse_on,
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &chunk,
                enhanced_query: &chunk,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();

    let out_off = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_sparse_off,
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &chunk,
                enhanced_query: &chunk,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();

    // Sparse relief reduces geo_isolated_ratio and increases confidence/geo_score.
    assert!(out_on.results[0].geo_isolated_ratio < out_off.results[0].geo_isolated_ratio);
    assert!(out_on.results[0].geo_score > out_off.results[0].geo_score);

    // Edge gates = all 1.0 must match un-gated 0 bits; edge gates = all 0.0 blocks sparse relief.
    let ones = vec![1.0; graph.weights.len()];
    let gated_ones = graph.view().with_edge_gates(&ones).unwrap();
    let out_ones = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_sparse_on,
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: gated_ones,
                tag_vectors,
                original_query: &chunk,
                enhanced_query: &chunk,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert_eq!(
        out_ones.results[0].score.to_bits(),
        out_on.results[0].score.to_bits()
    );
    assert_eq!(
        out_ones.results[0].geo_isolated_ratio.to_bits(),
        out_on.results[0].geo_isolated_ratio.to_bits()
    );

    let zeros = vec![0.0; graph.weights.len()];
    let gated_zeros = graph.view().with_edge_gates(&zeros).unwrap();
    let out_zeros = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_sparse_on,
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: gated_zeros,
                tag_vectors,
                original_query: &chunk,
                enhanced_query: &chunk,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert_eq!(
        out_zeros.results[0].geo_isolated_ratio.to_bits(),
        out_off.results[0].geo_isolated_ratio.to_bits()
    );
}

#[test]
fn low_trust_fallback_restores_knn_scores_and_preserves_input_order() {
    let dim = 4;
    let mut tag_data = Vec::new();
    for i in 0..4 {
        tag_data.extend_from_slice(&unit_vec(dim, i));
    }
    let tag_vectors = VectorMatrixView::new(&tag_data, 4, dim).unwrap();
    // Only 1 node -> field_trusted == false (< min_field_tags = 3).
    let untrusted_field = build_test_field(
        &[DtscNodeInput {
            node: NodeId(0),
            energy: 1.0,
            normalized_energy: 0.0,
            source_type: "seed".into(),
        }],
        tag_vectors,
        &DtscFieldConfig::default(),
    );
    assert!(!untrusted_field.field_trusted);

    let graph = OwnedGraph::from_edges(4, &[], vec![0.0; 4], vec![1.0; 4]);
    // Deliberately unsorted baseline scores: [0.30, 0.90, 0.60].
    let pool = OwnedPool::from_candidates(
        dim,
        &[
            (0.30, unit_vec(dim, 0), vec![vec![(NodeId(0), 1)]]),
            (0.90, unit_vec(dim, 1), vec![vec![(NodeId(1), 1)]]),
            (0.60, unit_vec(dim, 2), vec![vec![(NodeId(2), 1)]]),
        ],
    );

    let q = unit_vec(dim, 0);
    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::default();
    let out = DtscReadoutOperator
        .execute(
            &ctx,
            &DtscConfig::default(),
            DtscReadoutInput {
                field: &untrusted_field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();

    assert!(!out.field_trusted);
    assert!(out.fallback_used);
    assert_eq!(out.fallback_reason, Some("query-energy-field-low-trust"));
    // Input order [0, 1, 2] MUST be preserved on fallback!
    assert_eq!(out.results[0].candidate, CandidateId(0));
    assert_eq!(out.results[1].candidate, CandidateId(1));
    assert_eq!(out.results[2].candidate, CandidateId(2));
    for item in &out.results {
        assert_eq!(item.score.to_bits(), item.original_knn_score.to_bits());
        assert_eq!(item.geo_bonus, 0.0);
        assert_eq!(item.geo_base_bonus, 0.0);
        assert_eq!(item.geo_aux_bonus, 0.0);
        assert_eq!(item.geo_effect, "neutral");
    }

    // Trusted field, but candidate has no contacts -> "no-candidate-curve-contacted-query-field".
    let trusted_field = build_test_field(
        &[
            DtscNodeInput {
                node: NodeId(0),
                energy: 1.0,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(1),
                energy: 0.9,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(2),
                energy: 0.8,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
        ],
        tag_vectors,
        &DtscFieldConfig::default(),
    );
    let cold_pool = OwnedPool::from_candidates(
        dim,
        &[
            (0.25, unit_vec(dim, 3), vec![vec![(NodeId(3), 1)]]),
            (0.75, unit_vec(dim, 3), vec![]),
        ],
    );
    let out_cold = DtscReadoutOperator
        .execute(
            &ctx,
            &DtscConfig::default(),
            DtscReadoutInput {
                field: &trusted_field,
                candidates: cold_pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert!(out_cold.fallback_used);
    assert_eq!(
        out_cold.fallback_reason,
        Some("no-candidate-curve-contacted-query-field")
    );
    assert_eq!(out_cold.results[0].candidate, CandidateId(0));
    assert_eq!(out_cold.results[1].candidate, CandidateId(1));
}

#[test]
fn multipath_csr_isolation_best_path_and_top_k_truncation() {
    let dim = 6;
    let mut tag_data = Vec::new();
    for i in 0..6 {
        tag_data.extend_from_slice(&unit_vec(dim, i));
    }
    let tag_vectors = VectorMatrixView::new(&tag_data, 6, dim).unwrap();
    let field = build_test_field(
        &[
            DtscNodeInput {
                node: NodeId(0),
                energy: 1.0,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(1),
                energy: 0.9,
                normalized_energy: 0.0,
                source_type: "core".into(),
            },
            DtscNodeInput {
                node: NodeId(2),
                energy: 0.85,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
        ],
        tag_vectors,
        &DtscFieldConfig::default(),
    );
    let graph = OwnedGraph::from_edges(6, &[(0, 1, 0.9), (1, 2, 0.8)], vec![0.1; 6], vec![1.0; 6]);

    let cand_vec = mix_vec(dim, 0, 0.6, 1, 0.6);
    let weak_path = vec![(NodeId(4), 1), (NodeId(5), 2)];
    let strong_path = vec![(NodeId(0), 1), (NodeId(1), 2), (NodeId(2), 3)];

    // Candidate 0: [weak, strong]
    // Candidate 1: [strong] alone
    // Candidate 2: [weak] alone
    // Candidate 3: split single-tag paths [(NodeId(0), 1)] and [(NodeId(1), 1)] (must NOT mix!).
    let pool = OwnedPool::from_candidates(
        dim,
        &[
            (0.50, cand_vec.clone(), vec![weak_path, strong_path.clone()]),
            (0.50, cand_vec.clone(), vec![strong_path]),
            (0.45, unit_vec(dim, 4), vec![vec![(NodeId(4), 1)]]),
            (
                0.50,
                cand_vec.clone(),
                vec![vec![(NodeId(0), 1)], vec![(NodeId(1), 1)]],
            ),
        ],
    );

    let q = mix_vec(dim, 0, 0.7, 1, 0.7);
    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::default();

    let out_all = DtscReadoutOperator
        .execute(
            &ctx,
            &DtscConfig::default(),
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();

    assert_eq!(out_all.results.len(), 4);
    let item0 = out_all
        .results
        .iter()
        .find(|r| r.candidate == CandidateId(0))
        .unwrap();
    let item1 = out_all
        .results
        .iter()
        .find(|r| r.candidate == CandidateId(1))
        .unwrap();
    let item3 = out_all
        .results
        .iter()
        .find(|r| r.candidate == CandidateId(3))
        .unwrap();

    // Candidate 0 picks strong_path and matches Candidate 1 bit-for-bit.
    assert_eq!(item0.score.to_bits(), item1.score.to_bits());
    assert_eq!(item0.geo_score.to_bits(), item1.geo_score.to_bits());
    assert_eq!(item0.geo_hit_count, 3);
    // Candidate 3 evaluates its two 1-tag paths separately (never mixes into 2 hits).
    assert_eq!(item3.geo_hit_count, 1);
    assert!(item0.score > item3.score);

    // Test top_k truncation = 2.
    let cfg_top2 = DtscConfig {
        top_k: 2,
        ..Default::default()
    };
    let out_top2 = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_top2,
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert_eq!(out_top2.results.len(), 2);
    assert_eq!(out_top2.results[0].candidate, CandidateId(0));
    assert_eq!(out_top2.results[1].candidate, CandidateId(1));
}

#[test]
fn config_serde_defaults_deny_unknown_fields_and_normalized() {
    // Empty map deserializes to exact Default values.
    let empty: Vec<(&str, &str)> = Vec::new();
    let de = MapDeserializer::<_, serde::de::value::Error>::new(empty.into_iter());
    let cfg = DtscConfig::deserialize(de).unwrap();
    assert_eq!(cfg.field.min_field_tags, 3);
    assert_eq!(cfg.curve.strong_contact_threshold, 0.16);
    assert_eq!(cfg.sparse.pair_saturation, 3);
    assert_eq!(cfg.reward.alpha, 0.35);
    assert_eq!(cfg.trust.min_strong_evidence, 1.0);
    assert!(!cfg.auxiliary.enabled);
    assert!(!cfg.auxiliary.identity_anchor.enabled);
    assert_eq!(cfg.top_k, 0);

    // Unknown field is rejected (`deny_unknown_fields`).
    let unknown = vec![("unknown_field_xyz", 1_u64)];
    let de_unknown = MapDeserializer::<_, serde::de::value::Error>::new(
        unknown
            .into_iter()
            .map(|(k, v)| (StrDeserializer::new(k), v.into_deserializer())),
    );
    assert!(DtscConfig::deserialize(de_unknown).is_err());

    // normalized() clamps cross-field invariants accurately.
    let mut bad = DtscConfig::default();
    bad.curve.strong_contact_threshold = 0.2;
    bad.curve.weak_contact_threshold = 0.9;
    bad.curve.position_decay = -2.0;
    bad.curve.min_geo_samples = 0;
    bad.sparse.min_contacts = 0;
    bad.sparse.max_relief = 5.0;
    bad.reward.direct_bonus_cap = 0.1;
    bad.reward.structural_bonus_cap = 0.5;
    bad.reward.thematic_bonus_cap = 0.5;
    bad.auxiliary.max_aux_bonus = 0.2;
    bad.auxiliary.identity_anchor.floor_exponent = 10.0;

    let norm = bad.normalized();
    assert_eq!(norm.curve.weak_contact_threshold, 0.2);
    assert_eq!(norm.curve.position_decay, 0.0);
    assert_eq!(norm.curve.min_geo_samples, 1);
    assert_eq!(norm.sparse.min_contacts, 2);
    assert_eq!(norm.sparse.max_relief, 0.8);
    assert_eq!(norm.reward.structural_bonus_cap, 0.1);
    assert_eq!(norm.reward.thematic_bonus_cap, 0.1);
    assert_eq!(norm.auxiliary.max_aux_bonus, 0.05);
    assert_eq!(norm.auxiliary.identity_anchor.floor_exponent, 4.0);
}

#[test]
fn readout_validates_inputs_and_cancellation() {
    let dim = 4;
    let tag_data = vec![1.0_f32, 0.0, 0.0, 0.0];
    let tag_vectors = VectorMatrixView::new(&tag_data, 1, dim).unwrap();
    let field = DtscField::default();
    let graph = OwnedGraph::from_edges(1, &[], vec![0.0], vec![1.0]);
    let pool = OwnedPool::from_candidates(dim, &[]);
    let q = vec![1.0_f32; dim];
    let mut scratch = ScratchPad::default();

    // Cancelled context.
    let cancel_flag = AtomicBool::new(true);
    let cancelled_ctx = ExecutionContext {
        cancel: CancellationToken::from_atomic(&cancel_flag),
        ..Default::default()
    };
    let err = DtscReadoutOperator
        .execute(
            &cancelled_ctx,
            &DtscConfig::default(),
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap_err();
    assert!(matches!(err, VcpError::Cancelled { .. }));

    // Zero dimension.
    let zero_tags = VectorMatrixView::empty(0);
    let zero_pool = OwnedPool::from_candidates(0, &[]);
    let ctx = ExecutionContext::default();
    let err_zero = DtscReadoutOperator
        .execute(
            &ctx,
            &DtscConfig::default(),
            DtscReadoutInput {
                field: &field,
                candidates: zero_pool.view(),
                graph: graph.view(),
                tag_vectors: zero_tags,
                original_query: &[],
                enhanced_query: &[],
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap_err();
    assert!(matches!(err_zero, VcpError::InvalidInput { .. }));

    // Mismatched query length.
    let err_dim = DtscReadoutOperator
        .execute(
            &ctx,
            &DtscConfig::default(),
            DtscReadoutInput {
                field: &field,
                candidates: pool.view(),
                graph: graph.view(),
                tag_vectors,
                original_query: &[1.0_f32; 2],
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap_err();
    assert!(matches!(
        err_dim,
        VcpError::DimensionMismatch {
            expected: 4,
            actual: 2,
            ..
        }
    ));
}

#[test]
fn low_trust_coverage_and_spread_fallback_and_bypass() {
    let dim = 6;
    let mut tag_data = Vec::new();
    for i in 0..6 {
        tag_data.extend_from_slice(&unit_vec(dim, i));
    }
    let tag_vectors = VectorMatrixView::new(&tag_data, 6, dim).unwrap();
    let field = build_test_field(
        &[
            DtscNodeInput {
                node: NodeId(0),
                energy: 1.0,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(1),
                energy: 0.9,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
            DtscNodeInput {
                node: NodeId(2),
                energy: 0.85,
                normalized_energy: 0.0,
                source_type: "seed".into(),
            },
        ],
        tag_vectors,
        &DtscFieldConfig::default(),
    );
    let graph = OwnedGraph::from_edges(6, &[], vec![0.1; 6], vec![1.0; 6]);
    let q = unit_vec(dim, 0);
    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::default();

    // 1 of 3 candidates has weak non-strong contact -> coverage (1/3 < 0.5) -> jointly low trust.
    let mut cfg_low = DtscConfig::default();
    cfg_low.field.field_similarity_threshold = 0.1;
    cfg_low.trust.min_geo_score = 0.5;
    cfg_low.trust.min_strong_evidence = 1.0;
    cfg_low.trust.min_candidate_coverage = 0.5;

    let pool_cov = OwnedPool::from_candidates(
        dim,
        &[
            (0.4, unit_vec(dim, 4), vec![vec![(NodeId(4), 1)]]),
            (0.8, unit_vec(dim, 5), vec![vec![(NodeId(5), 1)]]),
            (0.6, unit_vec(dim, 5), vec![vec![(NodeId(5), 1)]]),
        ],
    );
    // NodeId(4) has cosine 0.60 with NodeId(0), giving weak interpolated potential (~0.108 >= 0.06, < 0.16).
    let mut tag_data_cov = tag_data.clone();
    let near0 = mix_vec(dim, 0, 0.6, 4, 0.8);
    tag_data_cov[4 * dim..5 * dim].copy_from_slice(&near0);
    let tv_cov = VectorMatrixView::new(&tag_data_cov, 6, dim).unwrap();

    let out_cov = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_low,
            DtscReadoutInput {
                field: &field,
                candidates: pool_cov.view(),
                graph: graph.view(),
                tag_vectors: tv_cov,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert!(out_cov.fallback_used);
    assert_eq!(
        out_cov.fallback_reason,
        Some("candidate-curve-readout-jointly-low-trust")
    );

    // 2 of 2 contributors with identical geo_score (< min_geo_spread) -> lack discrimination.
    let pool_spread = OwnedPool::from_candidates(
        dim,
        &[
            (0.4, unit_vec(dim, 4), vec![vec![(NodeId(4), 1)]]),
            (0.8, unit_vec(dim, 4), vec![vec![(NodeId(4), 1)]]),
        ],
    );
    let out_spread = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_low,
            DtscReadoutInput {
                field: &field,
                candidates: pool_spread.view(),
                graph: graph.view(),
                tag_vectors: tv_cov,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert!(out_spread.fallback_used);
    assert_eq!(
        out_spread.fallback_reason,
        Some("candidate-curve-scores-lack-discrimination")
    );

    // When fallback_to_knn_on_low_trust = false, fallback is bypassed for low coverage/spread.
    cfg_low.field.fallback_to_knn_on_low_trust = false;
    let out_bypass = DtscReadoutOperator
        .execute(
            &ctx,
            &cfg_low,
            DtscReadoutInput {
                field: &field,
                candidates: pool_cov.view(),
                graph: graph.view(),
                tag_vectors: tv_cov,
                original_query: &q,
                enhanced_query: &q,
                geometry: DtscGeometryState::default(),
            },
            &mut scratch,
        )
        .unwrap();
    assert!(!out_bypass.fallback_used);
    assert_eq!(out_bypass.fallback_reason, None);
    // Sorted descending by score on non-fallback.
    assert!(out_bypass.results[0].score >= out_bypass.results[1].score);
}

#[cfg(feature = "schemars")]
#[test]
fn schemars_generates_dtsc_config_schema() {
    let schema = schemars::schema_for!(DtscConfig);
    assert!(schema.as_object().is_some());
}
