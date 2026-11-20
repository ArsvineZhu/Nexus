// SPDX-License-Identifier: Apache-2.0

use nexus_math::nexus_core::{
    CancellationToken, CandidateId, CandidateRowsView, CsrGraph, CsrStorage, ExecutionContext,
    NodeId, PathId, QueryCosines, TransportGraph, TransportGraphView, VcpError, VcpResult,
    VectorMatrixView,
};
use nexus_math::{
    DtscAuxConfig, DtscConfig, DtscDecision, DtscEvidenceConfig, DtscEvidenceInput,
    DtscEvidenceValues, DtscFallbackOrder, DtscFallbackReason, DtscFieldBuildInput,
    DtscFieldConfig, DtscFieldSampler, DtscFieldValues, DtscGeometryState, DtscIdentityConfig,
    DtscNodeInput, DtscOutput, DtscReadoutConfig, DtscRewardConfig, DtscSourceType,
    DtscTrustConfig, build_dtsc_field, dtsc_cosine, prepare_dtsc_evidence, readout_dtsc,
};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};

struct CandidateRows {
    matrix: Vec<f32>,
    dim: usize,
    document_ids: Vec<u64>,
    root_offsets: Vec<u32>,
    candidate_path_offsets: Vec<u32>,
    path_tag_offsets: Vec<u32>,
    path_tags: Vec<NodeId>,
    path_positions: Vec<u32>,
}

type Steps = Vec<(NodeId, u32)>;
type Candidate = (Vec<f32>, Vec<Steps>);

impl CandidateRows {
    fn new(dim: usize, items: &[Candidate]) -> Self {
        let mut rows = Self {
            matrix: Vec::new(),
            dim,
            document_ids: (0..items.len() as u64).collect(),
            root_offsets: vec![0; items.len() + 1],
            candidate_path_offsets: vec![0],
            path_tag_offsets: vec![0],
            path_tags: Vec::new(),
            path_positions: Vec::new(),
        };
        for (vector, paths) in items {
            rows.matrix.extend_from_slice(vector);
            for path in paths {
                for &(node, position) in path {
                    rows.path_tags.push(node);
                    rows.path_positions.push(position);
                }
                rows.path_tag_offsets.push(rows.path_tags.len() as u32);
            }
            rows.candidate_path_offsets
                .push((rows.path_tag_offsets.len() - 1) as u32);
        }
        rows
    }

    fn view(&self) -> CandidateRowsView<'_> {
        CandidateRowsView {
            candidate_count: self.document_ids.len(),
            document_ids: &self.document_ids,
            candidate_vectors: VectorMatrixView::new(
                &self.matrix,
                self.document_ids.len(),
                self.dim,
            )
            .unwrap(),
            candidate_root_offsets: &self.root_offsets,
            candidate_roots: &[],
            candidate_path_offsets: &self.candidate_path_offsets,
            path_tag_offsets: &self.path_tag_offsets,
            path_tags: &self.path_tags,
            path_positions: &self.path_positions,
        }
    }
}

fn graph(
    node_count: usize,
    edges: &[(usize, usize, f64)],
    inbound: Vec<f64>,
    anchor_gains: Vec<f64>,
) -> TransportGraph {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(source, target, weight) in edges {
        adjacency[source].push((NodeId(target as u32), weight));
    }
    let mut row_offsets = vec![0];
    let mut targets = Vec::new();
    let mut weights = Vec::new();
    for row in &mut adjacency {
        row.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(target, weight) in row.iter() {
            targets.push(target);
            weights.push(weight);
        }
        row_offsets.push(targets.len());
    }
    let max_inbound = inbound.iter().copied().fold(0.0, f64::max);
    let csr = CsrGraph::new(CsrStorage {
        node_count,
        row_offsets,
        targets,
        is_static_wormhole: vec![false; weights.len()],
        weights,
        inbound,
        max_inbound,
        anchor_gains,
        target_sorted_indices: None,
    })
    .unwrap();
    TransportGraph::new(csr).unwrap()
}

fn empty_graph(node_count: usize) -> TransportGraph {
    graph(
        node_count,
        &[],
        vec![0.0; node_count],
        vec![1.0; node_count],
    )
}

fn unit_vector(dim: usize, hot_index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[hot_index] = 1.0;
    vector
}

fn path(tags: &[u32]) -> Steps {
    tags.iter()
        .enumerate()
        .map(|(i, &node)| (NodeId(node), i as u32 + 1))
        .collect()
}

fn seeds(values: &[(u32, f64)]) -> Vec<DtscNodeInput> {
    values
        .iter()
        .map(|&(node, energy)| DtscNodeInput {
            node: NodeId(node),
            mass: energy,

            source_type: DtscSourceType::Seed,
        })
        .collect()
}

fn build_field(
    nodes: &[DtscNodeInput],
    tag_vectors: VectorMatrixView<'_, f32>,
    config: &DtscFieldConfig,
) -> DtscFieldValues {
    build_dtsc_field(
        DtscFieldBuildInput { nodes, tag_vectors },
        config,
        &ExecutionContext::default(),
    )
    .unwrap()
}

fn evidence(
    field: &DtscFieldValues,
    query: &QueryCosines<'_, '_>,
    graph: TransportGraphView<'_>,
    tag_vectors: VectorMatrixView<'_, f32>,
    config: &DtscEvidenceConfig,
) -> VcpResult<DtscEvidenceValues> {
    prepare_dtsc_evidence(
        DtscEvidenceInput {
            field,
            query,
            graph,
            tag_vectors,
            geometry: DtscGeometryState::default(),
        },
        config,
        &ExecutionContext::default(),
    )
}

fn readout(evidence: &DtscEvidenceValues, config: &DtscReadoutConfig) -> DtscOutput {
    readout_dtsc(evidence, config, &ExecutionContext::default()).unwrap()
}

#[test]
fn evidence_is_candidate_local_and_fallback_uses_the_entire_pool() {
    let tags = VectorMatrixView::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 2).unwrap();
    let nodes = seeds(&[(0, 1.0), (1, 1.0), (2, 1.0)]);
    let field = build_field(&nodes, tags, &DtscFieldConfig::default());
    close("entropy", field.field_entropy(), 1.0);
    let graph = empty_graph(3);
    let contacted = (vec![1.0, 0.0], vec![path(&[0])]);
    let no_path = (vec![0.0, 1.0], vec![]);
    let a = CandidateRows::new(2, std::slice::from_ref(&contacted));
    let b = CandidateRows::new(2, &[no_path.clone(), contacted]);
    let ctx = ExecutionContext::default();
    let q = [1.0, 0.0];
    let a_query = QueryCosines::prepare(&ctx, a.view(), &q, &q).unwrap();
    let b_query = QueryCosines::prepare(&ctx, b.view(), &q, &q).unwrap();
    let scoring = DtscEvidenceConfig::default();
    let a_evidence = evidence(&field, &a_query, graph.view(), tags, &scoring).unwrap();
    let b_evidence = evidence(&field, &b_query, graph.view(), tags, &scoring).unwrap();
    let a_contact = &a_evidence.items()[0];
    let b_contact = &b_evidence.items()[1];
    assert_eq!(
        a_contact.scores.geo_score.to_bits(),
        b_contact.scores.geo_score.to_bits()
    );
    assert_eq!(
        a_contact.scores.geo_bonus.to_bits(),
        b_contact.scores.geo_bonus.to_bits()
    );
    assert_eq!(b_contact.chosen_path, Some(PathId(0)));
    assert!(b_contact.scores.geo_bonus > 0.0);

    let mut config = DtscReadoutConfig {
        trust: DtscTrustConfig {
            min_geo_score: 1.0,
            min_strong_evidence: 100.0,
            min_candidate_coverage: 0.75,
            ..Default::default()
        },
        ..Default::default()
    };
    let alone = readout(&a_evidence, &config);
    assert_eq!(alone.decision, DtscDecision::Scored);
    assert!(alone.results[0].scores.score > 1.0);
    let mixed = readout(&b_evidence, &config);
    assert_eq!(
        mixed.decision,
        DtscDecision::Fallback {
            reason: DtscFallbackReason::InsufficientCandidateCoverage,
            order: DtscFallbackOrder::OriginalCosine,
        }
    );
    assert_eq!(
        mixed
            .results
            .iter()
            .map(|item| item.scores.score)
            .collect::<Vec<_>>(),
        [1.0, 0.0]
    );
    for item in &mixed.results {
        assert_eq!(item.scores.score, item.scores.original_cosine);
        assert_eq!(item.scores.geo_bonus, 0.0);
        assert_eq!(item.scores.geo_base_bonus, 0.0);
        assert_eq!(item.scores.geo_aux_bonus, 0.0);
        assert_eq!(item.scores.geo_effect, "neutral");
    }
    assert!(b_evidence.items()[1].scores.geo_bonus > 0.0);

    config.top_k = 1;
    let dense_fallback = readout(&b_evidence, &config);
    assert_eq!(dense_fallback.results[0].candidate, CandidateId(1));
    config.fallback_order = DtscFallbackOrder::InputOrder;
    let host_fallback = readout(&b_evidence, &config);
    assert_eq!(
        host_fallback.decision,
        DtscDecision::Fallback {
            reason: DtscFallbackReason::InsufficientCandidateCoverage,
            order: DtscFallbackOrder::InputOrder,
        }
    );
    assert_eq!(host_fallback.results[0].candidate, CandidateId(0));

    config.fallback_on_low_trust = false;
    config.fallback_order = DtscFallbackOrder::OriginalCosine;
    let first = readout(&b_evidence, &config);
    assert_eq!(first.decision, DtscDecision::Scored);
    assert_eq!(first.available, 2);
    assert_eq!(first.results.len(), 1);
    assert_eq!(first.results[0].candidate, CandidateId(1));
    close(
        "default weighted score",
        first.results[0].scores.score,
        b_contact.scores.original_cosine + 0.15 * b_contact.scores.geo_bonus,
    );

    let short = build_field(&nodes[..1], tags, &DtscFieldConfig::default());
    let short_evidence = evidence(&short, &a_query, graph.view(), tags, &scoring).unwrap();
    assert_eq!(
        readout(&short_evidence, &config).decision,
        DtscDecision::Fallback {
            reason: DtscFallbackReason::InsufficientFieldSupport,
            order: DtscFallbackOrder::OriginalCosine,
        }
    );
    let cold = CandidateRows::new(2, &[no_path]);
    let cold_query = QueryCosines::prepare(&ctx, cold.view(), &q, &q).unwrap();
    let cold_evidence = evidence(&field, &cold_query, graph.view(), tags, &scoring).unwrap();
    assert_eq!(
        readout(&cold_evidence, &config).decision,
        DtscDecision::Fallback {
            reason: DtscFallbackReason::NoCurveContact,
            order: DtscFallbackOrder::OriginalCosine,
        }
    );
}

#[test]
fn identity_auxiliary_adds_the_expected_reward() {
    let tags = VectorMatrixView::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 2).unwrap();
    let field = build_field(
        &seeds(&[(0, 1.0), (1, 1.0), (2, 1.0)]),
        tags,
        &DtscFieldConfig::default(),
    );
    let graph = empty_graph(3);
    let rows = CandidateRows::new(2, &[(vec![1.0, 0.0], vec![path(&[0])])]);
    let q = [1.0, 0.0];
    let query = QueryCosines::prepare(&ExecutionContext::default(), rows.view(), &q, &q).unwrap();
    let mut config = DtscEvidenceConfig {
        reward: DtscRewardConfig {
            alpha: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let disabled = evidence(&field, &query, graph.view(), tags, &config).unwrap();
    let baseline = &disabled.items()[0];
    assert_eq!(baseline.scores.geo_base_bonus, 0.0);
    assert_eq!(baseline.scores.geo_aux_bonus, 0.0);
    assert_eq!(baseline.scores.auxiliary_evidence, None);
    config.auxiliary = Some(DtscAuxConfig::default());
    let geometry = evidence(&field, &query, graph.view(), tags, &config).unwrap();
    let item = &geometry.items()[0];
    let auxiliary = item.scores.auxiliary_evidence.unwrap();
    assert!(auxiliary.geometry_floor > 0.0);
    assert_eq!(auxiliary.identity_floor, 0.0);
    close(
        "geometry-only auxiliary bonus",
        item.scores.geo_aux_bonus,
        auxiliary.geometry_floor,
    );

    config.auxiliary = Some(DtscAuxConfig {
        identity_anchor: Some(DtscIdentityConfig::default()),
        ..Default::default()
    });
    let enabled = evidence(&field, &query, graph.view(), tags, &config).unwrap();
    let result = readout(
        &enabled,
        &DtscReadoutConfig {
            geometry_weight: 1.0,
            ..Default::default()
        },
    );
    let item = &result.results[0];
    let auxiliary = item.scores.auxiliary_evidence.unwrap();
    assert!(auxiliary.shadow_score > 0.0);
    close("identity floor", auxiliary.identity_floor, 0.018);
    close("auxiliary bonus", item.scores.geo_aux_bonus, 0.018);
    close("score", item.scores.score, 1.018);
    assert!(item.scores.geo_bonus <= config.reward.direct_bonus_cap);
}

#[test]
fn inferred_contact_contributes_structure_without_a_direct_anchor() {
    let tags = VectorMatrixView::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 2).unwrap();
    let mut nodes = seeds(&[(0, 1.0), (1, 1.0), (2, 1.0)]);
    let graph = empty_graph(3);
    let rows = CandidateRows::new(2, &[(vec![1.0, 0.0], vec![path(&[0])])]);
    let q = [1.0, 0.0];
    let query = QueryCosines::prepare(&ExecutionContext::default(), rows.view(), &q, &q).unwrap();
    let score = |nodes: &[DtscNodeInput]| {
        let field = build_field(nodes, tags, &DtscFieldConfig::default());
        evidence(
            &field,
            &query,
            graph.view(),
            tags,
            &DtscEvidenceConfig::default(),
        )
        .unwrap()
        .items()[0]
            .scores
            .clone()
    };

    let direct = score(&nodes);
    nodes[0].source_type = DtscSourceType::Inferred;
    let inferred = score(&nodes);

    assert_eq!(direct.geo_evidence_class, "direct");
    assert_eq!(inferred.geo_evidence_class, "structural");
    assert_eq!(inferred.geo_exact_hits, direct.geo_exact_hits);
    assert!(inferred.geo_bonus > 0.0 && inferred.geo_bonus < direct.geo_bonus);
}

#[test]
fn sparse_association_uses_effective_transport_edges() {
    let directions = [
        [0.9, 0.4, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.85, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.88, 0.45, 0.0, 0.0],
    ]
    .concat();
    let tags = VectorMatrixView::new(&directions, 5, 4).unwrap();
    let field = build_field(
        &seeds(&[(0, 1.0), (2, 0.9), (4, 0.85)]),
        tags,
        &DtscFieldConfig::default(),
    );
    let graph = graph(
        5,
        &[(0, 2, 0.5), (2, 4, 0.7), (0, 4, 0.5)],
        vec![0.1; 5],
        vec![1.0; 5],
    );
    let q = unit_vector(4, 0);
    let rows = CandidateRows::new(4, &[(q.clone(), vec![path(&[0, 1, 2, 3, 4])])]);
    let query = QueryCosines::prepare(&ExecutionContext::default(), rows.view(), &q, &q).unwrap();
    let mut config = DtscEvidenceConfig::default();
    let with_relief = evidence(&field, &query, graph.view(), tags, &config).unwrap();
    let zeros = [0.0; 3];
    let gated = graph.view().with_edge_gates(&zeros).unwrap();
    let with_gate = evidence(&field, &query, gated, tags, &config).unwrap();
    config.sparse.enabled = false;
    let without_relief = evidence(&field, &query, graph.view(), tags, &config).unwrap();
    let on = &with_relief.items()[0];
    let off = &without_relief.items()[0];
    assert!(on.scores.geo_isolated_ratio < off.scores.geo_isolated_ratio);
    assert!(on.scores.geo_score > off.scores.geo_score);
    assert_eq!(
        off.scores.geo_isolated_ratio.to_bits(),
        with_gate.items()[0].scores.geo_isolated_ratio.to_bits()
    );
}

#[test]
fn full_path_scoring_keeps_path_identity_and_stable_ties() {
    let dim = 6;
    let tag_data = (0..dim)
        .flat_map(|i| unit_vector(dim, i))
        .collect::<Vec<_>>();
    let tags = VectorMatrixView::new(&tag_data, dim, dim).unwrap();
    let field = build_field(
        &seeds(&[(0, 1.0), (1, 0.9), (2, 0.85)]),
        tags,
        &DtscFieldConfig::default(),
    );
    let graph = graph(
        dim,
        &[(0, 1, 0.9), (1, 2, 0.8)],
        vec![0.1; dim],
        vec![1.0; dim],
    );
    let q = vec![0.7, 0.7, 0.0, 0.0, 0.0, 0.0];
    let strong = path(&[0, 1, 2]);
    let rows = CandidateRows::new(
        dim,
        &[
            (
                q.clone(),
                vec![path(&[4, 5]), strong.clone(), strong.clone()],
            ),
            (q.clone(), vec![strong]),
            (q.clone(), vec![path(&[0]), path(&[1])]),
        ],
    );
    let query = QueryCosines::prepare(&ExecutionContext::default(), rows.view(), &q, &q).unwrap();
    let raw = evidence(
        &field,
        &query,
        graph.view(),
        tags,
        &DtscEvidenceConfig::default(),
    )
    .unwrap();
    assert_eq!(raw.items()[0].chosen_path, Some(PathId(1)));
    assert_eq!(
        raw.items()[0].scores.score.to_bits(),
        raw.items()[1].scores.score.to_bits()
    );
    assert_eq!(raw.items()[0].scores.geo_hit_count, 3);
    assert_eq!(raw.items()[2].scores.geo_hit_count, 1);
    assert!(raw.items()[0].scores.score > raw.items()[2].scores.score);
    let result = readout(
        &raw,
        &DtscReadoutConfig {
            top_k: 2,
            ..Default::default()
        },
    );
    assert_eq!(
        result
            .results
            .iter()
            .map(|item| item.candidate)
            .collect::<Vec<_>>(),
        [CandidateId(0), CandidateId(1)]
    );
}

#[test]
fn trust_rejects_an_undifferentiated_pool_until_policy_changes() {
    let tags = VectorMatrixView::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 2).unwrap();
    let field = build_field(
        &seeds(&[(0, 1.0), (1, 1.0), (2, 1.0)]),
        tags,
        &DtscFieldConfig::default(),
    );
    let graph = empty_graph(3);
    let row = (vec![1.0, 0.0], vec![path(&[0])]);
    let rows = CandidateRows::new(2, &[row.clone(), row]);
    let q = [1.0, 0.0];
    let ctx = ExecutionContext::default();
    let query = QueryCosines::prepare(&ctx, rows.view(), &q, &q).unwrap();
    let evidence = evidence(
        &field,
        &query,
        graph.view(),
        tags,
        &DtscEvidenceConfig::default(),
    )
    .unwrap();
    let mut config = DtscReadoutConfig {
        trust: DtscTrustConfig {
            min_geo_score: 1.0,
            min_strong_evidence: 100.0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        readout(&evidence, &config).decision,
        DtscDecision::Fallback {
            reason: DtscFallbackReason::InsufficientScoreSpread,
            order: DtscFallbackOrder::OriginalCosine,
        }
    );
    config.fallback_on_low_trust = false;
    assert_eq!(readout(&evidence, &config).decision, DtscDecision::Scored);

    let flag = AtomicBool::new(false);
    let ctx = ExecutionContext {
        cancel: CancellationToken::from_atomic(&flag),
        ..Default::default()
    };
    flag.store(true, Ordering::Relaxed);
    let cancelled = readout_dtsc(&evidence, &config, &ctx);
    assert!(matches!(cancelled, Err(VcpError::Cancelled { .. })));
}

#[test]
fn config_deserialization_and_validation_follow_stage_ownership() {
    let config: DtscConfig = serde_json::from_value(serde_json::json!({
        "scoring": {
            "evidence": {"reward": {"alpha": 0.2}},
            "readout": {"top_k": 7, "trust": {"min_field_entropy": 0.5}}
        }
    }))
    .unwrap();
    config.validate().unwrap();
    assert_eq!(config.scoring.readout.trust.min_field_entropy, 0.5);
    assert_eq!(config.scoring.evidence.reward.alpha, 0.2);
    assert_eq!(config.scoring.readout.top_k, 7);
    assert!(serde_json::from_value::<DtscConfig>(serde_json::json!({"top_k": 7})).is_err());
    let mut invalid_evidence = config.scoring.evidence;
    invalid_evidence.curve.weak_contact_threshold = 0.9;
    assert!(matches!(
        invalid_evidence.validate(),
        Err(VcpError::InvalidInput { .. })
    ));
    let mut invalid_readout = config.scoring.readout;
    invalid_readout.trust.min_strong_evidence = f64::NAN;
    assert!(matches!(
        invalid_readout.validate(),
        Err(VcpError::InvalidInput { .. })
    ));
}

#[cfg(feature = "schemars")]
#[test]
fn config_schema_describes_scoring_inputs() {
    let schema = serde_json::to_value(schemars::schema_for!(DtscConfig)).unwrap();
    let scoring = &schema["$defs"]["DtscScoringConfig"]["properties"];
    assert!(scoring.get("evidence").is_some() && scoring.get("readout").is_some());
}

fn fixture_matrix(input: &Value, dim: usize) -> Vec<f32> {
    let rows: Vec<(usize, Vec<f32>)> =
        serde_json::from_value(input["tag_vectors"].clone()).unwrap();
    let count = rows.iter().map(|(id, _)| *id).max().unwrap_or(0);
    let mut values = vec![0.0; count * dim];
    for (id, vector) in rows {
        values[(id - 1) * dim..id * dim].copy_from_slice(&vector);
    }
    values
}

fn fixture_nodes(input: &Value) -> Vec<DtscNodeInput> {
    input["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| DtscNodeInput {
            node: NodeId((node["id"].as_u64().unwrap() - 1) as u32),
            mass: node["energy"].as_f64().unwrap(),

            source_type: serde_json::from_value(node["sourceType"].clone()).unwrap(),
        })
        .collect()
}

fn fixture_weights(input: &Value, field: &str, count: usize, default: f64) -> Vec<f64> {
    let mut values = vec![default; count];
    for (id, mass) in serde_json::from_value::<Vec<(usize, f64)>>(input[field].clone()).unwrap() {
        values[id - 1] = mass;
    }
    values
}

#[track_caller]
fn close(label: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{label}: {actual} != {expected}"
    );
}

#[test]
fn nous_wave_field_corpus() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/vcp-dtsc-field.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let input = &case["input"];
        let expected = &case["expected"];
        let dim = input["dimension"].as_u64().unwrap() as usize;
        let data = fixture_matrix(input, dim);
        let tags = VectorMatrixView::new(&data, data.len() / dim, dim).unwrap();
        let inbound = fixture_weights(input, "inbound", tags.rows, 0.0);
        let custom = &input["config"];
        let config = DtscFieldConfig {
            min_field_tags: custom["minFieldTags"].as_u64().unwrap_or(3) as usize,
            max_field_nodes: custom["maxFieldNodes"].as_u64().unwrap_or(48) as usize,
            field_energy_mass_ratio: custom["fieldEnergyMassRatio"].as_f64().unwrap_or(0.95),
            ..Default::default()
        };
        let field = build_field(&fixture_nodes(input), tags, &config);
        close(
            "field entropy",
            field.field_entropy(),
            expected["fieldEntropy"].as_f64().unwrap(),
        );
        close(
            "total energy",
            field.total_energy(),
            expected["totalEnergy"].as_f64().unwrap(),
        );
        let retained = expected["fieldNodes"].as_array().unwrap();
        assert_eq!(field.field_nodes().len(), retained.len());
        for (node, expected) in field.field_nodes().iter().zip(retained) {
            assert_eq!(u64::from(node.node.0) + 1, expected["id"].as_u64().unwrap());
            close(
                "retained potential",
                node.potential,
                expected["potential"].as_f64().unwrap(),
            );
        }
        let sampler = DtscFieldSampler::new(&field, &inbound);
        let targets: Vec<(u32, Vec<f32>)> =
            serde_json::from_value(input["sample_tags"].clone()).unwrap();
        for ((id, vector), expected) in targets.iter().zip(expected["samples"].as_array().unwrap())
        {
            let result = sampler.sample(NodeId(id - 1), vector);
            close(
                "sample potential",
                result.potential,
                expected["potential"].as_f64().unwrap(),
            );
            assert_eq!(result.exact, expected["exact"].as_bool().unwrap());
            assert_eq!(
                result.source_type,
                serde_json::from_value::<DtscSourceType>(expected["sourceType"].clone()).unwrap()
            );
        }
        let graph = empty_graph(tags.rows);
        let empty = CandidateRows::new(dim, &[]);
        let zero = vec![0.0; dim];
        let query = QueryCosines::prepare(&ExecutionContext::default(), empty.view(), &zero, &zero)
            .unwrap();
        let raw = evidence(
            &field,
            &query,
            graph.view(),
            tags,
            &DtscEvidenceConfig::default(),
        )
        .unwrap();
        let strict = readout(&raw, &DtscReadoutConfig::default());
        let permissive = readout(
            &raw,
            &DtscReadoutConfig {
                fallback_on_low_trust: false,
                ..Default::default()
            },
        );
        assert_eq!(strict.field, permissive.field);
        let expected_reason = if expected["fieldTrusted"] == false {
            DtscFallbackReason::LowFieldEntropy
        } else {
            DtscFallbackReason::NoCurveContact
        };
        assert_eq!(
            strict.decision,
            DtscDecision::Fallback {
                reason: expected_reason,
                order: DtscFallbackOrder::OriginalCosine,
            }
        );
        assert_eq!(
            permissive.decision,
            DtscDecision::Fallback {
                reason: DtscFallbackReason::NoCurveContact,
                order: DtscFallbackOrder::OriginalCosine,
            }
        );
    }
}

#[test]
fn nous_wave_curve_and_fallback_corpus() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/vcp-dtsc.json")).unwrap();
    let ctx = ExecutionContext::default();
    for case in fixture["cases"].as_array().unwrap() {
        let input = &case["input"];
        let dim = input["dimension"].as_u64().unwrap() as usize;
        if dim == 0 {
            let tags = VectorMatrixView::empty(0);
            let field = build_field(&[], tags, &DtscFieldConfig::default());
            let graph = empty_graph(0);
            let rows = CandidateRows::new(0, &[]);
            let query = QueryCosines::prepare(&ctx, rows.view(), &[], &[]).unwrap();
            let result = evidence(
                &field,
                &query,
                graph.view(),
                tags,
                &DtscEvidenceConfig::default(),
            );
            assert!(matches!(result, Err(VcpError::InvalidInput { .. })));
            continue;
        }

        let data = fixture_matrix(input, dim);
        let tags = VectorMatrixView::new(&data, data.len() / dim, dim).unwrap();
        let edges: Vec<(usize, usize, f64)> =
            serde_json::from_value(input["edges"].clone()).unwrap();
        let edges: Vec<_> = edges
            .into_iter()
            .map(|(a, b, weight)| (a - 1, b - 1, weight))
            .collect();
        let graph = graph(
            tags.rows,
            &edges,
            fixture_weights(input, "inbound", tags.rows, 0.0),
            fixture_weights(input, "anchor_gain", tags.rows, 1.0),
        );
        let candidates = input["candidates"].as_array().unwrap();
        let specs: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                let curve = &candidate["curve"];
                let steps = curve["tags"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|tag| {
                        (
                            NodeId(tag["id"].as_u64().unwrap() as u32 - 1),
                            tag["position"].as_u64().unwrap() as u32,
                        )
                    })
                    .collect();
                (
                    serde_json::from_value(curve["chunk_vector"].clone()).unwrap(),
                    vec![steps],
                )
            })
            .collect();
        let rows = CandidateRows::new(dim, &specs);
        let original: Vec<f32> = serde_json::from_value(input["original_vector"].clone()).unwrap();
        let enhanced: Vec<f32> = serde_json::from_value(input["enhanced_vector"].clone()).unwrap();
        let query = QueryCosines::prepare(&ctx, rows.view(), &original, &enhanced).unwrap();
        let field = build_field(&fixture_nodes(input), tags, &DtscFieldConfig::default());
        let raw = evidence(
            &field,
            &query,
            graph.view(),
            tags,
            &DtscEvidenceConfig::default(),
        )
        .unwrap();
        let top_k = input["top_k"].as_u64().unwrap() as usize;
        let actual = readout(
            &raw,
            &DtscReadoutConfig {
                top_k,
                fallback_order: DtscFallbackOrder::InputOrder,
                geometry_weight: 1.0,
                ..Default::default()
            },
        );
        let expected = &case["expected"];
        let diagnostics = &expected["diagnostics"];
        let expected_decision = match diagnostics["fallbackReason"].as_str() {
            None => DtscDecision::Scored,
            Some(reason) => DtscDecision::Fallback {
                reason: match reason {
                    "query-energy-field-low-trust" if actual.field.positive_nodes < 3 => {
                        DtscFallbackReason::InsufficientFieldSupport
                    }
                    "query-energy-field-low-trust" => DtscFallbackReason::LowFieldEntropy,
                    "no-candidate-curve-contacted-query-field" => {
                        DtscFallbackReason::NoCurveContact
                    }
                    "candidate-curve-readout-jointly-low-trust" => {
                        DtscFallbackReason::InsufficientCandidateCoverage
                    }
                    "candidate-curve-scores-lack-discrimination" => {
                        DtscFallbackReason::InsufficientScoreSpread
                    }
                    _ => panic!("unknown fixture reason {reason}"),
                },
                order: DtscFallbackOrder::InputOrder,
            },
        };
        assert_eq!(actual.decision, expected_decision);
        close(
            "curve field entropy",
            actual.field.entropy,
            diagnostics["fieldEntropy"].as_f64().unwrap(),
        );

        // Apply the new base score to the captured geometric rewards, indexed by the captured ID.
        let captured = expected["results"].as_array().unwrap();
        let mut expected_scores = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            let expected = captured
                .iter()
                .find(|row| row["id"] == candidate["curve"]["id"])
                .unwrap();
            let cosine = dtsc_cosine(&specs[i].0, &original);
            let bonus = expected["geoBonus"].as_f64().unwrap();
            expected_scores.push((CandidateId(i as u32), cosine, cosine + bonus, expected));
        }
        if !diagnostics["fallbackUsed"].as_bool().unwrap() {
            expected_scores.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.1.total_cmp(&a.1)));
        }
        if top_k != 0 {
            expected_scores.truncate(top_k);
        }
        assert_eq!(actual.results.len(), expected_scores.len());
        for (item, (candidate, cosine, score, expected)) in
            actual.results.iter().zip(expected_scores)
        {
            assert_eq!(item.candidate, candidate);
            close("original cosine", item.scores.original_cosine, cosine);
            close("score", item.scores.score, score);
            for (label, value) in [
                ("geoScore", item.scores.geo_score),
                ("geoBonus", item.scores.geo_bonus),
                ("geoConfidence", item.scores.geo_confidence),
                ("geoIsolatedRatio", item.scores.geo_isolated_ratio),
            ] {
                close(label, value, expected[label].as_f64().unwrap());
            }
        }
    }
}
