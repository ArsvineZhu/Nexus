// SPDX-License-Identifier: Apache-2.0

use nexus_math::dtsc::{
    DtscAuxConfig, DtscEvidenceConfig, DtscFallbackReason, DtscFieldConfig, DtscIdentityConfig,
    DtscReadoutConfig, DtscRewardConfig, DtscTrustConfig, EnergyNode, EvidenceClass, EvidenceInput,
    EvidenceOptions, EvidenceSource, FieldOptions, GeometryState, Path, ReadoutOptions,
    SampleInput, build_field, evidence, sample_field,
};
use nexus_math::{
    CancelToken, DtscDecision, Error, Parallelism, RunOptions, Scores, TransportGraph, VectorRows,
    WeightedEdge,
};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let source = match name {
        "field" => include_str!("fixtures/vcp-dtsc-field.json"),
        "curve" => include_str!("fixtures/vcp-dtsc.json"),
        _ => unreachable!(),
    };
    serde_json::from_str(source).unwrap()
}

fn fixture_kind(value: &Value) -> Option<EvidenceSource> {
    match value.as_str().unwrap() {
        "core" => Some(EvidenceSource::Core),
        "seed" => Some(EvidenceSource::Lexical),
        "emergent" => Some(EvidenceSource::Emergent),
        "inferred" => Some(EvidenceSource::Inferred),
        "unknown" => None,
        _ => panic!("unknown field source type: {value}"),
    }
}

fn nodes(input: &Value) -> Vec<EnergyNode> {
    input["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| EnergyNode {
            concept: node["id"].as_u64().unwrap(),
            energy: node["energy"].as_f64().unwrap(),
            kind: fixture_kind(&node["sourceType"]).unwrap(),
            hop: node["hop"].as_u64().unwrap_or(0) as usize,
            seed: node["seedId"].as_u64(),
        })
        .collect()
}

fn fixture_vectors(value: &Value) -> (Vec<u64>, Vec<f32>) {
    let rows: Vec<(u64, Vec<f32>)> = serde_json::from_value(value.clone()).unwrap();
    rows.into_iter().fold(
        (Vec::new(), Vec::new()),
        |(mut ids, mut values), (id, row)| {
            ids.push(id);
            values.extend(row);
            (ids, values)
        },
    )
}

fn graph(ids: &[u64], input: &Value) -> TransportGraph {
    let edges: Vec<(u64, u64, f64)> = serde_json::from_value(input["edges"].clone()).unwrap();
    let edges = edges
        .into_iter()
        .map(|(from, to, weight)| WeightedEdge {
            from,
            to,
            weight,
            wormhole: false,
        })
        .collect::<Vec<_>>();
    let mut graph = TransportGraph::from_edges(ids, &edges, &Default::default()).unwrap();
    let anchors: Vec<(u64, f64)> = serde_json::from_value(input["anchor_gain"].clone()).unwrap();
    let gains = ids
        .iter()
        .map(|id| {
            anchors
                .iter()
                .find(|(anchor, _)| anchor == id)
                .map_or(1.0, |(_, gain)| *gain)
        })
        .collect::<Vec<_>>();
    graph.set_anchor_gains(&gains).unwrap();
    graph
}

#[track_caller]
fn close(label: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{label}: {actual} != {expected}"
    );
}

fn energy(concept: u64, energy: f64, kind: EvidenceSource) -> EnergyNode {
    EnergyNode {
        concept,
        energy,
        kind,
        hop: 0,
        seed: None,
    }
}

#[test]
fn samples_match_nous_wave_float64_reference_for_valid_vectors() {
    let captured = fixture("field");
    let graph_fixture = fixture("curve");
    for case in captured["cases"].as_array().unwrap() {
        let input = &case["input"];
        let (ids, values) = fixture_vectors(&input["tag_vectors"]);
        let concepts = VectorRows::new(&ids, &values, 3).unwrap();
        // Both probes captured the same persisted transport graph.
        let graph = graph(&ids, &graph_fixture["cases"][0]["input"]);
        let config = &input["config"];
        let defaults = DtscFieldConfig::default();
        let options = FieldOptions {
            config: DtscFieldConfig {
                min_field_tags: config["minFieldTags"]
                    .as_u64()
                    .map_or(defaults.min_field_tags, |n| n as usize),
                max_field_nodes: config["maxFieldNodes"]
                    .as_u64()
                    .map_or(defaults.max_field_nodes, |n| n as usize),
                field_energy_mass_ratio: config["fieldEnergyMassRatio"]
                    .as_f64()
                    .unwrap_or(defaults.field_energy_mass_ratio),
                ..defaults
            },
            ..Default::default()
        };
        let field = build_field(concepts, &nodes(input), &options).unwrap();
        let report = evidence(
            EvidenceInput {
                field: &field,
                concepts,
                graph: Some(&graph),
                query: concepts.row(0).unwrap(),
                enhanced_query: None,
                candidates: VectorRows::new(&[], &[], 3).unwrap(),
                paths: &[],
                geometry: None,
            },
            &Default::default(),
        )
        .unwrap();

        let expected = &case["expected"];
        close(
            "entropy",
            report.field_stats().entropy,
            expected["fieldEntropy"].as_f64().unwrap(),
        );
        close(
            "mass",
            report.field_stats().total_mass,
            expected["totalEnergy"].as_f64().unwrap(),
        );
        assert_eq!(
            report.field_stats().positive_nodes,
            expected["exactField"].as_array().unwrap().len()
        );
        assert_eq!(
            report.field_stats().retained_nodes,
            expected["fieldNodes"].as_array().unwrap().len()
        );

        let (sample_ids, samples) = fixture_vectors(&input["sample_tags"]);
        let actual_samples = sample_ids
            .iter()
            .zip(samples.as_chunks::<3>().0)
            .map(|(&id, vector)| SampleInput {
                concept: ids.contains(&id).then_some(id),
                vector,
            })
            .collect::<Vec<_>>();
        // The zero vector in the capture is now a public input error. A vector
        // with no known identity continues to exercise vector-only sampling.
        assert!(matches!(
            sample_field(
                &field,
                Some(&graph),
                &actual_samples[4..],
                &Default::default()
            ),
            Err(Error::Input {
                field: "sample.vector",
                ..
            })
        ));
        let actual = sample_field(
            &field,
            Some(&graph),
            &actual_samples[..4],
            &Default::default(),
        )
        .unwrap();
        for (actual, expected) in actual.iter().zip(expected["samples"].as_array().unwrap()) {
            close(
                "potential",
                actual.potential,
                expected["potential"].as_f64().unwrap(),
            );
            assert_eq!(actual.exact, expected["exact"].as_bool().unwrap());
            assert_eq!(actual.kind, fixture_kind(&expected["sourceType"]));
        }
    }
}

#[test]
fn exact_sampling_retains_identities_outside_the_interpolation_top_n() {
    let rows = VectorRows::new(&[17, 10, 90], &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2).unwrap();
    let field = build_field(
        rows,
        &[
            energy(17, 1.0, EvidenceSource::Inferred),
            energy(10, 1.0, EvidenceSource::Core),
            energy(90, 0.5, EvidenceSource::Emergent),
        ],
        &FieldOptions {
            config: DtscFieldConfig {
                min_field_tags: 1,
                max_field_nodes: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let samples = sample_field(
        &field,
        None,
        &[
            SampleInput {
                concept: Some(17),
                vector: &[1.0, 0.0],
            },
            SampleInput {
                concept: Some(90),
                vector: &[0.0, 1.0],
            },
            SampleInput {
                concept: None,
                vector: &[1.0, 0.0],
            },
            SampleInput {
                concept: None,
                vector: &[0.0, 1.0],
            },
        ],
        &Default::default(),
    )
    .unwrap();
    assert_eq!(samples[0].potential, 1.0);
    assert!(samples[0].exact);
    assert_eq!(samples[0].kind, Some(EvidenceSource::Inferred));
    assert_eq!(samples[1].potential, 0.5);
    assert!(samples[1].exact);
    assert_eq!(samples[1].kind, Some(EvidenceSource::Emergent));
    assert_eq!(samples[2].potential, 1.0);
    assert!(!samples[2].exact);
    assert_eq!(samples[2].kind, Some(EvidenceSource::Core));
    assert_eq!(samples[3].kind, None);
    assert!(samples.iter().all(|s| s.specificity == 1.0));
    assert!(matches!(
        sample_field(
            &field,
            None,
            &[SampleInput {
                concept: Some(99),
                vector: &[1.0, 0.0]
            }],
            &Default::default()
        ),
        Err(Error::Input {
            field: "sample.concept",
            id: Some(99),
            ..
        })
    ));
}

#[test]
fn curve_and_applied_scores_match_nous_wave_reference() {
    let captured = fixture("curve");
    for case in captured["cases"].as_array().unwrap() {
        let input = &case["input"];
        let dim = input["dimension"].as_u64().unwrap() as usize;
        if dim == 0 {
            assert!(matches!(
                VectorRows::new(&[], &[], dim),
                Err(Error::Input {
                    field: "dimension",
                    ..
                })
            ));
            continue;
        }
        let (ids, vectors) = fixture_vectors(&input["tag_vectors"]);
        let concepts = VectorRows::new(&ids, &vectors, dim).unwrap();
        let graph = graph(&ids, input);
        let field = build_field(concepts, &nodes(input), &Default::default()).unwrap();
        let baseline = Scores::new(input["candidates"].as_array().unwrap().iter().map(|row| {
            (
                row["curve"]["id"].as_u64().unwrap(),
                row["score"].as_f64().unwrap(),
            )
        }))
        .unwrap();
        let candidate_ids = baseline.values().map(|(id, _)| id).collect::<Vec<_>>();
        let mut candidate_values = Vec::new();
        let mut tags = Vec::new();
        for row in input["candidates"].as_array().unwrap() {
            let values: Vec<f32> =
                serde_json::from_value(row["curve"]["chunk_vector"].clone()).unwrap();
            candidate_values.extend(values);
            tags.push(
                row["curve"]["tags"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|tag| tag["id"].as_u64().unwrap())
                    .collect::<Vec<_>>(),
            );
        }
        let paths = candidate_ids
            .iter()
            .zip(&tags)
            .map(|(&id, concepts)| Path {
                candidate: id,
                ordinal: 0,
                concepts,
            })
            .collect::<Vec<_>>();
        let query: Vec<f32> = serde_json::from_value(input["original_vector"].clone()).unwrap();
        let enhanced: Vec<f32> = serde_json::from_value(input["enhanced_vector"].clone()).unwrap();
        let report = evidence(
            EvidenceInput {
                field: &field,
                concepts,
                graph: Some(&graph),
                query: &query,
                enhanced_query: Some(&enhanced),
                candidates: VectorRows::new(&candidate_ids, &candidate_values, dim).unwrap(),
                paths: &paths,
                geometry: None,
            },
            &EvidenceOptions {
                run: RunOptions {
                    parallelism: Parallelism::Rayon,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let result = report.apply(&baseline, &Default::default()).unwrap();
        let expected = &case["expected"];
        let decision = if expected["diagnostics"]["fallbackUsed"].as_bool().unwrap() {
            DtscDecision::Baseline {
                reason: DtscFallbackReason::InsufficientFieldSupport,
            }
        } else {
            DtscDecision::Applied
        };
        assert_eq!(result.decision, decision);
        close(
            "entropy",
            report.field_stats().entropy,
            expected["diagnostics"]["fieldEntropy"].as_f64().unwrap(),
        );
        let ranked = result.scores.top(None);
        for (actual, expected) in ranked
            .rows()
            .iter()
            .zip(expected["results"].as_array().unwrap())
        {
            assert_eq!(actual.id, expected["id"].as_u64().unwrap());
            close("score", actual.value, expected["score"].as_f64().unwrap());
            let geometry = &report
                .items()
                .iter()
                .find(|row| row.candidate == actual.id)
                .unwrap()
                .metrics;
            for (key, value) in [
                ("geoScore", geometry.geo_score),
                ("geoBonus", geometry.geo_bonus),
                ("geoConfidence", geometry.confidence),
                ("geoContinuity", geometry.continuity),
                ("geoIsolatedRatio", geometry.isolated_ratio),
            ] {
                close(key, value, expected[key].as_f64().unwrap());
            }
        }
    }
}

#[test]
fn evidence_preserves_entire_path_identity_and_explicit_baseline() {
    let ids = [90, 40, 17];
    let concept_values = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let concepts = VectorRows::new(&ids, &concept_values, 2).unwrap();
    let nodes = [
        energy(90, 1.0, EvidenceSource::Core),
        energy(40, 0.8, EvidenceSource::Inferred),
        energy(17, 0.8, EvidenceSource::Emergent),
    ];
    let field = build_field(concepts, &nodes, &Default::default()).unwrap();
    let paths = [
        Path {
            candidate: 100,
            ordinal: 7,
            concepts: &[40],
        },
        Path {
            candidate: 100,
            ordinal: 3,
            concepts: &[90, 90],
        },
        Path {
            candidate: 100,
            ordinal: 4,
            concepts: &[90],
        },
        Path {
            candidate: 200,
            ordinal: 0,
            concepts: &[17],
        },
    ];
    let query = [1.0, 0.0];
    let candidate_vectors =
        VectorRows::new(&[100, 200, 300], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], 2).unwrap();
    let report = evidence(
        EvidenceInput {
            field: &field,
            concepts,
            graph: None,
            query: &query,
            enhanced_query: None,
            candidates: candidate_vectors,
            paths: &paths,
            geometry: None,
        },
        &Default::default(),
    )
    .unwrap();
    let direct = &report.items()[0];
    let structural = &report.items()[1];
    assert_eq!(direct.path, Some(3));
    assert_eq!(direct.metrics.exact_hits, 1);
    assert_eq!(direct.metrics.class, EvidenceClass::Direct);
    assert_eq!(structural.metrics.class, EvidenceClass::Structural);
    assert!(direct.metrics.geo_bonus > structural.metrics.geo_bonus);
    assert!(report.items()[2].path.is_none());

    let primary = Scores::new([(300, 1.7), (100, -0.25), (200, 1.4)]).unwrap();
    for (baseline, gain) in [
        (&primary, 1.0),
        (&primary, 0.0),
        (
            &Scores::new([(200, 2.0), (100, 2.0), (300, -2.0)]).unwrap(),
            0.5,
        ),
    ] {
        let options = ReadoutOptions {
            config: DtscReadoutConfig {
                correction_gain: gain,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = report.apply(baseline, &options).unwrap();
        assert_eq!(result.decision, DtscDecision::Applied);
        let expected = baseline
            .values()
            .map(|(id, baseline)| {
                let bonus = report
                    .items()
                    .iter()
                    .find(|row| row.candidate == id)
                    .unwrap()
                    .metrics
                    .geo_bonus;
                (id, baseline + gain * bonus)
            })
            .collect::<Vec<_>>();
        assert_eq!(result.scores.values().collect::<Vec<_>>(), expected);
        assert_eq!(direct.metrics.geo_bonus, direct.metrics.base_bonus);
    }
    assert!(matches!(
        report.apply(
            &Scores::new([(100, 1.0), (200, 2.0), (400, 1.0)]).unwrap(),
            &Default::default()
        ),
        Err(Error::Input {
            field: "baseline",
            id: Some(400),
            ..
        })
    ));
}

#[test]
fn identity_floor_and_geometry_run_from_their_own_inputs() {
    let query = [1.0, 0.0];
    let concepts = VectorRows::new(&[1], &query, 2).unwrap();
    let field = build_field(
        concepts,
        &[energy(1, 1.0, EvidenceSource::Core)],
        &FieldOptions {
            config: DtscFieldConfig {
                min_field_tags: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let paths = [Path {
        candidate: 1,
        ordinal: 0,
        concepts: &[1],
    }];
    let input = |geometry| EvidenceInput {
        field: &field,
        concepts,
        graph: None,
        query: &query,
        enhanced_query: Some(&query),
        candidates: concepts,
        paths: &paths,
        geometry,
    };
    let mut options = EvidenceOptions {
        config: DtscEvidenceConfig {
            reward: DtscRewardConfig {
                alpha: 0.0,
                ..Default::default()
            },
            auxiliary: Some(DtscAuxConfig {
                max_aux_bonus: 0.012,
                geometry: None,
                identity: Some(DtscIdentityConfig::default()),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let identity = evidence(input(None), &options).unwrap().items()[0].metrics;
    close(
        "identity floor",
        identity.auxiliary.unwrap().identity_floor,
        0.018,
    );
    assert!(identity.auxiliary.unwrap().shadow_score.is_none());
    close("limited identity bonus", identity.geo_bonus, 0.012);

    options.config.auxiliary.as_mut().unwrap().geometry = Some(Default::default());
    assert!(matches!(
        evidence(input(None), &options),
        Err(Error::Input {
            field: "geometry",
            ..
        })
    ));
    let state = GeometryState {
        epa_concentration: 0.9,
        epa_entropy: 0.1,
        epa_resonance: 0.4,
        pyramid_coverage: 0.95,
        pyramid_novelty: 0.2,
        pyramid_depth: 1.0,
    };
    options.config.auxiliary.as_mut().unwrap().identity = None;
    let geometry = evidence(input(Some(state)), &options).unwrap().items()[0].metrics;
    assert!(geometry.auxiliary.unwrap().shadow_score.unwrap() > 0.1);
    assert!(geometry.geo_bonus > 0.0);
}

#[test]
fn trust_guards_use_geometric_limits_and_preserve_supplied_ties_on_fallback() {
    let query = [1.0, 0.0];
    let concepts = VectorRows::new(&[1], &query, 2).unwrap();
    let candidate_rows =
        VectorRows::new(&[10, 20, 30], &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2).unwrap();
    let paths = [
        Path {
            candidate: 10,
            ordinal: 0,
            concepts: &[1],
        },
        Path {
            candidate: 20,
            ordinal: 0,
            concepts: &[1],
        },
    ];
    let node = energy(1, 1.0, EvidenceSource::Core);
    let field = build_field(
        concepts,
        &[node],
        &FieldOptions {
            config: DtscFieldConfig {
                min_field_tags: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let prepare = |paths| {
        evidence(
            EvidenceInput {
                field: &field,
                concepts,
                graph: None,
                query: &query,
                enhanced_query: None,
                candidates: candidate_rows,
                paths,
                geometry: None,
            },
            &Default::default(),
        )
        .unwrap()
    };
    let baseline = Scores::new([(30, 2.0), (20, -3.0), (10, -3.0)]).unwrap();
    let pool = prepare(&paths);
    assert_eq!(
        pool.apply(&baseline, &Default::default()).unwrap().decision,
        DtscDecision::Baseline {
            reason: DtscFallbackReason::LowFieldEntropy
        }
    );
    let mut options = ReadoutOptions {
        config: DtscReadoutConfig {
            trust: DtscTrustConfig {
                min_field_entropy: 0.0,
                min_candidate_coverage: 0.9,
                min_geo_score: 1.0,
                min_strong_evidence: 10.0,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let no_contact = prepare(&[]);
    assert_eq!(
        no_contact.apply(&baseline, &options).unwrap().decision,
        DtscDecision::Baseline {
            reason: DtscFallbackReason::NoContact
        }
    );
    let fallback = pool.apply(&baseline, &options).unwrap();
    assert_eq!(
        fallback.decision,
        DtscDecision::Baseline {
            reason: DtscFallbackReason::InsufficientCoverage
        }
    );
    assert_eq!(fallback.scores, baseline);
    assert_eq!(fallback.scores.top(None).ids(), &[30, 20, 10]);
    options.config.trust.min_candidate_coverage = 0.0;
    assert_eq!(
        pool.apply(&baseline, &options).unwrap().decision,
        DtscDecision::Baseline {
            reason: DtscFallbackReason::InsufficientSpread
        }
    );
    options.config.trust.min_strong_evidence = 0.0;
    options.config.trust.min_candidate_coverage = 0.9;
    assert_eq!(
        pool.apply(&baseline, &options).unwrap().decision,
        DtscDecision::Applied
    );
    let token = CancelToken::default();
    token.cancel();
    options.run.cancel = Some(token);
    assert!(matches!(
        pool.apply(&baseline, &options),
        Err(Error::Cancelled)
    ));
}
