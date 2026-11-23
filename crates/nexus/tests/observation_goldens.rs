// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::operators::{
    EpaOptions, GlobalSpanOptions, LocalConeOptions, SeedInput, analyze_epa, global_span,
    local_cone, select_seeds,
};
use nexus::{Concept, ConceptData, EpaGateMode, GlobalCandidateMode, Query, Relation, Source};
use serde::Deserialize;

#[derive(Deserialize)]
struct HypernymFixture {
    revision: String,
    hypernym: Vec<HypernymCase>,
}

#[derive(Deserialize)]
struct HypernymCase {
    case: String,
    confidence: f64,
    contribution: f64,
    explicit: bool,
    names: Vec<String>,
    parents: Vec<u64>,
    projection: Vec<f64>,
    query: Vec<f32>,
    vectors: Vec<Vec<f32>>,
}

#[test]
fn upstream_hypernym_shadow_and_differentia() {
    let frozen: HypernymFixture =
        serde_json::from_str(include_str!("fixtures/vcp_5adb12b.json")).unwrap();
    assert_eq!(frozen.revision, "5adb12bb55e616e22bd0e0dbfdc13032c069941a");
    for case in frozen.hypernym {
        let ids = [501, 719];
        let concepts = ConceptData::new(
            case.query.len(),
            ids.into_iter()
                .zip(case.names)
                .zip(case.vectors)
                .map(|((id, label), vector)| Concept::new(id, vector).with_label(label))
                .collect(),
        )
        .unwrap();
        let vector_rows = concepts.view().vectors().as_rows();
        let query_direction = nexus::math::normalize(&case.query).unwrap();
        let tags = ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                let similarity = query_direction
                    .as_slice()
                    .iter()
                    .zip(vector_rows.row(index).unwrap())
                    .map(|(&q, &x)| f64::from(q) * f64::from(x))
                    .sum::<f64>()
                    .clamp(-1.0, 1.0);
                nexus::PyramidTag {
                    concept: id,
                    similarity,
                    contribution: if index == 0 { case.contribution } else { 1.0 },
                    handshake_magnitude: 0.0,
                }
            })
            .collect();
        let pyramid = nexus::PyramidResult {
            levels: vec![nexus::PyramidLevel {
                level: 0,
                tags,
                projection_magnitude: 0.0,
                residual_magnitude: 0.0,
                residual_energy_ratio: 0.0,
                energy_explained: 0.0,
            }],
            ..Default::default()
        };
        let cores = if case.explicit { &ids[..1] } else { &[] };
        let selected = select_seeds(
            SeedInput {
                query: Query::new(&case.query).with_core_concepts(cores),
                concepts: concepts.view(),
                epa: None,
                pyramid: Some(&pyramid),
            },
            &Default::default(),
        )
        .unwrap();
        let parents: Vec<_> = case
            .parents
            .into_iter()
            .map(|id| ids[id as usize - 1])
            .collect();
        assert_eq!(selected.pruned_hypernyms, parents, "{}", case.case);
        let actual = selected
            .differentia
            .as_ref()
            .map_or(&[][..], |vector| vector.as_slice());
        assert_eq!(actual.len(), case.projection.len(), "{}", case.case);
        assert!(
            actual
                .iter()
                .zip(case.projection)
                .all(|(&a, b)| (f64::from(a) - b).abs() < 1e-7),
            "{}",
            case.case
        );
        // Concept and query directions are normalized at the input boundary.
        assert!(
            (selected.differentia_confidence - case.confidence).abs() < 2e-8,
            "{}",
            case.case
        );
    }
}

#[test]
fn epa_uses_signed_projections_and_business_ids_for_theme_bridges() {
    let ids = [101, 107, 205, 211];
    let concepts = ConceptData::new(
        2,
        ids.into_iter()
            .zip([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
            .map(|(id, vector)| Concept::new(id, vector.to_vec()))
            .collect(),
    )
    .unwrap();
    let sources = [Source::independent(17, vec![])];
    let relations: Vec<_> = [(101, 205), (205, 101), (107, 211), (211, 107)]
        .into_iter()
        .map(|(from, to)| Relation {
            from,
            to,
            support: 4.0,
            source: 17,
        })
        .collect();
    let mut local_options = LocalConeOptions::default();
    local_options.config.min_neighbors = 1;
    local_options.config.semantic_gate_enabled = false;
    let local = local_cone(concepts.view(), &sources, &relations, &local_options).unwrap();
    let mut global_options = GlobalSpanOptions::default();
    // The default semantic cap is larger than this rank-two basis.
    global_options.config.candidate_mode = GlobalCandidateMode::Anchor;
    let span = global_span(concepts.view(), &local, &global_options).unwrap();
    assert_eq!(span.axis_count(), 2);
    let mut options = EpaOptions::default();
    options.config.gate_mode = EpaGateMode::Raw;
    let signed = select_epa(&span, &[-0.8, 0.6], &options);
    assert_eq!(signed.axes.len(), 2);
    assert!(signed.axes[0].projection < 0.0 || signed.axes[1].projection < 0.0);
    assert_eq!(signed.bridges[0].from, signed.axes[0].concept);
    assert_eq!(signed.bridges[0].to, signed.axes[1].concept);
    assert!((signed.bridges[0].strength - 0.48).abs() < 1e-7);
    assert!((signed.bridges[0].balance - 0.5625).abs() < 1e-7);
    let entropy = -0.64_f64 * 0.64_f64.log2() - 0.36_f64 * 0.36_f64.log2();
    assert!((signed.concentration - (1.0 - entropy)).abs() < 1e-7);
    for (mode, expected) in [
        (EpaGateMode::BackgroundRelative, 0.0),
        (EpaGateMode::RelativeLinear, 1.0),
        (EpaGateMode::RelativeSqrt, 1.0),
    ] {
        options.config.gate_mode = mode;
        let value = select_epa(&span, &[-0.8, 0.6], &options);
        assert!((value.entropy - entropy).abs() < 1e-7);
        assert!((value.in_domain_gate - expected).abs() < 1e-7);
    }
    options.config.gate_mode = EpaGateMode::Raw;

    let smaller = select_epa(&span, &[0.96_f32.sqrt(), 0.2], &options);
    assert_eq!(smaller.axes.len(), 1);
    assert_eq!(smaller.bridges.len(), 1);
    assert!(ids.contains(&smaller.bridges[0].to));
    assert!((smaller.bridges[0].strength - (0.96_f64 * 0.04).sqrt()).abs() < 1e-7);
}

fn select_epa(
    span: &nexus::operators::GlobalSpan,
    query: &[f32],
    options: &EpaOptions,
) -> nexus::EpaAnalysis {
    analyze_epa(query, span, options).unwrap().unwrap()
}
