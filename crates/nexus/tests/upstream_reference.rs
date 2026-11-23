// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use nexus::math::{self, DedupOptions, VectorRows};
use nexus::operators::{LocalConeOptions, local_cone};
use nexus::{Concept, ConceptData, Relation, Source};
use serde_json::{Value, from_value};

fn reference() -> Value {
    serde_json::from_str(include_str!("fixtures/vcp_5adb12b.json")).unwrap()
}

#[test]
fn local_cone_matches_frozen_residuals_on_the_original_columns() {
    let source = reference();
    for case in source["geometry"].as_array().unwrap() {
        let vectors: Vec<Vec<f32>> = from_value(case["vectors"].clone()).unwrap();
        let dimension = vectors[0].len();
        let vectors = vectors
            .into_iter()
            .enumerate()
            .map(|(i, vector)| Concept::new(i as u64, vector))
            .collect();
        let concepts = ConceptData::new(dimension, vectors).unwrap();
        let weights: Vec<f64> = from_value(case["weights"].clone()).unwrap();
        let relations: Vec<_> = weights
            .into_iter()
            .enumerate()
            .map(|(i, support)| Relation {
                from: 0,
                to: i as u64 + 1,
                support,
                source: 100,
            })
            .collect();
        let mut options = LocalConeOptions::default();
        options.config.min_neighbors = 1;
        options.config.semantic_gate_enabled = false;
        let local = local_cone(
            concepts.view(),
            &[Source::independent(100, vec![])],
            &relations,
            &options,
        )
        .unwrap();
        let value = local.residual(0).unwrap();
        let upstream = case["residual_ratio"].as_f64().unwrap();
        assert!(
            (value.ratio - upstream).abs() < 1e-7,
            "{}: {} vs {upstream}",
            case["case"],
            value.ratio
        );
        assert!((value.energy - value.ratio.powi(2)).abs() < f64::EPSILON);
    }
}

#[test]
fn dedup_frozen_release_cases_use_supplied_order_and_require_valid_directions() {
    let source = reference();
    for case in source["dedup"].as_array().unwrap() {
        let vectors: Vec<Vec<f32>> = from_value(case["vectors"].clone()).unwrap();
        let dimension = vectors[0].len();
        let ids: Vec<_> = (1..=vectors.len() as u64).collect();
        let values: Vec<_> = vectors.into_iter().flatten().collect();
        let matrix = VectorRows::new(&ids, &values, dimension);
        match case["case"].as_str().unwrap() {
            "zero_vector" => assert!(matches!(
                matrix,
                Err(nexus::Error::Input { id: Some(2), .. })
            )),
            name => {
                let mut options = DedupOptions::default();
                options.config.similarity_threshold = case["threshold"].as_f64().unwrap();
                let result = math::semantic_dedup(matrix.unwrap(), &options).unwrap();
                let upstream: Vec<u64> = from_value(case["survivors"].clone()).unwrap();
                if name == "aligned_rank" {
                    assert_eq!(result.kept, upstream);
                } else {
                    assert_eq!(upstream, [2, 3]); // Upstream starts from query rank here.
                    assert_eq!(result.kept, [1, 3]);
                    assert_eq!(result.folded[0].candidate, 2);
                    assert_eq!(result.folded[0].representative, 1);
                }
            }
        }
    }
}
