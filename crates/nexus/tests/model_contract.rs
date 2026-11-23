// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::{
    Candidate, Concept, Error, Model, ObserveOptions, PathRef, Query, Relation, RelationData,
    ScoreEvidence, Source,
};

fn build(sources: Vec<Source>, relations: Vec<Relation>) -> nexus::Result<Model> {
    Model::build(
        RelationData {
            dimension: 3,
            concepts: vec![
                Concept::new(10, vec![1.0, 0.0, 0.0]),
                Concept::new(20, vec![0.0, 1.0, 0.0]),
                Concept::new(40, vec![0.0, 0.0, 1.0]),
            ],
            sources,
            relations,
        },
        &Default::default(),
    )
}

fn relation(source: u64, support: f64) -> Relation {
    Relation {
        from: 10,
        to: 20,
        support,
        source,
    }
}

#[test]
fn repeated_facts_in_one_source_preserve_transport_and_observation() {
    let path = vec![10, 20, 10, 40];
    let single = build(
        vec![Source::independent(7, vec![path.clone()])],
        vec![relation(7, 2.0)],
    )
    .unwrap();
    let duplicate = build(
        vec![Source::independent(7, vec![path.clone(), path.clone()])],
        vec![relation(7, 2.0), relation(7, 1.0), relation(7, 2.0)],
    )
    .unwrap();
    let independent = build(
        vec![
            Source::independent(7, vec![path]),
            Source::independent(9, vec![]),
        ],
        vec![relation(7, 2.0), relation(9, 6.0)],
    )
    .unwrap();

    let mut opts = ObserveOptions::default();
    opts.config.pyramid = None;
    opts.config.sense.max_hops = 1;
    opts.config.sense.minimum_injected_current = 0.0;
    let query = Query::new(&[1.0, 0.0, 0.0]).with_core_concepts(&[10]);
    let observation = single.observe(query, &opts).unwrap();
    let same_facts = duplicate.observe(query, &opts).unwrap();
    assert_eq!(observation.analysis(), same_facts.analysis());
    assert_eq!(observation.fields(), same_facts.fields());
    let conductance = |obs: &nexus::Observation| {
        obs.analysis()
            .sense
            .observed_edges
            .iter()
            .find(|edge| (edge.from, edge.to) == (10, 20))
            .unwrap()
            .conductance
    };
    assert!(conductance(&independent.observe(query, &opts).unwrap()) > conductance(&observation));

    let candidates = single
        .candidates(
            (0..10)
                .map(|id| Candidate::new(id, vec![1.0, 0.0, 0.0]).with_sources(vec![7]))
                .collect(),
        )
        .unwrap();
    assert_eq!(
        observation
            .evaluate(&candidates, &Default::default())
            .unwrap()
            .cosine()
            .iter()
            .len(),
        10
    );
    let after_binding = single.observe(query, &opts).unwrap();
    assert_eq!(after_binding.analysis(), observation.analysis());
    assert_eq!(after_binding.fields(), observation.fields());
    assert!(
        after_binding
            .recall_vectors()
            .zip(observation.recall_vectors())
            .all(|(next, prior)| next.kind == prior.kind && next.vector == prior.vector)
    );
}

#[test]
fn selected_subpaths_retain_original_positions_for_known_and_unknown_sources() {
    let path = vec![10, 20, 10, 40];
    let model = build(
        vec![
            Source::independent(7, vec![path.clone()]),
            Source::unknown_origin(9, vec![path]),
        ],
        vec![],
    )
    .unwrap();
    let row = || vec![1.0, 1.0, 0.0];
    let candidates = model
        .candidates(vec![
            Candidate::new(20, row()).with_paths(vec![PathRef {
                source: 7,
                path: 0,
                positions: Some(vec![1, 2]),
            }]),
            Candidate::new(40, row()).with_paths(vec![PathRef {
                source: 9,
                path: 0,
                positions: Some(vec![1, 2]),
            }]),
            Candidate::new(10, row())
                .with_sources(vec![7])
                .with_paths(vec![]),
            Candidate::new(70, row()).with_sources(vec![7]),
        ])
        .unwrap();
    let observation = model
        .observe(
            Query::new(&[1.0, 1.0, 0.0]).with_core_concepts(&[20]),
            &Default::default(),
        )
        .unwrap();
    let evaluation = observation
        .evaluate(&candidates, &Default::default())
        .unwrap();
    let scores = evaluation.river(&Default::default()).unwrap();
    for item in scores.iter() {
        let ScoreEvidence::River(evidence) = &item.evidence else {
            panic!("River evidence")
        };
        if item.id == 10 {
            assert!(evidence.selected_path.is_none());
            continue;
        }
        let selected = evidence.selected_path.as_ref().unwrap();
        assert_eq!(selected.source, if item.id == 40 { 9 } else { 7 });
        assert_eq!(selected.path, 0);
        if item.id == 70 {
            assert_eq!(selected.positions, [0, 1, 3]);
            assert_eq!(selected.concepts, [10, 20, 40]);
        } else {
            assert_eq!(selected.positions, [1, 2]);
            assert_eq!(selected.concepts, [20, 10]);
        }
    }
    let reordered = candidates.select([70, 40, 10, 20]).unwrap();
    let reordered_scores = observation
        .evaluate(&reordered, &Default::default())
        .unwrap()
        .river(&Default::default())
        .unwrap();
    for row in reordered_scores.iter() {
        assert_eq!(row, scores.iter().find(|prior| prior.id == row.id).unwrap());
    }
}

#[test]
fn model_and_candidate_references_are_validated_at_binding() {
    for sources in [
        vec![Source::derived(7, vec![], vec![])],
        vec![Source::derived(7, vec![7], vec![])],
        vec![
            Source::derived(7, vec![9], vec![]),
            Source::derived(9, vec![7], vec![]),
        ],
        vec![Source::derived(7, vec![9], vec![])],
        vec![Source::independent(7, vec![vec![20, 123]])],
    ] {
        assert!(matches!(
            build(sources, vec![]),
            Err(Error::Input { id: Some(_), .. })
        ));
    }
    assert!(matches!(
        build(
            vec![Source::independent(7, vec![])],
            vec![relation(7, f64::INFINITY)]
        ),
        Err(Error::Input { id: Some(7), .. }),
    ));
    let model = build(
        vec![Source::independent(7, vec![vec![10, 20, 10, 40]])],
        vec![],
    )
    .unwrap();
    for path in [
        PathRef {
            source: 999,
            path: 0,
            positions: None,
        },
        PathRef {
            source: 7,
            path: 1,
            positions: None,
        },
        PathRef {
            source: 7,
            path: 0,
            positions: Some(vec![4]),
        },
        PathRef {
            source: 7,
            path: 0,
            positions: Some(vec![1, 1]),
        },
    ] {
        assert!(matches!(
            model.candidates(vec![
                Candidate::new(82, vec![1.0, 0.0, 0.0]).with_paths(vec![path]),
            ]),
            Err(Error::Input { id: Some(82), .. })
        ));
    }
    for vector in [vec![0.0; 3], vec![f32::NAN, 0.0, 1.0], vec![1.0, 0.0]] {
        assert!(matches!(
            model.candidates(vec![Candidate::new(82, vector)]),
            Err(Error::Input {
                field: "candidate.vector",
                id: Some(82),
                ..
            })
        ));
    }
}
