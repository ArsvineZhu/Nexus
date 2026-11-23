// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::math::{EnergyNode, EvidenceSource, TransportGraph, WeightedEdge};
use nexus::operators::{
    FieldProjectionInput, FieldProjectionOptions, FieldSolveOptions, SenseInput, SenseOptions,
    project_fields, sense, solve_fields,
};
use nexus::{
    AnchorRole, CancelToken, Concept, ConceptData, ConceptSearch, ConceptSearchRequest, EdgeGate,
    Error, Model, ObserveOptions, Query, RecallKind, Relation, RelationData, Residual, Result,
    Seed, Seeds, Source, VectorAnchor,
};
use std::sync::Mutex;

fn model(concepts: Vec<Concept>) -> Model {
    Model::build(
        RelationData {
            dimension: 2,
            concepts,
            sources: vec![],
            relations: vec![],
        },
        &Default::default(),
    )
    .unwrap()
}

#[test]
fn empty_model_preserves_external_anchors_and_cancellation() {
    let model = model(vec![]);
    let anchors = [VectorAnchor {
        vector: &[0.0, 9.0],
        role: AnchorRole::Related,
    }];
    let query = Query::new(&[8.0, 0.0]).with_anchors(&anchors);
    let mut options = ObserveOptions::default();
    let token = CancelToken::default();
    token.cancel();
    options.run.cancel = Some(token);
    assert!(matches!(
        model.observe(query, &options),
        Err(Error::Cancelled)
    ));

    let obs = model.observe(query, &Default::default()).unwrap();
    let expected = nexus::math::normalize(&[0.82, 0.18]).unwrap();
    let enhanced = obs.vector(RecallKind::Enhanced).unwrap();
    assert!(
        enhanced
            .iter()
            .zip(expected.as_slice())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
    assert!(obs.active_concepts().is_empty());
    assert!(obs.transitions().is_empty());
    assert!(obs.fields().values.local.values.is_empty());
    assert_eq!(
        obs.recall_vectors().map(|v| v.kind).collect::<Vec<_>>(),
        [RecallKind::Original, RecallKind::Enhanced]
    );
}

#[derive(Default)]
struct RecordLayers(Mutex<Vec<(Vec<f32>, Vec<u64>)>>);

impl ConceptSearch for RecordLayers {
    fn search(&self, request: ConceptSearchRequest<'_>) -> Result<Vec<u64>> {
        self.0
            .lock()
            .unwrap()
            .push((request.residual.to_vec(), request.excluded.to_vec()));
        Ok(match request.level {
            0 => vec![72],
            1 => vec![u64::MAX],
            _ => vec![],
        })
    }
}

#[test]
fn pyramid_search_gets_unnormalized_next_residual_and_business_ids() {
    let model = model(vec![
        Concept::new(72, vec![1.0, 0.0]),
        Concept::new(u64::MAX, vec![0.0, 1.0]),
    ]);
    let search = RecordLayers::default();
    let options = ObserveOptions {
        concept_search: Some(&search),
        ..Default::default()
    };
    let obs = model.observe(Query::new(&[3.0, 4.0]), &options).unwrap();
    let calls = search.0.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].1.is_empty());
    assert_eq!(calls[1].1, [72]);
    assert!(calls[1].0[0].abs() < 1e-6 && (calls[1].0[1] - 0.8).abs() < 1e-6);
    let pyramid = obs.analysis().pyramid.as_ref().unwrap();
    assert_eq!(pyramid.levels[1].tags[0].concept, u64::MAX);
    assert_eq!(pyramid.residuals[1].vector, calls[1].0);
    assert!((pyramid.coverage - 1.0).abs() < 1e-6);
}

#[test]
fn disabled_pyramid_preserves_literal_candidates_and_omits_measured_coverage() {
    let model = model(vec![
        Concept::new(10, vec![-1.0, 0.0]).with_label("explicit"),
        Concept::new(90, vec![0.0, 1.0]).with_aliases(vec!["数据库".into(), "Cache".into()]),
    ]);
    let mut options = ObserveOptions::default();
    let query = Query::new(&[1.0, 0.0])
        .with_text("数据库 cache")
        .with_core_concepts(&[10, 10]);
    let enabled = model.observe(query, &options).unwrap();
    options.config.pyramid = None;
    let disabled = model.observe(query, &options).unwrap();
    assert_eq!(enabled.analysis().pyramid.as_ref().unwrap().coverage, 0.0);
    assert!(disabled.analysis().pyramid.is_none());
    assert!(
        enabled.analysis().seeds.dynamic_core_boost > disabled.analysis().seeds.dynamic_core_boost
    );
    let seeds = &disabled.analysis().seeds.items;
    assert_eq!(
        seeds
            .iter()
            .filter(|node| node.kind == EvidenceSource::Core)
            .count(),
        1
    );
    assert!(
        seeds
            .iter()
            .any(|node| node.kind == EvidenceSource::Lexical && node.concept == 90)
    );
}

#[test]
fn sense_beam_keeps_only_temporal_events_whose_destination_survives() {
    let concepts = ConceptData::new(
        2,
        vec![
            Concept::new(20, vec![1.0, 0.0]),
            Concept::new(70, vec![0.0, 1.0]),
            Concept::new(17, vec![1.0, 1.0]),
        ],
    )
    .unwrap();
    let graph = TransportGraph::from_edges(
        &[20, 17, 70],
        &[
            WeightedEdge {
                from: 20,
                to: 70,
                weight: 0.4,
                wormhole: false,
            },
            WeightedEdge {
                from: 20,
                to: 17,
                weight: 0.4,
                wormhole: false,
            },
        ],
        &Default::default(),
    )
    .unwrap();
    let seeds = Seeds {
        items: vec![Seed {
            concept: 20,
            weight: 1.0,
            kind: EvidenceSource::Core,
            residual_level: 0,
        }],
        ..Default::default()
    };
    let mut options = SenseOptions::default();
    options.config.beam_width = 1;
    options.config.max_hops = 1;
    let result = sense(
        SenseInput {
            graph: &graph,
            concepts: concepts.view(),
            seeds: &seeds,
            residuals: &[],
        },
        &options,
    )
    .unwrap();
    assert_eq!(
        result
            .observed_edges
            .iter()
            .map(|edge| edge.to)
            .collect::<Vec<_>>(),
        [17, 70]
    );
    assert_eq!(result.transitions.len(), 1);
    assert_eq!(result.transitions[0].to, 17);
    assert_eq!(result.stats.pruned_states, 1);

    // Strictly deeper residuals add routing within the existing neighbor budget.
    let residual = Residual {
        level: 3,
        vector: vec![0.0, 0.3],
        is_tail: true,
    };
    options.config.neighbor_width = 1;
    options.config.dynamic_wormholes.as_mut().unwrap().slots = 1;
    let result = sense(
        SenseInput {
            graph: &graph,
            concepts: concepts.view(),
            seeds: &seeds,
            residuals: &[residual],
        },
        &options,
    )
    .unwrap();
    assert_eq!(result.observed_edges.len(), 1);
    assert_eq!(result.transitions[0].to, 70);
    assert!(result.transitions[0].wormhole);
}

#[test]
fn one_field_iteration_reports_the_residual_of_the_returned_iterate() {
    let graph = TransportGraph::from_edges(
        &[91, 17],
        &[
            WeightedEdge {
                from: 91,
                to: 17,
                weight: 0.5,
                wormhole: false,
            },
            WeightedEdge {
                from: 17,
                to: 91,
                weight: 0.25,
                wormhole: false,
            },
        ],
        &Default::default(),
    )
    .unwrap();
    let mut options = FieldSolveOptions::default();
    options.config.max_iterations = 1;
    let fields = solve_fields(&graph, &[(91, 5.0)], &options).unwrap();
    for (field, alpha) in [
        (&fields.local, options.config.local_alpha),
        (&fields.transfer, options.config.transfer_alpha),
    ] {
        let &[(91, a), (17, b)] = field.values.as_slice() else {
            panic!("field node order")
        };
        let residual = ((1.0 - alpha) + alpha * b * 0.25 - a).abs() + (alpha * a * 0.5 - b).abs();
        let fixed_a = (1.0 - alpha) / (1.0 - alpha * alpha * 0.125);
        let fixed_b = alpha * fixed_a * 0.5;
        let error = (fixed_a - a).abs() + (fixed_b - b).abs();
        assert_eq!(field.iterations, 1);
        assert!((field.residual_l1 - residual).abs() < 1e-12);
        assert!((field.error_bound_l1 - residual / (1.0 - alpha * 0.5)).abs() < 1e-12);
        assert!(error <= field.error_bound_l1 + 1e-12);
    }

    let concepts = ConceptData::new(
        2,
        vec![
            Concept::new(91, vec![1.0, 0.0]),
            Concept::new(17, vec![0.0, 1.0]),
        ],
    )
    .unwrap();
    let sense = nexus::SenseResult {
        nodes: vec![EnergyNode {
            concept: 91,
            energy: 5.0,
            kind: EvidenceSource::Core,
            hop: 0,
            seed: Some(91),
        }],
        ..Default::default()
    };
    let projection = project_fields(
        FieldProjectionInput {
            graph: &graph,
            concepts: concepts.view(),
            sense: &sense,
            fields: &fields,
        },
        &FieldProjectionOptions::default(),
    )
    .unwrap();
    assert_eq!(projection.local.support, [91]);
    assert_eq!(projection.transfer.support, [91, 17]);
}

#[test]
fn zero_outer_budget_retains_the_whole_observed_core_with_zero_net_direction() {
    let ids: Vec<_> = (0..260).collect();
    let concepts = ConceptData::new(
        2,
        ids.iter()
            .map(|&id| {
                let sign = if id % 2 == 0 { 1.0 } else { -1.0 };
                Concept::new(id, vec![sign, 0.0])
            })
            .collect(),
    )
    .unwrap();
    let graph = TransportGraph::from_edges(&ids, &[], &Default::default()).unwrap();
    let source: Vec<_> = ids.iter().map(|&id| (id, 1.0)).collect();
    let fields = solve_fields(&graph, &source, &Default::default()).unwrap();
    let sense = nexus::SenseResult {
        nodes: ids
            .iter()
            .map(|&id| EnergyNode {
                concept: id,
                energy: 1.0,
                kind: EvidenceSource::Core,
                hop: 0,
                seed: Some(id),
            })
            .collect(),
        ..Default::default()
    };
    let mut options = FieldProjectionOptions::default();
    options.config.max_outer_nodes = Some(0);
    let projected = project_fields(
        FieldProjectionInput {
            graph: &graph,
            concepts: concepts.view(),
            sense: &sense,
            fields: &fields,
        },
        &options,
    )
    .unwrap();
    assert_eq!(projected.local.support, ids);
    assert_eq!(projected.transfer.support, ids);
    assert!(projected.local.direction.is_none() && projected.transfer.direction.is_none());
}

#[test]
fn query_edge_gates_attenuate_the_same_graph_for_sense_and_fields() {
    let model = Model::build(
        RelationData {
            dimension: 2,
            concepts: vec![
                Concept::new(10, vec![1.0, 0.0]),
                Concept::new(20, vec![0.0, 1.0]),
            ],
            sources: vec![Source::independent(30, vec![])],
            relations: vec![Relation {
                from: 10,
                to: 20,
                support: 1.0,
                source: 30,
            }],
        },
        &Default::default(),
    )
    .unwrap();
    let mut options = ObserveOptions::default();
    options.config.pyramid = None;
    let query = Query::new(&[1.0, 0.0]).with_core_concepts(&[10]);
    let open = model.observe(query, &options).unwrap();
    let gates = [
        EdgeGate {
            from: 10,
            to: 20,
            factor: 0.0,
        },
        EdgeGate {
            from: 10,
            to: 20,
            factor: 1.0,
        },
    ];
    let closed = model
        .observe(query.with_edge_gates(&gates), &options)
        .unwrap();
    assert_eq!(open.transitions()[0].to, 20);
    assert!(open.fields().values.transfer.values[1].1 > 0.0);
    assert!(closed.transitions().is_empty());
    assert_eq!(closed.fields().values.transfer.values[1], (20, 0.0));
    assert_eq!(
        model.observe(query, &options).unwrap().transitions(),
        open.transitions()
    );
}

#[test]
fn sense_emits_raw_current_before_fir_and_preserves_seed_layer() {
    let concepts = ConceptData::new(
        2,
        vec![
            Concept::new(10, vec![1.0, 0.0]),
            Concept::new(20, vec![0.0, 1.0]),
            Concept::new(50, vec![-1.0, 0.0]),
        ],
    )
    .unwrap();
    let graph = TransportGraph::from_edges(
        &[10, 20, 50],
        &[
            WeightedEdge {
                from: 10,
                to: 20,
                weight: 0.5,
                wormhole: false,
            },
            WeightedEdge {
                from: 20,
                to: 10,
                weight: 0.4,
                wormhole: true,
            },
        ],
        &Default::default(),
    )
    .unwrap();
    let seeds = Seeds {
        items: vec![
            Seed {
                concept: 10,
                weight: 0.15,
                kind: EvidenceSource::Core,
                residual_level: 0,
            },
            Seed {
                concept: 50,
                weight: 1.0,
                kind: EvidenceSource::Lexical,
                residual_level: 0,
            },
        ],
        ..Default::default()
    };
    let input = SenseInput {
        graph: &graph,
        concepts: concepts.view(),
        seeds: &seeds,
        residuals: &[],
    };
    let mut options = SenseOptions::default();
    options.config.beam_width = 1;
    options.config.max_hops = 0;
    let zero = sense(input, &options).unwrap();
    assert_eq!(
        zero.nodes
            .iter()
            .map(|node| (node.concept, node.energy))
            .collect::<Vec<_>>(),
        [(50, 1.0), (10, 0.15)]
    );

    options.config.max_hops = 2;
    options.config.base_momentum = 1.0;
    options.config.minimum_injected_current = 0.0;
    let weighted = sense(input, &options).unwrap();
    let first = 0.15 * 0.5 * 0.25;
    assert!((weighted.transitions[0].flow - first).abs() < 1e-12);
    assert_eq!(weighted.transitions.len(), 1);
    let mass = |id| {
        weighted
            .nodes
            .iter()
            .find(|node| node.concept == id)
            .unwrap()
            .energy
    };
    let denominator = 1.0 + 0.6 + 0.6 * 0.6;
    assert!((mass(20) - first * 0.6 / denominator).abs() < 1e-12);
    assert!((mass(10) - 0.15 / denominator).abs() < 1e-12);

    options.config.firing_threshold = 0.0;
    options.config.fir_gamma = 0.0;
    let unweighted = sense(input, &options).unwrap();
    assert_eq!(unweighted.nodes, zero.nodes);
    let returned = unweighted.transitions[1];
    assert_eq!(
        (returned.previous, returned.from, returned.to),
        (Some(10), 20, 10)
    );
    assert!(returned.wormhole && returned.immediate_return);
    assert!((returned.flow - first * 0.4 * 0.70 * 0.15).abs() < 1e-12);
}

#[test]
fn pyramid_rejects_bad_host_ids_and_cancellation_during_the_second_layer() {
    struct Reply<'a> {
        first: &'a [u64],
        second: &'a [u64],
        cancel: Option<CancelToken>,
    }
    impl ConceptSearch for Reply<'_> {
        fn search(&self, request: ConceptSearchRequest<'_>) -> Result<Vec<u64>> {
            if request.level == 0 {
                return Ok(self.first.to_vec());
            }
            if let Some(token) = &self.cancel {
                token.cancel();
            }
            Ok(self.second.to_vec())
        }
    }

    let model = model(vec![
        Concept::new(72, vec![1.0, 0.0]),
        Concept::new(91, vec![0.0, 1.0]),
    ]);
    let query = Query::new(&[3.0, 4.0]);
    for (first, second) in [
        (&[72, 72][..], &[][..]),
        (&[72, 500][..], &[][..]),
        (&[72, 91, 500][..], &[][..]),
        (&[72][..], &[72][..]),
    ] {
        let reply = Reply {
            first,
            second,
            cancel: None,
        };
        let mut options = ObserveOptions {
            concept_search: Some(&reply),
            ..Default::default()
        };
        options.config.pyramid.as_mut().unwrap().top_k = 2;
        assert!(matches!(
            model.observe(query, &options),
            Err(Error::Input {
                field: "pyramid.search",
                ..
            })
        ));
    }

    let token = CancelToken::default();
    let reply = Reply {
        first: &[72],
        second: &[91],
        cancel: Some(token.clone()),
    };
    let mut options = ObserveOptions {
        concept_search: Some(&reply),
        ..Default::default()
    };
    options.run.cancel = Some(token);
    assert!(matches!(
        model.observe(query, &options),
        Err(Error::Cancelled)
    ));
    let subsequent = model.observe(query, &Default::default()).unwrap();
    assert!((subsequent.analysis().pyramid.as_ref().unwrap().coverage - 1.0).abs() < 1e-6);
}
