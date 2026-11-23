// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use nexus::operators::{
    CrossSourceOptions, RiverAnalysisOptions, RiverScoringOptions, bridge_witnesses,
};
use nexus::*;

fn concepts() -> Vec<Concept> {
    [
        (0, [1.0, 0.0, 0.0]),
        (1, [0.8, 0.6, 0.0]),
        (2, [0.7, 0.6, 0.4]),
        (3, [0.0, 0.0, 1.0]),
    ]
    .into_iter()
    .map(|(id, vector)| Concept::new(id, vector.to_vec()))
    .collect()
}

fn observe_options() -> ObserveOptions<'static> {
    let mut options = ObserveOptions::default();
    options.config.pyramid = None;
    options.config.sense.firing_threshold = 0.0;
    options.config.sense.minimum_injected_current = 0.001;
    options
}

fn bridge_model(independent: bool, third: bool) -> nexus::Result<Model> {
    let other = if third {
        vec![0, 1, 2, 3]
    } else {
        vec![0, 1, 3]
    };
    let mut options = BuildOptions::default();
    options.config.graph.transport.wormhole_evidence_threshold = 3.0;
    Model::build(
        RelationData {
            dimension: 3,
            concepts: concepts(),
            sources: vec![
                Source::independent(20, vec![vec![0, 1, 2, 3], vec![2, 0, 1]]),
                if independent {
                    Source::independent(21, vec![other])
                } else {
                    Source::derived(21, vec![20], vec![other])
                },
                Source::derived(22, vec![20], vec![vec![0, 1, 2, 3]]),
            ],
            relations: vec![Relation {
                from: 0,
                to: 1,
                support: 100.0,
                source: 20,
            }],
        },
        &options,
    )
}

#[test]
fn real_wormhole_checks_independent_roots_and_joint_third_concept() -> nexus::Result<()> {
    let options = observe_options();
    let vector = [1.0, 0.1, 0.1];
    for (independent, third) in [(true, true), (false, true), (true, false)] {
        let model = bridge_model(independent, third)?;
        let query = Query::new(&vector).with_core_concepts(&[0, 3]);
        let observation = model.observe(query, &options)?;
        assert!(
            observation
                .analysis()
                .sense
                .observed_edges
                .iter()
                .any(|edge| edge.from == 0 && edge.to == 1 && edge.wormhole)
        );
        let witnesses = bridge_witnesses(&observation, &CrossSourceOptions::default())?;
        assert_eq!(
            witnesses.iter().any(|w| w.source_a == 20
                && w.source_b == 21
                && w.third == 2
                && w.other_foci == [3]),
            independent && third,
            "witnesses {witnesses:?}, positive flows {:?}",
            observation.analysis().sense.observed_edges
        );
        assert!(
            witnesses
                .iter()
                .all(|w| (w.source_a, w.source_b) != (20, 22))
        );
        if independent && third {
            let candidate = model.candidates(vec![
                Candidate::new(50, vec![0.7, 0.7, 0.1]).with_sources(vec![20]),
            ])?;
            let eval = observation.evaluate(&candidate, &Default::default())?;
            let evidence = eval.river_evidence(&Default::default())?;
            assert!(
                evidence
                    .paths(50)
                    .unwrap()
                    .iter()
                    .any(|path| path.cross_source_transport_factor > 0.0)
            );
        }
        let gates = [EdgeGate {
            from: 0,
            to: 1,
            factor: 0.0,
        }];
        let gated = model.observe(query.with_edge_gates(&gates), &options)?;
        assert!(
            bridge_witnesses(&gated, &Default::default())?
                .iter()
                .all(|w| (w.from, w.to) != (0, 1))
        );
    }
    Ok(())
}

#[test]
fn river_selects_one_complete_path_then_applies_shared_budget() -> nexus::Result<()> {
    let model = bridge_model(true, true)?;
    let catalog = model.candidates(vec![
        Candidate::new(90, vec![0.72, 0.57, 0.37]).with_sources(vec![20]),
        Candidate::new(91, vec![-1.0, 0.0, 0.0]),
        Candidate::new(92, vec![0.1, 0.0, 1.0]).with_sources(vec![21]),
    ])?;
    let vector = [1.0, 0.1, 0.1];
    let query = Query::new(&vector).with_core_concepts(&[0, 3]);
    let observe = observe_options();
    let observation = model.observe(query, &observe)?;
    let selected = catalog.select([92, 90])?;
    let evaluation = observation.evaluate(&catalog, &Default::default())?;
    let first = evaluation.river_evidence(&RiverAnalysisOptions::default())?;
    let second = observation
        .evaluate(&selected, &Default::default())?
        .river_evidence(&Default::default())?;
    assert_eq!(first.paths(90), second.paths(90));
    assert_eq!(first.paths(90).unwrap().len(), 2);
    let river = evaluation.river(&Default::default())?.top(None);
    let request = RiverRequestOptions {
        observe: observe.config,
        limit: None,
        ..Default::default()
    };
    assert_eq!(model.rank_river(query, &catalog, &request)?, river);
    let lane = river.rows().iter().find(|row| row.id == 91).unwrap();
    assert!(lane.value.abs() < f64::EPSILON);
    drop(evaluation);
    drop(observation);

    for cap in [0.0, 0.08, 0.5] {
        let config = RiverScoringConfig {
            topology_budget: cap,
            omega_gamma: 1.0,
        };
        let scores = first.readout(&RiverScoringOptions {
            config,
            run: RunOptions::default(),
        })?;
        let row = scores.iter().find(|row| row.id == 90).unwrap();
        let ScoreEvidence::River(actual) = &row.evidence else {
            panic!("missing River evidence")
        };
        let path = first
            .paths(90)
            .unwrap()
            .iter()
            .find(|p| p.selected_path == actual.selected_path)
            .unwrap();
        let gate = first.omega().omega;
        let p = cap * path.path_topology_factor * gate;
        let requested = (cap * path.graph_request_factor * gate).min(cap - p);
        let transport = cap
            * gate
            * path
                .cross_source_transport_factor
                .max(path.ordinary_transport_factor);
        let g = if cap > 0.0 {
            (cap * (1.0 - (1.0 - requested / cap) * (1.0 - transport / cap))).min(cap - p)
        } else {
            0.0
        };
        assert!((row.value - (path.base_score + p + g).clamp(0.0, 1.0)).abs() < 1e-14);
        assert!(actual.path_bonus + actual.graph_bonus <= cap);
        assert_eq!(actual.selected_path.as_ref().unwrap().source, 20);
        assert!((actual.field_support - path.field_support).abs() < f64::EPSILON);
    }
    Ok(())
}

#[test]
fn dtsc_applies_shared_geometry_to_negative_and_large_explicit_values() -> nexus::Result<()> {
    let model = bridge_model(true, true)?;
    let catalog = model.candidates(vec![
        Candidate::new(50, vec![0.7, 0.7, 0.3]).with_sources(vec![20]),
        Candidate::new(51, vec![0.2, 0.0, 1.0]),
    ])?;
    let query = [0.8, 0.6, 0.3];
    let obs = model.observe(
        Query::new(&query).with_core_concepts(&[0, 1, 2, 3]),
        &observe_options(),
    )?;
    let eval = obs.evaluate(&catalog, &Default::default())?;
    let baseline = eval.baseline([(51, -3.0), (50, 2.0)])?;
    let geometry = eval.dtsc_evidence(&Default::default())?;
    let mut options = DtscReadoutOptions::default();
    options.config.fallback_on_low_trust = false;
    let adjusted = geometry.apply(&baseline, &options)?;
    assert_eq!(
        adjusted.summary(),
        &ReadoutSummary::Dtsc {
            decision: DtscDecision::Applied
        }
    );
    assert!(geometry.metrics(50).unwrap().geo_bonus > 0.0);
    for row in adjusted.iter() {
        let ScoreEvidence::Dtsc(item) = &row.evidence else {
            panic!("missing geometry")
        };
        assert!((row.value - (item.baseline_value + item.correction)).abs() < f64::EPSILON);
        assert!(
            (item.correction - geometry.metrics(row.id).unwrap().geo_bonus).abs() < f64::EPSILON
        );
    }
    assert!(adjusted.top(Some(1)).rows()[0].value > 2.0);
    assert!((adjusted.iter().nth(1).unwrap().value + 3.0).abs() < f64::EPSILON);
    options.config.correction_gain = 0.0;
    assert_eq!(
        geometry
            .apply(&baseline, &options)?
            .values()
            .collect::<Vec<_>>(),
        baseline.values().collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn thinking_uses_retained_events_and_separates_document_and_candidate_ids() -> nexus::Result<()> {
    let mut build = BuildOptions::default();
    build.config.graph.facts.min_reverse_gain = 0.0;
    build.config.graph.facts.reverse_gain = 0.0;
    build.config.graph.transport.wormhole_evidence_threshold = 5.0;
    let model = Model::build(
        RelationData {
            dimension: 3,
            concepts: concepts(),
            sources: vec![
                Source::independent(30, vec![vec![0, 1]]),
                Source::independent(31, vec![vec![1, 2]]),
            ],
            relations: vec![],
        },
        &build,
    )?;
    let query = [1.0, 0.0, 0.0];
    let mut opts = observe_options();
    opts.config.sense.beam_width = 1;
    let observation = model.observe(Query::new(&query).with_core_concepts(&[0]), &opts)?;
    let catalog = model.candidates(vec![
        Candidate::new(200, vec![0.9, 0.3, 0.0])
            .with_sources(vec![30])
            .with_document_id(203),
        Candidate::new(201, vec![0.7, 0.7, 0.3]).with_sources(vec![31]),
        Candidate::new(202, vec![0.7, 0.7, 0.3])
            .with_sources(vec![31])
            .with_document_id(203),
        Candidate::new(203, vec![0.0, 0.0, 1.0]),
    ])?;
    let stages = [
        ThinkingStage {
            label: "start",
            candidates: &[200],
            limit: 1,
        },
        ThinkingStage {
            label: "continue",
            candidates: &[200, 202, 201, 203],
            limit: 3,
        },
    ];
    let eval = observation.evaluate(&catalog, &Default::default())?;
    let thinking = eval.thinking(&stages, &Default::default())?;
    let first = &thinking.stages[0].selections[0];
    let next = &thinking.stages[1].selections[0];
    assert_eq!(first.relation, ThinkingRelation::Root);
    assert_eq!(next.relation, ThinkingRelation::Continuation);
    assert_eq!(next.parents, [200]);
    let (a, b) = (first.route.unwrap(), next.route.unwrap());
    assert_eq!(a.hop + 1, b.hop);
    assert_eq!(a.to, b.from);
    assert_eq!(b.previous, Some(a.from));
    assert_eq!(thinking.stages[1].selections[1].candidate, 203);
    assert!(thinking.stages[1].selections[1].parents.is_empty());
    assert_eq!(
        thinking.stages[1].selections[1].relation,
        ThinkingRelation::Supplement
    );
    Ok(())
}

#[test]
fn thinking_rejects_an_edge_with_the_wrong_previous_concept() -> nexus::Result<()> {
    let mut build = BuildOptions::default();
    build.config.graph.facts.min_reverse_gain = 0.0;
    build.config.graph.facts.reverse_gain = 0.0;
    build.config.graph.transport.wormhole_evidence_threshold = 3.0;
    let model = Model::build(
        RelationData {
            dimension: 3,
            concepts: concepts(),
            sources: vec![
                Source::independent(60, vec![vec![0, 1]]),
                Source::independent(61, vec![vec![1, 2]]),
                Source::independent(62, vec![]),
            ],
            relations: vec![Relation {
                from: 3,
                to: 1,
                support: 100.0,
                source: 62,
            }],
        },
        &build,
    )?;
    let mut observe = observe_options();
    observe.config.sense.base_momentum = 1.0;
    observe.config.sense.max_hops = 2;
    let vector = [0.6, 0.1, 0.6];
    let obs = model.observe(Query::new(&vector).with_core_concepts(&[0, 3]), &observe)?;
    assert!(
        obs.transitions()
            .iter()
            .any(|t| t.hop == 1 && (t.from, t.to) == (0, 1))
    );
    assert!(
        obs.transitions()
            .iter()
            .any(|t| t.hop == 2 && (t.previous, t.from, t.to) == (Some(3), 1, 2))
    );
    assert!(
        !obs.transitions()
            .iter()
            .any(|t| t.hop == 2 && (t.previous, t.from, t.to) == (Some(0), 1, 2))
    );
    let candidates = model.candidates(vec![
        Candidate::new(70, vec![0.9, 0.4, 0.0]).with_sources(vec![60]),
        Candidate::new(71, vec![0.7, 0.6, 0.4]).with_sources(vec![61]),
    ])?;
    let stages = [
        ThinkingStage {
            label: "root",
            candidates: &[70],
            limit: 1,
        },
        ThinkingStage {
            label: "must match previous",
            candidates: &[71],
            limit: 1,
        },
    ];
    let plan = obs
        .evaluate(&candidates, &Default::default())?
        .thinking(&stages, &Default::default())?;
    assert_eq!(
        plan.stages[0].selections[0].relation,
        ThinkingRelation::Root
    );
    assert_eq!(
        plan.stages[1].selections[0].relation,
        ThinkingRelation::Supplement
    );
    assert!(plan.stages[1].selections[0].parents.is_empty());
    Ok(())
}
