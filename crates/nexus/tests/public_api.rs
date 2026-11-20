// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::core::{ContextFragmentsView, EpaBasisView, VcpError, VcpResult, VectorMatrixView};
use nexus::math::{DtscReadoutConfig, RrfConfig, WeightedRanking, rrf};
use nexus::vcp::{
    ContextAggregateInput, ContextConfig, EpaAnalyzeInput, EpaConfig, FieldConfig,
    ObservationConfig, PropagationCompletion, RiverScoringConfig, ThinkingConfig, aggregate,
    analyze_epa,
};
use nexus::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

const IDS: [u64; 3] = [100, 200, 300];
const MATRIX: [f32; 9] = [1., 0., 0., 0., 1., 0., 0., 0., 1.];

fn concept(id: u64, name: &str, vector: &[f32]) -> Concept {
    Concept {
        id,
        name: name.into(),
        vector: vector.into(),
        aliases: vec![],
    }
}

fn orthogonal() -> VcpResult<Nexus<'static>> {
    Nexus::prepare(
        Dataset {
            concepts: vec![
                concept(30, "gamma", &[0., 0., 8.]),
                concept(10, "alpha", &[f32::from_bits(1), 0., 0.]),
                concept(20, "beta", &[0., f32::MAX, 0.]),
            ],
            documents: IDS
                .iter()
                .zip(MATRIX.as_chunks::<3>().0)
                .map(|(&id, row)| Document {
                    id,
                    document_id: None,
                    vector: row.iter().map(|x| x * 2.).collect(),
                    paths: vec![],
                })
                .collect(),
            ..Default::default()
        },
        &Default::default(),
        &ExecutionContext::default(),
    )
}

fn alpha(
    _: &ExecutionContext<'_>,
    request: TagRecallRequest<'_>,
    out: &mut Vec<u64>,
) -> VcpResult<()> {
    if request.layer == 0 {
        out.push(10);
    }
    Ok(())
}

#[test]
fn host_recall_uses_business_ids_and_preserves_residual_energy() -> VcpResult<()> {
    let nexus = orthogonal()?;
    let ctx = ExecutionContext::default();
    let observed = RefCell::new(Vec::new());
    let recall = |_: &ExecutionContext<'_>, req: TagRecallRequest<'_>, out: &mut Vec<u64>| {
        observed
            .borrow_mut()
            .push((req.residual.to_vec(), req.excluded_concepts.to_vec()));
        match req.layer {
            0 => out.push(10),
            1 => out.push(20),
            _ => {}
        }
        Ok(())
    };
    let obs = nexus.observe(
        &Query::new(&[4., 3., 0.], "alpha beta").with_tag_recall(&recall),
        &Default::default(),
        &ctx,
    )?;
    assert!((obs.values().pyramid.coverage - 1.0).abs() < 1e-6);
    assert_eq!(observed.borrow()[0].0, [0.8, 0.6, 0.]);
    assert!((observed.borrow()[1].0[1] - 0.6).abs() < 1e-6);
    assert_eq!(observed.borrow()[1].1, [10]);
    let config = ObservationConfig {
        pyramid: Some(nexus::vcp::PyramidConfig {
            pyramid_top_k: 2,
            ..Default::default()
        }),
        ..Default::default()
    };
    for bad in [vec![555], vec![10, 10], vec![10, 20, 30], vec![10]] {
        let recall = |_: &ExecutionContext<'_>, _: TagRecallRequest<'_>, out: &mut Vec<u64>| {
            out.extend(&bad);
            Ok(())
        };
        let bad = Query::new(&[4., 3., 0.], "alpha beta").with_tag_recall(&recall);
        assert!(matches!(
            nexus.observe(&bad, &config, &ctx),
            Err(VcpError::InvalidInput { .. })
        ));
    }
    let cancel = |_: &ExecutionContext<'_>, _: TagRecallRequest<'_>, _: &mut Vec<u64>| {
        Err(VcpError::Cancelled { stage: "host" })
    };
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&cancel);
    assert!(matches!(
        nexus.rank_tag_memo(&query, CandidateInput::Ids(&IDS), &config, &ctx),
        Err(VcpError::Cancelled { stage: "host" })
    ));
    Ok(())
}

#[test]
fn ordered_candidates_and_bound_results_use_real_parent_objects() -> VcpResult<()> {
    let nexus = orthogonal()?;
    let other = orthogonal()?;
    let ctx = ExecutionContext::default();
    let matrix = VectorMatrixView::new(&MATRIX, 3, 3)?;
    let owned = IDS
        .iter()
        .zip(MATRIX.as_chunks::<3>().0)
        .map(|(&id, row)| Document::new(id, row.iter().map(|x| 8. * x).collect()))
        .collect();
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&alpha);
    let obs = nexus.observe(&query, &Default::default(), &ctx)?;
    let set = nexus.candidates(CandidateInput::Ids(&IDS))?;
    let eval = obs.evaluate(&set, &ctx)?;
    let expected = [(100, 1.), (200, 0.), (300, 0.)];
    for input in [
        CandidateInput::Ids(&IDS),
        CandidateInput::Owned(owned),
        CandidateInput::Borrowed(BorrowedCandidates::new(&IDS, matrix)),
    ] {
        let result = nexus.rank_tag_memo(&query, input, &Default::default(), &ctx)?;
        assert!(result.items.iter().map(|x| (x.id, x.score)).eq(expected));
    }
    let shuffled = nexus.candidates(CandidateInput::Ids(&[200, 100, 300]))?;
    assert!(
        obs.evaluate(&shuffled, &ctx)?
            .tag_memo(&ctx)?
            .items
            .iter()
            .map(|x| (x.id, x.score))
            .eq(expected)
    );
    let wrong_set = other.candidates(CandidateInput::Ids(&IDS))?;
    assert!(matches!(
        obs.evaluate(&wrong_set, &ctx),
        Err(VcpError::InvalidInput { .. })
    ));
    let second = nexus.observe(&query, &Default::default(), &ctx)?;
    let fields = second.fields(&Default::default(), &ctx)?;
    assert!(matches!(
        eval.river_evidence(&fields, &Default::default(), &ctx),
        Err(VcpError::InvalidInput { .. })
    ));
    let field = second.dtsc_field(&Default::default(), &ctx)?;
    assert!(matches!(
        eval.dtsc_evidence(&field, &Default::default(), &ctx),
        Err(VcpError::InvalidInput { .. })
    ));
    let indexes = other.prepare_recall(&Default::default(), &ctx)?;
    assert!(matches!(
        obs.recall(&indexes, &RecallRequest::top_k(2), &ctx),
        Err(VcpError::InvalidInput { .. })
    ));
    let indexes = nexus.prepare_recall(&Default::default(), &ctx)?;
    assert!(
        obs.recall(&indexes, &RecallRequest::top_k(0), &ctx)?
            .ids()
            .is_empty()
    );
    assert_eq!(
        obs.recall(
            &indexes,
            &RecallRequest {
                allowed: Some(&[200]),
                ..RecallRequest::top_k(3)
            },
            &ctx,
        )?
        .ids(),
        &[200]
    );

    for input in [
        CandidateInput::Ids(&[100, 100]),
        CandidateInput::Owned(vec![
            Document::new(100, vec![1., 0., 0.]),
            Document::new(100, vec![0., 1., 0.]),
        ]),
        CandidateInput::Borrowed(BorrowedCandidates::new(&[100, 100, 200], matrix)),
    ] {
        assert!(matches!(
            nexus.candidates(input),
            Err(VcpError::InvalidInput { .. })
        ));
    }
    let bad = VectorMatrixView::new(&[2., 0., 0.], 1, 3)?;
    assert!(matches!(
        nexus.candidates(CandidateInput::Borrowed(BorrowedCandidates::new(
            &[100],
            bad
        ))),
        Err(VcpError::InvalidInput { .. })
    ));
    assert!(matches!(
        nexus.candidates(CandidateInput::Borrowed(
            BorrowedCandidates::new(&IDS, matrix).with_document_ids(&[100])
        )),
        Err(VcpError::InvalidInput { .. })
    ));
    Ok(())
}

#[test]
fn built_in_indexes_honor_query_scope_and_concept_limit() -> VcpResult<()> {
    let ctx = ExecutionContext::default();
    let nexus = Nexus::prepare(
        Dataset {
            concepts: (1..=65)
                .map(|id| Concept::new(id, format!("concept {id}"), vec![1., 0.]))
                .collect(),
            documents: vec![
                Document::new(100, vec![1., 0.]),
                Document {
                    document_id: Some(700),
                    ..Document::new(200, vec![0., 1.])
                },
            ],
            ..Default::default()
        },
        &Default::default(),
        &ctx,
    )?;
    let indexes = nexus.prepare_recall(&Default::default(), &ctx)?;
    let observation = indexes.observe(
        &Query::new(&[1., 0.], "concept"),
        &ObservationConfig {
            pyramid: Some(nexus::vcp::PyramidConfig {
                pyramid_top_k: 65,
                ..Default::default()
            }),
            ..Default::default()
        },
        &ctx,
    )?;
    assert_eq!(observation.values().pyramid.levels[0].tags.len(), 65);
    let unrestricted = RecallRequest::top_k(1);
    let candidates = observation.recall(&indexes, &unrestricted, &ctx)?;
    assert_eq!(candidates.ids(), &[100]);
    assert_eq!(
        observation
            .evaluate(&candidates, &ctx)?
            .river(&Default::default(), &ctx)?
            .items[0]
            .id,
        100
    );
    let hits = indexes.search_documents(&[0., 4.], &unrestricted, &ctx)?;
    assert_eq!((hits[0].id, hits[0].document_id), (200, 700));
    let restricted = RecallRequest {
        allowed: Some(&[100]),
        ..unrestricted
    };
    assert_eq!(
        indexes.search_documents(&[0., 4.], &restricted, &ctx)?[0].id,
        100
    );
    Ok(())
}

#[test]
fn owned_evidence_survives_parent_assets_and_old_context() -> VcpResult<()> {
    let nexus = orthogonal()?;
    let river_config = RiverQueryConfig::default();
    let dtsc_config = DtscQueryConfig::default();
    let fresh = ExecutionContext::default();
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&alpha);
    let expected = nexus.rank_river(&query, CandidateInput::Ids(&IDS), &river_config, &fresh)?;
    let scored = nexus.rank_dtsc(&query, CandidateInput::Ids(&IDS), &dtsc_config, &fresh)?;
    let (river, dtsc) = {
        let cancelled = AtomicBool::new(false);
        let old = ExecutionContext {
            cancel: CancellationToken::from_atomic(&cancelled),
            ..Default::default()
        };
        let set = nexus.candidates(CandidateInput::Borrowed(BorrowedCandidates::new(
            &IDS,
            VectorMatrixView::new(&MATRIX, 3, 3)?,
        )))?;
        let observation = nexus.observe(&query, &river_config.observation, &old)?;
        cancelled.store(true, Ordering::Relaxed);
        assert!(matches!(
            observation.evaluate(&set, &old),
            Err(VcpError::Cancelled { .. })
        ));
        let evaluation = observation.evaluate(&set, &fresh)?;
        let fields = observation.fields(&river_config.river.fields, &fresh)?;
        let field = observation.dtsc_field(&dtsc_config.dtsc.field, &fresh)?;
        (
            evaluation.river_evidence(&fields, &river_config.river.analysis, &fresh)?,
            evaluation.dtsc_evidence(&field, &dtsc_config.dtsc.scoring.evidence, &fresh)?,
        )
    };
    drop(nexus);
    assert_eq!(
        river.readout(&river_config.river.scoring, &fresh)?.items,
        expected.items
    );
    let retuned = river.readout(
        &RiverScoringConfig {
            unified_budget_ct: 0.04,
            ..river_config.river.scoring
        },
        &fresh,
    )?;
    assert!(
        retuned
            .items
            .iter()
            .all(|item| item.scores.topology_bonus <= 0.04)
    );
    let result = dtsc.readout(&dtsc_config.dtsc.scoring.readout, &fresh)?;
    assert!(
        result
            .items
            .iter()
            .map(|item| (item.id, item.scores.score))
            .eq(scored.items.iter().map(|item| (item.id, item.scores.score)))
    );
    let top = dtsc.readout(
        &DtscReadoutConfig {
            top_k: 1,
            ..Default::default()
        },
        &fresh,
    )?;
    assert_eq!((top.items.len(), top.available), (1, 3));
    assert_eq!(top.propagation, PropagationCompletion::Complete);
    Ok(())
}

#[test]
fn context_to_epa_and_river_to_rrf() -> VcpResult<()> {
    let ctx = ExecutionContext::default();
    let fragments = ContextFragmentsView {
        count: 1,
        keys: &["fragment"],
        roles: &["user"],
        stages: &["query"],
        parent_keys: &[None],
        sequences: &[0],
        weights: &[1.],
        relevances: &[1.],
        validities: &[1.],
        distances: &[0.],
        half_lives: &[None],
        source_keys: &[None],
        source_ranges: &[None],
    };
    let input = ContextAggregateInput {
        raw_query: Some(&[1., 0.]),
        fragments,
        fragment_vectors: VectorMatrixView::new(&[1., 0.], 1, 2)?,
    };
    let cfg = ContextConfig::default();
    let epa_config = EpaConfig::default();
    let basis = Some(EpaBasisView {
        dim: 2,
        k_sem: 1,
        mean: &[0., 0.],
        basis: &[1., 0., 0., 1.],
        basis_count: 2,
        coverage: &[0.75, 1.],
    });
    let direct = aggregate(input, &cfg, &ctx)?;
    let direct = analyze_epa(
        EpaAnalyzeInput {
            query: &direct.effective_vector,
            basis,
        },
        &epa_config,
        &ctx,
    )?;
    let nexus = orthogonal()?;
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&alpha);
    let river = nexus.rank_river(&query, CandidateInput::Ids(&IDS), &Default::default(), &ctx)?;
    let ids: Vec<_> = river.items.iter().map(|item| item.id).collect();
    let fused = rrf(
        &[
            WeightedRanking {
                ids: &ids,
                weight: 1.,
            },
            WeightedRanking {
                ids: &[200, 900],
                weight: 1.,
            },
        ],
        &RrfConfig::default(),
        &ctx,
    )?;
    assert_eq!(direct.spectral_concentration, 1.0);
    assert_eq!(ids, IDS);
    assert!(
        fused
            .items
            .iter()
            .map(|item| item.id)
            .eq([200, 100, 900, 300])
    );
    Ok(())
}

#[test]
fn thinking_shares_observation_trace_and_cancellation() -> VcpResult<()> {
    let nexus = Nexus::prepare(
        Dataset {
            concepts: vec![
                concept(10, "alpha", &[1., 0.]),
                concept(20, "beta", &[0.8, 0.6]),
            ],
            sources: vec![SourceFact {
                id: 1000,
                checksum: 1,
                tags: vec![10, 20],
                required_sources: vec![],
            }],
            documents: vec![Document {
                id: 100,
                document_id: Some(99),
                vector: vec![1., 0.],
                paths: vec![Sequence {
                    source: 1000,
                    tags: vec![(10, 0), (20, 1)],
                }],
            }],
            ..Default::default()
        },
        &Default::default(),
        &ExecutionContext::default(),
    )?;
    let cancel = AtomicBool::new(false);
    let ctx = ExecutionContext {
        cancel: CancellationToken::from_atomic(&cancel),
        ..Default::default()
    };
    let query = Query {
        core_concepts: &[10],
        ..Query::new(&[1., 0.], "alpha").with_tag_recall(&alpha)
    };
    let obs = nexus.observe(&query, &Default::default(), &ctx)?;
    let set = nexus.candidates(CandidateInput::Ids(&[100]))?;
    let eval = obs.evaluate(&set, &ctx)?;
    let stages = [Stage {
        name: "Explore",
        limit: 1,
        candidates: &[100],
    }];
    let config = ThinkingConfig::default();
    let direct = eval.thinking(&stages, &config, &ctx)?;
    let actual = &direct.stages[0].results[0];
    assert_eq!(actual.document_id, 99);
    let route = actual.route.expect("a grounded transition");
    assert_eq!((route.source, route.target), (10, 20));
    let no_trace = ObservationConfig {
        sense: nexus::vcp::SenseConfig {
            max_transition_records: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let missing = nexus.observe(&query, &no_trace, &ctx)?;
    assert!(matches!(
        missing
            .evaluate(&set, &ctx)?
            .thinking(&stages, &config, &ctx),
        Err(VcpError::ThinkingUnavailable { .. })
    ));
    let fields = FieldConfig {
        solve: nexus::vcp::DualFieldConfig {
            max_iterations: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        obs.fields(&fields, &ctx),
        Err(VcpError::NotConverged { .. })
    ));
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        eval.thinking(&stages, &config, &ctx),
        Err(VcpError::Cancelled { .. })
    ));
    Ok(())
}

#[test]
fn river_uses_one_complete_maximum_scoring_path() -> VcpResult<()> {
    let ctx = ExecutionContext::default();
    let nexus = Nexus::prepare(
        Dataset {
            concepts: vec![
                concept(10, "narrow concept", &[1., 0.]),
                concept(20, "concept", &[0.66, (1.0_f32 - 0.66_f32.powi(2)).sqrt()]),
            ],
            sources: vec![
                SourceFact {
                    id: 500,
                    checksum: 1,
                    tags: vec![10, 20],
                    required_sources: vec![],
                },
                SourceFact {
                    id: 600,
                    checksum: 2,
                    tags: vec![10, 20],
                    required_sources: vec![],
                },
            ],
            documents: vec![Document {
                id: 100,
                document_id: None,
                vector: vec![1., 0.],
                paths: vec![
                    Sequence {
                        source: 600,
                        tags: vec![(20, 0)],
                    },
                    Sequence {
                        source: 500,
                        tags: vec![(10, 0)],
                    },
                ],
            }],
            ..Default::default()
        },
        &Default::default(),
        &ctx,
    )?;
    let recall = |_: &ExecutionContext<'_>, req: TagRecallRequest<'_>, out: &mut Vec<u64>| {
        if req.layer == 0 {
            out.extend([10, 20]);
        }
        Ok(())
    };
    let query = Query::new(&[1., 0.], "narrow concept").with_tag_recall(&recall);
    let result = nexus.rank_river(
        &query,
        CandidateInput::Ids(&[100]),
        &Default::default(),
        &ctx,
    )?;
    let selected = &result.items[0];
    assert_eq!(
        selected.chosen_path,
        Some(PathRef {
            candidate_id: 100,
            path_index: 0,
            source_id: Some(600),
        })
    );
    assert!(
        (selected.scores.score - 0.87564).abs() < 1e-5,
        "{}",
        selected.scores.score
    );
    assert!(
        selected.scores.path_bonus + selected.scores.graph_bonus
            <= RiverScoringConfig::default().unified_budget_ct
    );
    assert_eq!(
        selected.scores.score,
        (selected.scores.base_score + selected.scores.path_bonus + selected.scores.graph_bonus)
            .clamp(0., 1.)
    );
    let imported = Nexus::from_prepared(nexus.prepared())?;
    assert_eq!(
        imported
            .rank_river(
                &query,
                CandidateInput::Ids(&[100]),
                &Default::default(),
                &ctx
            )?
            .items[0]
            .chosen_path,
        selected.chosen_path,
    );
    Ok(())
}
#[test]
fn asset_roundtrip_retains_residuals_and_host_only_assets_skip_indexes() -> VcpResult<()> {
    let ctx = ExecutionContext::default();
    let owned = Nexus::prepare(
        Dataset {
            concepts: vec![
                concept(10, "alpha", &[1., 0.]),
                concept(20, "beta", &[1., 0.]),
                concept(30, "other", &[0., 1.]),
            ],
            sources: vec![SourceFact {
                id: 1,
                checksum: 1,
                tags: vec![10, 20],
                required_sources: vec![],
            }],
            relations: [(10, 20), (20, 10)]
                .into_iter()
                .map(|(from, to)| Relation {
                    from,
                    to,
                    source: 1,
                    evidence: RelationEvidence::Positive { mass: 1.0 },
                })
                .collect(),
            ..Default::default()
        },
        &AssetConfig {
            local: nexus::vcp::LocalConeAgsConfig {
                min_neighbors: 1,
                ..Default::default()
            },
            ..Default::default()
        },
        &ctx,
    )?;
    let before = owned.prepared();
    let borrowed = Nexus::from_prepared(before)?;
    let after = borrowed.prepared();
    let radii = |assets: PreparedDataset<'_>| -> VcpResult<Vec<f64>> {
        let residuals = assets.tag_residuals.expect("prepared geometry");
        assert!(residuals.iter().map(|r| r.residual_ratio).eq([0., 0., 1.]));
        let index = nexus::vcp::hyperbolic::tag_field(
            assets.tags,
            assets.basis.map_or(&[], |basis| basis.mean),
            assets.graph.inbound,
            residuals,
            Default::default(),
            &ctx,
        )?;
        Ok((0..assets.tags.rows)
            .map(|i| index.radius_of(i).unwrap())
            .collect())
    };
    assert_eq!(radii(before)?, radii(after)?);
    let config = RecallConfig {
        concepts: Some(Default::default()),
        documents: None,
        ..Default::default()
    };
    borrowed.prepare_recall(&config, &ctx)?;
    let host_only = Nexus::from_prepared(PreparedDataset {
        tag_residuals: None,
        ..after
    })?;
    assert!(matches!(
        host_only.prepare_recall(&config, &ctx),
        Err(VcpError::InvalidInput {
            stage: "recall",
            ..
        })
    ));
    let query = Query::new(&[1., 0.], "alpha").with_tag_recall(&alpha);
    assert_eq!(
        host_only
            .observe(&query, &Default::default(), &ctx)?
            .values()
            .pyramid
            .coverage,
        1.
    );
    Ok(())
}

#[test]
fn request_attenuation_reuses_candidates_and_readouts_report_propagation_loss() -> VcpResult<()> {
    let ctx = ExecutionContext::default();
    let nexus = Nexus::prepare(
        Dataset {
            concepts: (10..=13)
                .map(|id| concept(id, "concept", &[1., 0.]))
                .collect(),
            documents: vec![Document {
                id: 100,
                document_id: Some(u64::MAX),
                vector: vec![1., 0.],
                paths: vec![Sequence {
                    source: 1,
                    tags: vec![(10, 0), (11, 1)],
                }],
            }],
            sources: vec![SourceFact {
                id: 1,
                checksum: 1,
                tags: vec![10, 11],
                required_sources: vec![],
            }],
            ..Default::default()
        },
        &Default::default(),
        &ctx,
    )?;
    let assets = nexus.prepared();
    let candidates = nexus.candidates(CandidateInput::Ids(&[100]))?;
    let query = Query {
        core_concepts: &[10],
        ..Query::new(&[1., 0.], "concept")
    };
    let cfg = ObservationConfig {
        pyramid: None,
        ..Default::default()
    };
    let plain = nexus.observe(&query, &cfg, &ctx)?;
    let neutral = {
        let gates = vec![1.; assets.graph.nnz()];
        nexus.observe(&query.with_edge_gates(&gates), &cfg, &ctx)?
    };
    let half = {
        let gates = vec![0.5; assets.graph.nnz()];
        nexus.observe(&query.with_edge_gates(&gates), &cfg, &ctx)?
    };
    assert_eq!(neutral.graph().weights, plain.graph().weights);
    assert_eq!(half.graph().inbound, plain.graph().inbound);
    assert_eq!(half.graph().targets, plain.graph().targets);
    assert!(half.graph().edge_gates.is_none());
    assert!(
        half.graph()
            .weights
            .iter()
            .zip(plain.graph().weights)
            .all(|(&a, &b)| a == b * 0.5)
    );
    assert_eq!(
        neutral.evaluate(&candidates, &ctx)?.tag_memo(&ctx)?.items[0].document_id,
        u64::MAX
    );
    let fields = half.fields(&Default::default(), &ctx)?;
    let baseline = plain.fields(&Default::default(), &ctx)?;
    let target = assets.concept_ids.iter().position(|&id| id == 11).unwrap();
    let mass = fields.values().dual_fields.transfer_field[target];
    assert!(mass > 0.0 && mass < baseline.values().dual_fields.transfer_field[target]);
    half.evaluate(&candidates, &ctx)?
        .river_evidence(&fields, &Default::default(), &ctx)?
        .readout(&Default::default(), &ctx)?;

    let query = Query {
        core_concepts: &[10, 11, 12, 13],
        ..query
    };
    let limited = ObservationConfig {
        sense: nexus::vcp::SenseConfig {
            max_propagation_states: 3,
            max_safe_hops: 0,
            ..cfg.sense
        },
        ..cfg
    };
    let expected = PropagationCompletion::Limited { dropped_states: 1 };
    let river = nexus.rank_river(
        &query,
        CandidateInput::Ids(&[100]),
        &RiverQueryConfig {
            observation: limited.clone(),
            ..Default::default()
        },
        &ctx,
    )?;
    let dtsc = nexus.rank_dtsc(
        &query,
        CandidateInput::Ids(&[100]),
        &DtscQueryConfig {
            observation: limited.clone(),
            ..Default::default()
        },
        &ctx,
    )?;
    let tag = nexus.rank_tag_memo(&query, CandidateInput::Ids(&[100]), &limited, &ctx)?;
    assert_eq!(
        [river.propagation, dtsc.propagation, tag.propagation],
        [expected; 3]
    );
    let obs = nexus.observe(&query, &limited, &ctx)?;
    assert!(
        matches!(obs.values().sense.trace, nexus::vcp::Trace::Complete(ref trace) if trace.is_empty())
    );
    assert!(matches!(
        obs.values().sense.complete_trace(),
        Err(VcpError::ThinkingUnavailable { .. })
    ));

    let no_work = ObservationConfig {
        pyramid: Some(nexus::vcp::PyramidConfig {
            pyramid_top_k: 0,
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(
        nexus
            .observe(&query, &no_work, &ctx)?
            .values()
            .pyramid
            .levels
            .is_empty()
    );
    assert!(matches!(
        nexus.observe(&query, &Default::default(), &ctx),
        Err(VcpError::InvalidInput { .. })
    ));
    Ok(())
}
