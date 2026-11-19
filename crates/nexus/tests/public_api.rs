// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::core::{ContextFragmentsView, EpaBasisView, VcpError, VcpResult, VectorMatrixView};
use nexus::math::{
    DtscConfig, DtscReadoutConfig, RrfConfig, RrfInput, RrfOperator, WeightedRanking,
};
use nexus::vcp::{
    ContextAggregateInput, ContextAggregateOperator, ContextConfig, EpaAnalyzeInput,
    EpaAnalyzeOperator, EpaConfig, FieldConfig, ObservationConfig, ThinkingConfig,
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
                .zip(MATRIX.chunks_exact(3))
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
        ctx,
    )?;
    assert!((obs.values().pyramid.coverage - 1.0).abs() < 1e-6);
    assert_eq!(observed.borrow()[0].0, [0.8, 0.6, 0.]);
    assert!((observed.borrow()[1].0[1] - 0.6).abs() < 1e-6);
    assert_eq!(observed.borrow()[1].1, [10]);
    let config = ObservationConfig {
        pyramid: nexus::vcp::PyramidConfig {
            pyramid_top_k: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    for bad in [vec![555], vec![10, 10], vec![10, 20, 30], vec![10]] {
        let recall = |_: &ExecutionContext<'_>, _: TagRecallRequest<'_>, out: &mut Vec<u64>| {
            out.extend(&bad);
            Ok(())
        };
        let bad = Query::new(&[4., 3., 0.], "alpha beta").with_tag_recall(&recall);
        assert!(matches!(
            nexus.observe(&bad, &config, ctx),
            Err(VcpError::InvalidInput { .. })
        ));
    }
    let cancel = |_: &ExecutionContext<'_>, _: TagRecallRequest<'_>, _: &mut Vec<u64>| {
        Err(VcpError::Cancelled { stage: "host" })
    };
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&cancel);
    assert!(matches!(
        nexus.rank_tag_memo(&query, CandidateInput::Ids(&IDS), &config, ctx),
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
        .zip(MATRIX.chunks_exact(3))
        .map(|(&id, row)| Candidate::new(id, row.iter().map(|x| 8. * x).collect()))
        .collect();
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&alpha);
    let obs = nexus.observe(&query, &Default::default(), ctx)?;
    let set = nexus.candidates(CandidateInput::Ids(&IDS))?;
    let eval = obs.evaluate(&set)?;
    let expected = [(100, 1.), (200, 0.), (300, 0.)];
    for input in [
        CandidateInput::Ids(&IDS),
        CandidateInput::Owned(owned),
        CandidateInput::Borrowed(BorrowedCandidates::new(&IDS, matrix)),
    ] {
        let result = nexus.rank_tag_memo(&query, input, &Default::default(), ctx)?;
        assert!(result.items.iter().map(|x| (x.id, x.score)).eq(expected));
    }
    let shuffled = nexus.candidates(CandidateInput::Ids(&[200, 100, 300]))?;
    assert!(
        obs.evaluate(&shuffled)?
            .tag_memo()?
            .items
            .iter()
            .map(|x| (x.id, x.score))
            .eq(expected)
    );
    let wrong_set = other.candidates(CandidateInput::Ids(&IDS))?;
    assert!(matches!(
        obs.evaluate(&wrong_set),
        Err(VcpError::InvalidInput { .. })
    ));
    let second = nexus.observe(&query, &Default::default(), ctx)?;
    let fields = second.fields(&Default::default())?;
    assert!(matches!(
        eval.river(&fields, &Default::default()),
        Err(VcpError::InvalidInput { .. })
    ));
    let field = second.dtsc_field(&Default::default())?;
    assert!(matches!(
        eval.dtsc(&field, &Default::default()),
        Err(VcpError::InvalidInput { .. })
    ));
    let indexes = other.prepare_recall(&Default::default(), &ctx)?;
    assert!(matches!(
        obs.recall(&indexes, &RecallRequest::top_k(2)),
        Err(VcpError::InvalidInput { .. })
    ));
    let indexes = nexus.prepare_recall(&Default::default(), &ctx)?;
    assert!(
        obs.recall(&indexes, &RecallRequest::top_k(0))?
            .ids()
            .is_empty()
    );
    assert_eq!(
        obs.recall(
            &indexes,
            &RecallRequest {
                allowed: Some(&[200]),
                ..RecallRequest::top_k(3)
            }
        )?
        .ids(),
        &[200]
    );

    for input in [
        CandidateInput::Ids(&[100, 100]),
        CandidateInput::Owned(vec![
            Candidate::new(100, vec![1., 0., 0.]),
            Candidate::new(100, vec![0., 1., 0.]),
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
fn evaluation_reuses_original_and_rank_methods_delegate() -> VcpResult<()> {
    let nexus = orthogonal()?;
    let ctx = ExecutionContext::default();
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&alpha);
    let changed = Query {
        core_concepts: &[20],
        text: "alpha beta",
        ..query
    };
    let next = nexus.observe(&changed, &Default::default(), ctx)?;
    let set = nexus.candidates(CandidateInput::Ids(&IDS))?;
    let reused = {
        let previous = nexus.observe(&query, &Default::default(), ctx)?;
        assert_ne!(
            previous.values().fused.enhanced_query,
            next.values().fused.enhanced_query
        );
        next.evaluate_reusing(&previous.evaluate(&set)?)?
    };
    let direct = next.evaluate(&set)?;
    assert_eq!(direct.tag_memo()?.items, reused.tag_memo()?.items);
    let other = nexus.observe(
        &Query {
            vector: &[0., 1., 0.],
            ..query
        },
        &Default::default(),
        ctx,
    )?;
    assert!(matches!(
        other.evaluate_reusing(&reused),
        Err(VcpError::InvalidInput { .. })
    ));
    let river = RiverQueryConfig::default();
    let fields = next.fields(&river.fields)?;
    let prepared = direct.prepare_river(&fields)?;
    assert_eq!(
        prepared.readout(&river.readout)?.items,
        nexus
            .rank_river(&changed, CandidateInput::Ids(&IDS), &river, ctx)?
            .items
    );
    let mut alternate = river.readout.clone();
    alternate.scoring.unified_budget_ct = 0.04;
    assert!(
        prepared
            .readout(&alternate)?
            .items
            .iter()
            .all(|item| item.topology_bonus <= 0.04)
    );
    let config = DtscConfig::default();
    let field = next.dtsc_field(&config.field)?;
    let evidence = direct.dtsc_evidence(&field, &config.scoring.evidence)?;
    let explicit = evidence.readout(&config.scoring.readout)?;
    let thin = nexus.rank_dtsc(
        &changed,
        CandidateInput::Ids(&IDS),
        &DtscQueryConfig::default(),
        ctx,
    )?;
    assert!(
        explicit
            .items
            .iter()
            .map(|item| (item.id, item.score))
            .eq(thin.items.iter().map(|item| (item.id, item.score)))
    );
    let top = evidence.readout(&DtscReadoutConfig {
        top_k: 1,
        ..config.scoring.readout
    })?;
    assert_eq!((top.items.len(), top.available), (1, 3));
    assert!(!top.diagnostics.truncated);
    Ok(())
}

#[test]
fn context_to_epa_and_river_to_rrf_in_plan() -> VcpResult<()> {
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
    let direct = ContextAggregateOperator.execute(&ctx, &cfg, input)?;
    let direct = EpaAnalyzeOperator.execute(
        &ctx,
        &epa_config,
        EpaAnalyzeInput {
            query: &direct.effective_vector,
            basis,
        },
    )?;
    let nexus = orthogonal()?;
    let query = Query::new(&[1., 0., 0.], "alpha").with_tag_recall(&alpha);
    let mut plan = Plan::new();
    let context = plan.add((), |ctx, _| {
        ContextAggregateOperator.execute(ctx, &cfg, input)
    })?;
    let epa = plan.add(context, |ctx, input| {
        EpaAnalyzeOperator.execute(
            ctx,
            &epa_config,
            EpaAnalyzeInput {
                query: &input.get().effective_vector,
                basis,
            },
        )
    })?;
    let river = plan.add((), |ctx, _| {
        nexus.rank_river(&query, CandidateInput::Ids(&IDS), &Default::default(), *ctx)
    })?;
    let fused = plan.add(river, |ctx, input| {
        let ids = input
            .get()
            .items
            .iter()
            .map(|item| item.id)
            .collect::<Vec<_>>();
        RrfOperator.execute(
            ctx,
            &RrfConfig::default(),
            RrfInput {
                rankings: &[
                    WeightedRanking {
                        ids: &ids,
                        weight: 1.,
                    },
                    WeightedRanking {
                        ids: &[200, 900],
                        weight: 1.,
                    },
                ],
            },
        )
    })?;
    let outputs = plan.run((epa, river, fused), &ctx)?;
    assert_eq!(outputs.get(epa)?, &direct);
    assert_eq!(direct.spectral_concentration, 1.0);
    assert!(outputs.get(river)?.items.iter().map(|item| item.id).eq(IDS));
    assert!(
        outputs
            .get(fused)?
            .items
            .iter()
            .map(|item| item.id)
            .eq([200, 100, 900, 300])
    );
    Ok(())
}

#[test]
fn thinking_in_plan_shares_observation_trace_and_cancellation() -> VcpResult<()> {
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
    let obs = nexus.observe(&query, &Default::default(), ctx)?;
    let set = nexus.candidates(CandidateInput::Ids(&[100]))?;
    let eval = obs.evaluate(&set)?;
    let stages = [Stage {
        name: "Explore",
        limit: 1,
        candidates: &[100],
    }];
    let config = ThinkingConfig::default();
    let direct = eval.thinking(&stages, &config)?;
    let actual = &direct.stages[0].results[0];
    assert_eq!(actual.document_id, 99);
    let route = actual.route.expect("a grounded transition");
    assert_eq!((route.source, route.target), (10, 20));
    let mut plan = Plan::new();
    let result = plan.add((), |_, _| eval.thinking(&stages, &config))?;
    assert_eq!(plan.run(result, &ctx)?.get(result)?.stages, direct.stages);

    let no_trace = ObservationConfig {
        sense: nexus::vcp::SenseConfig {
            max_transition_records: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let missing = nexus.observe(&query, &no_trace, ctx)?;
    assert!(matches!(
        missing.evaluate(&set)?.thinking(&stages, &config),
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
        obs.fields(&fields),
        Err(VcpError::NotConverged { .. })
    ));
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        eval.thinking(&stages, &config),
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
            sources: vec![SourceFact {
                id: 500,
                checksum: 1,
                tags: vec![10, 20],
                required_sources: vec![],
            }],
            documents: vec![Document {
                id: 100,
                document_id: None,
                vector: vec![1., 0.],
                paths: vec![
                    Sequence {
                        source: 500,
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
        ctx,
    )?;
    let selected = &result.items[0];
    assert_eq!(
        selected.chosen_path,
        Some(PathRef {
            candidate_id: 100,
            path_index: 0
        })
    );
    assert!(
        (selected.score - 0.87567).abs() < 1e-5,
        "{}",
        selected.score
    );
    assert!(selected.path_bonus + selected.graph_bonus <= 0.08);
    assert_eq!(
        selected.score,
        (selected.base_score + selected.path_bonus + selected.graph_bonus).clamp(0., 1.)
    );
    Ok(())
}
