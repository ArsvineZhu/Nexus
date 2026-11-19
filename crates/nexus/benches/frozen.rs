// SPDX-License-Identifier: Apache-2.0

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nexus::*;
use nexus_core::kernel::{cosine_f32, norm_sq_f32};
use nexus_core::*;
use nexus_vcp::*;
use serde::Deserialize;
use std::cell::RefCell;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture {
    dim: usize,
    docs: Vec<DocumentId>,
    candidate_root_offsets: Vec<u32>,
    candidate_roots: Vec<RootId>,
    candidate_path_offsets: Vec<u32>,
    path_tag_offsets: Vec<u32>,
    path_tags: Vec<NodeId>,
    path_positions: Vec<u32>,
    labels: Vec<String>,
    alias_offsets: Vec<u32>,
    aliases: Vec<String>,
    graph: CsrStorage,
    provenance_offsets: Vec<u32>,
    provenance_groups: Vec<GroupId>,
    provenance_masses: Vec<f64>,
    group_offsets: Vec<u32>,
    group_roots: Vec<RootId>,
    group_kinds: Vec<SupportKind>,
    basis: Option<EpaBasis>,
    query_texts: Vec<String>,
}

fn vectors(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read frozen vector file");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

struct QueryCase {
    scores: Vec<f64>,
    ranks: Vec<u32>,
    recalled: Vec<Vec<(NodeId, f64)>>,
    stage: Vec<CandidateId>,
}

impl Fixture {
    fn candidates<'a>(
        &'a self,
        matrix: VectorMatrixView<'a>,
        zeros: &'a [f64],
        query: &'a QueryCase,
    ) -> CandidatePoolView<'a> {
        CandidatePoolView {
            candidate_count: self.docs.len(),
            baseline_scores: &query.scores,
            union_scores: &query.scores,
            union_ranks: &query.ranks,
            bm25_scores: zeros,
            bm25_reference: 0.0,
            anchor_scores: zeros,
            curve_document_ids: &self.docs,
            candidate_vectors: matrix,
            candidate_root_offsets: &self.candidate_root_offsets,
            candidate_roots: &self.candidate_roots,
            candidate_path_offsets: &self.candidate_path_offsets,
            path_tag_offsets: &self.path_tag_offsets,
            path_tags: &self.path_tags,
            path_positions: &self.path_positions,
        }
    }
}

fn run_frozen(criterion: &mut Criterion) {
    let Some(path) = std::env::var_os("NEXUS_FROZEN_FIXTURE") else {
        eprintln!("Set NEXUS_FROZEN_FIXTURE to an exported dataset. See docs/benchmarks.md.");
        return;
    };
    let path = Path::new(&path);
    let spec = std::fs::read(path.join("fixture.json")).expect("read frozen fixture metadata");
    let fixture: Fixture = serde_json::from_slice(&spec).expect("decode frozen fixture metadata");
    let candidate_values = vectors(&path.join("candidates.f32"));
    let tag_values = vectors(&path.join("tags.f32"));
    let query_values = vectors(&path.join("queries.f32"));
    let count = fixture.docs.len();
    let dim = fixture.dim;
    let queries: Vec<_> = query_values.chunks_exact(dim).collect();
    assert_eq!(queries.len(), fixture.query_texts.len());
    assert!(!queries.is_empty(), "fixture must contain queries");
    let threads = std::env::var("NEXUS_BENCH_THREADS").map_or(1, |value| {
        value
            .parse::<usize>()
            .expect("NEXUS_BENCH_THREADS must be a positive integer")
    });
    assert!(threads > 0);
    let pool = (threads > 1).then(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build benchmark Rayon pool")
    });
    let ctx = ExecutionContext::default().with_parallel(
        pool.as_ref()
            .map_or(ParallelPolicy::Sequential, ParallelPolicy::CustomPool),
    );
    let zeros = vec![0.0; count];
    let norms: Vec<_> = candidate_values
        .chunks_exact(dim)
        .map(|row| norm_sq_f32(row, ctx.precision))
        .collect();
    let candidates = VectorMatrixView::new(&candidate_values, count, dim)
        .unwrap()
        .with_sq_norms(&norms)
        .unwrap();
    let tags = VectorMatrixView::new(&tag_values, fixture.labels.len(), dim).unwrap();
    let config = ObservationConfig {
        sense: SenseConfig {
            max_transition_records: 16_000,
            ..Default::default()
        },
        ..Default::default()
    };
    let graph = CsrGraph::new(fixture.graph.clone()).unwrap();
    let lexical = TagLexicalMetaView {
        node_count: tags.rows,
        names: TextRows::Strings(&fixture.labels),
        alias_offsets: &fixture.alias_offsets,
        aliases: TextRows::Strings(&fixture.aliases),
    };
    let provenance = ProvenanceView {
        edge_count: fixture.graph.targets.len(),
        edge_prov_offsets: &fixture.provenance_offsets,
        prov_groups: &fixture.provenance_groups,
        prov_masses: &fixture.provenance_masses,
    };
    let groups = SupportGroupView {
        group_count: fixture.group_kinds.len(),
        group_root_offsets: &fixture.group_offsets,
        group_roots: &fixture.group_roots,
        group_kinds: &fixture.group_kinds,
    };
    let assets = PreparedDataset {
        concept_ids: &[],
        tags,
        graph: graph.view(),
        lexical,
        basis: fixture.basis.as_ref().map(EpaBasis::as_view),
        provenance,
        groups,
        documents: None,
        tag_search: None,
        sources: RootFactView::empty(),
        counterevidence: CounterevidenceAdjView::empty(),
    };
    let nexus = Nexus::from_prepared(assets).unwrap();
    let mut cases = Vec::with_capacity(queries.len());
    let mut route_queries = Vec::new();
    for (index, query) in queries.iter().enumerate() {
        let recorded = RefCell::new(Vec::new());
        let recall = |_,
                      residual: &[f32],
                      limit,
                      out: &mut Vec<_>,
                      _: Option<&EpaAnalysis>,
                      excluded: &[NodeId]| {
            out.extend(tags.data.chunks_exact(dim).enumerate().map(|(index, row)| {
                (
                    NodeId(index as u32),
                    cosine_f32(row, residual, ctx.precision),
                )
            }));
            out.retain(|(node, _)| !excluded.contains(node));
            out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            out.truncate(limit);
            recorded.borrow_mut().push(out.clone());
            Ok(())
        };
        let observation = nexus
            .observe_with_recall(
                &Query::new(query, &fixture.query_texts[index]),
                &config,
                &recall,
                &ctx,
            )
            .expect("observe the captured query");

        let scores: Vec<_> = candidate_values
            .chunks_exact(dim)
            .map(|row| cosine_f32(row, query, ctx.precision))
            .collect();
        let mut ranks = vec![0; count];
        let mut ordered: Vec<_> = (0..count).collect();
        ordered.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        for (rank, &candidate) in ordered.iter().enumerate() {
            ranks[candidate] = rank as u32 + 1;
        }
        let mut case = QueryCase {
            recalled: recorded.into_inner(),
            scores,
            ranks,
            stage: Vec::new(),
        };
        // One real host stage drawn from the strongest dense candidates, with
        // length bounded by the public Thinking contract.
        case.stage = ordered
            .into_iter()
            .take(256)
            .map(|index| CandidateId(index as u32))
            .collect();
        let stage = [ThinkingStageInputView {
            stage_label: "frozen",
            k: 1,
            candidates: &case.stage,
        }];
        let input = ThinkingInput {
            trace: observation.complete_trace().unwrap(),
            candidates: fixture.candidates(candidates, &zeros, &case),
            query_cosines: Some(&case.scores),
            stages: &stage,
            graph: assets.graph,
            provenance,
            support_groups: groups,
            tag_vectors: tags,
        };
        let thinking = ThinkingPlanOperator
            .execute(&ctx, &ThinkingConfig::default(), input)
            .expect("read the frozen Thinking stage");
        if thinking.stages[0]
            .results
            .iter()
            .any(|item| item.route.is_some())
        {
            route_queries.push(index);
        }
        cases.push(case);
    }
    eprintln!(
        "Frozen cohort: {} queries, {count} × {dim} candidates, {} queries with a grounded Thinking route, {threads} threads",
        cases.len(),
        route_queries.len()
    );

    let execute = |fanout: bool, index: usize| {
        let case = &cases[index];
        let recall = |level: usize,
                      _: &[f32],
                      limit: usize,
                      out: &mut Vec<_>,
                      _: Option<&EpaAnalysis>,
                      _: &[NodeId]| {
            if let Some(layer) = case.recalled.get(level) {
                out.extend(layer.iter().take(limit).copied());
            }
            Ok(())
        };
        let pool = fixture.candidates(candidates, &zeros, case);
        let stages = [ThinkingStageInputView {
            stage_label: "frozen",
            k: 1,
            candidates: &case.stage,
        }];
        let mut plan = Plan::new();
        let observation = plan
            .add(Observe, config.clone(), (), |_| {
                Ok(ObservationInput {
                    query: queries[index],
                    query_text: &fixture.query_texts[index],
                    candidates: &recall,
                    graph: assets.graph,
                    tag_vectors: assets.tags,
                    lexical_meta: assets.lexical,
                    epa_basis: assets.basis,
                    core_tags: &[],
                    ghost_vectors: None,
                    ghost_is_core: &[],
                })
            })
            .unwrap();
        let fields = plan
            .add(
                ComputeFields,
                FieldConfig::default(),
                observation,
                |inputs| Ok(nexus.field_input(inputs.get()?)),
            )
            .unwrap();
        let calibration = plan
            .add(
                PoolCalibrateOperator,
                PoolCalibrateConfig {
                    precision: fanout.then_some(PrecisionMode::StrictLeftFold),
                    ..PoolCalibrateConfig::default()
                },
                (observation, fields),
                |input| {
                    let (observation, fields) = input.get()?;
                    Ok(PoolCalibrateInput {
                        original_query: &observation.original_query,
                        enhanced_query: &observation.fused.enhanced_query,
                        fields,
                        candidates: pool,
                    })
                },
            )
            .unwrap();
        let river = plan
            .add(
                ReadRiver,
                RiverConfig {
                    evidence: EvidencePolicy::Nous,
                    ..Default::default()
                },
                (observation, fields, calibration),
                |inputs| {
                    let (observation, fields, calibrated) = inputs.get()?;
                    Ok(RiverInput {
                        assets,
                        observation,
                        fields,
                        pool: calibrated,
                        candidates: pool,
                    })
                },
            )
            .unwrap();
        if fanout {
            let dtsc = plan
                .add(
                    ReadDtsc,
                    nexus::math::DtscConfig::default(),
                    (observation, calibration),
                    |inputs| {
                        let (observation, calibrated) = inputs.get()?;
                        Ok(ObservedDtscInput {
                            assets,
                            observation,
                            pool: Some(calibrated),
                            candidates: pool,
                        })
                    },
                )
                .unwrap();
            let thinking = plan
                .add(
                    ThinkingPlanOperator,
                    ThinkingConfig::default(),
                    observation,
                    |inputs| {
                        Ok(ThinkingInput {
                            trace: inputs.get()?.complete_trace()?,
                            candidates: pool,
                            query_cosines: Some(&case.scores),
                            stages: &stages,
                            graph: assets.graph,
                            provenance,
                            support_groups: groups,
                            tag_vectors: tags,
                        })
                    },
                )
                .unwrap();
            plan.run((river, dtsc, thinking), &ctx)
                .expect("run the unified plan")
        } else {
            plan.run(river, &ctx).expect("run the River plan")
        }
    };

    let report_quantiles = |name: &str, fanout: bool, indices: &[usize]| {
        let mut samples: Vec<_> = indices
            .iter()
            .map(|&index| {
                let start = Instant::now();
                black_box(execute(fanout, index));
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        samples.sort_unstable_by(f64::total_cmp);
        let [p50, p95, p99] = [0.50_f64, 0.95, 0.99]
            .map(|p| samples[((samples.len() as f64 * p).ceil() as usize).saturating_sub(1)]);
        eprintln!(
            "{name} ({} queries): P50 {p50:.3} ms, P95 {p95:.3} ms, P99 {p99:.3} ms",
            samples.len()
        );
    };
    report_quantiles("same_pool", false, &(0..queries.len()).collect::<Vec<_>>());
    if !route_queries.is_empty() {
        report_quantiles("shared_river_dtsc_thinking", true, &route_queries);
    }
    let mut group = criterion.benchmark_group(format!("frozen_{threads}t"));
    group.throughput(Throughput::Elements(count as u64));
    let mut query_index = 0;
    group.bench_function("same_pool", |b| {
        b.iter(|| {
            let value = execute(false, query_index % queries.len());
            query_index = query_index.wrapping_add(1);
            black_box(value);
        })
    });
    if !route_queries.is_empty() {
        let mut query_index = 0;
        group.bench_function("shared_river_dtsc_thinking", |b| {
            b.iter(|| {
                let value = execute(true, route_queries[query_index % route_queries.len()]);
                query_index = query_index.wrapping_add(1);
                black_box(value);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, run_frozen);
criterion_main!(benches);
