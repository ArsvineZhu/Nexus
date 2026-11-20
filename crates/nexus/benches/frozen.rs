// SPDX-License-Identifier: Apache-2.0

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nexus::math::DtscConfig;
use nexus::*;
use nexus_core::kernel::{cosine_f32, normalize_direction};
use nexus_core::*;
use nexus_vcp::{EpaBasis, EvidencePolicy, ObservationConfig, RiverAnalysisConfig, ThinkingConfig};
use serde::Deserialize;
use std::cell::RefCell;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Fixture {
    dim: usize,
    docs: Vec<u64>,
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
    let (rows, remainder) = bytes.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "frozen vector file ends inside a float"
    );
    rows.iter().copied().map(f32::from_le_bytes).collect()
}

struct QueryCase {
    recalled: Vec<Vec<u64>>,
    stage: Vec<u64>,
}

fn directions(values: &[f32], dim: usize) -> Vec<f32> {
    values
        .chunks_exact(dim)
        .flat_map(|row| normalize_direction(row, dim, "bench").unwrap())
        .collect()
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
    let candidate_values = directions(&candidate_values, dim);
    let tag_values = directions(&tag_values, dim);
    let matrix = VectorMatrixView::new(&candidate_values, count, dim).unwrap();
    let tags = VectorMatrixView::new(&tag_values, fixture.labels.len(), dim).unwrap();
    let graph = TransportGraph::new(CsrGraph::new(fixture.graph.clone()).unwrap()).unwrap();
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
    let root_count = fixture
        .group_roots
        .iter()
        .chain(&fixture.candidate_roots)
        .map(|id| id.index() + 1)
        .max()
        .unwrap_or(0);
    let source_ids: Vec<_> = (0..root_count as u64).collect();
    let source_checksums = vec![0; root_count];
    let source_offsets = vec![0; root_count + 1];
    let ids: Vec<_> = (0..count as u64).collect();
    let document_ids = &fixture.docs;
    let concept_ids: Vec<_> = (1..=tags.rows as u64).collect();
    let nexus = Nexus::from_prepared(PreparedDataset {
        concept_ids: &concept_ids,
        source_ids: &source_ids,
        tags,
        graph: graph.view(),
        lexical,
        basis: fixture.basis.as_ref().map(EpaBasis::as_view),
        provenance,
        groups,
        documents: Some(Catalog {
            ids: &ids,
            document_ids: Some(document_ids),
            vectors: matrix,
            root_offsets: &fixture.candidate_root_offsets,
            roots: &fixture.candidate_roots,
            path_offsets: &fixture.candidate_path_offsets,
            path_sources: None,
            tag_offsets: &fixture.path_tag_offsets,
            tags: &fixture.path_tags,
            positions: &fixture.path_positions,
        }),
        // The study exports attribution groups and candidate roots but no root witness facts.
        sources: RootFactView {
            root_count,
            checksum_hashes: &source_checksums,
            root_tag_offsets: &source_offsets,
            root_tags: &[],
            root_req_offsets: &source_offsets,
            root_required_roots: &[],
        },
        counterevidence: CounterevidenceAdjView::empty(),
        tag_residuals: None,
    })
    .unwrap();
    let candidates = nexus.candidates(CandidateInput::Ids(&ids)).unwrap();
    let config = ObservationConfig::default();
    let thinking_config = ThinkingConfig::default();
    let analysis = RiverAnalysisConfig {
        evidence: EvidencePolicy::DiscountWhenUnattributed,
        ..Default::default()
    };
    let dtsc_config = DtscConfig::default();
    let mut cases = Vec::with_capacity(queries.len());
    let mut route_queries = Vec::new();
    for (index, vector) in queries.iter().enumerate() {
        let recorded = RefCell::new(Vec::new());
        let recall = |_: &ExecutionContext<'_>, req: TagRecallRequest<'_>, out: &mut Vec<u64>| {
            let mut hits: Vec<_> = tags
                .data
                .chunks_exact(dim)
                .zip(&concept_ids)
                .filter(|(_, id)| !req.excluded_concepts.contains(id))
                .map(|(row, &id)| (id, cosine_f32(row, req.residual, PrecisionMode::FastSimd)))
                .collect();
            hits.sort_by(|a, b| b.1.total_cmp(&a.1));
            out.extend(hits.into_iter().take(req.limit).map(|(id, _)| id));
            recorded.borrow_mut().push(out.clone());
            Ok(())
        };
        let query = Query::new(vector, &fixture.query_texts[index]).with_tag_recall(&recall);
        let observation = nexus.observe(&query, &config, &ctx).unwrap();
        let evaluation = observation.evaluate(&candidates, &ctx).unwrap();
        let mut scored: Vec<_> = candidate_values
            .chunks_exact(dim)
            .zip(&ids)
            .map(|(row, &id)| (id, cosine_f32(row, vector, PrecisionMode::StrictLeftFold)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let stage: Vec<_> = scored.into_iter().take(256).map(|(id, _)| id).collect();
        let thinking = evaluation
            .thinking(
                &[Stage {
                    name: "frozen",
                    limit: 1,
                    candidates: &stage,
                }],
                &thinking_config,
                &ctx,
            )
            .unwrap();
        if thinking.stages[0]
            .results
            .iter()
            .any(|item| item.route.is_some())
        {
            route_queries.push(index);
        }
        cases.push(QueryCase {
            recalled: recorded.into_inner(),
            stage,
        });
    }
    eprintln!(
        "Frozen cohort: {} queries, {count} x {dim} candidates, {} grounded Thinking queries, {threads} threads",
        cases.len(),
        route_queries.len()
    );

    let execute = |fanout: bool, index: usize| {
        let case = &cases[index];
        let recall = |_: &ExecutionContext<'_>, req: TagRecallRequest<'_>, out: &mut Vec<u64>| {
            if let Some(hits) = case.recalled.get(req.layer) {
                out.extend(hits.iter().copied().take(req.limit));
            }
            Ok(())
        };
        let query =
            Query::new(queries[index], &fixture.query_texts[index]).with_tag_recall(&recall);
        let observation = nexus.observe(&query, &config, &ctx).unwrap();
        let evaluation = observation.evaluate(&candidates, &ctx).unwrap();
        let fields = observation.fields(&Default::default(), &ctx).unwrap();
        let river = evaluation
            .river_evidence(&fields, &analysis, &ctx)
            .unwrap()
            .readout(&Default::default(), &ctx)
            .unwrap();
        assert_eq!(river.items.len(), count);
        black_box(river);
        if fanout {
            let field = observation.dtsc_field(&dtsc_config.field, &ctx).unwrap();
            black_box(
                evaluation
                    .dtsc_evidence(&field, &dtsc_config.scoring.evidence, &ctx)
                    .unwrap()
                    .readout(&dtsc_config.scoring.readout, &ctx)
                    .unwrap(),
            );
            let thinking = evaluation
                .thinking(
                    &[Stage {
                        name: "frozen",
                        limit: 1,
                        candidates: &case.stage,
                    }],
                    &thinking_config,
                    &ctx,
                )
                .unwrap();
            assert!(
                thinking.stages[0]
                    .results
                    .iter()
                    .any(|item| item.route.is_some())
            );
            black_box(thinking);
        }
    };

    let report = |name: &str, fanout: bool, indices: &[usize]| {
        let mut samples: Vec<_> = indices
            .iter()
            .map(|&i| {
                let start = Instant::now();
                execute(fanout, i);
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
    report("same_pool", false, &(0..queries.len()).collect::<Vec<_>>());
    if !route_queries.is_empty() {
        report("shared_river_dtsc_thinking", true, &route_queries);
    }
    let mut group = criterion.benchmark_group(format!("frozen_{threads}t"));
    group.throughput(Throughput::Elements(count as u64));
    let mut query_index = 0;
    group.bench_function("same_pool", |b| {
        b.iter(|| {
            execute(false, query_index % queries.len());
            query_index = query_index.wrapping_add(1);
        })
    });
    if !route_queries.is_empty() {
        let mut query_index = 0;
        group.bench_function("shared_river_dtsc_thinking", |b| {
            b.iter(|| {
                execute(true, route_queries[query_index % route_queries.len()]);
                query_index = query_index.wrapping_add(1);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, run_frozen);
criterion_main!(benches);
