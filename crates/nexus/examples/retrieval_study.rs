// SPDX-License-Identifier: CC-BY-NC-SA-4.0
//! Replay judged queries as a caller of the public computation API.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

use nexus::math::{self, RankedList};
use nexus::{
    Candidate, Candidates, Concept, Evaluation, Model, Observation, PathRef, Query, RecallKind,
    Relation, RelationData, Source, ThinkingStage,
};
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    producer: String,
    dimension: usize,
    concepts: Vec<NamedConcept>,
    sources: Vec<SourceRow>,
    relations: Vec<Link>,
    candidates: Vec<CandidateRow>,
    queries: Vec<Case>,
}

#[derive(Deserialize)]
struct NamedConcept {
    id: u64,
    label: Option<String>,
    aliases: Vec<String>,
}
#[derive(Deserialize)]
struct SourceRow {
    id: u64,
    #[serde(default)]
    parents: Vec<u64>,
    paths: Vec<Vec<u64>>,
}
#[derive(Deserialize)]
struct Link {
    from: u64,
    to: u64,
    support: f64,
    source: u64,
}
#[derive(Deserialize)]
struct CandidateRow {
    id: u64,
    document: u64,
    sources: Vec<u64>,
    paths: Vec<Reference>,
}
#[derive(Deserialize)]
struct Reference {
    source: u64,
    path: usize,
    positions: Option<Vec<usize>>,
}
#[derive(Deserialize)]
struct Case {
    split: String,
    text: String,
    relevant_documents: Vec<u64>,
}

fn vectors(file: &Path, dimension: usize, count: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(file)?;
    let expected = dimension.checked_mul(count).and_then(|n| n.checked_mul(4));
    if dimension == 0 || expected != Some(bytes.len()) {
        return Err(format!("invalid matrix length in {}", file.display()).into());
    }
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    Ok(values
        .chunks_exact(dimension)
        .map(<[f32]>::to_vec)
        .collect())
}

struct Stats {
    queries: usize,
    recall: f64,
    ndcg: f64,
}

impl Stats {
    fn new() -> Self {
        Self {
            queries: 0,
            recall: 0.0,
            ndcg: 0.0,
        }
    }

    fn add(&mut self, ids: &[u64], docs: &HashMap<u64, u64>, relevant: &[u64]) {
        let mut seen = HashSet::new();
        let documents = ids.iter().map(|id| docs[id]).filter(|id| seen.insert(*id));
        let mut dcg = 0.0;
        let mut matches = 0;
        fn discount(i: usize) -> f64 {
            1.0 / ((i + 2) as f64).log2()
        }
        for (i, id) in documents.enumerate() {
            if !relevant.contains(&id) {
                continue;
            }
            matches += 1;
            if i < 8 {
                dcg += discount(i);
            }
        }
        self.queries += 1;
        self.recall += matches as f64 / relevant.len() as f64;
        let ideal = (0..8.min(relevant.len())).map(discount).sum::<f64>();
        self.ndcg += dcg / ideal;
    }
}

fn recall(
    obs: &Observation,
    lane: RecallKind,
    catalog: &Candidates<'_>,
    limit: usize,
) -> nexus::Result<Vec<u64>> {
    let Some(direction) = obs.vector(lane) else {
        return Ok(Vec::new());
    };
    Ok(
        math::cosine_scores(direction, catalog.vectors(), &Default::default())?
            .top(Some(limit))
            .ids()
            .to_vec(),
    )
}

fn readouts(
    eval: &Evaluation<'_>,
    ids: &[u64],
) -> nexus::Result<(nexus::Ranking, nexus::Ranking, nexus::ThinkingResult)> {
    let baseline = eval.tag_memo();
    let river = eval.river(&Default::default())?.top(None);
    let dtsc = eval.dtsc(&baseline, &Default::default())?.top(None);
    let thinking = eval.thinking(
        &[ThinkingStage {
            label: "route",
            candidates: ids,
            limit: 4,
        }],
        &Default::default(),
    )?;
    Ok((river, dtsc, thinking))
}

fn compare_reuse(obs: &Observation, candidates: &Candidates<'_>, paths: usize) -> Result<()> {
    // Equal searches and scores, with and without separate candidate preparation.
    let ids = candidates.ids();
    let mut shared = Duration::ZERO;
    let mut separate = Duration::ZERO;
    for i in 0..12 {
        let eval = || obs.evaluate(candidates, &Default::default());
        let together = || -> nexus::Result<_> { readouts(&eval()?, ids) };
        let apart = || -> nexus::Result<_> {
            let river = eval()?.river(&Default::default())?.top(None);
            let dtsc = {
                let eval = eval()?;
                eval.dtsc(&eval.tag_memo(), &Default::default())?.top(None)
            };
            let thinking = eval()?.thinking(
                &[ThinkingStage {
                    label: "route",
                    candidates: ids,
                    limit: 4,
                }],
                &Default::default(),
            )?;
            Ok((river, dtsc, thinking))
        };
        let instant = Instant::now();
        let (a, b) = if i % 2 == 0 {
            let a = together()?;
            shared += instant.elapsed();
            let start = Instant::now();
            let b = apart()?;
            separate += start.elapsed();
            (a, b)
        } else {
            let b = apart()?;
            separate += instant.elapsed();
            let start = Instant::now();
            let a = together()?;
            shared += start.elapsed();
            (a, b)
        };
        if a != b {
            return Err("shared and separate evaluation results differ".into());
        }
    }
    println!(
        "paired, 12 runs | shared {:.3} ms | separate {:.3} ms | candidates {} | dimension {} | paths {} | observed nodes {} | observed edges {} | transitions {}",
        shared.as_secs_f64() * 1_000.0 / 12.0,
        separate.as_secs_f64() * 1_000.0 / 12.0,
        candidates.len(),
        candidates.vectors().dimension(),
        paths,
        obs.active_concepts().len(),
        obs.analysis().sense.observed_edges.len(),
        obs.transitions().len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("provide the exported fixture directory")?;
    let path = Path::new(&path);
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path.join("fixture.json"))?)?;
    let dim = fixture.dimension;
    let concept_values = vectors(&path.join("concepts.f32"), dim, fixture.concepts.len())?;
    let concepts = fixture
        .concepts
        .into_iter()
        .zip(concept_values)
        .map(|(meta, vector)| Concept {
            id: meta.id,
            vector,
            label: meta.label,
            aliases: meta.aliases,
        })
        .collect();
    let source_paths = fixture
        .sources
        .iter()
        .map(|row| row.paths.len())
        .sum::<usize>();
    let path_references = fixture
        .candidates
        .iter()
        .map(|row| row.paths.len())
        .sum::<usize>();
    let relation_count = fixture.relations.len();
    let sources = fixture
        .sources
        .into_iter()
        .map(|row| {
            if row.parents.is_empty() {
                Source::unknown_origin(row.id, row.paths)
            } else {
                Source::derived(row.id, row.parents, row.paths)
            }
        })
        .collect();
    let relations = fixture
        .relations
        .into_iter()
        .map(|link| Relation {
            from: link.from,
            to: link.to,
            support: link.support,
            source: link.source,
        })
        .collect();
    let started = Instant::now();
    let model = Model::build(
        RelationData {
            dimension: dim,
            concepts,
            sources,
            relations,
        },
        &Default::default(),
    )?;
    let built = started.elapsed();
    let documents: HashMap<_, _> = fixture
        .candidates
        .iter()
        .map(|row| (row.id, row.document))
        .collect();
    let candidate_paths: HashMap<_, _> = fixture
        .candidates
        .iter()
        .map(|row| (row.id, row.paths.len()))
        .collect();
    let data = vectors(&path.join("candidates.f32"), dim, fixture.candidates.len())?;
    let candidates = fixture
        .candidates
        .into_iter()
        .zip(data)
        .map(|(meta, vector)| {
            let paths = meta
                .paths
                .into_iter()
                .map(|path| PathRef {
                    source: path.source,
                    path: path.path,
                    positions: path.positions,
                })
                .collect();
            Candidate::new(meta.id, vector)
                .with_sources(meta.sources)
                .with_paths(paths)
                .with_document_id(meta.document)
        })
        .collect();
    let catalog = model.candidates(candidates)?;
    let queries = vectors(&path.join("queries.f32"), dim, fixture.queries.len())?;
    println!(
        "{} | {} | {} queries | {} candidates | {} concepts | {} dims | {} source paths | {} positive relations | {} candidate path references | one thread | build {:.2} ms",
        fixture.name,
        fixture.producer,
        queries.len(),
        catalog.len(),
        model.concepts().ids().len(),
        dim,
        source_paths,
        relation_count,
        path_references,
        built.as_secs_f64() * 1_000.0
    );

    let mut stats = BTreeMap::<(&str, &str), Stats>::new();
    let mut observe_time = Duration::ZERO;
    let mut shared_time = Duration::ZERO;
    let mut compared = false;
    for (case, query) in fixture.queries.iter().zip(&queries) {
        let start = Instant::now();
        let obs = model.observe(Query::new(query).with_text(&case.text), &Default::default())?;
        observe_time += start.elapsed();
        let dense = recall(&obs, RecallKind::Original, &catalog, 64)?;
        let enhanced = recall(&obs, RecallKind::Enhanced, &catalog, 64)?;
        let local = recall(&obs, RecallKind::LocalField, &catalog, 32)?;
        let transfer = recall(&obs, RecallKind::TransferField, &catalog, 32)?;
        let interleaved: Vec<_> = (0..local.len().max(transfer.len()))
            .flat_map(|i| local.get(i).into_iter().chain(transfer.get(i)))
            .copied()
            .collect();
        let expanded =
            math::merge_ranked_ids(&[&dense[..dense.len().min(48)], &interleaved, &dense]);
        let expanded = &expanded[..expanded.len().min(64)];
        let dense_candidates = catalog.select(dense.iter().copied())?;
        if !compared
            && matches!(case.split.as_str(), "development" | "holdout")
            && !obs.transitions().is_empty()
        {
            let paths = dense.iter().map(|id| candidate_paths[id]).sum();
            compare_reuse(&obs, &dense_candidates, paths)?;
            compared = true;
        }
        let start = Instant::now();
        let eval = obs.evaluate(&dense_candidates, &Default::default())?;
        let (river, dtsc, _) = readouts(&eval, &dense)?;
        shared_time += start.elapsed();
        let enhanced_fixed = eval.tag_memo().top(None);
        let fused = math::reciprocal_rank_fusion(
            &[
                RankedList {
                    ids: &dense,
                    weight: 1.0,
                },
                RankedList {
                    ids: river.ids(),
                    weight: 1.0,
                },
                RankedList {
                    ids: dtsc.ids(),
                    weight: 1.0,
                },
            ],
            &Default::default(),
        )?
        .top(None);
        let expanded_candidates = catalog.select(expanded.iter().copied())?;
        let expanded_eval = obs.evaluate(&expanded_candidates, &Default::default())?;
        let expanded_river = expanded_eval.river(&Default::default())?.top(None);
        for (name, ids) in [
            ("dense", dense.as_slice()),
            ("enhanced", enhanced.as_slice()),
            ("dense48+fields16", expanded),
            ("dense/tag-memo", enhanced_fixed.ids()),
            ("dense/river", river.ids()),
            ("dense/dtsc", dtsc.ids()),
            ("dense/rrf", fused.ids()),
            ("fields/river", expanded_river.ids()),
        ] {
            stats
                .entry((&case.split, name))
                .or_insert_with(Stats::new)
                .add(ids, &documents, &case.relevant_documents);
        }
    }
    for ((split, name), stat) in stats {
        println!(
            "{split:11} {name:18} n={:3} Recall@64={:.4} nDCG@8={:.4}",
            stat.queries,
            stat.recall / stat.queries as f64,
            stat.ndcg / stat.queries as f64
        );
    }
    let count = queries.len().max(1) as f64;
    println!(
        "mean observe {:.3} ms | mean shared evaluate+River+DTSC+Thinking {:.3} ms",
        observe_time.as_secs_f64() * 1_000.0 / count,
        shared_time.as_secs_f64() * 1_000.0 / count
    );
    Ok(())
}
