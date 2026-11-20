// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::math::DtscConfig;
use nexus::vcp::{ObservationConfig, ThinkingConfig};
use nexus::*;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Tag {
    name: String,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct Passage {
    title: String,
    tags: Vec<u64>,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct QueryFixture {
    text: String,
    expected_document: u64,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct Fixture {
    model: String,
    dimension: usize,
    tags: Vec<Tag>,
    documents: Vec<Passage>,
    queries: Vec<QueryFixture>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("expected an embedding fixture path")?;
    let data: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    let titles: Vec<_> = data.documents.iter().map(|doc| doc.title.clone()).collect();
    let ids: Vec<_> = (0..titles.len() as u64).collect();
    let ctx = ExecutionContext::default();
    let nexus = Nexus::prepare(
        Dataset {
            concepts: data
                .tags
                .into_iter()
                .enumerate()
                .map(|(id, tag)| Concept {
                    id: id as u64,
                    name: tag.name,
                    vector: tag.vector,
                    aliases: Vec::new(),
                })
                .collect(),
            documents: data
                .documents
                .into_iter()
                .enumerate()
                .map(|(id, doc)| Document {
                    id: id as u64,
                    document_id: None,
                    vector: doc.vector,
                    paths: vec![Sequence {
                        source: id as u64,
                        tags: doc
                            .tags
                            .into_iter()
                            .enumerate()
                            .map(|(position, tag)| (tag, position as u32 + 1))
                            .collect(),
                    }],
                })
                .collect(),
            sources: ids
                .iter()
                .map(|&id| SourceFact {
                    id,
                    checksum: id + 1,
                    tags: vec![],
                    required_sources: vec![],
                })
                .collect(),
            ..Dataset::default()
        },
        &AssetConfig::default(),
        &ctx,
    )?;
    let assets = nexus.prepared();
    println!(
        "{} · {} dimensions · {} documents · {} transport edges",
        data.model,
        data.dimension,
        ids.len(),
        assets.graph.nnz()
    );
    let indexes = nexus.prepare_recall(&RecallConfig::default(), &ctx)?;
    let mut top_one = 0;
    let mut grounded = 0;
    let dtsc_config = DtscConfig::default();
    for probe in &data.queries {
        let query = Query::new(&probe.vector, &probe.text);
        let observation = indexes.observe(&query, &ObservationConfig::default(), &ctx)?;
        let candidates = observation.recall(&indexes, &RecallRequest::top_k(ids.len()), &ctx)?;
        let evaluation = observation.evaluate(&candidates, &ctx)?;
        let result = evaluation.river(&Default::default(), &ctx)?;
        let dtsc = evaluation.dtsc(&dtsc_config, &ctx)?;
        let thinking = evaluation.thinking(
            &[Stage {
                name: "Explore",
                limit: 2,
                candidates: candidates.ids(),
            }],
            &ThinkingConfig::default(),
            &ctx,
        )?;
        let top = result.items.first().ok_or("River returned no candidates")?;
        let expected_rank = result
            .items
            .iter()
            .position(|item| item.id == probe.expected_document)
            .ok_or("expected passage missing from River")?
            + 1;
        let routes = thinking.stages[0]
            .results
            .iter()
            .filter(|item| item.route.is_some())
            .count();
        top_one += usize::from(expected_rank == 1);
        grounded += usize::from(routes > 0);
        println!(
            "{} | River: {} ({:.4}) | expected rank: {expected_rank} | DTSC: {} | Thinking routes: {routes}",
            probe.text,
            titles[top.id as usize],
            top.scores.score,
            dtsc.items.len(),
        );
    }
    println!(
        "Top-1: {top_one}/{}; queries with grounded Thinking routes: {grounded}",
        data.queries.len()
    );
    if top_one != data.queries.len() || grounded == 0 {
        return Err("the expected retrieval or grounded Thinking result was absent".into());
    }
    Ok(())
}
