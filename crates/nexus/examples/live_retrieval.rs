// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::core::{CandidateId, PrecisionMode, ThinkingStageInputView};
use nexus::math::DtscConfig;
use nexus::vcp::hyperbolic::SearchConfig;
use nexus::vcp::{
    ComputeFields, FieldConfig, ObservationConfig, PoolCalibrateInput, PoolCalibrateOperator,
    ThinkingConfig, ThinkingInput, ThinkingPlanOperator,
};
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
            ..Dataset::default()
        },
        &AssetConfig::default(),
    )?;
    let assets = nexus.prepared();
    println!(
        "{} · {} dimensions · {} documents · {} transport edges",
        data.model,
        data.dimension,
        ids.len(),
        assets.graph.nnz()
    );
    let borrowed = Nexus::from_prepared(assets)?;
    let ctx = ExecutionContext::default();
    let mut config = RiverPipelineConfig::default();
    config.readout.calibration.precision = Some(PrecisionMode::StrictLeftFold);
    let candidates = Candidates::recall(ids.len());
    let mut top_one = 0;
    let mut grounded = 0;
    for (index, probe) in data.queries.iter().enumerate() {
        let query = Query::new(&probe.vector, &probe.text);
        let result = nexus.run(&query, &candidates, Pipeline::river(), &config)?;
        if index == 0 {
            let tag_memo = nexus.run(
                &query,
                &candidates,
                Pipeline::tag_memo(),
                &TagMemoConfig::default(),
            )?;
            assert_eq!(tag_memo.ranking.scores.len(), ids.len());
            let dtsc = nexus.run(
                &query,
                &candidates,
                Pipeline::dtsc(),
                &DtscPipelineConfig::default(),
            )?;
            assert_eq!(dtsc.scores.results.len(), ids.len());
        }
        let top = result
            .scores
            .items
            .first()
            .ok_or("River returned no candidate")?;
        let expected_rank = result
            .scores
            .items
            .iter()
            .position(|item| {
                result.candidate_ids[item.candidate.index()] == probe.expected_document
            })
            .ok_or("expected passage missing from River")?
            + 1;
        top_one += usize::from(expected_rank == 1);
        let observed = borrowed.observe(&query, &ObservationConfig::default())?;
        let dtsc = borrowed.dtsc(&observed, &candidates, &DtscConfig::default())?;
        let stages = [Stage {
            name: "Explore",
            limit: 2,
            documents: &ids,
        }];
        let plan = borrowed.thinking(
            observed.complete_trace()?,
            &stages,
            &ThinkingConfig::default(),
        )?;
        let thought = plan.scores.stages[0]
            .results
            .iter()
            .filter(|item| item.route.is_some())
            .count();
        grounded += usize::from(thought > 0);
        println!(
            "{} | River: {} ({:.4}, topology {:.4}) | expected rank: {expected_rank} | tags: {} | DTSC: {} | Thinking routes: {thought}",
            probe.text,
            titles[result.candidate_ids[top.candidate.index()] as usize],
            top.quality_score,
            top.topology_bonus,
            observed.sense.nodes.len(),
            dtsc.scores.results.len()
        );

        let batch =
            borrowed.candidates(query.vector, &candidates, SearchConfig::default(), &ctx)?;
        let thinking_candidates: Vec<_> = (0..batch.ids().len())
            .map(|slot| CandidateId::from_index(slot, "example"))
            .collect::<Result<_, _>>()?;
        let plan_stages = [ThinkingStageInputView {
            stage_label: "Explore",
            k: 2,
            candidates: &thinking_candidates,
        }];
        let mut plan = Plan::new();
        let observation = plan.add(ObserveQuery, ObservationConfig::default(), (), |_| {
            Ok(QueryInput {
                nexus: &borrowed,
                query,
            })
        })?;
        let fields = plan.add(
            ComputeFields,
            FieldConfig::default(),
            observation,
            |input| Ok(borrowed.field_input(input.get()?)),
        )?;
        let calibration = plan.add(
            PoolCalibrateOperator,
            config.readout.calibration.clone(),
            (observation, fields),
            |input| {
                let (observation, fields) = input.get()?;
                Ok(PoolCalibrateInput {
                    original_query: &observation.original_query,
                    enhanced_query: &observation.fused.enhanced_query,
                    fields,
                    candidates: batch.view(),
                })
            },
        )?;
        let river = plan.add(
            ReadRiver,
            config.readout.clone(),
            (observation, fields, calibration),
            |input| {
                let (observation, fields, pool) = input.get()?;
                Ok(RiverInput {
                    assets,
                    observation,
                    fields,
                    pool,
                    candidates: batch.view(),
                })
            },
        )?;
        let dtsc_step = plan.add(
            ReadDtsc,
            DtscConfig::default(),
            (observation, calibration),
            |input| {
                let (observation, pool) = input.get()?;
                Ok(ObservedDtscInput {
                    assets,
                    observation,
                    pool: Some(pool),
                    candidates: batch.view(),
                })
            },
        )?;
        let combined = if thought == 0 {
            plan.run((river, dtsc_step), &ctx)?
        } else {
            let thinking = plan.add(
                ThinkingPlanOperator,
                ThinkingConfig::default(),
                observation,
                |input| {
                    Ok(ThinkingInput {
                        trace: input.get()?.complete_trace()?,
                        candidates: batch.view(),
                        stages: &plan_stages,
                        graph: assets.graph,
                        provenance: assets.provenance,
                        support_groups: assets.groups,
                        tag_vectors: assets.tags,
                        query_cosines: None,
                    })
                },
            )?;
            let result = plan.run((river, dtsc_step, thinking), &ctx)?;
            assert!(
                result.get(thinking)?.stages[0]
                    .results
                    .iter()
                    .any(|item| item.route.is_some())
            );
            result
        };
        assert_eq!(combined.get(river)?.items, result.scores.items);
        assert_eq!(
            combined
                .get(dtsc_step)?
                .results
                .iter()
                .map(|x| x.score)
                .collect::<Vec<_>>(),
            dtsc.scores
                .results
                .iter()
                .map(|x| x.score)
                .collect::<Vec<_>>()
        );
    }
    println!(
        "Top-1: {top_one}/{} · queries with grounded Thinking routes: {grounded}",
        data.queries.len()
    );
    if top_one != data.queries.len() || grounded == 0 {
        return Err("the expected retrieval or grounded Thinking result was absent".into());
    }
    Ok(())
}
