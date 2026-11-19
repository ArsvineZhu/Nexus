// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::core::{VcpResult, VectorMatrixView};
use nexus::math::{DtscConfig, RrfConfig, RrfInput, RrfOperator, WeightedRanking};
use nexus::vcp::{ObservationConfig, ThinkingConfig};
use nexus::*;

fn main() -> VcpResult<()> {
    let ctx = ExecutionContext::default();
    let nexus = Nexus::prepare(
        Dataset {
            concepts: vec![
                Concept {
                    id: 20,
                    name: "beta".into(),
                    aliases: vec![],
                    vector: vec![0.8, 0.6],
                },
                Concept {
                    id: 10,
                    name: "alpha".into(),
                    aliases: vec![],
                    vector: vec![1.0, 0.0],
                },
            ],
            sources: vec![SourceFact {
                id: 1000,
                checksum: 1,
                tags: vec![10, 20],
                required_sources: vec![],
            }],
            documents: vec![Document {
                id: 100,
                document_id: Some(700),
                vector: vec![1.0, 0.0],
                paths: vec![Sequence {
                    source: 1000,
                    tags: vec![(10, 0), (20, 1)],
                }],
            }],
            ..Default::default()
        },
        &AssetConfig::default(),
        &ctx,
    )?;

    let recall = |_: &ExecutionContext<'_>, request: TagRecallRequest<'_>, out: &mut Vec<u64>| {
        if request.layer == 0 && request.limit > 0 {
            out.push(10);
        }
        Ok(())
    };
    let query = Query {
        core_concepts: &[10],
        ..Query::new(&[1.0, 0.0], "alpha").with_tag_recall(&recall)
    };
    let catalog_rank = nexus.rank_river(
        &query,
        CandidateInput::Ids(&[100]),
        &RiverQueryConfig::default(),
        ctx,
    )?;
    println!("Catalog River winner: {}", catalog_rank.items[0].id);

    let ids = [100, 200];
    let vector_data = [1.0, 0.0, 0.0, 1.0];
    let matrix = VectorMatrixView::new(&vector_data, ids.len(), 2)?;
    let paths = [Sequence {
        source: 1000,
        tags: vec![(10, 0), (20, 1)],
    }];
    let path_rows = [paths.as_slice(), &[]];
    let borrowed = BorrowedCandidates::new(&ids, matrix)
        .with_document_ids(&[700, 800])
        .with_paths(&path_rows);
    let candidates = nexus.candidates(CandidateInput::Borrowed(borrowed))?;
    let observation = nexus.observe(&query, &ObservationConfig::default(), ctx)?;
    let evaluation = observation.evaluate(&candidates)?;
    let fields = observation.fields(&Default::default())?;
    let river = evaluation.prepare_river(&fields)?;
    let mut config = RiverConfig::default();
    let standard = river.readout(&config)?;
    config.scoring.unified_budget_ct = 0.04;
    let smaller_budget = river.readout(&config)?;
    let config = DtscConfig::default();
    let field = observation.dtsc_field(&config.field)?;
    let evidence = evaluation.dtsc_evidence(&field, &config.scoring.evidence)?;
    let dtsc = evidence.readout(&config.scoring.readout)?;
    let stages = [Stage {
        name: "Explore",
        limit: 1,
        candidates: &ids[..1],
    }];

    let mut plan = Plan::new();
    let river_result = plan.add((), |_, _| Ok(standard))?;
    let fused = plan.add(river_result, |ctx, inputs| {
        let ids: Vec<_> = inputs.get().items.iter().map(|item| item.id).collect();
        RrfOperator.execute(
            ctx,
            &RrfConfig::default(),
            RrfInput {
                rankings: &[
                    WeightedRanking {
                        ids: &ids,
                        weight: 1.0,
                    },
                    WeightedRanking {
                        ids: &[200, 900],
                        weight: 1.0,
                    },
                ],
            },
        )
    })?;
    let thinking = plan.add((), |_, _| {
        evaluation.thinking(&stages, &ThinkingConfig::default())
    })?;
    let output = plan.run((fused, thinking), observation.context())?;
    let route = output.get(thinking)?.stages[0]
        .results
        .iter()
        .find_map(|item| item.route);
    println!(
        "Borrowed TagMemo winner: {}",
        evaluation.tag_memo()?.items[0].id
    );
    println!(
        "River at C=0.04: {}; DTSC candidates: {}",
        smaller_budget.items[0].id, dtsc.available
    );
    println!(
        "Thinking transition: {:?}",
        route.map(|edge| (edge.source, edge.target))
    );
    println!(
        "RRF IDs: {:?}",
        output
            .get(fused)?
            .items
            .iter()
            .map(|item| item.id)
            .collect::<Vec<_>>()
    );
    Ok(())
}
