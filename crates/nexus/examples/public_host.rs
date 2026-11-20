// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::core::{VcpResult, VectorMatrixView};
use nexus::math::{DtscConfig, RrfConfig, WeightedRanking, rrf};
use nexus::vcp::{ObservationConfig, RiverAnalysisConfig, RiverScoringConfig, ThinkingConfig};
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
            documents: vec![
                Document {
                    id: 100,
                    document_id: Some(700),
                    vector: vec![1.0, 0.0],
                    paths: vec![Sequence {
                        source: 1000,
                        tags: vec![(10, 0), (20, 1)],
                    }],
                },
                Document::new(200, vec![0.0, 1.0]),
            ],
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
    let query = Query::new(&[1.0, 0.0], "alpha")
        .with_core_concepts(&[10])
        .with_tag_recall(&recall);
    let catalog_rank = nexus.rank_river(
        &query,
        CandidateInput::Ids(&[200, 100]),
        &RiverQueryConfig::default(),
        &ctx,
    )?;
    let winner = &catalog_rank.items[0];
    println!(
        "Catalog River winner: {} from source {:?}",
        winner.id,
        winner.chosen_path.and_then(|path| path.source_id)
    );

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
    let observation = nexus.observe(&query, &ObservationConfig::default(), &ctx)?;
    let evaluation = observation.evaluate(&candidates, &ctx)?;
    let fields = observation.fields(&Default::default(), &ctx)?;
    let river = evaluation.river_evidence(&fields, &RiverAnalysisConfig::default(), &ctx)?;
    let mut config = RiverScoringConfig::default();
    let standard = river.readout(&config, &ctx)?;
    config.unified_budget_ct = 0.04;
    let retuned = river.readout(&config, &ctx)?;
    let config = DtscConfig::default();
    let field = observation.dtsc_field(&config.field, &ctx)?;
    let evidence = evaluation.dtsc_evidence(&field, &config.scoring.evidence, &ctx)?;
    let dtsc = evidence.readout(&config.scoring.readout, &ctx)?;
    let stages = [Stage {
        name: "Explore",
        limit: 1,
        candidates: &ids[..1],
    }];

    let indexes = nexus.prepare_recall(&Default::default(), &ctx)?;
    let another_lane = indexes.search_documents(&[0.0, 1.0], &RecallRequest::top_k(1), &ctx)?;
    let other_ids: Vec<_> = another_lane.iter().map(|item| item.id).collect();
    let river_ids: Vec<_> = standard.items.iter().map(|item| item.id).collect();
    let fused = rrf(
        &[
            WeightedRanking {
                ids: &river_ids,
                weight: 1.0,
            },
            WeightedRanking {
                ids: &other_ids,
                weight: 1.0,
            },
        ],
        &RrfConfig::default(),
        &ctx,
    )?;
    let thinking = evaluation.thinking(&stages, &ThinkingConfig::default(), &ctx)?;
    let route = thinking.stages[0]
        .results
        .iter()
        .find_map(|item| item.route);
    println!(
        "Borrowed TagMemo winner: {}",
        evaluation.tag_memo(&ctx)?.items[0].id
    );
    println!(
        "River at C=0.04: {}; DTSC candidates: {}",
        retuned.items[0].id, dtsc.available
    );
    println!(
        "Thinking transition: {:?}",
        route.map(|edge| (edge.source, edge.target))
    );
    println!(
        "RRF IDs: {:?}",
        fused.items.iter().map(|item| item.id).collect::<Vec<_>>()
    );
    Ok(())
}
