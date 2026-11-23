// SPDX-License-Identifier: CC-BY-NC-SA-4.0
mod fixture;

use nexus::{Query, ThinkingStage, math};

fn main() -> nexus::Result<()> {
    let model = fixture::model()?;
    let catalog = fixture::candidates(&model)?;
    let index = math::WarpedIndex::build(catalog.vectors(), &Default::default())?;
    let observation = model.observe(
        Query::new(&fixture::QUERY)
            .with_text("增量更新")
            .with_core_concepts(&[30]),
        &Default::default(),
    )?;

    let directions: Vec<&[f32]> = observation
        .recall_vectors()
        .map(|direction| direction.vector)
        .collect();
    let hits = index.search_many(&directions, &Default::default())?;
    let lists: Vec<&[u64]> = hits.iter().map(|result| result.ids()).collect();
    let ids = math::merge_ranked_ids(&lists);
    let candidates = catalog.select(ids)?;
    let evaluation = observation.evaluate(&candidates, &Default::default())?;

    let original = evaluation.cosine();
    let enhanced = evaluation.tag_memo();
    let river = evaluation.river(&Default::default())?;

    // 同一份几何证据可以用不同底分；不重新运行查询或候选分析。
    let geometry = evaluation.dtsc_evidence(&Default::default())?;
    let dtsc_enhanced = geometry.apply(&enhanced, &Default::default())?;
    let dtsc_river = geometry.apply(&river, &Default::default())?;

    let original_rank = original.top(None);
    let river_rank = river.top(None);
    let dtsc_rank = dtsc_enhanced.top(None);
    let fused = math::reciprocal_rank_fusion(
        &[
            math::RankedList {
                ids: original_rank.ids(),
                weight: 1.0,
            },
            math::RankedList {
                ids: river_rank.ids(),
                weight: 1.0,
            },
            math::RankedList {
                ids: dtsc_rank.ids(),
                weight: 1.0,
            },
        ],
        &Default::default(),
    )?
    .top(Some(3));
    let fused_candidates = candidates.select(fused.ids().iter().copied())?;
    let deduped = math::semantic_dedup(fused_candidates.vectors(), &Default::default())?;
    println!("RRF: {:?}, distinct: {:?}", fused.ids(), deduped.kept);
    println!("DTSC over River: {:?}", dtsc_river.top(Some(3)).ids());

    let first = &candidates.ids()[..candidates.len().min(2)];
    let stages = [
        ThinkingStage {
            label: "起点",
            candidates: first,
            limit: 1,
        },
        ThinkingStage {
            label: "延续",
            candidates: candidates.ids(),
            limit: 2,
        },
    ];
    let thinking = evaluation.thinking(&stages, &Default::default())?;
    for stage in thinking.stages {
        for choice in stage.selections {
            println!(
                "{} id={} parents={:?}",
                stage.label, choice.candidate, choice.parents
            );
        }
    }

    // 相同候选在另一查询继续复用。旧Scores不能串入新的Evaluation。
    let other_query = [0.0_f32, 0.0, 0.1, 1.0];
    let other = model.observe(Query::new(&other_query), &Default::default())?;
    let other_evaluation = other.evaluate(&candidates, &Default::default())?;
    let other_scores = other_evaluation.tag_memo();
    let other_rank = other_evaluation
        .dtsc(&other_scores, &Default::default())?
        .top(Some(3));
    println!("reused candidates: {:?}", other_rank.ids());
    Ok(())
}
