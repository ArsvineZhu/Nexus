// SPDX-License-Identifier: CC-BY-NC-SA-4.0
mod fixture;

use nexus::{Baseline, DtscRequestOptions, Query, RiverRequestOptions, ScoreEvidence};

fn main() -> nexus::Result<()> {
    let model = fixture::model()?;
    let candidates = fixture::candidates(&model)?;
    let query = Query::new(&fixture::QUERY)
        .with_text("增量更新")
        .with_core_concepts(&[30]);

    let river = model.rank_river(
        query,
        &candidates,
        &RiverRequestOptions {
            limit: Some(3),
            ..Default::default()
        },
    )?;
    for row in river.rows() {
        println!("id={} value={}", row.id, row.value);
        if let ScoreEvidence::River(evidence) = &row.evidence
            && let Some(path) = &evidence.selected_path
        {
            println!(
                "source={} path={} positions={:?}",
                path.source, path.path, path.positions
            );
        }
    }

    let dtsc = model.rank_dtsc(
        Query::new(&fixture::QUERY)
            .with_text("增量更新")
            .with_core_concepts(&[30]),
        &candidates,
        &DtscRequestOptions {
            baseline: Baseline::TagMemo,
            limit: Some(3),
            ..Default::default()
        },
    )?;
    let second = [0.0, 0.0, 0.0, 1.0];
    let reranked = model.rank_tag_memo(Query::new(&second), &candidates, &Default::default())?;
    println!("same catalog, other query: {:?}", reranked.ids());
    drop(candidates);
    drop(model);
    println!("owned DTSC result: {:?}", dtsc.ids());
    Ok(())
}
