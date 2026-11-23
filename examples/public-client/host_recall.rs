// SPDX-License-Identifier: CC-BY-NC-SA-4.0
mod fixture;

use nexus::{Candidates, Observation, Query, Ranking, math};

struct HostIndex {
    ids: Vec<u64>,
    values: Vec<f32>,
    dimension: usize,
}

impl HostIndex {
    fn recall(&self, vector: &[f32], limit: usize) -> nexus::Result<Vec<u64>> {
        let scores = math::cosine_scores(
            vector,
            math::VectorRows::new(&self.ids, &self.values, self.dimension)?,
            &Default::default(),
        )?;
        Ok(scores.top(Some(limit)).ids().to_vec())
    }
}

fn finish(
    observation: &Observation,
    catalog: &Candidates<'_>,
    recalled: Vec<u64>,
) -> nexus::Result<Ranking> {
    let candidates = catalog.select(recalled)?;
    let evaluation = observation.evaluate(&candidates, &Default::default())?;
    let baseline = evaluation.tag_memo();
    Ok(evaluation
        .dtsc(&baseline, &Default::default())?
        .top(Some(3)))
}

fn main() -> nexus::Result<()> {
    let model = fixture::model()?;
    let catalog = fixture::candidates(&model)?;
    let host = HostIndex {
        ids: fixture::ROWS.iter().map(|row| row.0).collect(),
        values: fixture::ROWS.iter().flat_map(|row| row.1).collect(),
        dimension: 4,
    };

    let observation = model.observe(
        Query::new(&fixture::QUERY)
            .with_text("增量更新")
            .with_core_concepts(&[30]),
        &Default::default(),
    )?;
    let recall_lists = observation
        .recall_vectors()
        .map(|direction| {
            let hits = host.recall(direction.vector, 3)?;
            println!("{:?} -> {:?}", direction.kind, hits);
            Ok(hits)
        })
        .collect::<nexus::Result<Vec<_>>>()?;
    let borrowed: Vec<&[u64]> = recall_lists.iter().map(Vec::as_slice).collect();
    let ids = math::merge_ranked_ids(&borrowed);
    println!("merged candidates: {ids:?}");
    let ranking = finish(&observation, &catalog, ids)?;
    println!("host recall -> DTSC: {:?}", ranking.ids());

    Ok(())
}

#[cfg(test)]
mod integration_types {
    use super::*;
    use std::future::Future;

    async fn after_host_recall<F, Fut>(
        model: &nexus::Model,
        catalog: &Candidates<'_>,
        query: &[f32],
        recall: F,
    ) -> nexus::Result<Ranking>
    where
        F: FnOnce(Vec<f32>) -> Fut,
        Fut: Future<Output = nexus::Result<Vec<u64>>>,
    {
        let observation = model.observe(Query::new(query), &Default::default())?;
        let direction = observation
            .vector(nexus::RecallKind::Enhanced)
            .expect("enhanced direction");
        let ids = recall(direction.to_vec()).await?;
        finish(&observation, catalog, ids)
    }

    #[test]
    fn owned_results_and_host_recall_are_thread_safe() -> nexus::Result<()> {
        fn shareable<T: Send + Sync>() {}
        shareable::<nexus::Model>();
        shareable::<Candidates<'_>>();
        shareable::<Observation>();
        shareable::<nexus::Scores>();
        shareable::<Ranking>();
        shareable::<nexus::DtscEvidence>();
        let model = fixture::model()?;
        let catalog = fixture::candidates(&model)?;
        let index = HostIndex {
            ids: fixture::ROWS.iter().map(|row| row.0).collect(),
            values: fixture::ROWS.iter().flat_map(|row| row.1).collect(),
            dimension: 4,
        };
        let future = after_host_recall(&model, &catalog, &fixture::QUERY, |query| async move {
            index.recall(&query, 3)
        });
        fn send<T: Send>(_: T) {}
        send(future);
        Ok(())
    }
}
