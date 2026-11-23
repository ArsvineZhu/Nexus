// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use nexus::math::{self, SearchRequest, VectorRows, WarpedIndex};

fn main() -> nexus::Result<()> {
    let ids = [101_u64, 102, 103];
    let values = [1.0_f32, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 1.0];
    let query = [1.0_f32, 0.2, 0.0];

    let cosine = math::cosine_scores(
        &query,
        VectorRows::new(&ids, &values, 3)?,
        &Default::default(),
    )?;
    let best = cosine.top(Some(2));
    assert_eq!(best.ids()[0], 101);
    for row in best.rows() {
        println!("cosine id={} value={}", row.id, row.value);
    }

    let index = WarpedIndex::build(VectorRows::new(&ids, &values, 3)?, &Default::default())?;
    let hits = index.search(
        &query,
        &SearchRequest {
            limit: 2,
            ..Default::default()
        },
    )?;
    for hit in hits.rows() {
        println!("warped id={} distance={}", hit.id, hit.distance);
    }
    Ok(())
}
