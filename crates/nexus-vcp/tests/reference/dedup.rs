// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::*;
use serde_json::json;

#[test]
fn nexus_golden_capture() {
    let cases = [
        (
            "aligned_rank",
            vec![1.0_f32, 0.0],
            vec![vec![1.0, 0.0], vec![0.98, 0.2], vec![0.0, 1.0]],
            0.92,
        ),
        (
            "distinct_query_order",
            vec![1.0, 0.0],
            vec![vec![0.98, 0.2], vec![1.0, 0.0], vec![0.0, 1.0]],
            0.92,
        ),
        (
            "zero_vector",
            vec![1.0, 0.0],
            vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]],
            0.92,
        ),
    ];
    let output: Vec<_> = cases
        .into_iter()
        .map(|(case, query, vectors, threshold)| {
            let db = rusqlite::Connection::open_in_memory().unwrap();
            db.execute_batch("CREATE TABLE chunks(id INTEGER PRIMARY KEY, vector BLOB)")
                .unwrap();
            let mut candidates = Vec::new();
            for (row, vector) in vectors.iter().enumerate() {
                let id = row as i64 + 1;
                let bytes: Vec<u8> = vector.iter().flat_map(|v| f32::to_ne_bytes(*v)).collect();
                db.execute(
                    "INSERT INTO chunks VALUES (?1, ?2)",
                    rusqlite::params![id, bytes],
                )
                .unwrap();
                candidates.push(DedupCandidate {
                    id,
                    score: 1.0 - row as f64 * 0.1,
                    original_index: row,
                });
            }
            let result =
                deduplicate_with_connection(&db, candidates, &query, 2, threshold, vectors.len())
                    .unwrap();
            json!({"case":case,"query":query,"vectors":vectors,"threshold":threshold,
            "survivors":result.candidates.iter().map(|c| c.id).collect::<Vec<_>>()})
        })
        .collect();
    println!("NEXUS_GOLDEN:{}", json!(["dedup", output]));
}
