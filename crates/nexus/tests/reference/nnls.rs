// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::*;
use serde_json::json;

#[test]
fn nexus_golden_capture() {
    let cases = [
        (
            "opposite",
            vec![0.94, 0.342],
            vec![1, 2],
            vec![vec![1.0, 0.0], vec![0.94, -0.342]],
        ),
        (
            "joint_fit",
            vec![1.0, 0.1],
            vec![1, 2],
            vec![vec![1.0, 1.0], vec![0.0, -1.0]],
        ),
        (
            "duplicate_and_zero",
            vec![0.3, 0.4, 0.5],
            vec![30, 10, 20, 40],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0; 3],
            ],
        ),
        (
            "near_collinear",
            vec![0.0, 1.0],
            vec![1, 2],
            vec![vec![1.0, 0.0], vec![1.0, 1e-7]],
        ),
        ("zero_target", vec![0.0, 0.0], vec![1], vec![vec![1.0, 0.0]]),
    ];
    let results: Vec<_> = cases
        .into_iter()
        .map(|(case, query, ids, vectors)| {
            let columns: Vec<_> = ids
                .iter()
                .copied()
                .zip(vectors.iter().map(Vec::as_slice))
                .collect();
            let result = solve(&query, &columns).unwrap();
            json!({"case":case, "query":query, "ids":ids, "vectors":vectors,
            "projection":result.projection, "residual":result.residual,
            "weights":result.contributions, "residual_ratio":result.diagnostics.residual_ratio})
        })
        .collect();
    println!("NEXUS_GOLDEN:{}", json!(["nnls", results]));
}
