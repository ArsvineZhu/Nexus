// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::*;

#[test]
fn nexus_golden_capture() {
    let cases = [
        (
            "opposed",
            vec![vec![0.94_f32, 0.342], vec![1.0, 0.0], vec![0.94, -0.342]],
            vec![2.0, 1.0],
        ),
        (
            "reconsider_after_projection",
            vec![
                vec![0.005, 0.9999875, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.999, 0.04471018, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![5.0, 4.0, 3.0],
        ),
        (
            "same_ray_tie",
            vec![
                vec![0.6, 0.8],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
            vec![1.0; 3],
        ),
        (
            "no_positive_fit",
            vec![vec![-1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![1.0; 2],
        ),
    ];
    let rows: Vec<_> = cases
        .into_iter()
        .map(|(case, vectors, weights)| {
            let normalized: Vec<_> = vectors
                .iter()
                .map(|vector| {
                    let mut unit: Vec<_> = vector.iter().copied().map(f64::from).collect();
                    assert!(normalize(&mut unit));
                    unit
                })
                .collect();
            let candidates: Vec<_> = weights
                .iter()
                .enumerate()
                .map(|(i, &weight): (usize, &f64)| AgsCandidate {
                    id: i as i64 + 2,
                    prior: weight.ln_1p(),
                })
                .collect();
            let refs: Vec<_> = normalized[1..].iter().map(Vec::as_slice).collect();
            let diagonal: Vec<_> = refs.iter().map(|row| dot(row, row)).collect();
            let correlations = dot_neighbors(&normalized[0], &refs);
            let ags = direct_axes_core(
                AgsTarget::Explain(&normalized[0]),
                &candidates,
                &refs,
                Some(&diagonal),
                Some(&correlations),
                24,
                0.015,
                false,
            );
            let selected: Vec<_> = ags
                .picked
                .iter()
                .map(|&(id, _)| (id, refs[(id - 2) as usize]))
                .collect();
            let cone = crate::memo_nnls::solve_qr_f64(&normalized[0], &selected).unwrap();
            let residual = cone.diagnostics.residual_ratio.sqrt().clamp(0.0, 1.0);
            json!({"case":case,"vectors":vectors,"weights":weights,
            "basis_used":selected.len(),"residual_ratio":residual})
        })
        .collect();
    println!("NEXUS_GOLDEN:{}", json!(["geometry", rows]));
}
