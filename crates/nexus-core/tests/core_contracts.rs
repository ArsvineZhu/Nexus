// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use nexus_core::kernel::{
    batch_cosines_f32, batch_dot_4_f64, cosine_f32, cosine_f32_with_norms, cosine_f64,
    csr_spmv_dual, dot_f32, norm_sq_f32, relax_field, sub_4_f64,
};
use nexus_core::nnls::{
    CHECK_TOL, solve_nnls_f32, solve_nnls_f64, solve_nnls_flat_f64, solve_nnls_with_ids_f64,
};
use nexus_core::{
    CancellationToken, CsrGraphView, CsrParts, GroupId, NnlsWorkspace, NodeId, ParallelPolicy,
    PrecisionMode, ProvenanceView, ResourceLimits, RootId, SupportGroupView, SupportKind, VcpError,
    VectorMatrixView,
};
use std::sync::atomic::AtomicBool;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn gen_f32_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed;
    (0..dim)
        .map(|_| {
            let bits = (lcg_next(&mut s) >> 40) as i32 - 8_388_608;
            (bits as f32) / 8_388_608.0
        })
        .collect()
}

fn scalar_ref_squared_norm(v: &[f32]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| {
        let xf = f64::from(x);
        acc + xf * xf
    })
}

fn scalar_ref_dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |acc, (&x, &y)| acc + f64::from(x) * f64::from(y))
}

fn scalar_ref_dtsc_cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let aa = scalar_ref_squared_norm(a);
    let bb = scalar_ref_squared_norm(b);
    if !(aa > 1e-12 && bb > 1e-12) {
        return 0.0;
    }
    scalar_ref_dot(a, b) / (aa.sqrt() * bb.sqrt())
}

// =========================================================================
// 1. NNLS & QR-NNLS Contract Tests
// =========================================================================

#[test]
fn nnls_opposite_and_secondary_directions() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    let a = [1.0, 0.0];
    let b = [0.94, -0.342];
    let q = [0.94, 0.342];
    let sol = solve_nnls_f64(&q, &[&a, &b], &limits, &cancel, &mut scratch).unwrap();
    assert_eq!(sol.weights[1], 0.0);
    assert!(sol.residual[1] > 0.34);
    assert!(sol.max_kkt_violation <= CHECK_TOL);

    let q_neg = [-1.0, 0.0];
    let sol_neg = solve_nnls_f64(&q_neg, &[&a], &limits, &cancel, &mut scratch).unwrap();
    assert_eq!(sol_neg.projection, vec![0.0, 0.0]);
    assert!((sol_neg.residual_ratio - 1.0).abs() < 1e-12);
}

#[test]
fn nnls_joint_fit_negative_initial_correlation() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    let a = [1.0, 1.0];
    let b = [0.0, -1.0];
    let q = [1.0, 0.1];
    let sol = solve_nnls_f64(&q, &[&a, &b], &limits, &cancel, &mut scratch).unwrap();
    assert!(sol.weights[1] > 0.8);
    assert!(sol.residual.iter().all(|r| r.abs() < 1e-6));
    assert!(sol.max_kkt_violation <= CHECK_TOL);
}

#[test]
fn nnls_dependent_rays_r2() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    let c1 = [1.0, 1.0];
    let c2 = [0.0, -1.0];
    let c3 = [-1.0, 0.0];
    let q = [-1.0, 0.0];
    let sol = solve_nnls_with_ids_f64(
        &q,
        &[(1, &c1), (2, &c2), (3, &c3)],
        &limits,
        &cancel,
        &mut scratch,
    )
    .unwrap();
    assert!(sol.residual.iter().all(|r| r.abs() < 1e-8));
    assert!(sol.weights[2] > 0.9);
}

#[test]
fn nnls_scale_duplicate_ids_zero_column_and_permutations() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    for scale in [1e-6_f64, 1.0, 1e6] {
        let a = [scale, 0.0, 0.0];
        let b = [0.0, scale, 0.0];
        let z = [0.0, 0.0, 0.0];
        let q = [0.3 * scale, 0.4 * scale, 0.5 * scale];

        let first = solve_nnls_with_ids_f64(
            &q,
            &[(30, &a), (10, &a), (20, &b), (40, &z)],
            &limits,
            &cancel,
            &mut scratch,
        )
        .unwrap();
        let second = solve_nnls_with_ids_f64(
            &q,
            &[(40, &z), (20, &b), (10, &a), (30, &a)],
            &limits,
            &cancel,
            &mut scratch,
        )
        .unwrap();

        assert_eq!(first.weights[0], 0.0); // id=30 loses to id=10
        assert_eq!(first.weights[3], 0.0); // zero column
        assert!((first.weights[1] - second.weights[2]).abs() < 1e-12);
        assert!((first.weights[2] - second.weights[1]).abs() < 1e-12);
        for (p1, p2) in first.projection.iter().zip(&second.projection) {
            assert!((p1 - p2).abs() <= 1e-10 * scale);
        }
    }

    let q_ratio = [0.3, 0.0, 0.4];
    let c1 = [1.0, 0.0, 0.0];
    let c0 = [0.0, 0.0, 0.0];
    let sol = solve_nnls_with_ids_f64(
        &q_ratio,
        &[(3, &c1), (1, &c1), (2, &c0)],
        &limits,
        &cancel,
        &mut scratch,
    )
    .unwrap();
    assert!((sol.residual_ratio - 0.64).abs() < 1e-10);
}

#[test]
fn nnls_near_collinear_cancellation() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    for eps in [1e-7_f64, 1e-4] {
        let a = [0.6, 0.8, 0.0];
        let b = [0.6, 0.8, eps];
        let q = [0.0, 0.0, 1.0];
        let sol = solve_nnls_f64(&q, &[&a, &b], &limits, &cancel, &mut scratch).unwrap();
        assert!(sol.residual[2] > 0.99);
        assert!(sol.weights.iter().all(|&w| w < 0.001));
        assert!(sol.max_kkt_violation <= CHECK_TOL);
    }
}

#[test]
fn nnls_3072d_128_cols_stress() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    let dim = 3072usize;
    let cols_count = 128usize;
    let mut cols = vec![vec![0.0_f64; dim]; cols_count];
    let mut q = vec![0.0_f64; dim];
    for i in 0..cols_count {
        cols[i][i] = 1.0;
        cols[i][256] = 0.05;
        let w = 0.01 + (i as f64) * 0.0001;
        q[i] += w;
        q[256] += w * 0.05;
    }
    q[3000] = 0.2;

    let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    let sol = solve_nnls_f64(&q, &refs, &limits, &cancel, &mut scratch).unwrap();
    assert_eq!(sol.active_columns, 128);
    assert_eq!(sol.iterations, 128);
    assert!((sol.residual[3000] - 0.2).abs() < 1e-6);
    assert!(sol.residual[..3000].iter().all(|r| r.abs() < 1e-6));
    assert!(sol.max_kkt_violation <= CHECK_TOL);
}

#[test]
fn nnls_256_cols_rank_deficient_d2048_qr_compression() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    let dim = 2048usize;
    let num_cols = 256usize;
    let mut cols_flat = vec![0.0_f64; num_cols * dim];
    for c in 0..num_cols {
        let base_col = c % 192; // force 64 exact rank-deficient duplicates
        cols_flat[c * dim + base_col] = 1.0;
        cols_flat[c * dim + 512 + (base_col % 64)] = 0.25;
        if c >= 192 {
            // Also inject slight linear combinations of earlier columns
            cols_flat[c * dim + ((base_col + 1) % 192)] = 0.1;
        }
    }

    let mut target = vec![0.0_f64; dim];
    for c in 0..120 {
        let weight = 0.02 + (c as f64) * 0.001;
        for r in 0..dim {
            target[r] += weight * cols_flat[c * dim + r];
        }
    }
    target[1500] = 0.35; // orthogonal component

    let sol = solve_nnls_flat_f64(
        &target,
        &cols_flat,
        num_cols,
        dim,
        &limits,
        &cancel,
        &mut scratch,
    )
    .unwrap();
    assert!(sol.weights.iter().all(|&w| w >= 0.0 && w.is_finite()));
    assert!(sol.max_kkt_violation <= CHECK_TOL);
    assert!((sol.residual[1500] - 0.35).abs() < 1e-6);

    // Also verify f32 wrapper on 128 columns
    let cols_f32: Vec<Vec<f32>> = (0..128)
        .map(|c| {
            cols_flat[c * dim..(c + 1) * dim]
                .iter()
                .map(|&v| v as f32)
                .collect()
        })
        .collect();
    let refs_f32: Vec<&[f32]> = cols_f32.iter().map(|c| c.as_slice()).collect();
    let target_f32: Vec<f32> = target.iter().map(|&v| v as f32).collect();
    let sol_f32 = solve_nnls_f32(&target_f32, &refs_f32, &limits, &cancel, &mut scratch).unwrap();
    assert!(sol_f32.max_kkt_violation <= CHECK_TOL);
}

#[test]
fn nnls_rejects_invalid_limits_and_cancellation() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();
    let col = [1.0, 0.0];

    // Empty query
    assert!(matches!(
        solve_nnls_f64(&[], &[&col], &limits, &cancel, &mut scratch),
        Err(VcpError::InvalidInput { .. })
    ));
    // Non-finite query
    assert!(matches!(
        solve_nnls_f64(&[f64::NAN, 1.0], &[&col], &limits, &cancel, &mut scratch),
        Err(VcpError::InvalidInput { .. })
    ));
    assert!(matches!(
        solve_nnls_f64(
            &[f64::INFINITY, 1.0],
            &[&col],
            &limits,
            &cancel,
            &mut scratch
        ),
        Err(VcpError::InvalidInput { .. })
    ));
    // Non-finite column
    let bad_col = [f64::NAN, 1.0];
    assert!(matches!(
        solve_nnls_f64(&[1.0, 0.0], &[&bad_col], &limits, &cancel, &mut scratch),
        Err(VcpError::InvalidInput { .. })
    ));
    // Column limit exceeded
    let many_cols = vec![col.as_slice(); 257];
    assert!(matches!(
        solve_nnls_f64(&[1.0, 0.0], &many_cols, &limits, &cancel, &mut scratch),
        Err(VcpError::ResourceLimitExceeded {
            limit_name: "nnls_max_columns",
            actual: 257,
            limit: 256,
            ..
        })
    ));
    // Cancelled
    let flag = AtomicBool::new(true);
    let cancelled = CancellationToken::from_atomic(&flag);
    assert!(matches!(
        solve_nnls_f64(&[1.0, 0.0], &[&col], &limits, &cancelled, &mut scratch),
        Err(VcpError::Cancelled { .. })
    ));
}

// =========================================================================
// 2. Kernel StrictLeftFold vs FastSimd & Batch/SpMV Tests
// =========================================================================

#[test]
fn kernel_strict_left_fold_zero_bit_diff() {
    let subnormal = f32::from_bits(1);
    let quiet_nan = f32::from_bits(0x7fa1_2345);
    let special: Vec<Vec<f32>> = vec![
        vec![],
        vec![0.0],
        vec![-0.0],
        vec![1e-6],
        vec![1e-7],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
        vec![f32::NAN],
        vec![quiet_nan],
        vec![f32::MAX],
        vec![subnormal],
        vec![1.0, f32::INFINITY],
        vec![1.0, f32::NAN],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
        vec![1e-6, 1e-6],
    ];

    let mut vectors = special;
    for (idx, &dim) in [1usize, 2, 3, 7, 16, 31, 128, 512, 2048].iter().enumerate() {
        vectors.push(gen_f32_vec(0x9E37_79B9_7F4A_7C15 + idx as u64, dim));
    }

    for (i, a) in vectors.iter().enumerate() {
        let got_aa = norm_sq_f32(a, PrecisionMode::StrictLeftFold);
        let ref_aa = scalar_ref_squared_norm(a);
        assert_eq!(
            got_aa.to_bits(),
            ref_aa.to_bits(),
            "norm_sq mismatch at {i}"
        );

        for (j, b) in vectors.iter().enumerate() {
            let got_cos = cosine_f32(a, b, PrecisionMode::StrictLeftFold);
            let ref_cos = scalar_ref_dtsc_cosine(a, b);
            assert_eq!(
                got_cos.to_bits(),
                ref_cos.to_bits(),
                "cosine mismatch at ({i}, {j}): got {got_cos} vs {ref_cos}"
            );
        }
    }
}

#[test]
fn kernel_fast_simd_within_tolerance_d2048() {
    let dim = 2048usize;
    let mut max_delta = 0.0_f64;
    for i in 0..64 {
        let a = gen_f32_vec(1000 + i * 2, dim);
        let b = gen_f32_vec(1001 + i * 2, dim);
        let strict = cosine_f32(&a, &b, PrecisionMode::StrictLeftFold);
        let simd = cosine_f32(&a, &b, PrecisionMode::FastSimd);
        let delta = (strict - simd).abs();
        max_delta = max_delta.max(delta);
        assert!(
            delta <= 2.61e-7,
            "FastSimd delta {delta:e} exceeded 2.61e-7 at pair {i}"
        );
        let dot_delta = (dot_f32(&a, &b, PrecisionMode::StrictLeftFold)
            - dot_f32(&a, &b, PrecisionMode::FastSimd))
        .abs();
        assert!(dot_delta <= 1e-10);
    }

    // Also test cosine_f64 & batch_dot_4_f64 & sub_4_f64
    let f64_a: Vec<f64> = (0..dim).map(|k| ((k % 17) as f64) * 0.1 - 0.8).collect();
    let f64_b: Vec<f64> = (0..dim).map(|k| ((k % 13) as f64) * 0.1 - 0.5).collect();
    let c_strict = cosine_f64(&f64_a, &f64_b, PrecisionMode::StrictLeftFold);
    let c_simd = cosine_f64(&f64_a, &f64_b, PrecisionMode::FastSimd);
    assert!((c_strict - c_simd).abs() <= 1e-12);

    let axes = [
        vec![1.0_f64; 16],
        vec![0.5_f64; 16],
        vec![-0.25_f64; 16],
        vec![2.0_f64; 16],
    ];
    let target = vec![3.0_f64; 16];
    let dots_s = batch_dot_4_f64(
        &target,
        [&axes[0], &axes[1], &axes[2], &axes[3]],
        PrecisionMode::StrictLeftFold,
    );
    let dots_f = batch_dot_4_f64(
        &target,
        [&axes[0], &axes[1], &axes[2], &axes[3]],
        PrecisionMode::FastSimd,
    );
    for k in 0..4 {
        assert!((dots_s[k] - dots_f[k]).abs() < 1e-12);
    }
    let mut residual = target.clone();
    sub_4_f64(
        &mut residual,
        [0.5, 0.5, 0.5, 0.5],
        [&axes[0], &axes[1], &axes[2], &axes[3]],
    );
    assert!(
        residual
            .iter()
            .all(|&v| (v - (3.0 - 0.5 * 3.25)).abs() < 1e-12)
    );
}

#[test]
fn kernel_batch_cosines_1way_and_4way_parallel_and_seq() {
    let rows = 80usize;
    let dim = 2048usize;
    let mut mat_data = Vec::with_capacity(rows * dim);
    let mut sq_norms = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = gen_f32_vec(5000 + r as u64, dim);
        sq_norms.push(norm_sq_f32(&row, PrecisionMode::StrictLeftFold));
        mat_data.extend_from_slice(&row);
    }
    let view = VectorMatrixView::new(&mat_data, rows, dim)
        .unwrap()
        .with_sq_norms(&sq_norms)
        .unwrap();

    let q0 = gen_f32_vec(9001, dim);
    let q1 = gen_f32_vec(9002, dim);
    let q2 = gen_f32_vec(9003, dim);
    let q3 = gen_f32_vec(9004, dim);

    let mut out1 = vec![0.0_f64; rows];
    batch_cosines_f32(
        &view,
        &[&q0],
        &mut [&mut out1],
        PrecisionMode::FastSimd,
        1e-12,
        ParallelPolicy::Sequential,
    )
    .unwrap();

    let mut out4_seq = vec![vec![0.0_f64; rows]; 4];
    let mut refs_seq: Vec<&mut [f64]> = out4_seq.iter_mut().map(|v| v.as_mut_slice()).collect();
    batch_cosines_f32(
        &view,
        &[&q0, &q1, &q2, &q3],
        &mut refs_seq,
        PrecisionMode::FastSimd,
        1e-12,
        ParallelPolicy::Sequential,
    )
    .unwrap();

    let mut out4_par = vec![vec![0.0_f64; rows]; 4];
    let mut refs_par: Vec<&mut [f64]> = out4_par.iter_mut().map(|v| v.as_mut_slice()).collect();
    batch_cosines_f32(
        &view,
        &[&q0, &q1, &q2, &q3],
        &mut refs_par,
        PrecisionMode::FastSimd,
        1e-12,
        ParallelPolicy::GlobalRayon,
    )
    .unwrap();

    let queries = [&q0, &q1, &q2, &q3];
    let mut strict = [(); 4].map(|_| vec![0.0; rows]);
    let mut strict_columns = strict.each_mut().map(Vec::as_mut_slice);
    batch_cosines_f32(
        &view,
        &[&q0, &q1, &q2, &q3],
        &mut strict_columns,
        PrecisionMode::StrictLeftFold,
        1e-12,
        ParallelPolicy::GlobalRayon,
    )
    .unwrap();
    for r in 0..rows {
        let row = view.row(r).unwrap();
        assert!((out1[r] - out4_seq[0][r]).abs() <= 1e-12);
        for k in 0..4 {
            assert_eq!(out4_seq[k][r].to_bits(), out4_par[k][r].to_bits());
            let exact = cosine_f32_with_norms(
                row,
                queries[k],
                sq_norms[r],
                norm_sq_f32(queries[k], PrecisionMode::StrictLeftFold),
                PrecisionMode::StrictLeftFold,
                1e-12,
            );
            assert!((out4_seq[k][r] - exact).abs() <= 2.61e-7);
            assert_eq!(strict[k][r].to_bits(), exact.to_bits());
        }
    }
}

#[test]
fn kernel_csr_spmv_dual_edge_gates_and_parallel() {
    let row_offsets = [0usize, 2, 3, 4];
    let targets = [NodeId(1), NodeId(2), NodeId(2), NodeId(0)];
    let weights = [0.6, 0.3, 0.8, 0.5];
    let is_wh = [false, true, false, false];
    let inbound = [0.5, 0.6, 1.1];
    let gains = [1.0, 1.0, 1.0];
    let sorted = [0u32, 1, 2, 3];

    let graph = CsrGraphView::new(CsrParts {
        node_count: 3,
        row_offsets: &row_offsets,
        targets: &targets,
        weights: &weights,
        is_static_wormhole: &is_wh,
        inbound: &inbound,
        max_inbound: 1.1,
        anchor_gains: &gains,
        target_sorted_indices: Some(&sorted),
        edge_gates: None,
    })
    .unwrap();
    graph.validate("test").unwrap();

    let local = [1.0, 0.5, 0.2];
    let transfer = [0.4, 0.0, 0.8];
    let mut loc_none = [0.0; 3];
    let mut trn_none = [0.0; 3];
    csr_spmv_dual(
        &graph,
        &local,
        &transfer,
        &mut loc_none,
        &mut trn_none,
        ParallelPolicy::Sequential,
    )
    .unwrap();

    // edge_gates = [1.0; nnz] must produce 0-bit diff vs None
    let ones = [1.0; 4];
    let gated_ones = graph.with_edge_gates(&ones).unwrap();
    let mut loc_ones = [0.0; 3];
    let mut trn_ones = [0.0; 3];
    csr_spmv_dual(
        &gated_ones,
        &local,
        &transfer,
        &mut loc_ones,
        &mut trn_ones,
        ParallelPolicy::Sequential,
    )
    .unwrap();
    for i in 0..3 {
        assert_eq!(loc_none[i].to_bits(), loc_ones[i].to_bits());
        assert_eq!(trn_none[i].to_bits(), trn_ones[i].to_bits());
    }

    // edge_gates = [0.0; nnz] blocks all edges
    let zeros = [0.0; 4];
    let gated_zeros = graph.with_edge_gates(&zeros).unwrap();
    let mut loc_z = [99.0; 3];
    let mut trn_z = [99.0; 3];
    csr_spmv_dual(
        &gated_zeros,
        &local,
        &transfer,
        &mut loc_z,
        &mut trn_z,
        ParallelPolicy::Sequential,
    )
    .unwrap();
    assert_eq!(loc_z, [0.0; 3]);
    assert_eq!(trn_z, [0.0; 3]);

    // relax_field smoke check
    let mut next = [0.0; 3];
    let delta = relax_field(&local, &loc_none, 0.5, &local, &mut next);
    assert!(delta > 0.0);

    // Parallel SpMV parity on larger graph (n=2048, nnz=16384)
    let big_n = 2048usize;
    let deg = 8usize;
    let mut big_offsets = Vec::with_capacity(big_n + 1);
    let mut big_targets = Vec::with_capacity(big_n * deg);
    let mut big_weights = Vec::with_capacity(big_n * deg);
    let mut big_wh = vec![false; big_n * deg];
    big_offsets.push(0);
    for u in 0..big_n {
        for d in 0..deg {
            big_targets.push(NodeId(((u + d + 1) % big_n) as u32));
            big_weights.push(0.1);
        }
        big_offsets.push(big_targets.len());
    }
    big_wh[0] = true;
    let big_inbound = vec![0.8; big_n];
    let big_gains = vec![1.0; big_n];
    let big_graph = CsrGraphView::new(CsrParts {
        node_count: big_n,
        row_offsets: &big_offsets,
        targets: &big_targets,
        weights: &big_weights,
        is_static_wormhole: &big_wh,
        inbound: &big_inbound,
        max_inbound: 0.8,
        anchor_gains: &big_gains,
        target_sorted_indices: None,
        edge_gates: None,
    })
    .unwrap();
    let big_l = vec![1.0; big_n];
    let big_t = vec![0.5; big_n];
    let mut seq_l = vec![0.0; big_n];
    let mut seq_t = vec![0.0; big_n];
    let mut par_l = vec![0.0; big_n];
    let mut par_t = vec![0.0; big_n];
    csr_spmv_dual(
        &big_graph,
        &big_l,
        &big_t,
        &mut seq_l,
        &mut seq_t,
        ParallelPolicy::Sequential,
    )
    .unwrap();
    csr_spmv_dual(
        &big_graph,
        &big_l,
        &big_t,
        &mut par_l,
        &mut par_t,
        ParallelPolicy::GlobalRayon,
    )
    .unwrap();
    for i in 0..big_n {
        assert!((seq_l[i] - par_l[i]).abs() < 1e-12);
        assert!((seq_t[i] - par_t[i]).abs() < 1e-12);
    }
}

// =========================================================================
// 3. CSR, Provenance, EpochTable, Views & Operator Trait Tests
// =========================================================================

#[test]
fn csr_target_sorted_indices_and_gate_validation() {
    // Row 0 has edges ordered by weight descending: targets [3, 1, 2] at edge indices [0, 1, 2]
    // Sorted by target ascending: [edge 1 (target 1), edge 2 (target 2), edge 0 (target 3)]
    let row_offsets = [0usize, 3, 3, 3, 3];
    let targets = [NodeId(3), NodeId(1), NodeId(2)];
    let weights = [0.9, 0.5, 0.2];
    let is_wh = [false, true, false];
    let inbound = [0.0, 0.5, 0.2, 0.9];
    let gains = [1.0; 4];
    let sorted = [1u32, 2, 0];

    let csr_indexed = CsrGraphView::new(CsrParts {
        node_count: 4,
        row_offsets: &row_offsets,
        targets: &targets,
        weights: &weights,
        is_static_wormhole: &is_wh,
        inbound: &inbound,
        max_inbound: 0.9,
        anchor_gains: &gains,
        target_sorted_indices: Some(&sorted),
        edge_gates: None,
    })
    .unwrap();
    csr_indexed.validate("test").unwrap();

    let csr_linear = CsrGraphView::new(CsrParts {
        target_sorted_indices: None,
        ..*csr_indexed
    })
    .unwrap();

    for dst in 0..4 {
        let d = NodeId(dst);
        assert_eq!(
            csr_indexed.find_edge(NodeId(0), d),
            csr_linear.find_edge(NodeId(0), d)
        );
        assert_eq!(
            csr_indexed.edge_weight(NodeId(0), d),
            csr_linear.edge_weight(NodeId(0), d)
        );
    }
    assert_eq!(csr_indexed.find_edge(NodeId(0), NodeId(1)), Some(1));
    assert_eq!(csr_indexed.edge_weight(NodeId(0), NodeId(1)), 0.5);
    assert!(csr_indexed.is_wormhole_edge(1));

    // Gate checks
    let gates = [0.5, 0.0, 1.0];
    let gated = csr_indexed.with_edge_gates(&gates).unwrap();
    assert!((gated.edge_weight(NodeId(0), NodeId(3)) - 0.45).abs() < 1e-12);
    assert_eq!(gated.edge_weight(NodeId(0), NodeId(1)), 0.0);

    // Reject out-of-range gate
    assert!(matches!(
        csr_indexed.with_edge_gates(&[1.2, 0.5, 0.5]),
        Err(VcpError::InvalidInput { .. })
    ));
    assert!(matches!(
        csr_indexed.with_edge_gates(&[0.5, 0.5]),
        Err(VcpError::DimensionMismatch { .. })
    ));

    // Reject bad target_sorted_indices (wrong order)
    let bad_sorted = [0u32, 1, 2];
    let bad_csr = CsrParts {
        target_sorted_indices: Some(&bad_sorted),
        ..*csr_indexed
    };
    assert!(matches!(
        bad_csr.validate("test"),
        Err(VcpError::InvalidInput { .. })
    ));
}

#[test]
fn provenance_multi_root_and_support_group_joint_exclusion() {
    // Mirrors Nous/native/vcp/tests/provenance.rs:4-58
    // Case 1: ungrouped roots mapped directly as GroupIds (fallback to RootId matching)
    let groups_empty = SupportGroupView::empty();
    let offsets = [0u32, 3];
    let prov_groups = [GroupId(101), GroupId(202), GroupId(303)];
    let prov_masses = [3.0, 1.0, 2.0];
    let prov = ProvenanceView {
        edge_count: 1,
        edge_prov_offsets: &offsets,
        prov_groups: &prov_groups,
        prov_masses: &prov_masses,
    };
    prov.validate(1, "test").unwrap();

    let frac = prov.independent_fraction_excluding(0, &[RootId(101), RootId(202)], &groups_empty);
    assert!((frac - 2.0 / 6.0).abs() < 1e-15);
    let floor_frac = prov.independent_fraction_excluding(
        0,
        &[RootId(101), RootId(202), RootId(303)],
        &groups_empty,
    );
    assert_eq!(floor_frac, 0.15);

    // Case 2: SupportGroup joint exclusion
    let sg_offsets = [0u32, 2, 3, 3];
    let sg_roots = [RootId(101), RootId(202), RootId(303)];
    let sg_kinds = [
        SupportKind::Interpretation,
        SupportKind::Source,
        SupportKind::Hypothesis,
    ];
    let sg = SupportGroupView {
        group_count: 3,
        group_root_offsets: &sg_offsets,
        group_roots: &sg_roots,
        group_kinds: &sg_kinds,
    };
    sg.validate("test").unwrap();

    let g_offsets = [0u32, 2, 3];
    let g_ids = [GroupId(0), GroupId(1), GroupId(2)];
    let g_masses = [2.0, 3.0, 1.0];
    let g_prov = ProvenanceView {
        edge_count: 2,
        edge_prov_offsets: &g_offsets,
        prov_groups: &g_ids,
        prov_masses: &g_masses,
    };
    g_prov.validate(2, "test").unwrap();

    // Excluding 101 excludes entire GroupId(0) because 101 is in its required_roots
    let raw = g_prov.independent_support_fraction(0, &[RootId(101)], &sg);
    assert_eq!(raw, Some(0.6));
    let raw_none = g_prov.independent_support_fraction(0, &[RootId(101), RootId(303)], &sg);
    assert_eq!(raw_none, Some(0.0));
    assert_eq!(
        g_prov.independent_fraction_excluding(0, &[RootId(101), RootId(303)], &sg),
        0.15
    );
    // GroupId(2) has empty required_roots => unconditionally excluded
    assert_eq!(g_prov.independent_support_fraction(1, &[], &sg), Some(0.0));
}

#[cfg(feature = "schemars")]
#[test]
fn schemars_generation_for_core_types() {
    let id_schema = schemars::schema_for!(NodeId);
    let diag_schema = schemars::schema_for!(nexus_core::Diagnostics);
    let json = serde_json::to_string(&id_schema).unwrap();
    assert!(json.contains("integer"));
    assert!(
        serde_json::to_string(&diag_schema)
            .unwrap()
            .contains("truncated")
    );
}

// =========================================================================
// 4. Iteration 2 Remediation Regression Tests
// =========================================================================

#[test]
fn batch_cosines_f32_dim_zero_no_panic_all_arities() {
    for rows in [1usize, 5, 80] {
        let matrix = VectorMatrixView::new(&[], rows, 0).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for policy in [
            ParallelPolicy::Sequential,
            ParallelPolicy::GlobalRayon,
            ParallelPolicy::CustomPool(&pool),
        ] {
            for k in [1usize, 2, 3, 4, 5] {
                let queries: Vec<&[f32]> = vec![&[]; k];
                let mut out_storage = vec![vec![99.0_f64; rows]; k];
                let mut out_slices: Vec<&mut [f64]> =
                    out_storage.iter_mut().map(|v| v.as_mut_slice()).collect();
                batch_cosines_f32(
                    &matrix,
                    &queries,
                    &mut out_slices,
                    PrecisionMode::FastSimd,
                    1e-12,
                    policy,
                )
                .unwrap();
                for col in &out_storage {
                    assert!(
                        col.iter().all(|&v| v == 0.0),
                        "expected all zeros for rows={rows}, k={k}"
                    );
                }
            }
        }
    }

    // Also verify CustomPool 4-way matches Sequential 0-bit on non-empty dim.
    let rows = 80usize;
    let dim = 64usize;
    let mut data = Vec::with_capacity(rows * dim);
    for r in 0..rows {
        data.extend_from_slice(&gen_f32_vec(7100 + r as u64, dim));
    }
    let mat = VectorMatrixView::new(&data, rows, dim).unwrap();
    let q0 = gen_f32_vec(8100, dim);
    let q1 = gen_f32_vec(8101, dim);
    let q2 = gen_f32_vec(8102, dim);
    let q3 = gen_f32_vec(8103, dim);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();

    let mut out_seq = vec![vec![0.0_f64; rows]; 4];
    let mut out_seq_refs: Vec<&mut [f64]> = out_seq.iter_mut().map(|v| v.as_mut_slice()).collect();
    batch_cosines_f32(
        &mat,
        &[&q0, &q1, &q2, &q3],
        &mut out_seq_refs,
        PrecisionMode::FastSimd,
        1e-12,
        ParallelPolicy::Sequential,
    )
    .unwrap();

    let mut out_pool = vec![vec![0.0_f64; rows]; 4];
    let mut out_pool_refs: Vec<&mut [f64]> =
        out_pool.iter_mut().map(|v| v.as_mut_slice()).collect();
    batch_cosines_f32(
        &mat,
        &[&q0, &q1, &q2, &q3],
        &mut out_pool_refs,
        PrecisionMode::FastSimd,
        1e-12,
        ParallelPolicy::CustomPool(&pool),
    )
    .unwrap();

    for k in 0..4 {
        for r in 0..rows {
            assert_eq!(out_seq[k][r].to_bits(), out_pool[k][r].to_bits());
        }
    }
}

#[test]
fn csr_find_edge_duplicate_targets_matches_linear_scan() {
    // Row 0: edges [0->2 (0.9), 1->1 (0.7), 2->2 (0.5), 3->2 (0.1)]
    // Row 1: empty
    // Row 2: edges [4->0 (0.8), 5->0 (0.3)]
    let row_offsets = [0usize, 4, 4, 6];
    let targets = [
        NodeId(2),
        NodeId(1),
        NodeId(2),
        NodeId(2),
        NodeId(0),
        NodeId(0),
    ];
    let weights = [0.9, 0.7, 0.5, 0.1, 0.8, 0.3];
    let is_wh = [false; 6];
    let inbound = [1.1, 0.7, 1.5];
    let gains = [1.0; 3];
    // Sorted by (target, edge_idx) ascending per row:
    // Row 0: edge 1 (target 1), edge 0 (target 2), edge 2 (target 2), edge 3 (target 2)
    // Row 2: edge 4 (target 0), edge 5 (target 0)
    let sorted = [1u32, 0, 2, 3, 4, 5];

    let csr_indexed = CsrGraphView::new(CsrParts {
        node_count: 3,
        row_offsets: &row_offsets,
        targets: &targets,
        weights: &weights,
        is_static_wormhole: &is_wh,
        inbound: &inbound,
        max_inbound: 1.5,
        anchor_gains: &gains,
        target_sorted_indices: Some(&sorted),
        edge_gates: None,
    })
    .unwrap();
    csr_indexed.validate("dup_test").unwrap();

    let csr_linear = CsrGraphView::new(CsrParts {
        target_sorted_indices: None,
        ..*csr_indexed
    })
    .unwrap();

    let gates = [0.5, 1.0, 0.25, 0.1, 0.75, 0.2];
    let gated_indexed = csr_indexed.with_edge_gates(&gates).unwrap();
    let gated_linear = csr_linear.with_edge_gates(&gates).unwrap();

    for src in 0..4 {
        for dst in 0..4 {
            let s = NodeId(src);
            let d = NodeId(dst);
            assert_eq!(csr_indexed.find_edge(s, d), csr_linear.find_edge(s, d));
            assert_eq!(
                csr_indexed.edge_weight(s, d).to_bits(),
                csr_linear.edge_weight(s, d).to_bits()
            );
            assert_eq!(
                gated_indexed.edge_weight(s, d).to_bits(),
                gated_linear.edge_weight(s, d).to_bits()
            );
        }
    }
    assert_eq!(csr_indexed.find_edge(NodeId(0), NodeId(2)), Some(0));
    assert_eq!(csr_indexed.edge_weight(NodeId(0), NodeId(2)), 0.9);
    assert_eq!(gated_indexed.edge_weight(NodeId(0), NodeId(2)), 0.45);
    assert_eq!(csr_indexed.find_edge(NodeId(2), NodeId(0)), Some(4));
    assert_eq!(csr_indexed.edge_weight(NodeId(2), NodeId(0)), 0.8);
}

#[test]
fn csr_spmv_dual_zero_bit_parity_across_policies_n2500_nnz25000() {
    let n = 2500usize;
    let deg = 10usize;
    let nnz = n * deg;
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut targets = Vec::with_capacity(nnz);
    let mut weights = Vec::with_capacity(nnz);
    let is_wh = vec![false; nnz];
    let mut inbound = vec![0.0_f64; n];
    let gains = vec![1.0_f64; n];

    row_offsets.push(0);
    for u in 0..n {
        // Route one edge per node to hub 0 and distribute remaining edges non-uniformly
        // so addition associativity is strongly exercised.
        for d in 0..deg {
            let dst = if d == 0 { 0 } else { (u * 13 + d * 37) % n };
            let w = 0.01 + (((u + d) % 19) as f64) * 0.037;
            targets.push(NodeId(dst as u32));
            weights.push(w);
            inbound[dst] += w;
        }
        row_offsets.push(targets.len());
    }
    let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);
    let graph = CsrGraphView::new(CsrParts {
        node_count: n,
        row_offsets: &row_offsets,
        targets: &targets,
        weights: &weights,
        is_static_wormhole: &is_wh,
        inbound: &inbound,
        max_inbound,
        anchor_gains: &gains,
        target_sorted_indices: None,
        edge_gates: None,
    })
    .unwrap();
    graph.validate("spmv_2500").unwrap();

    let local: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 23) as f64) * 0.11).collect();
    let transfer: Vec<f64> = (0..n).map(|i| 0.25 + ((i % 17) as f64) * 0.07).collect();

    let pool2 = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool7 = rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build()
        .unwrap();

    let mut seq_l = vec![0.0; n];
    let mut seq_t = vec![0.0; n];
    csr_spmv_dual(
        &graph,
        &local,
        &transfer,
        &mut seq_l,
        &mut seq_t,
        ParallelPolicy::Sequential,
    )
    .unwrap();

    for policy in [
        ParallelPolicy::GlobalRayon,
        ParallelPolicy::CustomPool(&pool2),
        ParallelPolicy::CustomPool(&pool7),
    ] {
        let mut par_l = vec![0.0; n];
        let mut par_t = vec![0.0; n];
        csr_spmv_dual(&graph, &local, &transfer, &mut par_l, &mut par_t, policy).unwrap();
        for i in 0..n {
            assert_eq!(
                seq_l[i].to_bits(),
                par_l[i].to_bits(),
                "local bit diff at {i}"
            );
            assert_eq!(
                seq_t[i].to_bits(),
                par_t[i].to_bits(),
                "transfer bit diff at {i}"
            );
        }
    }
}

#[test]
fn nnls_qr_tie_breaking_and_1e160_scaled_norm() {
    let limits = ResourceLimits::default();
    let cancel = CancellationToken::none();
    let mut scratch = NnlsWorkspace::default();

    // 1. Deterministic (id, slot) tie-breaking under Householder QR (D = 2048 > count = 18)
    let dim = 2048usize;
    let bg_cols: Vec<Vec<f64>> = (0..16)
        .map(|k| {
            (0..dim)
                .map(|r| ((r as f64 * 0.19) + (k as f64 * 1.31)).cos())
                .collect()
        })
        .collect();

    for seed in 0..20usize {
        let base: Vec<f64> = (0..dim)
            .map(|r| ((r as f64 + (seed as f64) * 7.13) * 0.37).sin() + 0.5)
            .collect();
        let target: Vec<f64> = (0..dim).map(|r| base[r] + 0.2 * bg_cols[0][r]).collect();

        // Different IDs: id=100 at slot 0, id=10 at slot 17 -> id=10 (slot 17) must win.
        let mut cols_diff: Vec<(i64, &[f64])> = Vec::with_capacity(18);
        cols_diff.push((100, &base));
        for (idx, col) in bg_cols.iter().enumerate() {
            cols_diff.push(((5 + idx) as i64, col.as_slice()));
        }
        cols_diff.push((10, &base));

        let sol_diff =
            solve_nnls_with_ids_f64(&target, &cols_diff, &limits, &cancel, &mut scratch).unwrap();
        assert_eq!(
            sol_diff.weights[0], 0.0,
            "diff-id seed {seed}: higher id=100 at slot 0 stole weight"
        );
        assert!(
            sol_diff.weights[17] > 0.5,
            "diff-id seed {seed}: lower id=10 at slot 17 got {}",
            sol_diff.weights[17]
        );
        assert!(sol_diff.max_kkt_violation <= CHECK_TOL);

        // Same ID: id=10 at slot 0, id=10 at slot 17 -> earlier slot 0 must win.
        let mut cols_same = cols_diff;
        cols_same[0].0 = 10;
        let sol_same =
            solve_nnls_with_ids_f64(&target, &cols_same, &limits, &cancel, &mut scratch).unwrap();
        assert!(
            sol_same.weights[0] > 0.5,
            "same-id seed {seed}: earlier slot 0 got {}",
            sol_same.weights[0]
        );
        assert_eq!(
            sol_same.weights[17], 0.0,
            "same-id seed {seed}: later slot 17 stole weight"
        );
        assert!(sol_same.max_kkt_violation <= CHECK_TOL);
    }

    // 2. Finite 1e160 scaled norm fallback (without overflowing f64 norm_sq)
    let a = [1e160_f64, 0.0];
    let q = [1e160_f64, 1e160_f64];
    let sol = solve_nnls_f64(&q, &[&a], &limits, &cancel, &mut scratch).unwrap();
    assert!((sol.weights[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    assert!(sol.target_norm.is_finite());
    assert!(sol.residual_norm.is_finite());
    assert!((sol.residual_norm - 1e160).abs() / 1e160 < 1e-12);
    assert!(sol.max_kkt_violation <= CHECK_TOL);

    let b = [1e160_f64, 1e160_f64];
    let sol_exact = solve_nnls_f64(&q, &[&b], &limits, &cancel, &mut scratch).unwrap();
    assert!((sol_exact.weights[0] - 1.0).abs() < 1e-12);
    assert!(sol_exact.residual_norm.is_finite());
    assert!(sol_exact.max_kkt_violation <= CHECK_TOL);
}
