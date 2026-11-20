// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::time::Instant;

use nexus_core::PrecisionMode;
use nexus_core::kernel::{batch_dot_4_f64, dot_f64, norm_sq_f64, normalize_f64, sub_4_f64};
use nexus_core::nnls::solve_nnls_with_ids_f64;
use nexus_core::*;
use rayon::prelude::*;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalConeAgsConfig {
    /// Maximum number of visited sequence position pairs.
    pub pair_work_limit: Option<usize>,
    pub max_neighbors: usize,
    pub min_neighbors: usize,
    pub max_basis: usize,
    pub min_gain: f64,
    pub position_decay: f64,
    pub star_mix: f64,
    pub star_neighbors: usize,
    pub semantic_gate_enabled: bool,
    pub semantic_gate_peak: f64,
    pub semantic_gate_sigma: f64,
    pub semantic_gate_floor: f64,
    pub semantic_gate_hard_floor: f64,
    pub anchor_gain_base: f64,
    pub anchor_gain_scale: f64,
    pub anchor_gain_gamma: f64,
    pub anchor_gain_min: f64,
    pub anchor_gain_max: f64,
    pub reorth_loss_threshold: f64,
}

impl Default for LocalConeAgsConfig {
    fn default() -> Self {
        Self {
            pair_work_limit: Some(2_000_000),
            max_neighbors: 96,
            min_neighbors: 3,
            max_basis: 24,
            min_gain: 0.015,
            position_decay: 0.15,
            star_mix: 0.8,
            star_neighbors: 48,
            semantic_gate_enabled: true,
            semantic_gate_peak: 0.65,
            semantic_gate_sigma: 0.25,
            semantic_gate_floor: 0.35,
            semantic_gate_hard_floor: -1.0,
            anchor_gain_base: 0.75,
            anchor_gain_scale: 1.25,
            anchor_gain_gamma: 1.0,
            anchor_gain_min: 0.5,
            anchor_gain_max: 2.0,
            reorth_loss_threshold: 1e-3,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalEpaPriorMode {
    #[default]
    Support,
    Explained,
    Residual,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalEpaCandidateMode {
    #[default]
    Star,
    Anchor,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlobalEpaConfig {
    pub k_max: usize,
    pub k_sem: usize,
    pub min_gain: f64,
    pub knee_delta: f64,
    pub epsilon: f64,
    pub min_support: f64,
    pub support_exponent: f64,
    pub residual_exponent: f64,
    pub prior_mode: GlobalEpaPriorMode,
    pub candidate_mode: GlobalEpaCandidateMode,
    pub candidate_pool: usize,
}

impl Default for GlobalEpaConfig {
    fn default() -> Self {
        Self {
            k_max: 32,
            k_sem: 8,
            min_gain: 0.02,
            knee_delta: 0.0,
            epsilon: 0.1,
            min_support: 3.0,
            support_exponent: 0.65,
            residual_exponent: 0.35,
            prior_mode: GlobalEpaPriorMode::Support,
            candidate_mode: GlobalEpaCandidateMode::Star,
            candidate_pool: 512,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalConeAgsInput<'call> {
    pub tag_vectors: VectorMatrixView<'call, f32>,
    pub sequences: SequenceSliceView<'call>,
    pub explicit_edges: Option<&'call [(NodeId, NodeId, f64)]>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IntrinsicResidual {
    pub node: NodeId,
    pub residual_ratio: f64,
    pub neighbor_count: usize,
    pub basis_used: usize,
    pub mgs2_fallback: bool,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GeometryState {
    pub node_count: usize,
    pub anchor_gains: Vec<f64>,
    pub local_residuals: Vec<f64>,
    pub intrinsics: Vec<IntrinsicResidual>,
    #[serde(default)]
    pub node_supports: Vec<f64>,
    #[serde(default)]
    pub star_directions: Vec<f64>,
    #[serde(default)]
    pub dim: usize,
    #[serde(default)]
    pub computed_nodes: Vec<bool>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GlobalEpaInput<'call> {
    pub tag_vectors: VectorMatrixView<'call, f32>,
    pub geometry: &'call GeometryState,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpaBasis {
    pub dim: usize,
    pub k_sem: usize,
    pub basis_count: usize,
    pub mean: Vec<f64>,
    pub basis: Vec<f64>,
    pub coverage: Vec<f64>,
    pub elapsed_ms: f64,
}

impl EpaBasis {
    pub fn as_view(&self) -> EpaBasisView<'_> {
        EpaBasisView {
            dim: self.dim,
            k_sem: self.k_sem,
            mean: &self.mean,
            basis: &self.basis,
            basis_count: self.basis_count,
            coverage: &self.coverage,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct AgsCandidate {
    id: i64,
    prior: f64,
}

struct AgsTrace {
    picked: Vec<(usize, i64, f64)>,
    basis: Vec<Vec<f64>>,
    mgs2_fallback: bool,
}

#[inline]
fn semantic_weight(
    sim: f64,
    enabled: bool,
    peak: f64,
    sigma: f64,
    floor: f64,
    hard_floor: f64,
) -> f64 {
    if !enabled {
        1.0
    } else if sim < hard_floor {
        0.0
    } else if sim <= 0.0 {
        floor
    } else {
        (0.5 + 0.8 * (-(sim - peak).powi(2) / (2.0 * sigma * sigma)).exp()).max(floor)
    }
}

fn dot_neighbors(target: &[f64], neighbor_vecs: &[&[f64]], mode: PrecisionMode) -> Vec<f64> {
    let mut out = vec![0.0; neighbor_vecs.len()];
    let (groups, tail) = neighbor_vecs.as_chunks::<4>();
    let (sums, tail_sums) = out.as_chunks_mut::<4>();
    for (&vecs, dst) in groups.iter().zip(sums) {
        *dst = batch_dot_4_f64(target, vecs, mode);
    }
    for (dst, &v) in tail_sums.iter_mut().zip(tail) {
        *dst = dot_f64(target, v, mode);
    }
    out
}

fn star_direction(target: &[f64], mean: &[f64], neighbors: &[(&[f64], f64)], mix: f64) -> Vec<f64> {
    let weight_sum: f64 = neighbors.iter().map(|(_, w)| *w).sum();
    let mix = if weight_sum > 1e-12 { mix } else { 0.0 };
    let mut direction: Vec<f64> = target
        .iter()
        .zip(mean)
        .map(|(v, m)| (1.0 - mix) * (v - m))
        .collect();
    if mix > 0.0 {
        for (vector, weight) in neighbors {
            for ((out, v), m) in direction.iter_mut().zip(*vector).zip(mean) {
                *out += mix * weight / weight_sum * (v - m);
            }
        }
    }
    if !normalize_f64(&mut direction, 1e-12, PrecisionMode::StrictLeftFold) {
        direction = target.iter().zip(mean).map(|(v, m)| v - m).collect();
        if !normalize_f64(&mut direction, 1e-12, PrecisionMode::StrictLeftFold) {
            direction.fill(0.0);
        }
    }
    direction
}

/// 2-pass Modified Gram-Schmidt in strict left-fold order for rank-degenerate fallback.
fn orthogonal_residual_mgs2(vector: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut residual = vector.to_vec();
    if basis.is_empty() {
        return residual;
    }
    for _ in 0..2 {
        for axis in basis {
            let coeff = dot_f64(&residual, axis, PrecisionMode::StrictLeftFold);
            for (value, &q) in residual.iter_mut().zip(axis) {
                *value -= coeff * q;
            }
        }
    }
    residual
}

/// Fused projection-cache + selective 4-way CGS / 2-pass MGS2 reorthogonalization.
fn orthogonal_residual_with_projections(
    vector: &[f64],
    basis: &[Vec<f64>],
    first_pass_coeffs: Option<&[f64]>,
    initial_norm_sq: f64,
    est_norm_sq: f64,
    reorth_loss_threshold: f64,
    mode: PrecisionMode,
) -> (Vec<f64>, f64, bool) {
    if basis.is_empty() {
        return (vector.to_vec(), initial_norm_sq.max(0.0).sqrt(), false);
    }
    if first_pass_coeffs.is_none() || est_norm_sq <= reorth_loss_threshold * initial_norm_sq {
        let residual = orthogonal_residual_mgs2(vector, basis);
        let norm = norm_sq_f64(&residual, PrecisionMode::StrictLeftFold).sqrt();
        return (residual, norm, true);
    }

    let coeffs = first_pass_coeffs.unwrap();
    let mut residual = vector.to_vec();

    // Pass 1: 4-way fused subtraction using cached <vector, b_m>.
    let (axis_groups, axis_tail) = basis.as_chunks::<4>();
    let (coeff_groups, coeff_tail) = coeffs.as_chunks::<4>();
    for (axes, &coefficients) in axis_groups.iter().zip(coeff_groups) {
        sub_4_f64(
            &mut residual,
            coefficients,
            axes.each_ref().map(Vec::as_slice),
        );
    }
    for (axis, &c) in axis_tail.iter().zip(coeff_tail) {
        for (value, &q) in residual.iter_mut().zip(axis) {
            *value -= c * q;
        }
    }

    // Pass 2: 4-way blocked CGS reorthogonalization pass.
    for axes in axis_groups {
        let refs = axes.each_ref().map(Vec::as_slice);
        let c = batch_dot_4_f64(&residual, refs, mode);
        sub_4_f64(&mut residual, c, refs);
    }
    for axis in axis_tail {
        let c = dot_f64(&residual, axis, mode);
        for (value, &q) in residual.iter_mut().zip(axis) {
            *value -= c * q;
        }
    }

    let norm = norm_sq_f64(&residual, mode).sqrt();
    (residual, norm, false)
}

#[inline]
fn project_active_into(
    axis: &[f64],
    vectors: &[&[f64]],
    active: &[usize],
    row: &mut [f64],
    mode: PrecisionMode,
) {
    let (groups, tail) = active.as_chunks::<4>();
    for &[a, b, c, d] in groups {
        let sums = batch_dot_4_f64(axis, [vectors[a], vectors[b], vectors[c], vectors[d]], mode);
        row[a] = sums[0];
        row[b] = sums[1];
        row[c] = sums[2];
        row[d] = sums[3];
    }
    for &index in tail {
        row[index] = dot_f64(axis, vectors[index], mode);
    }
}

#[allow(clippy::too_many_arguments)]
fn direct_axes_core(
    target: Option<&[f64]>,
    candidates: &[AgsCandidate],
    vectors: &[&[f64]],
    precomputed_diagonal: Option<&[f64]>,
    precomputed_correlations: Option<&[f64]>,
    max_basis: usize,
    min_gain: f64,
    reorth_loss_threshold: f64,
    keep_basis: bool,
    mode: PrecisionMode,
) -> AgsTrace {
    let n = candidates.len();
    if n == 0 || max_basis == 0 {
        return AgsTrace {
            picked: Vec::new(),
            basis: Vec::new(),
            mgs2_fallback: false,
        };
    }

    let owned_diagonal: Vec<f64>;
    let diagonal: &[f64] = match precomputed_diagonal {
        Some(d) => d,
        None => {
            owned_diagonal = vectors.iter().map(|v| norm_sq_f64(v, mode)).collect();
            &owned_diagonal
        }
    };

    let mut active: Vec<usize> = (0..n)
        .filter(|&j| {
            candidates[j].prior.is_finite()
                && candidates[j].prior > 0.0
                && diagonal[j].is_finite()
                && diagonal[j] > 0.0
        })
        .collect();

    let owned_correlations: Vec<f64>;
    let correlations: &[f64] = match (target, precomputed_correlations) {
        (Some(_), Some(c)) => c,
        (Some(t), None) => {
            let mut row = vec![0.0; n];
            project_active_into(t, vectors, &active, &mut row, mode);
            owned_correlations = row;
            &owned_correlations
        }
        (None, _) => &[],
    };

    let limit = max_basis.min(n);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(limit);
    let mut proj_history: Vec<f64> = Vec::new();
    let mut row_buf: Vec<f64> = Vec::new();
    let mut proj_sq_sum = vec![0.0; n];
    let mut proj_target_sum = vec![0.0; n];
    let mut picked = Vec::with_capacity(limit);
    let mut mgs2_fallback = false;

    while picked.len() < limit {
        let mut best: Option<(usize, f64)> = None;
        let mut best_pos = 0;
        let mut write_pos = 0;
        let current_k = basis.len();

        for read_pos in 0..active.len() {
            let i = active[read_pos];
            let norm_sq = diagonal[i] - proj_sq_sum[i];
            let (orth, signed) = if norm_sq <= 1e-8 * diagonal[i] {
                let residual = orthogonal_residual_mgs2(vectors[i], &basis);
                let orth = norm_sq_f64(&residual, PrecisionMode::StrictLeftFold).sqrt();
                let signed = match target {
                    Some(t) if orth > 1e-6 && orth.is_finite() => {
                        dot_f64(t, &residual, PrecisionMode::StrictLeftFold) / orth
                    }
                    _ => orth,
                };
                (orth, signed)
            } else {
                let orth = norm_sq.sqrt();
                let signed = match target {
                    Some(_) => (correlations[i] - proj_target_sum[i]) / orth,
                    None => orth,
                };
                (orth, signed)
            };

            if !orth.is_finite() || orth <= 1e-6 || !signed.is_finite() {
                continue;
            }

            let cur_pos = write_pos;
            active[write_pos] = i;
            write_pos += 1;

            // V5.2 Explain-mode pre-filter: skip scoring for this round ONLY,
            // keeping candidate `i` in `active` for subsequent rounds.
            if target.is_some() && signed.abs() < min_gain {
                continue;
            }

            let score = signed.abs() * orth * candidates[i].prior;
            if best.as_ref().is_none_or(|&(j, old)| {
                score > old || (score == old && candidates[i].id < candidates[j].id)
            }) {
                best = Some((i, score));
                best_pos = cur_pos;
            }
        }
        active.truncate(write_pos);
        let Some((i, _)) = best else { break };
        active.remove(best_pos);

        let est_norm_sq = diagonal[i] - proj_sq_sum[i];
        let coeffs = (current_k > 0 && !proj_history.is_empty())
            .then(|| &proj_history[i * limit..i * limit + current_k]);
        let (mut axis, norm, used_mgs2) = orthogonal_residual_with_projections(
            vectors[i],
            &basis,
            coeffs,
            diagonal[i],
            est_norm_sq,
            reorth_loss_threshold,
            mode,
        );
        if used_mgs2 {
            mgs2_fallback = true;
        }
        if !norm.is_finite() || norm <= 1e-6 {
            continue;
        }
        for value in &mut axis {
            *value /= norm;
        }

        let coefficient = target.map(|t| dot_f64(t, &axis, mode));
        let gain = coefficient.map_or(norm, f64::abs);
        if gain < min_gain {
            if target.is_some() {
                continue;
            } else {
                break;
            }
        }

        picked.push((i, candidates[i].id, gain));
        if picked.len() < limit && !active.is_empty() {
            if proj_history.is_empty() {
                proj_history.resize(n * limit, 0.0);
                row_buf.resize(n, 0.0);
            }
            project_active_into(&axis, vectors, &active, &mut row_buf, mode);
            if let Some(c) = coefficient {
                for &j in &active {
                    let p = row_buf[j];
                    proj_history[j * limit + current_k] = p;
                    proj_sq_sum[j] += p * p;
                    proj_target_sum[j] += p * c;
                }
            } else {
                for &j in &active {
                    let p = row_buf[j];
                    proj_history[j * limit + current_k] = p;
                    proj_sq_sum[j] += p * p;
                }
            }
        }
        basis.push(axis);
    }

    if !keep_basis {
        basis.clear();
    }
    AgsTrace {
        picked,
        basis,
        mgs2_fallback,
    }
}

fn effective_k(coverage: &[f64], k_sem: usize, epsilon: f64, delta: f64) -> usize {
    if coverage.is_empty() {
        return 0;
    }
    let minimum = k_sem.min(coverage.len()).max(1);
    for k in minimum..=coverage.len() {
        let previous = if k > 1 { coverage[k - 2] } else { 0.0 };
        if 1.0 - coverage[k - 1] <= epsilon || coverage[k - 1] - previous < delta {
            return k;
        }
    }
    coverage.len()
}

struct UnitVectors {
    vecs: Vec<Option<Vec<f64>>>,
    diag_norms: Vec<f64>,
    valid_count: usize,
}

fn build_unit_vectors(
    tag_vectors: &VectorMatrixView<'_, f32>,
    stage: &'static str,
) -> VcpResult<UnitVectors> {
    let rows = tag_vectors.rows;
    let dim = tag_vectors.dim;
    if tag_vectors.data.iter().any(|v| !v.is_finite()) {
        return Err(VcpError::InvalidInput {
            stage,
            detail: "tag_vectors contains non-finite values".into(),
        });
    }
    let mut vecs = Vec::with_capacity(rows);
    let mut diag_norms = Vec::with_capacity(rows);
    let mut valid_count = 0usize;
    for r in 0..rows {
        let slice = tag_vectors.row(r).unwrap_or(&[]);
        if dim == 0 || slice.is_empty() {
            vecs.push(None);
            diag_norms.push(0.0);
            continue;
        }
        let mut vec_f64: Vec<f64> = slice.iter().map(|&v| f64::from(v)).collect();
        if normalize_f64(&mut vec_f64, 1e-12, PrecisionMode::StrictLeftFold) {
            let diag = norm_sq_f64(&vec_f64, PrecisionMode::StrictLeftFold);
            vecs.push(Some(vec_f64));
            diag_norms.push(diag);
            valid_count += 1;
        } else {
            vecs.push(None);
            diag_norms.push(0.0);
        }
    }
    Ok(UnitVectors {
        vecs,
        diag_norms,
        valid_count,
    })
}

pub fn local_cone<'call>(
    input: LocalConeAgsInput<'call>,
    config: &LocalConeAgsConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<GeometryState> {
    const STAGE: &str = "local_cone";
    let start_time = Instant::now();
    ctx.check_cancelled(STAGE)?;
    if config.max_neighbors == 0
        || config.min_neighbors == 0
        || config.min_neighbors > config.max_neighbors
        || config.max_basis == 0
        || config.star_neighbors == 0
    {
        return Err(crate::observation::invalid(
            STAGE,
            "min_neighbors and max_neighbors must be ordered and positive; max_basis and star_neighbors must be positive",
        ));
    }
    for (name, value, min, max) in [
        ("min_gain", config.min_gain, 0.0, 1.0),
        ("position_decay", config.position_decay, 0.0, f64::INFINITY),
        ("star_mix", config.star_mix, 0.0, 1.0),
        (
            "semantic_gate_sigma",
            config.semantic_gate_sigma,
            0.02,
            f64::INFINITY,
        ),
        ("semantic_gate_peak", config.semantic_gate_peak, -1.0, 1.0),
        (
            "semantic_gate_hard_floor",
            config.semantic_gate_hard_floor,
            -1.0,
            1.0,
        ),
        ("semantic_gate_floor", config.semantic_gate_floor, 0.0, 1.0),
        (
            "anchor_gain_min",
            config.anchor_gain_min,
            0.0,
            f64::INFINITY,
        ),
        (
            "anchor_gain_max",
            config.anchor_gain_max,
            config.anchor_gain_min,
            f64::INFINITY,
        ),
        (
            "anchor_gain_base",
            config.anchor_gain_base,
            0.0,
            f64::INFINITY,
        ),
        (
            "anchor_gain_gamma",
            config.anchor_gain_gamma,
            0.0,
            f64::INFINITY,
        ),
        (
            "anchor_gain_scale",
            config.anchor_gain_scale,
            0.0,
            f64::INFINITY,
        ),
        (
            "reorth_loss_threshold",
            config.reorth_loss_threshold,
            0.0,
            1.0,
        ),
    ] {
        crate::parameter(STAGE, name, value, min, max)?;
    }

    input.tag_vectors.validate(STAGE)?;
    let node_count = input.tag_vectors.rows;
    let dim = input.tag_vectors.dim;
    input.sequences.validate(node_count, STAGE)?;

    if let Some(edges) = input.explicit_edges {
        for &(src, dst, weight) in edges {
            if src.index() >= node_count || dst.index() >= node_count {
                return Err(VcpError::InvalidInput {
                    stage: STAGE,
                    detail: "explicit_edges contains out-of-bounds NodeId".into(),
                });
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(VcpError::InvalidInput {
                    stage: STAGE,
                    detail: "explicit_edges weight must be finite and non-negative".into(),
                });
            }
        }
    }

    if node_count == 0 {
        return Ok(GeometryState {
            node_count: 0,
            anchor_gains: Vec::new(),
            local_residuals: Vec::new(),
            intrinsics: Vec::new(),
            node_supports: Vec::new(),
            star_directions: Vec::new(),
            dim,
            computed_nodes: Vec::new(),
            elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let UnitVectors {
        vecs: unit_vecs,
        diag_norms,
        valid_count,
    } = build_unit_vectors(&input.tag_vectors, STAGE)?;

    let mut adjacency: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); node_count];
    if let Some(edges) = input.explicit_edges {
        for &(src, dst, weight) in edges {
            if src == dst || weight <= 0.0 {
                continue;
            }
            *adjacency[src.index()].entry(dst.index()).or_default() += weight;
        }
    }

    let position_decay = config.position_decay;
    if config.pair_work_limit == Some(0) {
        return Err(crate::observation::invalid(
            STAGE,
            "pair_work_limit must be positive",
        ));
    }
    let mut pair_work = 0usize;
    let mut seq_buf: Vec<(u32, NodeId)> = Vec::new();

    for seq_idx in 0..input.sequences.sequence_count {
        ctx.check_cancelled(STAGE)?;
        let (tags, positions) = input.sequences.sequence(SequenceId(seq_idx as u32));
        if tags.len() < 2 {
            continue;
        }
        seq_buf.clear();
        seq_buf.extend(positions.iter().copied().zip(tags.iter().copied()));
        let is_sorted = seq_buf.windows(2).all(|w| w[0].0 <= w[1].0);
        if !is_sorted {
            seq_buf.sort_by_key(|step| step.0);
        }

        for i in 0..seq_buf.len() {
            let (p, a) = seq_buf[i];
            for &(q, b) in &seq_buf[i + 1..] {
                if pair_work.is_multiple_of(1024) {
                    ctx.check_cancelled(STAGE)?;
                }
                pair_work += 1;
                if let Some(pair_limit) = config.pair_work_limit.filter(|&limit| pair_work > limit)
                {
                    return Err(VcpError::ResourceLimitExceeded {
                        stage: STAGE,
                        limit_name: "pair_work_limit",
                        actual: pair_work,
                        limit: pair_limit,
                    });
                }
                if a == b {
                    continue;
                }
                let delta = if p > 0 && q > 0 {
                    p.abs_diff(q).max(1) as f64
                } else {
                    1.0
                };
                let weight = (-position_decay * (delta - 1.0)).exp();
                let ai = a.index();
                let bi = b.index();
                *adjacency[ai].entry(bi).or_default() += weight;
                *adjacency[bi].entry(ai).or_default() += weight;
            }
        }
    }

    let max_keep = config.max_neighbors.max(config.star_neighbors);
    let cmp_edge = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));

    let mut node_supports = Vec::with_capacity(node_count);
    let mut row_offsets = Vec::with_capacity(node_count + 1);
    let mut valid_edges: Vec<(usize, f64)> = Vec::new();
    row_offsets.push(0);

    for (i, neighbors) in adjacency.into_iter().enumerate() {
        let support: f64 = neighbors.values().sum();
        node_supports.push(support);
        if unit_vecs[i].is_some() {
            let mut filtered: Vec<(usize, f64)> = neighbors
                .into_iter()
                .filter(|&(nid, w)| w > 0.0 && unit_vecs[nid].is_some())
                .collect();
            if filtered.len() > max_keep {
                filtered.select_nth_unstable_by(max_keep - 1, cmp_edge);
                filtered.truncate(max_keep);
            }
            filtered.sort_unstable_by(cmp_edge);
            valid_edges.extend(filtered);
        }
        row_offsets.push(valid_edges.len());
    }

    // Compute global mean of valid unit vectors for star_direction cache.
    let mut mean = vec![0.0; dim];
    if dim > 0 && valid_count > 0 {
        for v in unit_vecs.iter().flatten() {
            for (m, &x) in mean.iter_mut().zip(v) {
                *m += x;
            }
        }
        for m in &mut mean {
            *m /= valid_count as f64;
        }
    }

    let max_neighbors = config.max_neighbors;
    let min_neighbors = config.min_neighbors;
    let max_basis = config.max_basis;
    let sigma = config.semantic_gate_sigma;
    let anchor_low = config.anchor_gain_min;
    let anchor_high = config.anchor_gain_max;

    let compute_node = |i: usize,
                        local_scratch: &mut nexus_core::NnlsWorkspace|
     -> VcpResult<(IntrinsicResidual, f64, bool, Vec<f64>)> {
        ctx.check_cancelled(STAGE)?;
        let node_id = NodeId(i as u32);
        let Some(target) = unit_vecs[i].as_deref() else {
            return Ok((
                IntrinsicResidual {
                    node: node_id,
                    residual_ratio: 1.0,
                    neighbor_count: 0,
                    basis_used: 0,
                    mgs2_fallback: false,
                },
                1.0,
                false,
                vec![0.0; dim],
            ));
        };

        let target_norm_sq = norm_sq_f64(target, PrecisionMode::StrictLeftFold);
        if !target_norm_sq.is_finite() || target_norm_sq <= 1e-24 {
            return Ok((
                IntrinsicResidual {
                    node: node_id,
                    residual_ratio: 1.0,
                    neighbor_count: 0,
                    basis_used: 0,
                    mgs2_fallback: false,
                },
                1.0,
                false,
                vec![0.0; dim],
            ));
        }

        let row_start = row_offsets[i];
        let row_end = row_offsets[i + 1];

        // Precompute star direction for global_epa parity.
        let star_end = row_end.min(row_start + config.star_neighbors);
        let star_slice = &valid_edges[row_start..star_end];
        let star_vecs: Vec<&[f64]> = star_slice
            .iter()
            .map(|&(nid, _)| unit_vecs[nid].as_deref().unwrap())
            .collect();
        let star_dots = dot_neighbors(target, &star_vecs, PrecisionMode::FastSimd);
        let weighted_star: Vec<(&[f64], f64)> = star_slice
            .iter()
            .zip(star_vecs.iter().zip(star_dots))
            .filter_map(|(&(_, w), (&v, raw_dot))| {
                let sim = raw_dot.clamp(-1.0, 1.0);
                let weight = w.ln_1p()
                    * semantic_weight(
                        sim,
                        config.semantic_gate_enabled,
                        config.semantic_gate_peak,
                        sigma,
                        config.semantic_gate_floor,
                        config.semantic_gate_hard_floor,
                    );
                (weight > 0.0).then_some((v, weight))
            })
            .collect();
        let star_dir = star_direction(target, &mean, &weighted_star, config.star_mix);

        // Local Cone-AGS candidate preparation.
        let local_end = row_end.min(row_start + max_neighbors);
        let neighbors = &valid_edges[row_start..local_end];
        let neighbor_vecs: Vec<&[f64]> = neighbors
            .iter()
            .map(|&(nid, _)| unit_vecs[nid].as_deref().unwrap())
            .collect();
        let raw_dots = dot_neighbors(target, &neighbor_vecs, PrecisionMode::FastSimd);

        let mut candidates = Vec::with_capacity(neighbors.len());
        let mut refs = Vec::with_capacity(neighbors.len());
        let mut diagonal = Vec::with_capacity(neighbors.len());
        let mut correlations = Vec::with_capacity(neighbors.len());

        for (&(nid, w), (&v, raw_dot)) in neighbors.iter().zip(neighbor_vecs.iter().zip(raw_dots)) {
            let sim = raw_dot.clamp(-1.0, 1.0);
            let semantic = semantic_weight(
                sim,
                config.semantic_gate_enabled,
                config.semantic_gate_peak,
                sigma,
                config.semantic_gate_floor,
                config.semantic_gate_hard_floor,
            );
            if semantic > 0.0 {
                candidates.push(AgsCandidate {
                    id: nid as i64,
                    prior: w.ln_1p() * semantic,
                });
                refs.push(v);
                diagonal.push(diag_norms[nid]);
                correlations.push(raw_dot);
            }
        }

        let candidate_count = candidates.len();
        if candidate_count < min_neighbors {
            return Ok((
                IntrinsicResidual {
                    node: node_id,
                    residual_ratio: 1.0,
                    neighbor_count: candidate_count,
                    basis_used: 0,
                    mgs2_fallback: false,
                },
                1.0,
                false,
                star_dir,
            ));
        }

        let trace = direct_axes_core(
            Some(target),
            &candidates,
            &refs,
            Some(&diagonal),
            Some(&correlations),
            max_basis,
            config.min_gain,
            config.reorth_loss_threshold,
            false,
            PrecisionMode::FastSimd,
        );

        let selected: Vec<(i64, &[f64])> = trace
            .picked
            .iter()
            .map(|&(cand_idx, id, _)| (id, refs[cand_idx]))
            .collect();

        let basis_used = selected.len();
        let r_t = if selected.is_empty() {
            1.0
        } else {
            let cone =
                solve_nnls_with_ids_f64(target, &selected, max_basis, &ctx.cancel, local_scratch)?;
            if cone.target_norm > 0.0 && cone.target_norm.is_finite() {
                (cone.residual_norm / cone.target_norm).clamp(0.0, 1.0)
            } else {
                1.0
            }
        };

        let anchor_gain = (config.anchor_gain_base
            + config.anchor_gain_scale * r_t.powf(config.anchor_gain_gamma))
        .clamp(anchor_low, anchor_high);

        Ok((
            IntrinsicResidual {
                node: node_id,
                residual_ratio: r_t,
                neighbor_count: candidate_count,
                basis_used,
                mgs2_fallback: trace.mgs2_fallback,
            },
            anchor_gain,
            true,
            star_dir,
        ))
    };

    let per_node: Vec<(IntrinsicResidual, f64, bool, Vec<f64>)> = if ctx.parallel.is_parallel() {
        ctx.parallel.install(|| {
            (0..node_count)
                .into_par_iter()
                .map_init(nexus_core::NnlsWorkspace::default, |local_scratch, i| {
                    compute_node(i, local_scratch)
                })
                .collect::<VcpResult<Vec<_>>>()
        })?
    } else {
        let mut out = Vec::with_capacity(node_count);
        let mut workspace = nexus_core::NnlsWorkspace::default();
        for i in 0..node_count {
            out.push(compute_node(i, &mut workspace)?);
        }
        out
    };

    let mut anchor_gains = Vec::with_capacity(node_count);
    let mut local_residuals = Vec::with_capacity(node_count);
    let mut intrinsics = Vec::with_capacity(node_count);
    let mut computed_nodes = Vec::with_capacity(node_count);
    let mut star_directions = Vec::with_capacity(node_count * dim);

    for (intrinsic, gain, computed, star_dir) in per_node {
        local_residuals.push(intrinsic.residual_ratio);
        intrinsics.push(intrinsic);
        anchor_gains.push(gain);
        computed_nodes.push(computed);
        star_directions.extend_from_slice(&star_dir);
    }

    Ok(GeometryState {
        node_count,
        anchor_gains,
        local_residuals,
        intrinsics,
        node_supports,
        star_directions,
        dim,
        computed_nodes,
        elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
    })
}

pub fn global_epa<'call>(
    input: GlobalEpaInput<'call>,
    config: &GlobalEpaConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<EpaBasis> {
    const STAGE: &str = "global_epa";
    let start_time = Instant::now();
    ctx.check_cancelled(STAGE)?;
    if config.candidate_pool == 0 || config.k_max == 0 {
        return Err(crate::observation::invalid(
            STAGE,
            "candidate_pool and k_max must be positive",
        ));
    }
    for (name, value, min) in [
        ("min_gain", config.min_gain, 0.0),
        ("knee_delta", config.knee_delta, 0.0),
        ("epsilon", config.epsilon, 0.0),
        ("min_support", config.min_support, 0.0),
        ("support_exponent", config.support_exponent, 0.0),
        ("residual_exponent", config.residual_exponent, 0.0),
    ] {
        crate::parameter(STAGE, name, value, min, f64::INFINITY)?;
    }

    input.tag_vectors.validate(STAGE)?;
    let node_count = input.tag_vectors.rows;
    let dim = input.tag_vectors.dim;
    if input.geometry.node_count != node_count {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: node_count,
            actual: input.geometry.node_count,
        });
    }

    if node_count == 0 || dim == 0 {
        return Ok(EpaBasis {
            dim,
            k_sem: config.k_sem,
            basis_count: 0,
            mean: vec![0.0; dim],
            basis: Vec::new(),
            coverage: Vec::new(),
            elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let UnitVectors {
        vecs: unit_vecs,
        diag_norms: _,
        valid_count,
    } = build_unit_vectors(&input.tag_vectors, STAGE)?;
    if valid_count == 0 {
        return Ok(EpaBasis {
            dim,
            k_sem: config.k_sem,
            basis_count: 0,
            mean: vec![0.0; dim],
            basis: Vec::new(),
            coverage: Vec::new(),
            elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let mut mean = vec![0.0; dim];
    for v in unit_vecs.iter().flatten() {
        for (m, &x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    for m in &mut mean {
        *m /= valid_count as f64;
    }

    let has_supports = input.geometry.node_supports.len() == node_count;
    let has_computed = input.geometry.computed_nodes.len() == node_count;
    let has_stars =
        input.geometry.dim == dim && input.geometry.star_directions.len() == node_count * dim;

    let mut pool: Vec<AgsCandidate> = input
        .geometry
        .intrinsics
        .iter()
        .filter_map(|r| {
            let idx = r.node.index();
            if idx >= node_count || unit_vecs[idx].is_none() {
                return None;
            }
            let is_computed = if has_computed {
                input.geometry.computed_nodes[idx]
            } else {
                r.basis_used > 0 || r.neighbor_count > 0
            };
            if !is_computed {
                return None;
            }
            let support = if has_supports {
                input.geometry.node_supports[idx]
            } else {
                r.neighbor_count as f64
            };
            if !support.is_finite() || support < config.min_support {
                return None;
            }
            let res = r.residual_ratio.clamp(0.0, 1.0);
            let factor = match config.prior_mode {
                GlobalEpaPriorMode::Explained => {
                    (1.0 - res * res).max(0.0).powf(config.residual_exponent)
                }
                GlobalEpaPriorMode::Residual => res.powf(config.residual_exponent),
                GlobalEpaPriorMode::Support => 1.0,
            };
            let prior = support.powf(config.support_exponent) * factor;
            (prior.is_finite() && prior > 0.0).then_some(AgsCandidate {
                id: idx as i64,
                prior,
            })
        })
        .collect();

    if pool.is_empty() {
        return Ok(EpaBasis {
            dim,
            k_sem: config.k_sem,
            basis_count: 0,
            mean,
            basis: Vec::new(),
            coverage: Vec::new(),
            elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        });
    }

    pool.sort_by(|a, b| b.prior.total_cmp(&a.prior).then(a.id.cmp(&b.id)));
    pool.truncate(config.candidate_pool);

    let centered: Vec<Vec<f64>> = pool
        .iter()
        .map(|c| {
            let idx = c.id as usize;
            if config.candidate_mode == GlobalEpaCandidateMode::Star && has_stars {
                let s = &input.geometry.star_directions[idx * dim..(idx + 1) * dim];
                let norm_sq = norm_sq_f64(s, PrecisionMode::StrictLeftFold);
                if norm_sq.is_finite() && norm_sq > 1e-24 {
                    return s.to_vec();
                }
            }
            let target = unit_vecs[idx].as_deref().unwrap();
            let mut dir: Vec<f64> = target.iter().zip(&mean).map(|(v, m)| v - m).collect();
            if !normalize_f64(&mut dir, 1e-12, PrecisionMode::StrictLeftFold) {
                dir.fill(0.0);
            }
            dir
        })
        .collect();

    let refs: Vec<&[f64]> = centered.iter().map(Vec::as_slice).collect();
    let mut trace = direct_axes_core(
        None,
        &pool,
        &refs,
        None,
        None,
        config.k_max,
        config.min_gain,
        1e-3,
        true,
        PrecisionMode::FastSimd,
    );

    ctx.check_cancelled(STAGE)?;

    let basis_refs: Vec<&[f64]> = trace.basis.iter().map(Vec::as_slice).collect();
    let compute_chunk = |chunk: &[Option<Vec<f64>>]| -> (f64, Vec<f64>) {
        let mut energy = 0.0;
        let mut captured = vec![0.0; basis_refs.len()];
        let mut x = vec![0.0; dim];
        for v in chunk.iter().flatten() {
            for ((out, &val), &m) in x.iter_mut().zip(v).zip(&mean) {
                *out = val - m;
            }
            energy += norm_sq_f64(&x, PrecisionMode::FastSimd);
            let (axis_groups, axis_tail) = basis_refs.as_chunks::<4>();
            let (cap_groups, cap_tail) = captured.as_chunks_mut::<4>();
            for (&axes, cap) in axis_groups.iter().zip(cap_groups) {
                let sums = batch_dot_4_f64(&x, axes, PrecisionMode::FastSimd);
                cap[0] += sums[0].powi(2);
                cap[1] += sums[1].powi(2);
                cap[2] += sums[2].powi(2);
                cap[3] += sums[3].powi(2);
            }
            for (c, axis) in cap_tail.iter_mut().zip(axis_tail) {
                *c += dot_f64(&x, axis, PrecisionMode::FastSimd).powi(2);
            }
        }
        (energy, captured)
    };

    let partial: Vec<(f64, Vec<f64>)> = if ctx.parallel.is_parallel() {
        ctx.parallel
            .install(|| unit_vecs.par_chunks(128).map(compute_chunk).collect())
    } else {
        unit_vecs.chunks(128).map(compute_chunk).collect()
    };

    let mut total_energy = 0.0;
    let mut energies = vec![0.0; trace.basis.len()];
    for (energy, values) in partial {
        total_energy += energy;
        for (e, v) in energies.iter_mut().zip(values) {
            *e += v;
        }
    }

    let mut cumulative = 0.0;
    let mut coverage: Vec<f64> = energies
        .iter()
        .map(|e| {
            cumulative += e;
            if total_energy > 1e-12 {
                (cumulative / total_energy).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();

    let k = effective_k(&coverage, config.k_sem, config.epsilon, config.knee_delta);
    trace.basis.truncate(k);
    coverage.truncate(k);

    let mut flat_basis = Vec::with_capacity(k * dim);
    for axis in &trace.basis {
        flat_basis.extend_from_slice(axis);
    }

    Ok(EpaBasis {
        dim,
        k_sem: config.k_sem,
        basis_count: k,
        mean,
        basis: flat_basis,
        coverage,
        elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
    })
}
