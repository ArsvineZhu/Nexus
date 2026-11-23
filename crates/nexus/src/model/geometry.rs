// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]

use super::{ConceptView, SourceBook};
use crate::{Result, config, parameter};
use nexus_math::kernel::PrecisionMode;
use nexus_math::kernel::{batch_dot_4_f64, dot_f64, norm_sq_f64, normalize_f64, sub_4_f64};
use nexus_math::nnls::solve_nnls_with_ids_f64;
use nexus_math::*;
use rayon::prelude::*;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalConeConfig {
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
}

impl Default for LocalConeConfig {
    fn default() -> Self {
        Self {
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
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalPriorMode {
    #[default]
    Support,
    Explained,
    Residual,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalCandidateMode {
    #[default]
    Star,
    Anchor,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlobalSpanConfig {
    pub k_max: usize,
    pub k_sem: usize,
    pub min_gain: f64,
    pub knee_delta: f64,
    pub epsilon: f64,
    pub min_support: f64,
    pub support_exponent: f64,
    pub residual_exponent: f64,
    pub prior_mode: GlobalPriorMode,
    pub candidate_mode: GlobalCandidateMode,
    pub candidate_pool: usize,
}

impl Default for GlobalSpanConfig {
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
            prior_mode: GlobalPriorMode::Support,
            candidate_mode: GlobalCandidateMode::Star,
            candidate_pool: 512,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalResidual {
    pub ratio: f64,
    pub energy: f64,
}

pub struct LocalGeometry {
    pub(super) concepts: std::sync::Arc<[u64]>,
    pub(super) anchor_gains: Vec<f64>,
    pub(crate) residuals: Vec<Option<LocalResidual>>,
    supports: Vec<f64>,
    stars: Vec<f64>,
}

impl LocalGeometry {
    pub fn residual(&self, id: u64) -> Option<LocalResidual> {
        let slot = self.concepts.iter().position(|&concept| concept == id)?;
        self.residuals[slot]
    }

    pub fn anchor_gain(&self, id: u64) -> Option<f64> {
        let slot = self.concepts.iter().position(|&concept| concept == id)?;
        Some(self.anchor_gains[slot])
    }

    pub(crate) fn check(&self, concepts: ConceptView<'_>) -> Result<()> {
        if !std::sync::Arc::ptr_eq(&self.concepts, concepts.identity()) {
            return Err(crate::Error::Association {
                message: "local geometry belongs to a different concept space".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct LocalConeOptions {
    pub config: LocalConeConfig,
    pub run: RunOptions,
}

#[derive(Clone, Default)]
pub struct GlobalSpanOptions {
    pub config: GlobalSpanConfig,
    pub run: RunOptions,
}

pub struct GlobalSpan {
    pub(crate) data: SpanData,
    mean: Vec<f32>,
    representatives: Vec<u64>,
}

impl GlobalSpan {
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }
    pub fn axis_count(&self) -> usize {
        self.data.coverage.len()
    }
    pub fn representatives(&self) -> &[u64] {
        &self.representatives
    }
}
pub(crate) struct SpanData {
    pub(crate) k_sem: usize,
    pub(crate) mean: Vec<f64>,
    pub(crate) axes: Vec<f64>,
    pub(crate) coverage: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct AgsCandidate {
    id: i64,
    prior: f64,
}

struct AgsTrace {
    picked: Vec<(usize, i64, f64)>,
    basis: Vec<Vec<f64>>,
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
        let (mut axis, norm, _) = orthogonal_residual_with_projections(
            vectors[i],
            &basis,
            coeffs,
            diagonal[i],
            est_norm_sq,
            reorth_loss_threshold,
            mode,
        );
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
    AgsTrace { picked, basis }
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

pub fn local_cone(
    concepts: ConceptView<'_>,
    sources: &[super::Source],
    relations: &[super::Relation],
    options: &LocalConeOptions,
) -> Result<LocalGeometry> {
    let book = SourceBook::new(concepts, sources, relations, &options.run)?;
    local_geometry(concepts, &book, &options.config, &options.run)
}

pub(super) fn local_geometry(
    concepts: ConceptView<'_>,
    book: &SourceBook,
    cfg: &LocalConeConfig,
    run: &RunOptions,
) -> Result<LocalGeometry> {
    const STAGE: &str = "local_cone";
    run.check_cancelled()?;
    if cfg.max_neighbors == 0
        || cfg.min_neighbors == 0
        || cfg.min_neighbors > cfg.max_neighbors
        || cfg.max_basis == 0
        || cfg.star_neighbors == 0
    {
        return Err(config(
            "local.neighbors",
            "neighbor and basis counts must be positive and min_neighbors must not exceed max_neighbors",
        ));
    }
    for (name, value, min, max) in [
        ("min_gain", cfg.min_gain, 0.0, f64::INFINITY),
        ("position_decay", cfg.position_decay, 0.0, f64::INFINITY),
        ("star_mix", cfg.star_mix, 0.0, 1.0),
        ("semantic_gate_peak", cfg.semantic_gate_peak, -1.0, 1.0),
        (
            "semantic_gate_hard_floor",
            cfg.semantic_gate_hard_floor,
            -1.0,
            1.0,
        ),
        ("semantic_gate_floor", cfg.semantic_gate_floor, 0.0, 1.0),
        ("anchor_gain_min", cfg.anchor_gain_min, 0.0, f64::INFINITY),
        (
            "anchor_gain_max",
            cfg.anchor_gain_max,
            cfg.anchor_gain_min,
            f64::INFINITY,
        ),
        ("anchor_gain_base", cfg.anchor_gain_base, 0.0, f64::INFINITY),
        (
            "anchor_gain_scale",
            cfg.anchor_gain_scale,
            0.0,
            f64::INFINITY,
        ),
    ] {
        parameter(STAGE, name, value, min, max)?;
    }
    for (field, value) in [
        ("semantic_gate_sigma", cfg.semantic_gate_sigma),
        ("anchor_gain_gamma", cfg.anchor_gain_gamma),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(config(field, "must be positive and finite"));
        }
    }

    let node_count = concepts.ids().len();
    let dim = concepts.dimension();
    let unit_vecs = geometry_vectors(concepts);
    let diag_norms: Vec<_> = unit_vecs
        .iter()
        .map(|row| norm_sq_f64(row, PrecisionMode::StrictLeftFold))
        .collect();
    let adjacency = book.local_neighbors(node_count, cfg.position_decay, run)?;
    let config = cfg;
    let max_keep = config.max_neighbors.max(config.star_neighbors);
    let cmp_edge = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));

    let mut node_supports = Vec::with_capacity(node_count);
    let mut row_offsets = Vec::with_capacity(node_count + 1);
    let mut valid_edges: Vec<(usize, f64)> = Vec::new();
    row_offsets.push(0);

    for neighbors in adjacency {
        let support: f64 = neighbors.values().sum();
        node_supports.push(support);
        let mut filtered: Vec<_> = neighbors.into_iter().filter(|&(_, w)| w > 0.0).collect();
        if filtered.len() > max_keep {
            filtered.select_nth_unstable_by(max_keep - 1, cmp_edge);
            filtered.truncate(max_keep);
        }
        filtered.sort_unstable_by(cmp_edge);
        valid_edges.extend(filtered);
        row_offsets.push(valid_edges.len());
    }

    let mean = mean_vector(&unit_vecs, dim);
    let max_neighbors = config.max_neighbors;
    let min_neighbors = config.min_neighbors;
    let max_basis = config.max_basis;
    let sigma = config.semantic_gate_sigma;
    let anchor_low = config.anchor_gain_min;
    let anchor_high = config.anchor_gain_max;
    let compute_node = |i: usize,
                        local_scratch: &mut nexus_math::NnlsWorkspace|
     -> Result<(Option<LocalResidual>, f64, Vec<f64>)> {
        run.check_cancelled()?;
        let target = &unit_vecs[i];
        let row_start = row_offsets[i];
        let row_end = row_offsets[i + 1];

        // Precompute star direction for global_epa parity.
        let star_end = row_end.min(row_start.saturating_add(config.star_neighbors));
        let star_slice = &valid_edges[row_start..star_end];
        let star_vecs: Vec<&[f64]> = star_slice
            .iter()
            .map(|&(nid, _)| unit_vecs[nid].as_slice())
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
        let local_end = row_end.min(row_start.saturating_add(max_neighbors));
        let neighbors = &valid_edges[row_start..local_end];
        let neighbor_vecs: Vec<&[f64]> = neighbors
            .iter()
            .map(|&(nid, _)| unit_vecs[nid].as_slice())
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
            return Ok((None, 1.0, star_dir));
        }

        let trace = direct_axes_core(
            Some(target),
            &candidates,
            &refs,
            Some(&diagonal),
            Some(&correlations),
            max_basis,
            config.min_gain,
            1e-3,
            false,
            PrecisionMode::FastSimd,
        );

        let selected: Vec<(i64, &[f64])> = trace
            .picked
            .iter()
            .map(|&(cand_idx, id, _)| (id, refs[cand_idx]))
            .collect();

        let r_t = if selected.is_empty() {
            1.0
        } else {
            let cone = solve_nnls_with_ids_f64(target, &selected, run, local_scratch)?;
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
            Some(LocalResidual {
                ratio: r_t,
                energy: r_t * r_t,
            }),
            anchor_gain,
            star_dir,
        ))
    };

    let per_node: Vec<_> = if run.parallelism.is_parallel() {
        (0..node_count)
            .into_par_iter()
            .map_init(nexus_math::NnlsWorkspace::default, |workspace, i| {
                compute_node(i, workspace)
            })
            .collect::<Result<_>>()?
    } else {
        let mut workspace = nexus_math::NnlsWorkspace::default();
        (0..node_count)
            .map(|i| compute_node(i, &mut workspace))
            .collect::<Result<_>>()?
    };
    let mut geometry = LocalGeometry {
        concepts: concepts.identity().clone(),
        anchor_gains: Vec::with_capacity(node_count),
        residuals: Vec::with_capacity(node_count),
        supports: node_supports,
        stars: Vec::with_capacity(node_count * dim),
    };
    for (residual, gain, star) in per_node {
        geometry.residuals.push(residual);
        geometry.anchor_gains.push(gain);
        geometry.stars.extend(star);
    }
    Ok(geometry)
}

fn geometry_vectors(concepts: ConceptView<'_>) -> Vec<Vec<f64>> {
    concepts
        .values()
        .chunks_exact(concepts.dimension())
        .map(|row| {
            let mut vector: Vec<_> = row.iter().copied().map(f64::from).collect();
            normalize_f64(&mut vector, 0.0, PrecisionMode::StrictLeftFold);
            vector
        })
        .collect()
}

fn mean_vector(vectors: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut mean = vec![0.0; dim];
    for row in vectors {
        for (value, &x) in mean.iter_mut().zip(row) {
            *value += x;
        }
    }
    if !vectors.is_empty() {
        for value in &mut mean {
            *value /= vectors.len() as f64;
        }
    }
    mean
}

pub fn global_span(
    concepts: ConceptView<'_>,
    local: &LocalGeometry,
    options: &GlobalSpanOptions,
) -> Result<GlobalSpan> {
    build_span(concepts, local, &options.config, &options.run)
}

pub(super) fn build_span(
    concepts: ConceptView<'_>,
    local: &LocalGeometry,
    cfg: &GlobalSpanConfig,
    run: &RunOptions,
) -> Result<GlobalSpan> {
    const STAGE: &str = "global_span";
    run.check_cancelled()?;
    local.check(concepts)?;
    if cfg.candidate_pool == 0 || cfg.k_sem == 0 || cfg.k_sem > cfg.k_max {
        return Err(config(
            "global.axes",
            "candidate_pool and k_sem must be positive and k_sem must not exceed k_max",
        ));
    }
    for (name, value, max) in [
        ("min_gain", cfg.min_gain, f64::INFINITY),
        ("knee_delta", cfg.knee_delta, f64::INFINITY),
        ("epsilon", cfg.epsilon, 1.0),
        ("min_support", cfg.min_support, f64::INFINITY),
        ("support_exponent", cfg.support_exponent, f64::INFINITY),
        ("residual_exponent", cfg.residual_exponent, f64::INFINITY),
    ] {
        parameter(STAGE, name, value, 0.0, max)?;
    }
    let dim = concepts.dimension();
    let unit_vecs = geometry_vectors(concepts);
    let mean = mean_vector(&unit_vecs, dim);
    let config = cfg;
    let mut pool: Vec<AgsCandidate> = local
        .residuals
        .iter()
        .enumerate()
        .filter_map(|(idx, residual)| {
            let res = residual.as_ref()?.ratio;
            let support = local.supports[idx];
            if support < config.min_support {
                return None;
            }
            let factor = match config.prior_mode {
                GlobalPriorMode::Explained => {
                    (1.0 - res * res).max(0.0).powf(config.residual_exponent)
                }
                GlobalPriorMode::Residual => res.powf(config.residual_exponent),
                GlobalPriorMode::Support => 1.0,
            };
            let prior = support.powf(config.support_exponent) * factor;
            (prior > 0.0).then_some(AgsCandidate {
                id: idx as i64,
                prior,
            })
        })
        .collect();
    if pool.iter().any(|item| !item.prior.is_finite()) {
        return Err(crate::numeric(STAGE, "global priors overflowed"));
    }
    pool.sort_by(|a, b| b.prior.total_cmp(&a.prior).then(a.id.cmp(&b.id)));
    pool.truncate(config.candidate_pool);

    let centered: Vec<Vec<f64>> = pool
        .iter()
        .map(|c| {
            let idx = c.id as usize;
            if config.candidate_mode == GlobalCandidateMode::Star {
                let s = &local.stars[idx * dim..(idx + 1) * dim];
                let norm_sq = norm_sq_f64(s, PrecisionMode::StrictLeftFold);
                if norm_sq.is_finite() && norm_sq > 1e-24 {
                    return s.to_vec();
                }
            }
            let target = &unit_vecs[idx];
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

    run.check_cancelled()?;

    let basis_refs: Vec<&[f64]> = trace.basis.iter().map(Vec::as_slice).collect();
    let compute_chunk = |chunk: &[Vec<f64>]| -> (f64, Vec<f64>) {
        let mut energy = 0.0;
        let mut captured = vec![0.0; basis_refs.len()];
        let mut x = vec![0.0; dim];
        for v in chunk {
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

    let partial: Vec<(f64, Vec<f64>)> = if run.parallelism.is_parallel() {
        unit_vecs.par_chunks(128).map(compute_chunk).collect()
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

    Ok(GlobalSpan {
        representatives: trace.picked[..k]
            .iter()
            .map(|&(_, id, _)| concepts.ids()[id as usize])
            .collect(),
        mean: mean.iter().map(|&x| x as f32).collect(),
        data: SpanData {
            k_sem: config.k_sem,
            mean,
            axes: flat_basis,
            coverage,
        },
    })
}
