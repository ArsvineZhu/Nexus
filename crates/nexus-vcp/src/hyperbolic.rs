// SPDX-License-Identifier: CC-BY-NC-SA-4.0

//! In-memory Chunk and Tag geometry derived from VCPToolBox's warped fields at 5adb12b.

pub use crate::warped_geometry::{WarpedProfile, clairaut_distance, clairaut_lower_bound};
use crate::{EpaAnalysis, IntrinsicResidual};
use nexus_core::kernel::{dot_f32, norm_sq_f32};
use nexus_core::{ExecutionContext, NodeId, PrecisionMode, VcpError, VcpResult, VectorMatrixView};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const STAGE: &str = "warped_search";
const INNER_RADIUS: f64 = 0.35;
const RADIUS_SPAN: f64 = 0.30;
const FOLD: PrecisionMode = PrecisionMode::StrictLeftFold;
const DEFAULT_VISIT_BUDGET: usize = 2_048;

#[derive(Clone, Copy, Debug)]
pub struct WarpedConfig {
    pub softening: f64,
    pub mean_subtraction: f64,
    pub throat: f64,
    pub angular_scale: f64,
}

impl Default for WarpedConfig {
    fn default() -> Self {
        Self {
            softening: 0.15,
            mean_subtraction: 0.7,
            throat: 1.5,
            angular_scale: 2.0,
        }
    }
}

fn invalid(detail: &'static str) -> VcpError {
    VcpError::InvalidInput {
        stage: STAGE,
        detail: detail.into(),
    }
}

impl WarpedConfig {
    fn validate(self) -> VcpResult<()> {
        if !self.softening.is_finite() || self.softening < 0.0 {
            return Err(invalid("softening must be finite and nonnegative"));
        }
        if !(0.0..=1.0).contains(&self.mean_subtraction) {
            return Err(invalid("mean_subtraction must lie in [0, 1]"));
        }
        if !self.throat.is_finite() || self.throat < 0.0 {
            return Err(invalid("throat must be finite and nonnegative"));
        }
        if !(self.angular_scale > 0.0 && self.angular_scale <= 2.0) {
            return Err(invalid("angular_scale must lie in (0, 2]"));
        }
        Ok(())
    }

    fn profile(self) -> WarpedProfile {
        if self.throat == 0.0 && self.angular_scale == 1.0 {
            WarpedProfile::Softening {
                alpha: self.softening,
            }
        } else {
            WarpedProfile::ThroatSoftening {
                alpha: self.softening,
                phi0: self.throat,
                kappa: self.angular_scale,
            }
        }
    }
}

fn normalize_axis(mean: &mut [f64]) {
    let length = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
    if length > 1e-12 {
        mean.iter_mut().for_each(|x| *x /= length);
    }
}

fn alignment(axis: &[f64], vector: &[f32], norm: f64) -> f64 {
    vector
        .iter()
        .zip(axis)
        .map(|(&x, &m)| f64::from(x) / norm * m)
        .sum()
}

fn checked_norm(vector: &[f32], dim: usize) -> VcpResult<f64> {
    if vector.len() != dim {
        return Err(VcpError::DimensionMismatch {
            stage: STAGE,
            expected: dim,
            actual: vector.len(),
        });
    }
    let norm = norm_sq_f32(vector, FOLD).sqrt();
    if !norm.is_finite() || norm <= 1e-12 {
        return Err(invalid(
            "warped search vectors must have finite, nonzero norm",
        ));
    }
    Ok(norm)
}

fn direction(axis: &[f64], vector: &[f32], norm: f64, beta: f64) -> VcpResult<Vec<f32>> {
    let aligned = alignment(axis, vector, norm);
    let mut result: Vec<_> = vector
        .iter()
        .zip(axis)
        .map(|(&x, &m)| f64::from(x) / norm - beta * aligned * m)
        .collect();
    let length = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    if length <= 1e-12 {
        return Err(invalid("mean subtraction annihilates direction"));
    }
    result.iter_mut().for_each(|x| *x /= length);
    Ok(result.into_iter().map(|x| x as f32).collect())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    pub slot: usize,
    pub distance: f64,
}

fn order(a: &SearchHit, b: &SearchHit) -> Ordering {
    a.distance.total_cmp(&b.distance).then(a.slot.cmp(&b.slot))
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    /// Scopes this size or smaller are scored exhaustively.
    pub exact_scope: usize,
    pub visit_budget: usize,
    /// Override the default precision rerank size of twice the requested limit.
    pub rerank: Option<usize>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            exact_scope: 128,
            visit_budget: DEFAULT_VISIT_BUDGET,
            rerank: None,
        }
    }
}

#[derive(Clone, Copy)]
struct Frontier(SearchHit);
impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        order(&other.0, &self.0)
    }
}

fn navigate(
    neighbors: &[Vec<usize>],
    seeds: impl IntoIterator<Item = usize>,
    budget: usize,
    proxy: impl Fn(usize) -> f64,
    admits: impl Fn(usize) -> bool,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<Vec<SearchHit>> {
    let mut frontier = BinaryHeap::new();
    let mut seen = vec![false; neighbors.len()];
    let mut visits = 0;
    let mut enqueue = |i, frontier: &mut BinaryHeap<Frontier>| {
        if visits < budget && !std::mem::replace(&mut seen[i], true) {
            visits += 1;
            frontier.push(Frontier(SearchHit {
                slot: i,
                distance: proxy(i),
            }));
        }
    };
    for seed in seeds {
        enqueue(seed, &mut frontier);
    }
    let mut results = Vec::new();
    while let Some(Frontier(hit)) = frontier.pop() {
        ctx.check_cancelled(STAGE)?;
        if admits(hit.slot) {
            results.push(hit);
        }
        for &next in &neighbors[hit.slot] {
            enqueue(next, &mut frontier);
        }
    }
    results.sort_unstable_by(order);
    Ok(results)
}

fn radial_proxy(rq: f64, rv: f64, aq: f64, av: f64, cosine: f64) -> f64 {
    (rq - rv).powi(2) + 2.0 * aq * av * (1.0 - cosine.clamp(-1.0, 1.0))
}

/// Immutable warped directions and a small navigable graph. External row slots are preserved.
pub struct WarpedIndex {
    axis: Vec<f64>,
    config: WarpedConfig,
    slots: Vec<usize>,
    directions: Vec<Vec<f32>>,
    radii: Vec<f64>,
    direction_norms: Vec<f64>,
    neighbors: Vec<Vec<usize>>,
    landmarks: Vec<usize>,
}

impl WarpedIndex {
    fn build(
        vectors: VectorMatrixView<'_>,
        axis: Vec<f64>,
        radii: impl IntoIterator<Item = (usize, f64)>,
        config: WarpedConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Self> {
        config.validate()?;
        let mut slots = Vec::new();
        let mut directions = Vec::new();
        let mut radial = Vec::new();
        for (slot, radius) in radii {
            ctx.check_cancelled(STAGE)?;
            let vector = vectors
                .row(slot)
                .ok_or_else(|| invalid("invalid vector slot"))?;
            let norm = checked_norm(vector, axis.len())?;
            directions.push(direction(&axis, vector, norm, config.mean_subtraction)?);
            slots.push(slot);
            radial.push(f64::from(radius as f32));
        }
        // Connect adjacent points in radial and principal coordinate orderings.
        // The radial ordering alone connects the graph, including filtered scopes.
        let n = slots.len();
        let mut neighbors = vec![Vec::new(); n];
        let mut landmarks = Vec::new();
        let mut ordered: Vec<_> = (0..n).collect();
        for coordinate in 0..axis.len().min(7) + 1 {
            let key = |i: usize| {
                if coordinate == 0 {
                    radial[i]
                } else {
                    f64::from(directions[i][coordinate - 1])
                }
            };
            ordered.sort_unstable_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
            for pair in ordered.windows(2) {
                neighbors[pair[0]].push(pair[1]);
                neighbors[pair[1]].push(pair[0]);
            }
            for &position in &[
                0,
                n / 4,
                n / 2,
                n.saturating_mul(3) / 4,
                n.saturating_sub(1),
            ] {
                landmarks.extend(ordered.get(position).copied());
            }
        }
        for row in &mut neighbors {
            row.sort_unstable();
            row.dedup();
        }
        landmarks.sort_unstable();
        landmarks.dedup();
        if n > DEFAULT_VISIT_BUDGET {
            // The coordinate chains connect the field. Add nearby warped
            // directions so a query can move through semantic neighborhoods.
            let profile = config.profile();
            let angular: Vec<_> = radial.iter().map(|&r| profile.phi(r)).collect();
            let close = |slot: usize| -> VcpResult<Vec<usize>> {
                let proxy = |i: usize| {
                    radial_proxy(
                        radial[slot],
                        radial[i],
                        angular[slot],
                        angular[i],
                        dot_f32(
                            directions[slot].as_slice(),
                            directions[i].as_slice(),
                            ctx.precision,
                        ),
                    )
                };
                let seeds = std::iter::once(slot).chain(landmarks.iter().copied());
                Ok(navigate(&neighbors, seeds, 128, proxy, |i| i != slot, ctx)?
                    .into_iter()
                    .take(12)
                    .map(|hit| hit.slot)
                    .collect())
            };
            let close = if ctx.parallel.is_parallel() {
                ctx.parallel.install(|| {
                    (0..n)
                        .into_par_iter()
                        .map(close)
                        .collect::<VcpResult<Vec<_>>>()
                })?
            } else {
                (0..n).map(close).collect::<VcpResult<Vec<_>>>()?
            };
            for (slot, near) in close.into_iter().enumerate() {
                neighbors[slot].extend(near);
                neighbors[slot].sort_unstable();
                neighbors[slot].dedup();
            }
        }
        let direction_norms = directions
            .iter()
            .map(|row| norm_sq_f32(row, FOLD).sqrt())
            .collect();
        Ok(Self {
            axis,
            config,
            slots,
            direction_norms,
            directions,
            radii: radial,
            neighbors,
            landmarks,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Radius of a prepared row, if it belongs to this field.
    pub fn radius_of(&self, slot: usize) -> Option<f64> {
        self.slots
            .binary_search(&slot)
            .ok()
            .map(|index| self.radii[index])
    }

    pub fn dimension(&self) -> usize {
        self.axis.len()
    }

    pub fn search(
        &self,
        query: &[f32],
        radius: f64,
        limit: usize,
        scope: Option<&[bool]>,
        config: SearchConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Vec<SearchHit>> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(invalid("query radius must be finite and nonnegative"));
        }
        if config.visit_budget == 0 || config.rerank == Some(0) {
            return Err(invalid("visit_budget and rerank must be positive"));
        }
        if scope.is_some_and(|mask| self.slots.iter().any(|&slot| slot >= mask.len())) {
            return Err(invalid("scope must cover the indexed vector slots"));
        }
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let norm = checked_norm(query, self.axis.len())?;
        let query = direction(&self.axis, query, norm, self.config.mean_subtraction)?;
        let admitted = |slot: usize| scope.is_none_or(|mask| mask[self.slots[slot]]);
        let count = (0..self.slots.len()).filter(|&i| admitted(i)).count();
        let profile = self.config.profile();
        let query_norm = norm_sq_f32(&query, FOLD).sqrt();
        let cosine =
            |i: usize| dot_f32(&query, &self.directions[i], ctx.precision).clamp(-1.0, 1.0);
        let points: Vec<usize> = if count <= config.exact_scope {
            (0..self.slots.len()).filter(|&i| admitted(i)).collect()
        } else {
            let angle = profile.phi(radius);
            let proxy = |i: usize| {
                radial_proxy(
                    radius,
                    self.radii[i],
                    angle,
                    profile.phi(self.radii[i]),
                    cosine(i),
                )
            };
            let budget = config.visit_budget.max(limit);
            let mut candidates = navigate(
                &self.neighbors,
                self.landmarks.iter().copied(),
                budget,
                proxy,
                admitted,
                ctx,
            )?;
            let rerank = config
                .rerank
                .unwrap_or_else(|| limit.saturating_mul(2).clamp(64, 256));
            candidates.truncate(rerank.max(limit));
            candidates.into_iter().map(|hit| hit.slot).collect()
        };
        let mut hits = Vec::with_capacity(points.len());
        for i in points {
            ctx.check_cancelled(STAGE)?;
            hits.push(SearchHit {
                slot: self.slots[i],
                distance: clairaut_distance(
                    profile,
                    radius,
                    self.radii[i],
                    (dot_f32(&query, &self.directions[i], FOLD)
                        / query_norm
                        / self.direction_norms[i])
                        .clamp(-1.0, 1.0),
                ),
            });
        }
        hits.sort_unstable_by(order);
        hits.truncate(limit);
        Ok(hits)
    }
}

pub struct ChunkField {
    index: WarpedIndex,
    quantiles: [f64; 64],
}

impl ChunkField {
    pub fn build(
        vectors: VectorMatrixView<'_>,
        config: WarpedConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Self> {
        vectors.validate(STAGE)?;
        if vectors.rows == 0 {
            return Err(invalid("Chunk field requires vectors"));
        }
        let mut axis = vec![0.0; vectors.dim];
        let mut norms = Vec::with_capacity(vectors.rows);
        for row in vectors.data.chunks_exact(vectors.dim) {
            ctx.check_cancelled(STAGE)?;
            let norm = checked_norm(row, vectors.dim)?;
            norms.push(norm);
            for (mean, &x) in axis.iter_mut().zip(row) {
                *mean += f64::from(x) / norm;
            }
        }
        normalize_axis(&mut axis);
        let mut sorted: Vec<_> = vectors
            .data
            .chunks_exact(vectors.dim)
            .zip(&norms)
            .map(|(row, &norm)| alignment(&axis, row, norm).abs().min(1.0))
            .collect();
        let mut points = sorted.clone();
        sorted.sort_unstable_by(f64::total_cmp);
        let quantiles = std::array::from_fn(|i| {
            let pos = i as f64 * (sorted.len() - 1) as f64 / 63.0;
            let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
            sorted[lo] + (sorted[hi] - sorted[lo]) * pos.fract()
        });
        points
            .iter_mut()
            .for_each(|x| *x = chunk_radius(*x, &quantiles));
        let index = WarpedIndex::build(vectors, axis, points.into_iter().enumerate(), config, ctx)?;
        Ok(Self { index, quantiles })
    }

    pub fn radius(&self, query: &[f32]) -> VcpResult<f64> {
        let norm = checked_norm(query, self.index.dimension())?;
        let aligned = alignment(&self.index.axis, query, norm).abs().min(1.0);
        Ok(f64::from(chunk_radius(aligned, &self.quantiles) as f32))
    }

    pub fn index(&self) -> &WarpedIndex {
        &self.index
    }

    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        scope: Option<&[bool]>,
        config: SearchConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Vec<SearchHit>> {
        self.index
            .search(query, self.radius(query)?, limit, scope, config, ctx)
    }
}

fn chunk_radius(value: f64, knots: &[f64; 64]) -> f64 {
    let lo = knots.partition_point(|&v| v < value);
    let hi = knots.partition_point(|&v| v <= value);
    let percentile = if lo != hi {
        (lo + hi - 1) as f64 / 126.0
    } else if lo == 0 {
        0.0
    } else if lo == 64 {
        1.0
    } else {
        ((lo - 1) as f64 + (value - knots[lo - 1]) / (knots[lo] - knots[lo - 1])) / 63.0
    };
    INNER_RADIUS + RADIUS_SPAN * (1.0 - percentile).powf(1.10)
}

pub fn tag_field(
    vectors: VectorMatrixView<'_>,
    mean: &[f64],
    inbound: &[f64],
    residuals: &[IntrinsicResidual],
    config: WarpedConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<WarpedIndex> {
    vectors.validate(STAGE)?;
    if (!mean.is_empty() && mean.len() != vectors.dim) || mean.iter().any(|x| !x.is_finite()) {
        return Err(invalid(
            "Tag geometry mean does not match vector dimensions",
        ));
    }
    if inbound.len() != vectors.rows || inbound.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(invalid("Tag inbound weights must align with vector rows"));
    }
    let mut axis = if mean.is_empty() {
        vec![0.0; vectors.dim]
    } else {
        mean.to_vec()
    };
    normalize_axis(&mut axis);
    let mut ranked: Vec<_> = inbound.iter().map(|mass| mass.ln_1p()).collect();
    ranked.sort_unstable_by(f64::total_cmp);
    let mut seen = vec![false; vectors.rows];
    let mut radii = Vec::new();
    for residual in residuals {
        let index = residual.node.index();
        let Some(&mass) = inbound.get(index) else {
            return Err(invalid("Tag residual slot is outside the matrix"));
        };
        if std::mem::replace(&mut seen[index], true)
            || !(0.0..=1.0).contains(&residual.residual_ratio)
        {
            return Err(invalid("Tag residuals must be unique and within [0, 1]"));
        }
        let mass = mass.ln_1p();
        let first = ranked.partition_point(|&x| x < mass);
        let last = ranked.partition_point(|&x| x <= mass);
        let quantile = if ranked.len() > 1 {
            (first + last - 1) as f64 / (2 * (ranked.len() - 1)) as f64
        } else {
            0.5
        };
        let position = (0.65 * (1.0 - quantile) + 0.35 * residual.residual_ratio).powf(1.15);
        radii.push((index, INNER_RADIUS + RADIUS_SPAN * position));
    }
    radii.sort_unstable_by_key(|row| row.0);
    WarpedIndex::build(vectors, axis, radii, config, ctx)
}

pub fn residual_radius(epa: Option<&EpaAnalysis>, level: usize, energy_ratio: f64) -> f64 {
    let to_radius =
        |position: f64| -> f32 { (INNER_RADIUS + RADIUS_SPAN * position.clamp(0.0, 1.0)) as f32 };
    let Some(epa) = epa.filter(|value| value.cache_available) else {
        return f64::from(to_radius(0.25 + 0.25 * level as f64));
    };
    let base_radius =
        to_radius(0.55 * epa.in_domain_gate * (1.0 - epa.entropy) + 0.30 * epa.logic_depth);
    if level == 0 {
        return f64::from(base_radius);
    }
    let base = (f64::from(base_radius) - INNER_RADIUS) / RADIUS_SPAN;
    f64::from(to_radius(
        base + 0.75 * (1.0 - base) * (1.0 - energy_ratio.clamp(0.0, 1.0).powf(0.35)),
    ))
}

pub fn tag_hits(hits: Vec<SearchHit>, output: &mut Vec<(NodeId, f64)>) -> VcpResult<()> {
    for hit in hits {
        output.push((NodeId::from_index(hit.slot, STAGE)?, -hit.distance));
    }
    Ok(())
}
