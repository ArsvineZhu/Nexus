// SPDX-License-Identifier: CC-BY-NC-SA-4.0

//! In-memory Chunk and Tag geometry derived from VCPToolBox's warped fields at 5adb12b.

use super::geometry::{WarpedProfile, clairaut_distance, clairaut_lower_bound};
use crate::model::{ConceptView, LocalGeometry};
use crate::observation::{ConceptSearchConfig, ConceptSearchMode};
use crate::{ConceptSearchRequest, EpaAnalysis};
use crate::{Error, Result, config, numeric};
use nexus_math::kernel::PrecisionMode;
use nexus_math::kernel::{dot_f32, norm_sq_f32};
use nexus_math::{CosineOptions, TransportGraph, cosine_scores};
use nexus_math::{RunOptions, VectorRows};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const STAGE: &str = "warped_search";
const INNER_RADIUS: f64 = 0.35;
const RADIUS_SPAN: f64 = 0.30;
const FOLD: PrecisionMode = PrecisionMode::StrictLeftFold;
const DEFAULT_VISIT_BUDGET: usize = 2_048;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WarpedGeometryConfig {
    pub softening: f64,
    pub mean_subtraction: f64,
    pub throat: f64,
    pub angular_scale: f64,
}

impl Default for WarpedGeometryConfig {
    fn default() -> Self {
        Self {
            softening: 0.15,
            mean_subtraction: 0.7,
            throat: 1.5,
            angular_scale: 2.0,
        }
    }
}

fn invalid(detail: &'static str) -> Error {
    crate::invalid(STAGE, detail)
}

impl WarpedGeometryConfig {
    fn validate(self) -> Result<()> {
        if !self.softening.is_finite() || self.softening < 0.0 {
            return Err(config(
                "softening",
                "softening must be finite and nonnegative",
            ));
        }
        if !(0.0..=1.0).contains(&self.mean_subtraction) {
            return Err(config(
                "mean_subtraction",
                "mean_subtraction must lie in [0, 1]",
            ));
        }
        if !self.throat.is_finite() || self.throat < 0.0 {
            return Err(config("throat", "throat must be finite and nonnegative"));
        }
        if !(self.angular_scale > 0.0 && self.angular_scale <= 2.0) {
            return Err(config("angular_scale", "angular_scale must lie in (0, 2]"));
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

fn checked_norm(vector: &[f32], dim: usize) -> Result<f64> {
    if vector.len() != dim {
        return Err(crate::invalid(
            "search.vector",
            format!("expected dimension {dim}, got {}", vector.len()),
        ));
    }
    let norm = norm_sq_f32(vector, FOLD).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid(
            "warped search vectors must have finite, nonzero norm",
        ));
    }
    Ok(norm)
}

fn direction(axis: &[f64], vector: &[f32], norm: f64, beta: f64) -> Result<Vec<f32>> {
    let aligned = alignment(axis, vector, norm);
    let mut result: Vec<_> = vector
        .iter()
        .zip(axis)
        .map(|(&x, &m)| f64::from(x) / norm - beta * aligned * m)
        .collect();
    let length = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    if length <= 1e-12 {
        return Err(numeric(STAGE, "mean subtraction annihilated direction"));
    }
    result.iter_mut().for_each(|x| *x /= length);
    Ok(result.into_iter().map(|x| x as f32).collect())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SlotHit {
    pub slot: usize,
    pub distance: f64,
}

impl Eq for SlotHit {}

impl PartialOrd for SlotHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SlotHit {
    fn cmp(&self, other: &Self) -> Ordering {
        order(self, other)
    }
}

fn order(a: &SlotHit, b: &SlotHit) -> Ordering {
    a.distance.total_cmp(&b.distance).then(a.slot.cmp(&b.slot))
}

#[derive(Clone, Copy, Debug)]
struct SearchConfig {
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
struct Frontier(SlotHit);
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
    ctx: &RunOptions,
) -> Result<Vec<SlotHit>> {
    let mut frontier = BinaryHeap::new();
    let mut seen = vec![false; neighbors.len()];
    let mut visits = 0;
    let mut enqueue = |i, frontier: &mut BinaryHeap<Frontier>| {
        if visits < budget && !std::mem::replace(&mut seen[i], true) {
            visits += 1;
            frontier.push(Frontier(SlotHit {
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
        ctx.check_cancelled()?;
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
struct WarpedField {
    axis: Vec<f64>,
    config: WarpedGeometryConfig,
    slots: Vec<usize>,
    directions: Vec<Vec<f32>>,
    radii: Vec<f64>,
    direction_norms: Vec<f64>,
    neighbors: Vec<Vec<usize>>,
    landmarks: Vec<usize>,
}

impl WarpedField {
    fn build(
        vectors: VectorRows<'_>,
        axis: Vec<f64>,
        radii: impl IntoIterator<Item = (usize, f64)>,
        config: WarpedGeometryConfig,
        ctx: &RunOptions,
    ) -> Result<Self> {
        config.validate()?;
        let mut slots = Vec::new();
        let mut directions = Vec::new();
        let mut radial = Vec::new();
        for (slot, radius) in radii {
            ctx.check_cancelled()?;
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
            let close = |slot: usize| -> Result<Vec<usize>> {
                let proxy = |i: usize| {
                    radial_proxy(
                        radial[slot],
                        radial[i],
                        angular[slot],
                        angular[i],
                        dot_f32(
                            directions[slot].as_slice(),
                            directions[i].as_slice(),
                            PrecisionMode::FastSimd,
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
            let close = if ctx.parallelism.is_parallel() {
                {
                    (0..n)
                        .into_par_iter()
                        .map(close)
                        .collect::<Result<Vec<_>>>()
                }?
            } else {
                (0..n).map(close).collect::<Result<Vec<_>>>()?
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

    fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.axis.len()
    }

    fn search(
        &self,
        query: &[f32],
        radius: f64,
        limit: usize,
        scope: Option<&[bool]>,
        config: SearchConfig,
        ctx: &RunOptions,
    ) -> Result<Vec<SlotHit>> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(invalid("query radius must be finite and nonnegative"));
        }
        if config.visit_budget == 0 {
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
        let cosine = |i: usize| {
            dot_f32(&query, &self.directions[i], PrecisionMode::FastSimd).clamp(-1.0, 1.0)
        };
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
            let budget = config.visit_budget;
            let mut candidates = navigate(
                &self.neighbors,
                self.landmarks.iter().copied(),
                budget,
                proxy,
                admitted,
                ctx,
            )?;
            let rerank = config.rerank.unwrap_or_else(|| limit.saturating_mul(2));
            candidates.truncate(rerank.max(limit));
            candidates.into_iter().map(|hit| hit.slot).collect()
        };
        let mut shortlist = BinaryHeap::<SlotHit>::new();
        for i in points {
            ctx.check_cancelled()?;
            let cosine =
                (dot_f32(&query, &self.directions[i], FOLD) / query_norm / self.direction_norms[i])
                    .clamp(-1.0, 1.0);
            if shortlist.len() >= limit
                && let Some(&worst) = shortlist.peek()
                && clairaut_lower_bound(profile, radius, self.radii[i], cosine) > worst.distance
            {
                continue;
            }
            let hit = SlotHit {
                slot: self.slots[i],
                distance: clairaut_distance(profile, radius, self.radii[i], cosine),
            };
            if !hit.distance.is_finite() {
                return Err(numeric(STAGE, "Clairaut distance is not finite"));
            }
            if shortlist.len() < limit {
                shortlist.push(hit);
            } else if order(&hit, shortlist.peek().expect("nonempty shortlist")).is_lt() {
                *shortlist.peek_mut().expect("nonempty shortlist") = hit;
            }
        }
        Ok(shortlist.into_sorted_vec())
    }
}

#[derive(Clone, Default)]
pub struct WarpedBuildOptions {
    pub geometry: WarpedGeometryConfig,
    pub run: RunOptions,
}

#[derive(Clone)]
pub struct SearchRequest<'a> {
    pub limit: usize,
    pub allowed: Option<&'a [u64]>,
    pub effort: SearchEffort,
    pub run: RunOptions,
}

impl Default for SearchRequest<'_> {
    fn default() -> Self {
        Self {
            limit: 64,
            allowed: None,
            effort: SearchEffort::Exact,
            run: RunOptions::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum SearchEffort {
    #[default]
    Exact,
    Approximate {
        visit_budget: usize,
        rerank: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceHit {
    pub id: u64,
    pub distance: f64,
}

pub struct SearchHits {
    rows: Vec<DistanceHit>,
    ids: Vec<u64>,
}

impl SearchHits {
    pub fn rows(&self) -> &[DistanceHit] {
        &self.rows
    }
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }
}

/// In-memory numeric search over owned Warped geometry and business IDs.
pub struct WarpedIndex {
    field: WarpedField,
    quantiles: [f64; 64],
    ids: Vec<u64>,
    slots: std::collections::HashMap<u64, usize>,
}

impl WarpedIndex {
    pub fn build(vectors: VectorRows<'_>, options: &WarpedBuildOptions) -> Result<Self> {
        options.run.check_cancelled()?;
        let axis = mean_axis(vectors, &options.run)?;
        let mut points = Vec::with_capacity(vectors.len());
        for i in 0..vectors.len() {
            options.run.check_cancelled()?;
            let row = vectors.row(i).expect("validated row");
            points.push(
                alignment(&axis, row, checked_norm(row, vectors.dimension())?)
                    .abs()
                    .min(1.0),
            );
        }
        let mut sorted = points.clone();
        sorted.sort_unstable_by(f64::total_cmp);
        let quantiles = if sorted.is_empty() {
            [0.0; 64]
        } else {
            std::array::from_fn(|i| {
                let pos = i as f64 * (sorted.len() - 1) as f64 / 63.0;
                let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                sorted[lo] + (sorted[hi] - sorted[lo]) * pos.fract()
            })
        };
        let radii = points
            .into_iter()
            .map(|x| chunk_radius(x, &quantiles))
            .enumerate();
        let field = WarpedField::build(vectors, axis, radii, options.geometry, &options.run)?;
        let ids = vectors.ids().to_vec();
        let slots = ids
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, id)| (id, slot))
            .collect();
        Ok(Self {
            field,
            quantiles,
            ids,
            slots,
        })
    }

    pub fn search(&self, query: &[f32], request: &SearchRequest<'_>) -> Result<SearchHits> {
        request.run.check_cancelled()?;
        let scope = request
            .allowed
            .map(|allowed| {
                let mut mask = vec![false; self.ids.len()];
                for id in allowed {
                    request.run.check_cancelled()?;
                    let slot = *self.slots.get(id).ok_or_else(|| {
                        crate::input(
                            "search.allowed",
                            Some(*id),
                            "ID does not belong to the index",
                        )
                    })?;
                    mask[slot] = true;
                }
                Ok(mask)
            })
            .transpose()?;
        let config = match request.effort {
            SearchEffort::Exact => SearchConfig {
                exact_scope: usize::MAX,
                ..Default::default()
            },
            SearchEffort::Approximate {
                visit_budget,
                rerank,
            } => {
                if visit_budget == 0 {
                    return Err(config("visit_budget", "must be positive"));
                }
                if rerank.is_some_and(|n| n < request.limit) {
                    return Err(config("rerank", "must be at least limit"));
                }
                SearchConfig {
                    exact_scope: 0,
                    visit_budget,
                    rerank,
                }
            }
        };
        let norm = checked_norm(query, self.field.dimension())?;
        let aligned = alignment(&self.field.axis, query, norm).abs().min(1.0);
        let radius = f64::from(chunk_radius(aligned, &self.quantiles) as f32);
        let rows: Vec<_> = self
            .field
            .search(
                query,
                radius,
                request.limit,
                scope.as_deref(),
                config,
                &request.run,
            )?
            .into_iter()
            .map(|hit| DistanceHit {
                id: self.ids[hit.slot],
                distance: hit.distance,
            })
            .collect();
        let ids = rows.iter().map(|hit| hit.id).collect();
        Ok(SearchHits { rows, ids })
    }

    pub fn search_many(
        &self,
        queries: &[&[f32]],
        request: &SearchRequest<'_>,
    ) -> Result<Vec<SearchHits>> {
        request.run.check_cancelled()?;
        queries
            .iter()
            .map(|query| self.search(query, request))
            .collect()
    }
}

fn mean_axis(vectors: VectorRows<'_>, run: &RunOptions) -> Result<Vec<f64>> {
    if std::alloc::Layout::array::<f64>(vectors.dimension()).is_err() {
        return Err(crate::invalid(
            "warped.dimension",
            "mean axis size overflows",
        ));
    }
    let mut axis = vec![0.0; vectors.dimension()];
    for slot in 0..vectors.len() {
        run.check_cancelled()?;
        let row = vectors.row(slot).expect("validated row");
        let norm = checked_norm(row, vectors.dimension())?;
        for (sum, &value) in axis.iter_mut().zip(row) {
            *sum += f64::from(value) / norm;
        }
    }
    if !vectors.is_empty() {
        for component in &mut axis {
            *component /= vectors.len() as f64;
        }
    }
    normalize_axis(&mut axis);
    Ok(axis)
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

pub struct ConceptIndex {
    field: WarpedField,
    space: std::sync::Arc<[u64]>,
}

impl ConceptIndex {
    pub fn build(
        concepts: ConceptView<'_>,
        local: &LocalGeometry,
        graph: &TransportGraph,
        options: &WarpedBuildOptions,
    ) -> Result<Self> {
        options.run.check_cancelled()?;
        local.check(concepts)?;
        if graph.nodes() != concepts.ids() {
            return Err(Error::Association {
                message: "graph and concepts have different IDs or order".into(),
            });
        }
        let vectors = concepts.vectors().as_rows();
        let axis = mean_axis(vectors, &options.run)?;
        let graph = graph.view();
        let mut ranked: Vec<_> = graph.inbound.iter().map(|mass| mass.ln_1p()).collect();
        ranked.sort_unstable_by(f64::total_cmp);
        let radii = graph.inbound.iter().zip(&local.residuals).enumerate().map(
            |(slot, (&mass, residual))| {
                let mass = mass.ln_1p();
                let first = ranked.partition_point(|&value| value < mass);
                let last = ranked.partition_point(|&value| value <= mass);
                let quantile = if ranked.len() > 1 {
                    (first + last - 1) as f64 / (2 * (ranked.len() - 1)) as f64
                } else {
                    0.5
                };
                let ratio = residual.map_or(1.0, |value| value.ratio);
                let position = (0.65 * (1.0 - quantile) + 0.35 * ratio).powf(1.15);
                (slot, INNER_RADIUS + RADIUS_SPAN * position)
            },
        );
        let field = WarpedField::build(vectors, axis, radii, options.geometry, &options.run)?;
        Ok(Self {
            field,
            space: concepts.identity().clone(),
        })
    }

    pub(crate) fn search_residual(
        &self,
        concepts: ConceptView<'_>,
        request: ConceptSearchRequest<'_>,
        epa: Option<&EpaAnalysis>,
        search: &ConceptSearchConfig,
        run: &RunOptions,
    ) -> Result<Vec<u64>> {
        let ConceptSearchRequest {
            residual: query,
            level,
            excluded,
            limit,
        } = request;
        if !std::sync::Arc::ptr_eq(&self.space, concepts.identity()) {
            return Err(Error::Association {
                message: "concept index belongs to a different concept space".into(),
            });
        }
        if search.visit_budget == 0 {
            return Err(config("concept_search.visit_budget", "must be positive"));
        }
        match search.mode {
            ConceptSearchMode::Cosine => {
                let scores = cosine_scores(
                    query,
                    concepts.vectors().as_rows(),
                    &CosineOptions { run: run.clone() },
                )?;
                Ok(scores
                    .top(None)
                    .ids()
                    .iter()
                    .copied()
                    .filter(|id| !excluded.contains(id))
                    .take(limit)
                    .collect())
            }
            ConceptSearchMode::Warped => {
                let mut mask = vec![true; concepts.ids().len()];
                for &id in excluded {
                    mask[concepts.node(id)?.index()] = false;
                }
                let ratio = norm_sq_f32(query, FOLD);
                let cfg = SearchConfig {
                    exact_scope: search.exact_scope,
                    visit_budget: search.visit_budget,
                    rerank: search.rerank,
                };
                let hits = self.field.search(
                    query,
                    residual_radius(epa, level, ratio),
                    limit,
                    Some(&mask),
                    cfg,
                    run,
                )?;
                Ok(hits.into_iter().map(|hit| self.space[hit.slot]).collect())
            }
        }
    }
}

fn residual_radius(epa: Option<&EpaAnalysis>, level: usize, energy_ratio: f64) -> f64 {
    let to_radius =
        |position: f64| -> f32 { (INNER_RADIUS + RADIUS_SPAN * position.clamp(0.0, 1.0)) as f32 };
    let Some(epa) = epa else {
        return f64::from(to_radius(0.25 + 0.25 * level as f64));
    };
    let base_radius =
        to_radius(0.55 * epa.in_domain_gate * (1.0 - epa.entropy) + 0.30 * epa.concentration);
    if level == 0 {
        return f64::from(base_radius);
    }
    let base = (f64::from(base_radius) - INNER_RADIUS) / RADIUS_SPAN;
    f64::from(to_radius(
        base + 0.75 * (1.0 - base) * (1.0 - energy_ratio.clamp(0.0, 1.0).powf(0.35)),
    ))
}
