// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::observation::{finite_vector, invalid, numeric_failure};
use nexus_core::kernel::{dot_f32, norm_sq_f32, normalize_f64};
use nexus_core::{
    ContextFragmentsView, ExecutionContext, PrecisionMode, VcpResult, VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextConfig {
    pub window: usize,
    pub context_mix: f64,
    pub null_mass: f64,
    pub default_role_weight: f64,
    pub role_weights: BTreeMap<String, f64>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            window: 10,
            context_mix: 0.2,
            null_mass: 1.0,
            default_role_weight: 1.0,
            role_weights: [
                ("user", 1.0),
                ("assistant", 0.5),
                ("work", 1.0),
                ("ego", 0.25),
                ("source", 1.0),
            ]
            .into_iter()
            .map(|(key, weight)| (key.into(), weight))
            .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContextAggregateInput<'call> {
    pub raw_query: Option<&'call [f32]>,
    pub fragments: ContextFragmentsView<'call>,
    pub fragment_vectors: VectorMatrixView<'call, f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextAllocation {
    pub row: usize,
    pub role: String,
    pub stage: String,
    pub contribution: f64,
    pub decayed_weight: f64,
    pub role_weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoleMassDiagnostics {
    pub role: String,
    pub stage: String,
    pub supplied: usize,
    pub budget: f64,
    pub contribution: f64,
    pub null_mass: f64,
}

#[derive(Clone, Debug)]
pub struct ContextAggregation {
    pub effective_vector: Vec<f32>,
    pub query_origin: &'static str,
    pub allocations: Vec<ContextAllocation>,
    pub role_diagnostics: Vec<RoleMassDiagnostics>,
    pub elapsed_ms: f64,
}

/// Return the part of [start, end) not already allocated to another fragment.
fn add_range(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) -> f64 {
    if end <= start {
        return 0.0;
    }
    ranges.sort_unstable();
    let mut cursor = start;
    let mut uncovered = 0;
    for &(left, right) in ranges.iter() {
        if right > cursor && left < end {
            uncovered += left.saturating_sub(cursor);
            cursor = cursor.max(right).min(end);
        }
    }
    uncovered += end - cursor;
    ranges.push((start, end));
    uncovered as f64
}

struct Parent {
    utility: f64,
    coverage: f64,
    rows: Vec<(usize, f64, f64)>,
}

fn prepare_parent(fragments: ContextFragmentsView<'_>, rows: Vec<usize>) -> Parent {
    let mut ranges = BTreeMap::<_, Vec<_>>::new();
    let mut sources = BTreeSet::new();
    let mut total = 0.0;
    let mut utility = 0.0;
    let mut samples = Vec::new();
    for row in rows {
        let source = fragments.source_keys[row].unwrap_or(fragments.keys[row]);
        let coverage = if let Some((start, end)) = fragments.source_ranges[row] {
            add_range(ranges.entry(source).or_default(), start, end)
        } else {
            f64::from(sources.insert(source))
        };
        if coverage > 0.0 {
            let decay = fragments.half_lives[row]
                .map_or(1.0, |half| (-fragments.distances[row] / half).exp2());
            let value = fragments.relevances[row]
                * fragments.validities[row]
                * fragments.weights[row]
                * decay;
            total += coverage;
            utility += coverage * value;
            samples.push((row, coverage, value));
        }
    }
    Parent {
        utility: if total > 0.0 { utility / total } else { 0.0 },
        coverage: total,
        rows: samples,
    }
}

pub fn aggregate(
    input: ContextAggregateInput<'_>,
    config: &ContextConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<ContextAggregation> {
    let start = std::time::Instant::now();
    let stage = "context.aggregate";
    ctx.check_cancelled(stage)?;
    let fragments = input.fragments;
    fragments.validate(stage)?;
    input.fragment_vectors.validate(stage)?;
    let dim = input.fragment_vectors.dim;
    if dim == 0
        || fragments.count != input.fragment_vectors.rows
        || !(1..=10_000).contains(&config.window)
        || !(0.0..=1.0).contains(&config.context_mix)
        || !config.null_mass.is_finite()
        || config.null_mass <= 0.0
        || std::iter::once(&config.default_role_weight)
            .chain(config.role_weights.values())
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(invalid(
            stage,
            "invalid context configuration or input dimensions",
        ));
    }
    type Grouped<'a> = BTreeMap<(&'a str, &'a str), BTreeMap<&'a str, Vec<usize>>>;
    let mut groups = Grouped::new();
    let mut seen = BTreeSet::new();
    let mut norm_buffer = vec![0.0; fragments.count];
    let norm = norm_buffer.as_mut_slice();
    for (row, scale) in norm.iter_mut().enumerate() {
        ctx.check_cancelled(stage)?;
        let [key, role, phase] = [
            fragments.keys[row],
            fragments.roles[row],
            fragments.stages[row],
        ];
        if [key, role, phase].into_iter().any(str::is_empty)
            || fragments.weights[row] < 0.0
            || fragments.distances[row] < 0.0
            || !(0.0..=1.0).contains(&fragments.relevances[row])
            || !(0.0..=1.0).contains(&fragments.validities[row])
            || fragments.half_lives[row].is_some_and(|v| v <= 0.0)
        {
            return Err(invalid(stage, "invalid context fragment"));
        }
        let vector = input.fragment_vectors.row(row).unwrap();
        finite_vector(stage, vector, dim)?;
        *scale = norm_sq_f32(vector, PrecisionMode::FastSimd).sqrt();
        if *scale > 1e-12 && seen.insert(key) {
            groups
                .entry((phase, role))
                .or_default()
                .entry(fragments.parent_keys[row].unwrap_or(key))
                .or_default()
                .push(row);
        }
    }
    let mut allocations = Vec::new();
    let mut role_diagnostics = Vec::new();
    let mut combined_buffer = vec![0.0; dim];
    let combined = combined_buffer.as_mut_slice();
    for ((phase, role), parents) in groups {
        ctx.check_cancelled(stage)?;
        let mut parents: Vec<_> = parents.into_values().collect();
        parents.sort_by_key(|rows| Reverse(rows.iter().map(|&row| fragments.sequences[row]).max()));
        let supplied = parents.len();
        let prepared: Vec<_> = parents
            .into_iter()
            .take(config.window)
            .map(|rows| prepare_parent(fragments, rows))
            .collect();
        let denominator =
            config.null_mass + prepared.iter().map(|parent| parent.utility).sum::<f64>();
        let budget = config.context_mix
            * config
                .role_weights
                .get(role)
                .copied()
                .unwrap_or(config.default_role_weight);
        if !denominator.is_finite() {
            return Err(numeric_failure(stage, "context utility overflowed"));
        }
        let mut contribution = 0.0;
        for parent in prepared.into_iter().filter(|parent| parent.utility > 0.0) {
            let parent_mass = budget * (parent.utility / denominator);
            contribution += parent_mass;
            for (row, coverage, value) in parent.rows {
                let mass = parent_mass * (coverage / parent.coverage);
                if mass == 0.0 {
                    continue;
                }
                for (out, &v) in combined
                    .iter_mut()
                    .zip(input.fragment_vectors.row(row).unwrap())
                {
                    *out += mass * (f64::from(v) / norm[row]);
                }
                allocations.push(ContextAllocation {
                    row,
                    role: role.into(),
                    stage: phase.into(),
                    contribution: mass,
                    decayed_weight: value,
                    role_weight: budget,
                });
            }
        }
        role_diagnostics.push(RoleMassDiagnostics {
            role: role.into(),
            stage: phase.into(),
            supplied,
            budget,
            contribution,
            null_mass: config.null_mass,
        });
    }
    let has_context = combined.iter().any(|v| *v != 0.0);
    let query_origin = if let Some(query) = input.raw_query {
        finite_vector(stage, query, dim)?;
        let scale = norm_sq_f32(query, PrecisionMode::FastSimd).sqrt();
        if scale <= 1e-12 {
            return Err(invalid(stage, "raw query has no direction"));
        }
        for (out, &v) in combined.iter_mut().zip(query) {
            *out += f64::from(v) / scale;
        }
        if has_context { "raw+context" } else { "raw" }
    } else {
        "context"
    };
    if !normalize_f64(combined, 1e-12, PrecisionMode::FastSimd) {
        return Err(invalid(stage, "query signals have no net direction"));
    }
    Ok(ContextAggregation {
        effective_vector: combined.iter().map(|&v| v as f32).collect(),
        query_origin,
        allocations,
        role_diagnostics,
        elapsed_ms: start.elapsed().as_secs_f64() * 1_000.0,
    })
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextSegmentConfig {
    pub similarity_threshold: f64,
}

impl Default for ContextSegmentConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.70,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContextSegmentInput<'call> {
    pub vectors: VectorMatrixView<'call, f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ContextSegments {
    pub segment_means: Vec<Vec<f32>>,
    pub ranges: Vec<SegmentRange>,
}

pub fn segment(
    input: ContextSegmentInput<'_>,
    config: &ContextSegmentConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<ContextSegments> {
    let stage = "context.segment";
    ctx.check_cancelled(stage)?;
    let vectors = input.vectors;
    vectors.validate(stage)?;
    if vectors.dim == 0 || !(-1.0..=1.0).contains(&config.similarity_threshold) {
        return Err(invalid(
            stage,
            "invalid segment dimensions or similarity threshold",
        ));
    }
    let mut result = ContextSegments::default();
    let mut start = 0;
    let mut previous = None;
    let mut mean_buffer = vec![0.0; vectors.dim];
    let mean = mean_buffer.as_mut_slice();
    let append =
        |range: SegmentRange, mean: &mut [f64], result: &mut ContextSegments| -> VcpResult<()> {
            if !normalize_f64(mean, 1e-12, PrecisionMode::FastSimd) {
                return Err(invalid(stage, "segment mean has no direction"));
            }
            result
                .segment_means
                .push(mean.iter().map(|&v| v as f32).collect());
            result.ranges.push(range);
            mean.fill(0.0);
            Ok(())
        };
    for (index, vector) in vectors.data.chunks_exact(vectors.dim).enumerate() {
        ctx.check_cancelled(stage)?;
        finite_vector(stage, vector, vectors.dim)?;
        let norm = norm_sq_f32(vector, PrecisionMode::FastSimd).sqrt();
        if norm <= 1e-12 {
            return Err(invalid(stage, "segment vector has no direction"));
        }
        if let Some((prior, prior_norm)) = previous {
            let cosine = dot_f32(vector, prior, PrecisionMode::FastSimd) / (norm * prior_norm);
            if cosine < config.similarity_threshold {
                append(SegmentRange { start, end: index }, mean, &mut result)?;
                start = index;
            }
        }
        for (out, &v) in mean.iter_mut().zip(vector) {
            *out += f64::from(v) / norm;
        }
        previous = Some((vector, norm));
    }
    if !vectors.is_empty() {
        append(
            SegmentRange {
                start,
                end: vectors.rows,
            },
            mean,
            &mut result,
        )?;
    }
    Ok(result)
}
