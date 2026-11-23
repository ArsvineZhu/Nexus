// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{EpaOptions, PyramidOptions, direction};
use crate::{
    ConceptId, ConceptSearch, ConceptSearchRequest, ConceptView, Error, Result, RunOptions, config,
    input as bad_input, invalid,
};
use crate::{model::GlobalSpan, warped::ConceptIndex};
use nexus_math::kernel::{PrecisionMode, cosine_f32, dot_f64, norm_sq_f32, norm_sq_f64};
use nexus_math::nnls::solve_nnls_with_ids_f64;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpaGateMode {
    #[default]
    BackgroundRelative,
    RelativeLinear,
    RelativeSqrt,
    Raw,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EpaConfig {
    pub gate_mode: EpaGateMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpaAxis {
    pub concept: ConceptId,
    pub energy: f64,
    pub projection: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpaBridge {
    pub from: ConceptId,
    pub to: ConceptId,
    pub strength: f64,
    pub balance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpaAnalysis {
    pub concentration: f64,
    pub entropy: f64,
    pub resonance: f64,
    pub in_domain_gate: f64,
    pub semantic_explained_ratio: f64,
    pub background_ratio: f64,
    pub baseline_coverage: f64,
    pub out_of_domain: f64,
    pub axes: Vec<EpaAxis>,
    pub bridges: Vec<EpaBridge>,
}

pub fn analyze_epa(
    query: &[f32],
    span: &GlobalSpan,
    options: &EpaOptions,
) -> Result<Option<EpaAnalysis>> {
    options.run.check_cancelled()?;
    let direction = direction(query, span.mean().len(), "query.vector")?;
    epa(direction.as_slice(), span, &options.config, &options.run)
}

pub(super) fn epa(
    query: &[f32],
    span: &GlobalSpan,
    config: &EpaConfig,
    run: &RunOptions,
) -> Result<Option<EpaAnalysis>> {
    run.check_cancelled()?;
    let basis = &span.data;
    let dim = basis.mean.len();
    if basis.axes.is_empty() || basis.k_sem == 0 {
        return Ok(None);
    }
    let representatives = span.representatives();
    let mut centered_buffer = vec![0.0; dim];
    let centered = centered_buffer.as_mut_slice();
    for ((slot, &value), &mean) in centered.iter_mut().zip(query).zip(&basis.mean) {
        *slot = f64::from(value) - mean;
    }
    let count = basis.k_sem.min(span.axis_count());
    let mode = PrecisionMode::FastSimd;
    let centered_energy = norm_sq_f64(centered, mode);
    let mut semantic = Vec::with_capacity(count);
    let mut captured = 0.0;
    let mut mean_captured = 0.0;
    for (index, axis) in basis.axes.chunks_exact(dim).enumerate() {
        run.check_cancelled()?;
        let projection = dot_f64(centered, axis, mode);
        captured += projection * projection;
        if index < count {
            semantic.push(projection);
            mean_captured += dot_f64(&basis.mean, axis, mode).powi(2);
        }
    }
    let energy = semantic.iter().map(|v| v * v).sum::<f64>();
    let raw_background = count as f64 / dim as f64;
    let baseline = basis.coverage[count - 1].max(1e-4);
    let background =
        ((raw_background + mean_captured) / (1.0 + norm_sq_f64(&basis.mean, mode))).clamp(0.0, 1.0);
    let ratio = if centered_energy > 1e-12 {
        (energy / centered_energy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gate = if centered_energy <= 1e-12 {
        0.0
    } else {
        match config.gate_mode {
            EpaGateMode::BackgroundRelative if baseline > background + 1e-8 => {
                (ratio - background) / (baseline - background)
            }
            EpaGateMode::BackgroundRelative => 0.0,
            EpaGateMode::RelativeLinear => ratio / baseline,
            EpaGateMode::RelativeSqrt => (ratio / baseline).sqrt(),
            EpaGateMode::Raw => ratio,
        }
        .clamp(0.0, 1.0)
    };
    let mut entropy = 0.0;
    let mut axes = Vec::new();
    if energy > 1e-12 {
        for (index, projection) in semantic.into_iter().enumerate() {
            let probability = projection * projection / energy;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
                axes.push((
                    index,
                    EpaAxis {
                        concept: representatives[index],
                        energy: probability,
                        projection,
                    },
                ));
            }
        }
        if count > 1 {
            entropy = (entropy / (count as f64).log2()).clamp(0.0, 1.0);
        }
    } else {
        entropy = 1.0;
    }
    axes.sort_unstable_by(|(i, a), (j, b)| b.energy.total_cmp(&a.energy).then(i.cmp(j)));
    let axes: Vec<_> = axes.into_iter().map(|(_, axis)| axis).collect();
    let mut bridges = Vec::new();
    if let Some(primary) = axes.first() {
        for secondary in &axes[1..] {
            let strength = (primary.energy * secondary.energy).sqrt();
            if strength > 0.15 {
                bridges.push(EpaBridge {
                    from: primary.concept,
                    to: secondary.concept,
                    strength,
                    balance: secondary.energy / primary.energy,
                });
            }
        }
    }
    Ok(Some(EpaAnalysis {
        concentration: (1.0 - entropy) * gate,
        entropy,
        resonance: bridges.iter().map(|bridge| bridge.strength).sum(),
        axes: axes.into_iter().filter(|axis| axis.energy > 0.05).collect(),
        bridges,
        out_of_domain: if centered_energy > 1e-12 {
            (1.0 - captured / centered_energy).clamp(0.0, 1.0).sqrt()
        } else {
            0.0
        },
        semantic_explained_ratio: ratio,
        in_domain_gate: gate,
        baseline_coverage: baseline,
        background_ratio: background,
    }))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptSearchMode {
    #[default]
    Warped,
    Cosine,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConceptSearchConfig {
    pub mode: ConceptSearchMode,
    pub exact_scope: usize,
    pub visit_budget: usize,
    pub rerank: Option<usize>,
}

impl Default for ConceptSearchConfig {
    fn default() -> Self {
        Self {
            mode: ConceptSearchMode::Warped,
            exact_scope: 128,
            visit_budget: 2048,
            rerank: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PyramidConfig {
    pub search: ConceptSearchConfig,
    pub max_levels: usize,
    pub top_k: usize,
    pub min_energy_ratio: f64,
    pub min_layer_explained: f64,
    pub tail_min_ratio: f64,
    pub append_tail: bool,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            search: ConceptSearchConfig::default(),
            max_levels: 3,
            top_k: 10,
            min_energy_ratio: 0.1,
            min_layer_explained: 0.03,
            tail_min_ratio: 0.02,
            append_tail: true,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PyramidInput<'a> {
    pub query: &'a [f32],
    pub concepts: ConceptView<'a>,
    pub epa: Option<&'a EpaAnalysis>,
    pub index: &'a ConceptIndex,
    pub concept_search: Option<&'a dyn ConceptSearch>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PyramidTag {
    pub concept: ConceptId,
    pub similarity: f64,
    pub contribution: f64,
    pub handshake_magnitude: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Handshake {
    pattern: f64,
    novelty: f64,
    noise: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PyramidLevel {
    pub level: usize,
    pub tags: Vec<PyramidTag>,
    pub projection_magnitude: f64,
    pub residual_magnitude: f64,
    pub residual_energy_ratio: f64,
    pub energy_explained: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PyramidResult {
    pub levels: Vec<PyramidLevel>,
    pub total_explained_energy: f64,
    pub coverage: f64,
    pub novelty: f64,
    pub coherence: f64,
    pub activation: f64,
    pub residuals: Vec<Residual>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Residual {
    pub level: usize,
    pub vector: Vec<f32>,
    pub is_tail: bool,
}

fn handshakes(query: &[f32], tags: &mut [PyramidTag], vectors: &[&[f32]]) -> Handshake {
    let mut directions = Vec::with_capacity(tags.len().min(5));
    let mut average = vec![0.0; query.len()];
    let mut direction = vec![0.0; query.len()];
    let count = tags.len();
    for (index, (tag, &vector)) in tags.iter_mut().zip(vectors).enumerate() {
        for ((out, &q), &v) in direction.iter_mut().zip(query).zip(vector) {
            *out = f64::from(q) - f64::from(v);
        }
        let distance = norm_sq_f64(&direction, PrecisionMode::FastSimd).sqrt();
        tag.handshake_magnitude = distance;
        if distance > 1e-9 {
            for (sum, value) in average.iter_mut().zip(&mut direction) {
                *value /= distance;
                *sum += *value;
            }
        }
        if index < 5 {
            directions.push(direction.clone());
        }
    }
    let coherence =
        (norm_sq_f64(&average, PrecisionMode::FastSimd).sqrt() / count as f64).clamp(0.0, 1.0);
    let mut pair_sum = 0.0;
    let mut pairs = 0;
    for (index, left) in directions.iter().enumerate() {
        for right in &directions[index + 1..] {
            pair_sum += dot_f64(left, right, PrecisionMode::FastSimd).abs();
            pairs += 1;
        }
    }
    let pattern = (pair_sum / f64::from(pairs.max(1))).clamp(0.0, 1.0);
    Handshake {
        pattern,
        novelty: coherence,
        noise: (1.0 - coherence) * (1.0 - pattern),
    }
}

pub fn analyze_pyramid(input: PyramidInput<'_>, options: &PyramidOptions) -> Result<PyramidResult> {
    options.run.check_cancelled()?;
    let original = direction(input.query, input.concepts.dimension(), "query.vector")?;
    pyramid(
        PyramidInput {
            query: original.as_slice(),
            ..input
        },
        &options.config,
        &options.run,
    )
}

pub(super) fn pyramid(
    input: PyramidInput<'_>,
    cfg: &PyramidConfig,
    run: &RunOptions,
) -> Result<PyramidResult> {
    const FIELD: &str = "pyramid.search";
    run.check_cancelled()?;
    for (field, value) in [
        ("pyramid.min_energy_ratio", cfg.min_energy_ratio),
        ("pyramid.min_layer_explained", cfg.min_layer_explained),
        ("pyramid.tail_min_ratio", cfg.tail_min_ratio),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(config(field, "must be in [0, 1]"));
        }
    }
    if cfg.max_levels == 0 || cfg.top_k == 0 || cfg.search.visit_budget == 0 {
        return Err(config(
            "pyramid",
            "max_levels, top_k and visit_budget must be positive",
        ));
    }

    let mut output = PyramidResult {
        novelty: 1.0,
        ..Default::default()
    };
    let precision = PrecisionMode::FastSimd;
    let original_energy = norm_sq_f32(input.query, precision);
    let mut remaining = original_energy;
    let mut residual = input.query.to_vec();
    let mut excluded = Vec::new();
    let mut seen = HashSet::new();
    let mut workspace = nexus_math::NnlsWorkspace::default();
    let mut first_handshake = None;
    for level in 0..cfg.max_levels {
        run.check_cancelled()?;
        let request = ConceptSearchRequest {
            residual: &residual,
            level,
            limit: cfg.top_k,
            excluded: &excluded,
        };
        let candidates = if let Some(search) = input.concept_search {
            search.search(request).map_err(|error| match error {
                Error::Cancelled | Error::External { .. } => error,
                other => Error::External {
                    operation: FIELD,
                    message: other.to_string(),
                },
            })?
        } else {
            input
                .index
                .search_residual(input.concepts, request, input.epa, &cfg.search, run)?
        };
        run.check_cancelled()?;
        if candidates.len() > cfg.top_k {
            return Err(invalid(
                FIELD,
                format!(
                    "returned {} concepts for limit {}",
                    candidates.len(),
                    cfg.top_k
                ),
            ));
        }
        if candidates.is_empty() {
            break;
        }
        let mut tags = Vec::with_capacity(candidates.len());
        let mut vectors = Vec::with_capacity(candidates.len());
        for concept in candidates {
            if !seen.insert(concept) {
                return Err(bad_input(
                    FIELD,
                    Some(concept),
                    "concept is repeated or excluded",
                ));
            }
            let slot = input
                .concepts
                .node(concept)
                .map_err(|_| bad_input(FIELD, Some(concept), "unknown concept"))?;
            let vector = input.concepts.row(slot);
            vectors.push(vector);
            tags.push(PyramidTag {
                concept,
                similarity: cosine_f32(&residual, vector, precision).clamp(-1.0, 1.0),
                contribution: 0.0,
                handshake_magnitude: 0.0,
            });
        }
        let columns: Vec<Vec<f64>> = vectors
            .iter()
            .map(|column| column.iter().copied().map(f64::from).collect())
            .collect();
        let target: Vec<_> = residual.iter().copied().map(f64::from).collect();
        let cone: Vec<_> = tags
            .iter()
            .zip(&columns)
            .map(|(tag, column)| ((tag.concept ^ (1 << 63)) as i64, column.as_slice()))
            .collect();
        let solution = solve_nnls_with_ids_f64(&target, &cone, run, &mut workspace)?;
        let energy = solution.residual_norm.powi(2);
        let explained = (remaining - energy).max(0.0) / original_energy;
        if explained <= 0.0 || explained < cfg.min_layer_explained {
            break;
        }
        for (tag, weight) in tags.iter_mut().zip(solution.direction_weights) {
            tag.contribution = weight;
        }
        excluded.extend(tags.iter().map(|tag| tag.concept));
        if first_handshake.is_none() {
            first_handshake = Some(handshakes(&residual, &mut tags, &vectors));
        } else {
            handshakes(&residual, &mut tags, &vectors);
        }
        output.total_explained_energy += explained;
        output.levels.push(PyramidLevel {
            level,
            tags,
            projection_magnitude: norm_sq_f64(&solution.projection, precision).sqrt(),
            residual_magnitude: solution.residual_norm,
            residual_energy_ratio: energy / original_energy,
            energy_explained: explained,
        });
        let next = solution
            .residual
            .into_iter()
            .map(|value| value as f32)
            .collect();
        output.residuals.push(Residual {
            level,
            vector: std::mem::replace(&mut residual, next),
            is_tail: false,
        });
        remaining = energy;
        if remaining <= 1e-12 || remaining / original_energy < cfg.min_energy_ratio {
            break;
        }
    }
    if let Some(first) = first_handshake {
        if cfg.append_tail && remaining > 1e-12 && remaining / original_energy >= cfg.tail_min_ratio
        {
            output.residuals.push(Residual {
                level: output.levels.len(),
                vector: residual,
                is_tail: true,
            });
        }
        output.coverage = output.total_explained_energy.clamp(0.0, 1.0);
        output.novelty = ((1.0 - output.coverage) * 0.7 + first.novelty * 0.3).clamp(0.0, 1.0);
        output.coherence = first.pattern;
        output.activation =
            (output.coverage * output.coherence * (1.0 - first.noise)).clamp(0.0, 1.0);
    }
    Ok(output)
}
