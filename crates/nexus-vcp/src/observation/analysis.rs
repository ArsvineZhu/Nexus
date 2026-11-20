// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{invalid, unit_direction};
use nexus_core::kernel::{cosine_f32, dot_f64, norm_sq_f32, norm_sq_f64};
use nexus_core::nnls::solve_nnls_with_ids_f64;
use nexus_core::{
    EpaBasisView, ExecutionContext, NodeId, PrecisionMode, VcpResult, VectorMatrixView,
};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpaGateMode {
    #[default]
    BackgroundRelative,
    RelativeLinear,
    RelativeSqrt,
    Raw,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EpaConfig {
    pub gate_mode: EpaGateMode,
}

#[derive(Clone, Copy, Debug)]
pub struct EpaAnalyzeInput<'call> {
    pub query: &'call [f32],
    pub basis: Option<EpaBasisView<'call>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpaAxis {
    pub index: usize,
    pub energy: f64,
    pub projection: f64,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpaBridge {
    pub from_axis: usize,
    pub to_axis: usize,
    pub strength: f64,
    pub balance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpaAnalysis {
    pub spectral_concentration: f64,
    pub entropy: f64,
    pub resonance: f64,
    pub dominant_axes: Vec<EpaAxis>,
    pub bridges: Vec<EpaBridge>,
    pub basis_count: usize,
    pub cache_available: bool,
    pub out_of_domain: f64,
    pub semantic_explained_ratio: f64,
    pub in_domain_gate: f64,
    pub baseline_coverage: f64,
    pub background_ratio: f64,
}

impl Default for EpaAnalysis {
    fn default() -> Self {
        Self {
            spectral_concentration: 0.5,
            entropy: 0.5,
            resonance: 0.0,
            dominant_axes: Vec::new(),
            bridges: Vec::new(),
            basis_count: 0,
            cache_available: false,
            out_of_domain: 1.0,
            semantic_explained_ratio: 0.0,
            in_domain_gate: 0.0,
            baseline_coverage: 0.0,
            background_ratio: 0.0,
        }
    }
}

pub fn analyze_epa(
    input: EpaAnalyzeInput<'_>,
    config: &EpaConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<EpaAnalysis> {
    let stage = "epa.analyze";
    ctx.check_cancelled(stage)?;
    let basis = input.basis.filter(|basis| !basis.is_empty());
    unit_direction(
        stage,
        input.query,
        basis.map_or(input.query.len(), |basis| basis.dim),
    )?;
    let Some(basis) = basis else {
        return Ok(EpaAnalysis::default());
    };
    basis.validate(stage)?;
    let mut centered_buffer = vec![0.0; basis.dim];
    let centered = centered_buffer.as_mut_slice();
    for ((slot, &value), &mean) in centered.iter_mut().zip(input.query).zip(basis.mean) {
        *slot = f64::from(value) - mean;
    }
    let count = basis.k_sem.max(1).min(basis.basis_count);
    let mode = PrecisionMode::FastSimd;
    let centered_energy = norm_sq_f64(centered, mode);
    let mut semantic = Vec::with_capacity(count);
    let mut captured = 0.0;
    let mut mean_captured = 0.0;
    for (index, axis) in basis.basis.chunks_exact(basis.dim).enumerate() {
        ctx.check_cancelled(stage)?;
        let projection = dot_f64(centered, axis, mode);
        captured += projection * projection;
        if index < count {
            semantic.push(projection);
            mean_captured += dot_f64(basis.mean, axis, mode).powi(2);
        }
    }
    let energy = semantic.iter().map(|v| v * v).sum::<f64>();
    let raw_background = count as f64 / basis.dim as f64;
    let baseline = if basis.coverage[count - 1] > 0.0 && basis.coverage[count - 1] <= 1.0 {
        basis.coverage[count - 1]
    } else {
        raw_background
    }
    .max(1e-4);
    let background =
        ((raw_background + mean_captured) / (1.0 + norm_sq_f64(basis.mean, mode))).clamp(0.0, 1.0);
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
    let mut dominant_axes = Vec::new();
    if energy > 1e-12 {
        for (index, projection) in semantic.into_iter().enumerate() {
            let probability = projection * projection / energy;
            if probability > 1e-9 {
                entropy -= probability * probability.log2();
            }
            if probability > 0.05 {
                dominant_axes.push(EpaAxis {
                    index,
                    energy: probability,
                    projection,
                    score: probability,
                });
            }
        }
        if basis.k_sem > 1 {
            entropy = (entropy / (basis.k_sem as f64).log2()).clamp(0.0, 1.0);
        }
    } else {
        entropy = 1.0;
    }
    dominant_axes
        .sort_unstable_by(|a, b| b.energy.total_cmp(&a.energy).then(a.index.cmp(&b.index)));
    let mut bridges = Vec::new();
    if let Some(primary) = dominant_axes.first() {
        for secondary in &dominant_axes[1..] {
            let strength = (primary.energy * secondary.energy).sqrt();
            if strength > 0.15 {
                bridges.push(EpaBridge {
                    from_axis: primary.index,
                    to_axis: secondary.index,
                    strength,
                    balance: secondary.energy / primary.energy,
                });
            }
        }
    }
    Ok(EpaAnalysis {
        spectral_concentration: (1.0 - entropy) * gate,
        entropy,
        resonance: bridges.iter().map(|bridge| bridge.strength).sum(),
        dominant_axes,
        bridges,
        basis_count: basis.basis_count,
        cache_available: true,
        out_of_domain: if centered_energy > 1e-12 {
            (1.0 - captured / centered_energy).clamp(0.0, 1.0).sqrt()
        } else {
            0.0
        },
        semantic_explained_ratio: ratio,
        in_domain_gate: gate,
        baseline_coverage: baseline,
        background_ratio: background,
    })
}

/// The host fills up to `limit` graph-local candidates for the current residual.
/// The buffer starts empty and its capacity is reused across layers.
pub type PyramidCandidates<'a> = dyn Fn(usize, &[f32], usize, &mut Vec<NodeId>, Option<&EpaAnalysis>, &[NodeId]) -> VcpResult<()>
    + 'a;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PyramidConfig {
    pub max_levels: usize,
    pub pyramid_top_k: usize,
    pub min_energy_ratio: f64,
    pub min_layer_explained_energy: f64,
    pub tail_residual_min_ratio: f64,
    pub append_tail_residual: bool,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            max_levels: 3,
            pyramid_top_k: 10,
            min_energy_ratio: 0.1,
            min_layer_explained_energy: 0.03,
            tail_residual_min_ratio: 0.02,
            append_tail_residual: true,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PyramidAnalyzeInput<'call> {
    pub query: &'call [f32],
    pub candidates: Option<&'call PyramidCandidates<'call>>,
    pub epa: Option<&'call EpaAnalysis>,
    pub tag_vectors: VectorMatrixView<'call, f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PyramidTagEntry {
    pub node: NodeId,
    pub similarity: f64,
    pub contribution: f64,
    pub handshake_magnitude: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HandshakeFeatures {
    pub direction_coherence: f64,
    pub pattern_strength: f64,
    pub novelty_signal: f64,
    pub noise_signal: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PyramidLevel {
    pub level: usize,
    pub tags: Vec<PyramidTagEntry>,
    pub projection_magnitude: f64,
    pub residual_magnitude: f64,
    pub residual_energy_ratio: f64,
    pub energy_explained: f64,
    pub handshake: HandshakeFeatures,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PyramidResidual {
    pub levels: Vec<PyramidLevel>,
    pub total_explained_energy: f64,
    pub coverage: f64,
    pub novelty: f64,
    pub coherence: f64,
    pub activation: f64,
    pub residuals: Vec<Vec<f32>>,
    pub has_tail_residual: bool,
}

fn handshakes(
    query: &[f32],
    tags: &mut [PyramidTagEntry],
    vectors: VectorMatrixView<'_>,
) -> HandshakeFeatures {
    let mut directions = Vec::with_capacity(tags.len().min(5));
    let mut average = vec![0.0; query.len()];
    let mut direction = vec![0.0; query.len()];
    let count = tags.len();
    for (index, tag) in tags.iter_mut().enumerate() {
        let vector = vectors
            .row(tag.node.index())
            .expect("candidate row was validated");
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
    let pattern = pair_sum / f64::from(pairs.max(1));
    HandshakeFeatures {
        direction_coherence: coherence,
        pattern_strength: pattern,
        novelty_signal: coherence,
        noise_signal: (1.0 - coherence) * (1.0 - pattern),
    }
}

pub fn analyze_pyramid(
    input: PyramidAnalyzeInput<'_>,
    config: &PyramidConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<PyramidResidual> {
    let stage = "pyramid.analyze";
    ctx.check_cancelled(stage)?;
    input.tag_vectors.validate(stage)?;
    unit_direction(stage, input.query, input.tag_vectors.dim)?;
    if ![
        config.min_energy_ratio,
        config.min_layer_explained_energy,
        config.tail_residual_min_ratio,
    ]
    .into_iter()
    .all(|v| (0.0..=1.0).contains(&v))
    {
        return Err(invalid(
            stage,
            "pyramid energy thresholds must be in [0, 1]",
        ));
    }
    let mut output = PyramidResidual {
        novelty: 1.0,
        ..PyramidResidual::default()
    };
    let precision = PrecisionMode::FastSimd;
    let original_energy = norm_sq_f32(input.query, precision);
    let limit = config.pyramid_top_k;
    if config.max_levels == 0 {
        return Err(invalid(stage, "max_levels must be positive"));
    }
    if original_energy <= 1e-12 || limit == 0 {
        return Ok(output);
    }
    let recall = input.candidates.ok_or_else(|| {
        invalid(
            stage,
            "pyramid decomposition requires a concept recall provider",
        )
    })?;
    let mut residual = input.query.to_vec();
    let mut remaining = original_energy;
    let mut candidates = Vec::new();
    let mut excluded = Vec::new();
    let mut columns = Vec::<Vec<f64>>::new();
    let mut seen = std::collections::HashSet::new();
    let mut workspace = nexus_core::NnlsWorkspace::default();
    for level in 0..config.max_levels {
        ctx.check_cancelled(stage)?;
        candidates.clear();
        recall(
            level,
            &residual,
            limit,
            &mut candidates,
            input.epa,
            &excluded,
        )?;
        ctx.check_cancelled(stage)?;
        if candidates.len() > limit {
            return Err(invalid(
                stage,
                "candidate provider exceeded the requested Top-K",
            ));
        }
        seen.clear();
        let mut tags = Vec::with_capacity(candidates.len());
        columns.clear();
        for &node in &candidates {
            let vector = input.tag_vectors.row(node.index()).ok_or_else(|| {
                invalid(
                    stage,
                    "candidate provider returned a node outside the graph",
                )
            })?;
            if seen.insert(node) {
                tags.push(PyramidTagEntry {
                    node,
                    similarity: cosine_f32(&residual, vector, precision),
                    contribution: 0.0,
                    handshake_magnitude: 0.0,
                });
                columns.push(vector.iter().map(|&v| f64::from(v)).collect());
            }
        }
        if tags.is_empty() {
            break;
        }
        let target: Vec<_> = residual.iter().map(|&v| f64::from(v)).collect();
        let cone: Vec<_> = tags
            .iter()
            .zip(&columns)
            .map(|(tag, column)| (i64::from(tag.node.0), column.as_slice()))
            .collect();
        let solution = solve_nnls_with_ids_f64(&target, &cone, limit, &ctx.cancel, &mut workspace)?;
        let energy = solution.residual_norm.powi(2);
        let explained = (remaining - energy).max(0.0) / original_energy;
        if explained <= 0.0 || explained < config.min_layer_explained_energy {
            break;
        }
        for (tag, weight) in tags.iter_mut().zip(solution.direction_weights) {
            tag.contribution = weight;
        }
        excluded.extend(tags.iter().map(|tag| tag.node));
        let handshake = handshakes(&residual, &mut tags, input.tag_vectors);
        output.total_explained_energy += explained;
        output.levels.push(PyramidLevel {
            level,
            tags,
            projection_magnitude: norm_sq_f64(&solution.projection, precision).sqrt(),
            residual_magnitude: solution.residual_norm,
            residual_energy_ratio: energy / original_energy,
            energy_explained: explained,
            handshake,
        });
        let next = solution.residual.into_iter().map(|v| v as f32).collect();
        output
            .residuals
            .push(std::mem::replace(&mut residual, next));
        remaining = energy;
        if remaining <= 1e-12 || remaining / original_energy < config.min_energy_ratio {
            break;
        }
    }
    if let Some(first) = output.levels.first() {
        if config.append_tail_residual
            && remaining > 1e-12
            && remaining / original_energy >= config.tail_residual_min_ratio
        {
            output.residuals.push(residual);
            output.has_tail_residual = true;
        }
        output.coverage = output.total_explained_energy.clamp(0.0, 1.0);
        output.novelty =
            ((1.0 - output.coverage) * 0.7 + first.handshake.novelty_signal * 0.3).clamp(0.0, 1.0);
        output.coherence = first.handshake.pattern_strength;
        output.activation =
            (output.coverage * output.coherence * (1.0 - first.handshake.noise_signal))
                .clamp(0.0, 1.0);
    }
    Ok(output)
}
