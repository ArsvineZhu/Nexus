// SPDX-License-Identifier: Apache-2.0

use super::config::{DtscConfig, DtscIdentityConfig};
use super::field::{DtscField, DtscFieldSampler, DtscSample};
use super::path_scoring::{contacts, path, shadow};
use super::{positive, unit};
use nexus_core::kernel::{batch_cosines_f32, cosine_f32_with_norms, norm_sq_f32};
use nexus_core::{
    CandidateId, CandidatePoolView, CsrGraphView, Diagnostics, ExecutionContext, NodeId, Operator,
    PathId, PrecisionMode, VcpError, VectorMatrixView,
};
use std::cmp::Ordering;
use std::time::Instant;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscGeometryState {
    pub epa_logic_depth: f64,
    pub epa_entropy: f64,
    pub epa_resonance: f64,
    pub pyramid_coverage: f64,
    pub pyramid_novelty: f64,
    pub pyramid_depth: f64,
}

impl Default for DtscGeometryState {
    fn default() -> Self {
        Self {
            epa_logic_depth: 0.5,
            epa_entropy: 0.5,
            epa_resonance: 0.0,
            pyramid_coverage: 0.5,
            pyramid_novelty: 0.5,
            pyramid_depth: 0.5,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AuxiliaryEvidence {
    pub shadow_score: f64,
    pub geometry_floor: f64,
    pub identity_floor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtscOrder {
    Score,
    Input,
}

pub struct DtscReadoutInput<'asset, 'query> {
    pub field: &'query DtscField,
    pub candidates: CandidatePoolView<'query>,
    pub graph: CsrGraphView<'asset>,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub original_query: &'query [f32],
    pub enhanced_query: &'query [f32],
    pub geometry: DtscGeometryState,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DtscItem {
    pub candidate: CandidateId,
    pub score: f64,
    pub original_knn_score: f64,
    pub geo_score: f64,
    pub normalized_geo: f64,
    pub geo_bonus: f64,
    pub geo_base_bonus: f64,
    pub geo_aux_bonus: f64,
    pub auxiliary_evidence: Option<AuxiliaryEvidence>,
    pub geo_effect: &'static str,
    pub geo_evidence_class: &'static str,
    pub geo_reward_eligible: bool,
    pub geo_confidence: f64,
    pub geo_exact_hits: usize,
    pub geo_strong_hits: usize,
    pub geo_hit_count: usize,
    pub geo_continuity: f64,
    pub geo_isolated_ratio: f64,
}

impl Default for DtscItem {
    fn default() -> Self {
        Self {
            candidate: CandidateId(0),
            score: 0.0,
            original_knn_score: 0.0,
            geo_score: 0.0,
            normalized_geo: 0.0,
            geo_bonus: 0.0,
            geo_base_bonus: 0.0,
            geo_aux_bonus: 0.0,
            auxiliary_evidence: None,
            geo_effect: "",
            geo_evidence_class: "",
            geo_reward_eligible: false,
            geo_confidence: 0.0,
            geo_exact_hits: 0,
            geo_strong_hits: 0,
            geo_hit_count: 0,
            geo_continuity: 0.0,
            geo_isolated_ratio: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DtscOutput {
    pub results: Vec<DtscItem>,
    pub field_trusted: bool,
    pub field_entropy: f64,
    pub order: DtscOrder,
    pub available: usize,
    pub fallback_used: bool,
    pub fallback_reason: Option<&'static str>,
    pub diagnostics: Diagnostics,
}

pub struct DtscReadoutOperator;

#[derive(Clone, Copy, Debug)]
pub(super) struct DtscEval {
    pub candidate: CandidateId,
    pub score: f64,
    pub original_knn_score: f64,
    pub geo_score: f64,
    pub normalized_geo: f64,
    pub geo_bonus: f64,
    pub geo_base_bonus: f64,
    pub geo_aux_bonus: f64,
    pub auxiliary_evidence: Option<AuxiliaryEvidence>,
    pub geo_effect: &'static str,
    pub geo_evidence_class: &'static str,
    pub geo_reward_eligible: bool,
    pub geo_confidence: f64,
    pub geo_exact_hits: usize,
    pub geo_direct_exact_hits: usize,
    pub geo_emergent_exact_hits: usize,
    pub geo_direct_semantic_hits: usize,
    pub geo_direct_semantic_strength: f64,
    pub geo_strong_hits: usize,
    pub geo_hit_count: usize,
    pub geo_weighted_coverage: f64,
    pub geo_mean_potential: f64,
    pub geo_max_potential: f64,
    pub geo_continuity: f64,
    pub geo_isolated_ratio: f64,
    pub geo_raw_isolated_ratio: f64,
    pub geo_sparse_association_confidence: f64,
    pub geo_sparse_association_pairs: usize,
    pub geo_action_quality: f64,
    pub geo_closure_quality: f64,
    pub geo_direction_consistency: f64,
    pub geo_vector_lift: f64,
    pub geo_direct_score: f64,
    pub geo_structural_score: f64,
    pub geo_thematic_score: f64,
    pub geo_closure_score: f64,
    pub geo_fused_shadow_score: f64,
}

impl Default for DtscEval {
    fn default() -> Self {
        Self {
            candidate: CandidateId(0),
            score: 0.0,
            original_knn_score: 0.0,
            geo_score: 0.0,
            normalized_geo: 0.0,
            geo_bonus: 0.0,
            geo_base_bonus: 0.0,
            geo_aux_bonus: 0.0,
            auxiliary_evidence: None,
            geo_effect: "",
            geo_evidence_class: "",
            geo_reward_eligible: false,
            geo_confidence: 0.0,
            geo_exact_hits: 0,
            geo_direct_exact_hits: 0,
            geo_emergent_exact_hits: 0,
            geo_direct_semantic_hits: 0,
            geo_direct_semantic_strength: 0.0,
            geo_strong_hits: 0,
            geo_hit_count: 0,
            geo_weighted_coverage: 0.0,
            geo_mean_potential: 0.0,
            geo_max_potential: 0.0,
            geo_continuity: 0.0,
            geo_isolated_ratio: 0.0,
            geo_raw_isolated_ratio: 0.0,
            geo_sparse_association_confidence: 0.0,
            geo_sparse_association_pairs: 0,
            geo_action_quality: 0.0,
            geo_closure_quality: 0.0,
            geo_direction_consistency: 0.0,
            geo_vector_lift: 0.0,
            geo_direct_score: 0.0,
            geo_structural_score: 0.0,
            geo_thematic_score: 0.0,
            geo_closure_score: 0.0,
            geo_fused_shadow_score: 0.0,
        }
    }
}

impl DtscEval {
    fn into_item(self) -> DtscItem {
        DtscItem {
            candidate: self.candidate,
            score: self.score,
            original_knn_score: self.original_knn_score,
            geo_score: self.geo_score,
            normalized_geo: self.normalized_geo,
            geo_bonus: self.geo_bonus,
            geo_base_bonus: self.geo_base_bonus,
            geo_aux_bonus: self.geo_aux_bonus,
            auxiliary_evidence: self.auxiliary_evidence,
            geo_effect: self.geo_effect,
            geo_evidence_class: self.geo_evidence_class,
            geo_reward_eligible: self.geo_reward_eligible,
            geo_confidence: self.geo_confidence,
            geo_exact_hits: self.geo_exact_hits,
            geo_strong_hits: self.geo_strong_hits,
            geo_hit_count: self.geo_hit_count,
            geo_continuity: self.geo_continuity,
            geo_isolated_ratio: self.geo_isolated_ratio,
        }
    }
}

pub(super) struct DtscCurveContext<'a> {
    pub config: &'a DtscConfig,
    pub field_samples: &'a [Option<(DtscSample, f64)>],
    pub graph: &'a CsrGraphView<'a>,
    pub tag_vectors: &'a VectorMatrixView<'a, f32>,
    pub geometry: &'a DtscGeometryState,
    pub max_inbound: f64,
}

pub(super) struct Sample<'a> {
    pub node: NodeId,
    pub vector: &'a [f32],
    pub field: &'a DtscSample,
    pub mass: f64,
    pub closure: f64,
    pub specificity: f64,
}

#[inline]
pub(super) fn is_direct(s: &Sample<'_>) -> bool {
    matches!(s.field.source_type.as_str(), "seed" | "core")
}

fn empty_eval(candidate: CandidateId, original_knn_score: f64, [a, b]: [f64; 2]) -> DtscEval {
    let lift = b - a;
    DtscEval {
        candidate,
        score: original_knn_score,
        original_knn_score,
        geo_effect: "neutral",
        geo_evidence_class: "neutral",
        geo_isolated_ratio: 1.0,
        geo_raw_isolated_ratio: 1.0,
        geo_vector_lift: lift,
        geo_closure_score: unit(0.35 * a + 0.45 * b + 0.2 * unit(0.5 + lift * 5.0)),
        ..Default::default()
    }
}

fn collect_samples<'a>(
    chunk_vector: &[f32],
    chunk_norm: f64,
    tags: &[NodeId],
    positions: &[u32],
    ctx: &'a DtscCurveContext<'a>,
    out: &mut Vec<Sample<'a>>,
) {
    out.clear();
    let cfg = &ctx.config.curve;
    let hub_floor = ctx.config.field.public_hub_floor.clamp(0.05, 1.0);
    let closure_denom = (1.0 - cfg.min_closure_similarity).max(1e-6);

    for (&tag, &pos) in tags.iter().zip(positions.iter()) {
        let Some(tag_vec) = ctx.tag_vectors.row(tag.index()) else {
            continue;
        };
        let Some((field, tag_norm)) = &ctx.field_samples[tag.index()] else {
            continue;
        };
        let cosine = cosine_f32_with_norms(
            tag_vec,
            chunk_vector,
            *tag_norm,
            chunk_norm,
            PrecisionMode::StrictLeftFold,
            1e-12,
        );
        let closure = unit((cosine - cfg.min_closure_similarity) / closure_denom);
        let specificity = if ctx.max_inbound <= 0.0 {
            1.0
        } else {
            hub_floor.max(1.0 - unit(ctx.graph.inbound_mass(tag) / ctx.max_inbound).sqrt())
        };
        let anchor = ctx.graph.anchor_gain(tag).clamp(0.5, 2.0);
        let positional = (-cfg.position_decay * positive(f64::from(pos) - 1.0)).exp();
        let mass = (closure * anchor * positional * specificity).max(0.02);
        out.push(Sample {
            node: tag,
            vector: tag_vec,
            field,
            mass,
            closure,
            specificity,
        });
    }
}

fn identity_floor(samples: &[Sample<'_>], cfg: &DtscIdentityConfig) -> f64 {
    let mut maximum = 0.0_f64;
    let mut best: Option<(&Sample<'_>, f64)> = None;

    for s in samples.iter().filter(|s| s.field.exact && is_direct(s)) {
        maximum = maximum.max(s.field.potential);
        let s_strength = unit(s.field.potential * s.specificity * s.closure.sqrt());
        match best {
            None => best = Some((s, s_strength)),
            Some((_, best_strength)) => {
                if s_strength.total_cmp(&best_strength) != Ordering::Less {
                    best = Some((s, s_strength));
                }
            }
        }
    }

    let Some((best_sample, value)) = best else {
        return 0.0;
    };

    if maximum < cfg.min_potential
        || best_sample.specificity < cfg.min_specificity
        || best_sample.closure < cfg.min_tag_chunk_closure
        || value < cfg.min_strength
    {
        0.0
    } else {
        cfg.floor_cap
            * unit((value - cfg.min_strength) / (1.0 - cfg.min_strength).max(1e-6))
                .powf(cfg.floor_exponent)
    }
}

fn reward(samples: &[Sample<'_>], ctx: &DtscCurveContext<'_>, score: &mut DtscEval) {
    let r = &ctx.config.reward;

    let (class, eligible, cap, channel) = if score.geo_direct_exact_hits > 0
        || score.geo_direct_semantic_hits >= r.direct_semantic_min_contacts
    {
        ("direct", true, r.direct_bonus_cap, score.geo_direct_score)
    } else if score.geo_exact_hits > 0
        || (score.geo_strong_hits >= 2
            && score.geo_continuity >= r.structural_continuity_min
            && score.geo_isolated_ratio <= r.thematic_max_isolated_ratio)
    {
        (
            "structural",
            true,
            r.structural_bonus_cap,
            score.geo_structural_score,
        )
    } else {
        (
            "thematic",
            score.geo_max_potential >= r.thematic_min_potential
                && score.geo_isolated_ratio <= r.thematic_max_isolated_ratio,
            r.thematic_bonus_cap,
            score.geo_thematic_score,
        )
    };

    score.geo_evidence_class = class;
    score.geo_reward_eligible = eligible;

    let qualified =
        class == "direct" && score.geo_direct_semantic_hits >= r.direct_semantic_min_contacts;
    let strength = if qualified {
        score.normalized_geo.max(score.geo_direct_semantic_strength)
    } else {
        score.normalized_geo
    };
    let confidence = if qualified {
        score.geo_confidence.max(r.direct_confidence_floor)
    } else {
        score.geo_confidence
    };

    score.geo_base_bonus = if eligible {
        (r.alpha * confidence * strength).min(cap.max(0.0))
    } else {
        0.0
    };

    if let Some(a) = &ctx.config.auxiliary {
        let floor_cap = match class {
            "direct" => a.direct_floor_cap,
            "structural" => a.structural_floor_cap,
            _ => a.thematic_floor_cap,
        };
        let mut geometry_floor = 0.0;
        let mut identity = 0.0;
        if eligible {
            let values = [
                (score.geo_fused_shadow_score, a.min_fused_score),
                (channel, a.min_class_evidence),
                (score.geo_closure_score, a.min_closure_score),
            ];
            if values.iter().all(|(v, min)| v >= min) {
                geometry_floor = floor_cap
                    * values
                        .iter()
                        .map(|(v, min)| unit((v - min) / (1.0 - min).max(1e-6)))
                        .product::<f64>()
                        .cbrt()
                        .powf(a.floor_exponent);
            }
            if let Some(config) = &a.identity_anchor {
                identity = identity_floor(samples, config);
            }
        }
        let target = cap.min(geometry_floor.max(identity));
        score.geo_aux_bonus = a.max_aux_bonus.min(positive(target - score.geo_base_bonus));
        score.auxiliary_evidence = Some(AuxiliaryEvidence {
            shadow_score: score.geo_fused_shadow_score,
            geometry_floor,
            identity_floor: identity,
        });
    }
    score.geo_bonus = cap.min(score.geo_base_bonus + score.geo_aux_bonus);
    score.score = score.original_knn_score + score.geo_bonus;
    score.geo_effect = if score.geo_bonus > 0.0 {
        "boost"
    } else {
        "neutral"
    };
}

#[allow(clippy::too_many_arguments)]
fn eval_path<'a>(
    base: DtscEval,
    query_closures: [f64; 2],
    chunk_vector: &[f32],
    chunk_norm: f64,
    tags: &[NodeId],
    positions: &[u32],
    ctx: &'a DtscCurveContext<'a>,
    samples_buf: &mut Vec<Sample<'a>>,
    isolated_buf: &mut Vec<usize>,
    connected_flags: &mut Vec<bool>,
) -> DtscEval {
    let mut score = base;
    collect_samples(chunk_vector, chunk_norm, tags, positions, ctx, samples_buf);
    if samples_buf.is_empty() {
        return score;
    }

    let weighted = contacts(samples_buf, ctx, &mut score);
    if score.geo_hit_count == 0 {
        return base;
    }

    path(
        samples_buf,
        weighted,
        ctx,
        &mut score,
        isolated_buf,
        connected_flags,
    );

    let target = ctx
        .config
        .curve
        .min_geo_samples
        .min((samples_buf.len() as f64).sqrt().ceil() as usize);
    let evidence =
        unit((score.geo_strong_hits as f64 + 0.75 * score.geo_exact_hits as f64) / target as f64);

    score.geo_confidence = unit(
        score.geo_weighted_coverage
            * (0.55 + 0.45 * evidence)
            * (1.0 - 0.65 * score.geo_isolated_ratio),
    );
    score.geo_score = score.geo_confidence
        * unit(
            0.3 * score.geo_mean_potential
                + 0.2 * score.geo_max_potential
                + 0.2 * score.geo_continuity
                + 0.15 * score.geo_action_quality
                + 0.15 * score.geo_closure_quality,
        );

    let r = &ctx.config.reward;
    score.normalized_geo = unit(
        (score.geo_score - r.geo_reward_floor) / (r.geo_reward_saturation - r.geo_reward_floor),
    );

    if ctx.config.auxiliary.is_some() {
        shadow(query_closures, target, ctx, &mut score);
    }
    reward(samples_buf, ctx, &mut score);
    score
}

fn fallback(scores: &[DtscEval], field: &DtscField, config: &DtscConfig) -> Option<&'static str> {
    if !field.field_trusted {
        return Some("query-energy-field-low-trust");
    }

    let contributors: Vec<&DtscEval> = scores.iter().filter(|s| s.geo_score > 0.0).collect();
    if contributors.is_empty() {
        return Some("no-candidate-curve-contacted-query-field");
    }
    if !config.field.fallback_to_knn_on_low_trust {
        return None;
    }

    let maximum = contributors.iter().map(|s| s.geo_score).fold(0.0, f64::max);
    let minimum = contributors
        .iter()
        .map(|s| s.geo_score)
        .fold(f64::INFINITY, f64::min);
    let evidence: f64 = contributors
        .iter()
        .map(|s| s.geo_strong_hits as f64 + 0.75 * s.geo_exact_hits as f64)
        .sum();
    let low = maximum < config.trust.min_geo_score.max(0.0)
        && evidence < config.trust.min_strong_evidence.max(0.0);

    if low
        && (contributors.len() as f64 / (scores.len().max(1) as f64))
            < config.trust.min_candidate_coverage.clamp(0.0, 1.0)
    {
        Some("candidate-curve-readout-jointly-low-trust")
    } else if low
        && contributors.len() > 1
        && maximum - minimum < config.trust.min_geo_spread.max(0.0)
    {
        Some("candidate-curve-scores-lack-discrimination")
    } else {
        None
    }
}

impl Operator for DtscReadoutOperator {
    type Config = DtscConfig;
    type Input<'asset, 'query>
        = DtscReadoutInput<'asset, 'query>
    where
        Self: 'asset + 'query;
    type Output = DtscOutput;

    fn name(&self) -> &'static str {
        "dtsc_readout"
    }

    fn execute<'asset, 'query>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'asset, 'query>,
    ) -> Result<Self::Output, VcpError> {
        self.readout(ctx, config, input, None)
    }
}

impl DtscReadoutOperator {
    /// Reuse strict left-fold cosine columns in the candidate matrix's row order.
    /// Columns must use the same normalization and squared-norm cutoff as DTSC.
    pub fn execute_with_query_cosines(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &DtscConfig,
        input: DtscReadoutInput<'_, '_>,
        query_cosines: [&[f64]; 2],
    ) -> Result<DtscOutput, VcpError> {
        self.readout(ctx, config, input, Some(query_cosines))
    }

    fn readout(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &DtscConfig,
        input: DtscReadoutInput<'_, '_>,
        query_cosines: Option<[&[f64]; 2]>,
    ) -> Result<DtscOutput, VcpError> {
        const STAGE: &str = "dtsc_readout";
        let start = Instant::now();
        ctx.check_cancelled(STAGE)?;

        input.candidates.validate(STAGE)?;
        input.tag_vectors.validate(STAGE)?;

        let dim = input.tag_vectors.dim;
        if dim == 0 {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "DTSC dimension must be positive".into(),
            });
        }
        if input.candidates.candidate_vectors.dim != dim {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: dim,
                actual: input.candidates.candidate_vectors.dim,
            });
        }
        if input.original_query.len() != dim {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: dim,
                actual: input.original_query.len(),
            });
        }
        if input.enhanced_query.len() != dim {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: dim,
                actual: input.enhanced_query.len(),
            });
        }

        config.validate()?;
        let sampler = DtscFieldSampler::new(input.field, input.graph.inbound, &config.field);
        let max_inbound = sampler.max_inbound;
        let mut field_samples = vec![None; input.tag_vectors.rows];
        let mut neighbors = Vec::new();
        for &node in input.candidates.path_tags {
            if let Some(slot) = field_samples.get_mut(node.index())
                && slot.is_none()
            {
                ctx.check_cancelled(STAGE)?;
                let row = input.tag_vectors.row(node.index()).unwrap();
                *slot = Some((
                    sampler.sample_with_buf(node, row, &mut neighbors),
                    norm_sq_f32(row, PrecisionMode::StrictLeftFold),
                ));
            }
        }
        let candidates = input.candidates.candidate_count;
        let (mut original_buffer, mut enhanced_buffer);
        let [original, enhanced] = if let Some(columns) = query_cosines {
            for column in columns {
                if column.len() != candidates {
                    return Err(VcpError::DimensionMismatch {
                        stage: STAGE,
                        expected: candidates,
                        actual: column.len(),
                    });
                }
            }
            columns
        } else {
            original_buffer = vec![0.0; candidates];
            let original = original_buffer.as_mut_slice();
            enhanced_buffer = vec![0.0; candidates];
            let enhanced = enhanced_buffer.as_mut_slice();
            // A host may have calculated the norms under a different precision policy.
            let candidate_matrix = VectorMatrixView {
                sq_norms_f64: None,
                ..input.candidates.candidate_vectors
            };
            batch_cosines_f32(
                &candidate_matrix,
                &[input.original_query, input.enhanced_query],
                &mut [&mut *original, &mut *enhanced],
                PrecisionMode::StrictLeftFold,
                1e-12,
                ctx.parallel,
            )?;
            [&*original, &*enhanced]
        };
        ctx.check_cancelled(STAGE)?;
        let curve_ctx = DtscCurveContext {
            config,
            field_samples: &field_samples,
            graph: &input.graph,
            tag_vectors: &input.tag_vectors,
            geometry: &input.geometry,
            max_inbound,
        };

        let mut evals = Vec::with_capacity(input.candidates.candidate_count);
        let mut samples_buf = Vec::new();
        let mut isolated_buf = Vec::new();
        let mut connected_flags = Vec::new();

        for c in 0..input.candidates.candidate_count {
            if c % 32 == 0 {
                ctx.check_cancelled(STAGE)?;
            }
            let cand = CandidateId(c as u32);
            let original_knn_score = input.candidates.baseline_scores[c];
            let chunk_vector = input.candidates.candidate_vectors.row(c).unwrap_or(&[]);
            let closures = [original[c], enhanced[c]].map(|cosine| unit((cosine + 1.0) / 2.0));
            let base = empty_eval(cand, original_knn_score, closures);
            let paths = input.candidates.candidate_paths(cand);

            if paths.is_empty() {
                evals.push(base);
            } else {
                let chunk_norm = norm_sq_f32(chunk_vector, PrecisionMode::StrictLeftFold);
                let mut best: Option<DtscEval> = None;
                for p in paths {
                    let (tags, positions) = input.candidates.path_steps(PathId(p as u32));
                    let cand_eval = eval_path(
                        base,
                        closures,
                        chunk_vector,
                        chunk_norm,
                        tags,
                        positions,
                        &curve_ctx,
                        &mut samples_buf,
                        &mut isolated_buf,
                        &mut connected_flags,
                    );
                    match best {
                        None => best = Some(cand_eval),
                        Some(ref prev) => {
                            let cmp = cand_eval
                                .score
                                .total_cmp(&prev.score)
                                .then_with(|| cand_eval.geo_score.total_cmp(&prev.geo_score));
                            if cmp == Ordering::Greater {
                                best = Some(cand_eval);
                            }
                        }
                    }
                }
                if let Some(chosen) = best {
                    evals.push(chosen);
                }
            }
        }

        let reason = fallback(&evals, input.field, config);
        if reason.is_some() {
            for s in &mut evals {
                s.geo_bonus = 0.0;
                s.geo_base_bonus = 0.0;
                s.geo_aux_bonus = 0.0;
                s.score = s.original_knn_score;
                s.geo_effect = "neutral";
            }
        } else {
            evals.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| b.original_knn_score.total_cmp(&a.original_knn_score))
            });
        }

        let available = evals.len();
        let limit = if config.top_k == 0 {
            evals.len()
        } else {
            config.top_k
        };
        evals.truncate(limit.max(1));

        let results = evals.into_iter().map(DtscEval::into_item).collect();

        Ok(DtscOutput {
            results,
            field_trusted: input.field.field_trusted,
            field_entropy: input.field.field_entropy,
            order: if reason.is_some() {
                DtscOrder::Input
            } else {
                DtscOrder::Score
            },
            available,
            fallback_used: reason.is_some(),
            fallback_reason: reason,
            diagnostics: Diagnostics {
                truncated: false,
                truncation_reason: None,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            },
        })
    }
}
