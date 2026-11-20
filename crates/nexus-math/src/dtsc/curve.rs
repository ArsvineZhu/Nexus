// SPDX-License-Identifier: Apache-2.0

use super::config::{DtscEvidenceConfig, DtscIdentityConfig};
use super::evidence::DtscItem;
use super::field::{DtscSample, DtscSourceType};
use super::path_scoring::{contacts, path, shadow};
use super::{positive, unit};
use nexus_core::kernel::cosine_f32_with_norms;
use nexus_core::{
    CandidateId, NodeId, PathId, PrecisionMode, TransportGraphView, VectorMatrixView,
};
use std::cmp::Ordering;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscGeometryState {
    pub epa_spectral_concentration: f64,
    pub epa_entropy: f64,
    pub epa_resonance: f64,
    pub pyramid_coverage: f64,
    pub pyramid_novelty: f64,
    pub pyramid_depth: f64,
}

impl Default for DtscGeometryState {
    fn default() -> Self {
        Self {
            epa_spectral_concentration: 0.5,
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

#[derive(Clone, Copy, Debug)]
pub(super) struct DtscEval {
    pub candidate: CandidateId,
    pub chosen_path: Option<PathId>,
    pub score: f64,
    pub original_cosine: f64,
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
    pub geo_indirect_exact_hits: usize,
    pub geo_direct_semantic_hits: usize,
    pub geo_direct_semantic_strength: f64,
    pub geo_strong_hits: usize,
    pub geo_hit_count: usize,
    pub geo_weighted_coverage: f64,
    pub geo_mean_potential: f64,
    pub geo_max_potential: f64,
    pub geo_continuity: f64,
    pub geo_isolated_ratio: f64,
    pub geo_action_quality: f64,
    pub geo_closure_quality: f64,
    pub geo_direction_consistency: f64,
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
            chosen_path: None,
            score: 0.0,
            original_cosine: 0.0,
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
            geo_indirect_exact_hits: 0,
            geo_direct_semantic_hits: 0,
            geo_direct_semantic_strength: 0.0,
            geo_strong_hits: 0,
            geo_hit_count: 0,
            geo_weighted_coverage: 0.0,
            geo_mean_potential: 0.0,
            geo_max_potential: 0.0,
            geo_continuity: 0.0,
            geo_isolated_ratio: 0.0,
            geo_action_quality: 0.0,
            geo_closure_quality: 0.0,
            geo_direction_consistency: 0.0,
            geo_direct_score: 0.0,
            geo_structural_score: 0.0,
            geo_thematic_score: 0.0,
            geo_closure_score: 0.0,
            geo_fused_shadow_score: 0.0,
        }
    }
}

impl DtscEval {
    pub(super) fn into_item(self) -> DtscItem {
        DtscItem {
            candidate: self.candidate,
            chosen_path: self.chosen_path,
            scores: super::DtscScores {
                score: self.score,
                original_cosine: self.original_cosine,
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
            },
        }
    }
}

pub(super) struct DtscCurveContext<'a> {
    pub config: &'a DtscEvidenceConfig,
    pub hub_floor: f64,
    pub field_samples: &'a [Option<(DtscSample, f64)>],
    pub graph: TransportGraphView<'a>,
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
    matches!(
        s.field.source_type,
        DtscSourceType::Seed | DtscSourceType::Core
    )
}

pub(super) fn empty_eval(candidate: CandidateId, original_cosine: f64) -> DtscEval {
    DtscEval {
        candidate,
        score: original_cosine,
        original_cosine,
        geo_effect: "neutral",
        geo_evidence_class: "neutral",
        geo_isolated_ratio: 1.0,
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
            ctx.hub_floor
                .max(1.0 - unit(ctx.graph.inbound_mass(tag) / ctx.max_inbound).sqrt())
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
    score.score = score.original_cosine + score.geo_bonus;
    score.geo_effect = if score.geo_bonus > 0.0 {
        "boost"
    } else {
        "neutral"
    };
}

#[allow(clippy::too_many_arguments)]
pub(super) fn eval_path<'a>(
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
