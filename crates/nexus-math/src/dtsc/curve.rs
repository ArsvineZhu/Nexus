// SPDX-License-Identifier: Apache-2.0

use super::config::{DtscEvidenceConfig, DtscIdentityConfig};
use super::field::{EvidenceSource, GraphView, Sample as FieldSample};
use super::path_scoring::{contacts, path, shadow};
use super::{positive, unit};
use crate::kernel::{PrecisionMode, cosine_f32_with_norms};
use crate::{Result, RunOptions, VectorRows};
use std::collections::HashMap;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GeometryState {
    pub epa_concentration: f64,
    pub epa_entropy: f64,
    pub epa_resonance: f64,
    pub pyramid_coverage: f64,
    pub pyramid_novelty: f64,
    pub pyramid_depth: f64,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AuxiliaryEvidence {
    pub shadow_score: Option<f64>,
    pub geometry_floor: f64,
    pub identity_floor: f64,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    Direct,
    Structural,
    Thematic,
    #[default]
    None,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveMetrics {
    pub geo_score: f64,
    pub normalized_geo: f64,
    pub base_bonus: f64,
    pub aux_bonus: f64,
    pub geo_bonus: f64,
    pub confidence: f64,
    pub continuity: f64,
    pub isolated_ratio: f64,
    pub class: EvidenceClass,
    pub eligible: bool,
    pub exact_hits: usize,
    pub strong_hits: usize,
    pub hit_count: usize,
    pub auxiliary: Option<AuxiliaryEvidence>,
}

impl Default for CurveMetrics {
    fn default() -> Self {
        Self {
            geo_score: 0.0,
            normalized_geo: 0.0,
            base_bonus: 0.0,
            aux_bonus: 0.0,
            geo_bonus: 0.0,
            confidence: 0.0,
            continuity: 0.0,
            isolated_ratio: 1.0,
            class: EvidenceClass::None,
            eligible: false,
            exact_hits: 0,
            strong_hits: 0,
            hit_count: 0,
            auxiliary: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DtscEval {
    pub metrics: CurveMetrics,
    pub geo_direct_exact_hits: usize,
    pub geo_indirect_exact_hits: usize,
    pub geo_direct_semantic_hits: usize,
    pub geo_direct_semantic_strength: f64,
    pub geo_weighted_coverage: f64,
    pub geo_mean_potential: f64,
    pub geo_max_potential: f64,
    pub geo_action_quality: f64,
    pub geo_closure_quality: f64,
    pub geo_direction_consistency: f64,
    pub geo_direct_score: f64,
    pub geo_structural_score: f64,
    pub geo_thematic_score: f64,
    pub geo_closure_score: f64,
    pub geo_fused_shadow_score: f64,
}

pub(super) struct DtscCurveContext<'a> {
    pub config: &'a DtscEvidenceConfig,
    pub run: &'a RunOptions,
    pub field_samples: &'a HashMap<u64, (usize, FieldSample, f64)>,
    pub graph: Option<GraphView<'a>>,
    pub tag_vectors: VectorRows<'a>,
    pub geometry: Option<GeometryState>,
}

pub(super) struct Sample {
    pub node: u64,
    pub row: usize,
    pub field: FieldSample,
    pub mass: f64,
    pub closure: f64,
    pub specificity: f64,
}

pub(super) fn is_direct(sample: &Sample) -> bool {
    matches!(
        sample.field.kind,
        Some(EvidenceSource::Core | EvidenceSource::Lexical)
    )
}

pub(super) type TagBodyCosine<'a> = dyn Fn(u64, u64) -> Option<f64> + Sync + 'a;

pub(super) struct CurveBody<'a> {
    pub candidate: u64,
    pub query_closures: (f64, Option<f64>),
    pub vector: &'a [f32],
    pub norm: f64,
    pub cached_cosines: Option<&'a TagBodyCosine<'a>>,
}

fn collect_samples(
    body: &CurveBody<'_>,
    tags: &[u64],
    ctx: &DtscCurveContext<'_>,
    out: &mut Vec<Sample>,
) -> Result<()> {
    out.clear();
    let cfg = &ctx.config.curve;
    for (position, &tag) in tags.iter().enumerate() {
        if position.is_multiple_of(64) {
            ctx.run.check_cancelled()?;
        }
        let (row, field, tag_norm) = &ctx.field_samples[&tag];
        let tag_vec = ctx.tag_vectors.row(*row).expect("validated concept row");
        let cosine = body
            .cached_cosines
            .and_then(|lookup| lookup(body.candidate, tag))
            .unwrap_or_else(|| {
                cosine_f32_with_norms(
                    tag_vec,
                    body.vector,
                    *tag_norm,
                    body.norm,
                    PrecisionMode::StrictLeftFold,
                    0.0,
                )
            });
        let closure = if cfg.min_closure_similarity == 1.0 {
            f64::from(cosine >= 1.0)
        } else {
            unit((cosine - cfg.min_closure_similarity) / (1.0 - cfg.min_closure_similarity))
        };
        let anchor = ctx.graph.map_or(1.0, |g| g.anchor_gain(tag));
        let positional = (-cfg.position_decay * position as f64).exp();
        let mass = (closure * anchor * positional * field.specificity).max(0.02);
        out.push(Sample {
            node: tag,
            row: *row,
            field: *field,
            mass,
            closure,
            specificity: field.specificity,
        });
    }
    Ok(())
}

fn identity_floor(samples: &[Sample], cfg: &DtscIdentityConfig) -> f64 {
    let strength = samples
        .iter()
        .filter(|s| {
            s.field.exact
                && is_direct(s)
                && s.field.potential >= cfg.min_potential
                && s.specificity >= cfg.min_specificity
                && s.closure >= cfg.min_tag_chunk_closure
        })
        .map(|s| unit(s.field.potential * s.specificity * s.closure.sqrt()))
        .fold(0.0_f64, f64::max);
    cfg.floor_cap
        * unit((strength - cfg.min_strength) / (1.0 - cfg.min_strength)).powf(cfg.floor_exponent)
}

fn reward(samples: &[Sample], ctx: &DtscCurveContext<'_>, score: &mut DtscEval) {
    let r = &ctx.config.reward;

    let (class, eligible, cap, channel) = if score.geo_direct_exact_hits > 0
        || score.geo_direct_semantic_hits >= r.direct_semantic_min_contacts
    {
        (
            EvidenceClass::Direct,
            true,
            r.direct_bonus_cap,
            score.geo_direct_score,
        )
    } else if score.metrics.exact_hits > 0
        || (score.metrics.strong_hits >= 2
            && score.metrics.continuity >= r.structural_continuity_min
            && score.metrics.isolated_ratio <= r.thematic_max_isolated_ratio)
    {
        (
            EvidenceClass::Structural,
            true,
            r.structural_bonus_cap,
            score.geo_structural_score,
        )
    } else {
        (
            EvidenceClass::Thematic,
            score.geo_max_potential >= r.thematic_min_potential
                && score.metrics.isolated_ratio <= r.thematic_max_isolated_ratio,
            r.thematic_bonus_cap,
            score.geo_thematic_score,
        )
    };

    score.metrics.class = class;
    score.metrics.eligible = eligible;

    let qualified = class == EvidenceClass::Direct
        && score.geo_direct_semantic_hits >= r.direct_semantic_min_contacts;
    let strength = if qualified {
        score
            .metrics
            .normalized_geo
            .max(score.geo_direct_semantic_strength)
    } else {
        score.metrics.normalized_geo
    };
    let confidence = if qualified {
        score.metrics.confidence.max(r.direct_confidence_floor)
    } else {
        score.metrics.confidence
    };

    score.metrics.base_bonus = if eligible {
        (r.alpha * confidence * strength).min(cap.max(0.0))
    } else {
        0.0
    };

    if let Some(aux) = &ctx.config.auxiliary {
        let (shadow_score, geometry_floor) = if let Some(geometry) = &aux.geometry {
            let floor_cap = match class {
                EvidenceClass::Direct => geometry.direct_floor_cap,
                EvidenceClass::Structural => geometry.structural_floor_cap,
                EvidenceClass::Thematic => geometry.thematic_floor_cap,
                EvidenceClass::None => 0.0,
            };
            let values = [
                (score.geo_fused_shadow_score, geometry.min_fused_score),
                (channel, geometry.min_class_evidence),
                (score.geo_closure_score, geometry.min_closure_score),
            ];
            let floor = if eligible {
                floor_cap
                    * values
                        .iter()
                        .map(|(v, min)| unit((v - min) / (1.0 - min)))
                        .product::<f64>()
                        .cbrt()
                        .powf(geometry.floor_exponent)
            } else {
                0.0
            };
            (Some(score.geo_fused_shadow_score), floor)
        } else {
            (None, 0.0)
        };
        let identity_floor = if eligible {
            aux.identity
                .as_ref()
                .map_or(0.0, |config| identity_floor(samples, config))
        } else {
            0.0
        };
        let target = cap.min(geometry_floor.max(identity_floor));
        score.metrics.aux_bonus = aux
            .max_aux_bonus
            .min(positive(target - score.metrics.base_bonus));
        score.metrics.auxiliary = Some(AuxiliaryEvidence {
            shadow_score,
            geometry_floor,
            identity_floor,
        });
    }
    score.metrics.geo_bonus = cap.min(score.metrics.base_bonus + score.metrics.aux_bonus);
}

#[derive(Default)]
pub(super) struct CurveScratch {
    pub samples: Vec<Sample>,
    pub isolated: Vec<usize>,
    pub connected: Vec<bool>,
}

pub(super) fn eval_path(
    body: &CurveBody<'_>,
    tags: &[u64],
    ctx: &DtscCurveContext<'_>,
    scratch: &mut CurveScratch,
) -> Result<CurveMetrics> {
    let mut score = DtscEval::default();
    let CurveScratch {
        samples,
        isolated,
        connected,
    } = scratch;
    collect_samples(body, tags, ctx, samples)?;
    if samples.is_empty() {
        return Ok(score.metrics);
    }
    let weighted = contacts(samples, ctx, &mut score);
    if score.metrics.hit_count == 0 && score.metrics.exact_hits == 0 {
        return Ok(score.metrics);
    }
    path(samples, weighted, ctx, &mut score, isolated, connected)?;

    let target = ctx
        .config
        .curve
        .min_geo_samples
        .min((samples.len() as f64).sqrt().ceil() as usize);
    let evidence = unit(
        (score.metrics.strong_hits as f64 + 0.75 * score.metrics.exact_hits as f64) / target as f64,
    );

    score.metrics.confidence = unit(
        score.geo_weighted_coverage
            * (0.55 + 0.45 * evidence)
            * (1.0 - 0.65 * score.metrics.isolated_ratio),
    );
    score.metrics.geo_score = score.metrics.confidence
        * unit(
            0.3 * score.geo_mean_potential
                + 0.2 * score.geo_max_potential
                + 0.2 * score.metrics.continuity
                + 0.15 * score.geo_action_quality
                + 0.15 * score.geo_closure_quality,
        );

    let r = &ctx.config.reward;
    score.metrics.normalized_geo = unit(
        (score.metrics.geo_score - r.geo_reward_floor)
            / (r.geo_reward_saturation - r.geo_reward_floor),
    );

    if let Some(geometry) = ctx.geometry {
        shadow(body.query_closures, target, ctx, geometry, &mut score);
    }
    reward(samples, ctx, &mut score);
    Ok(score.metrics)
}
