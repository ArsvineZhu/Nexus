// SPDX-License-Identifier: Apache-2.0

use super::{DtscEvidenceValues, DtscFallbackOrder, DtscFieldStats, DtscItem, DtscReadoutConfig};
use nexus_core::{ExecutionContext, VcpResult};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtscFallbackReason {
    InsufficientFieldSupport,
    LowFieldEntropy,
    NoCurveContact,
    InsufficientCandidateCoverage,
    InsufficientScoreSpread,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtscDecision {
    Scored,
    Fallback {
        reason: DtscFallbackReason,
        order: DtscFallbackOrder,
    },
}

#[derive(Clone, Debug)]
pub struct DtscOutput {
    pub results: Vec<DtscItem>,
    pub field: DtscFieldStats,
    pub decision: DtscDecision,
    pub available: usize,
    pub elapsed_ms: f64,
}

pub fn readout_dtsc(
    evidence: &DtscEvidenceValues,
    config: &DtscReadoutConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<DtscOutput> {
    let start = Instant::now();
    ctx.check_cancelled("dtsc_readout")?;
    config.validate()?;
    let reason = fallback(evidence, config);
    let available = evidence.items().len();
    let limit = if config.top_k == 0 {
        available
    } else {
        config.top_k
    };

    let keep_input_order =
        reason.is_some() && config.fallback_order == DtscFallbackOrder::InputOrder;
    let mut results = if keep_input_order {
        evidence.items().iter().take(limit).cloned().collect()
    } else {
        evidence.items().to_vec()
    };
    let geometry_weight = if reason.is_some() {
        0.0
    } else {
        config.geometry_weight
    };
    for item in &mut results {
        let scores = &mut item.scores;
        scores.geo_bonus *= geometry_weight;
        scores.geo_base_bonus *= geometry_weight;
        scores.geo_aux_bonus *= geometry_weight;
        scores.score = scores.original_cosine + scores.geo_bonus;
        scores.geo_effect = if scores.geo_bonus > 0.0 {
            "boost"
        } else {
            "neutral"
        };
    }
    if !keep_input_order {
        results.sort_by(|a, b| {
            b.scores.score.total_cmp(&a.scores.score).then_with(|| {
                b.scores
                    .original_cosine
                    .total_cmp(&a.scores.original_cosine)
            })
        });
        results.truncate(limit);
    }
    let decision = reason.map_or(DtscDecision::Scored, |reason| DtscDecision::Fallback {
        reason,
        order: config.fallback_order,
    });

    ctx.check_cancelled("dtsc_readout")?;
    Ok(DtscOutput {
        results,
        field: evidence.field_stats(),
        decision,
        available,
        elapsed_ms: start.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn fallback(
    evidence: &DtscEvidenceValues,
    config: &DtscReadoutConfig,
) -> Option<DtscFallbackReason> {
    use DtscFallbackReason::*;
    if !evidence.field_has_support() {
        return Some(InsufficientFieldSupport);
    }
    if config.fallback_on_low_trust && evidence.field.entropy < config.trust.min_field_entropy {
        return Some(LowFieldEntropy);
    }
    let stats = &evidence.stats;
    if stats.contributors == 0 {
        return Some(NoCurveContact);
    }
    if !config.fallback_on_low_trust {
        return None;
    }

    let trust = &config.trust;
    let low =
        stats.maximum < trust.min_geo_score && stats.strong_evidence < trust.min_strong_evidence;
    let coverage = stats.contributors as f64 / evidence.items().len() as f64;
    if low && coverage < trust.min_candidate_coverage {
        Some(InsufficientCandidateCoverage)
    } else if low && stats.contributors > 1 && stats.maximum - stats.minimum < trust.min_geo_spread
    {
        Some(InsufficientScoreSpread)
    } else {
        None
    }
}
