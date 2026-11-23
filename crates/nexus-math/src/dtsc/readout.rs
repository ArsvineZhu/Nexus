// SPDX-License-Identifier: Apache-2.0

use super::{DtscReadoutConfig, Evidence};
use crate::{Error, Result, RunOptions, Scores};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtscFallbackReason {
    InsufficientFieldSupport,
    LowFieldEntropy,
    NoContact,
    InsufficientCoverage,
    InsufficientSpread,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtscDecision {
    Applied,
    Baseline { reason: DtscFallbackReason },
}

#[derive(Clone, Debug, Default)]
pub struct ReadoutOptions {
    pub config: DtscReadoutConfig,
    pub run: RunOptions,
}

#[derive(Clone, Debug)]
pub struct Readout {
    pub scores: Scores,
    pub decision: DtscDecision,
}

impl Evidence {
    pub fn apply(&self, baseline: &Scores, options: &ReadoutOptions) -> Result<Readout> {
        let run = &options.run;
        let config = &options.config;
        run.check_cancelled()?;
        config.validate()?;
        if self.items().len() != baseline.iter().len() {
            return Err(Error::dimension(
                "baseline",
                self.items().len(),
                baseline.iter().len(),
            ));
        }

        let decision = fallback(self, config).map_or(DtscDecision::Applied, |reason| {
            DtscDecision::Baseline { reason }
        });
        let gain = if decision == DtscDecision::Applied {
            config.correction_gain
        } else {
            0.0
        };
        let scores = baseline
            .values()
            .enumerate()
            .map(|(index, (id, value))| {
                if index.is_multiple_of(64) {
                    run.check_cancelled()?;
                }
                let bonus = self
                    .item(id)
                    .ok_or_else(|| Error::Input {
                        field: "baseline",
                        id: Some(id),
                        message: "score for an unknown candidate".into(),
                    })?
                    .metrics
                    .geo_bonus;
                let adjusted = if gain == 0.0 || bonus == 0.0 {
                    value
                } else {
                    value + gain * bonus
                };
                if adjusted.is_finite() {
                    Ok((id, adjusted))
                } else {
                    Err(Error::numeric(
                        "dtsc.apply",
                        format!("adjusted score overflowed for candidate {id}"),
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        run.check_cancelled()?;
        Ok(Readout {
            scores: Scores::new(scores)?,
            decision,
        })
    }
}

fn fallback(evidence: &Evidence, config: &DtscReadoutConfig) -> Option<DtscFallbackReason> {
    use DtscFallbackReason::*;
    if !evidence.field_has_support() {
        return Some(InsufficientFieldSupport);
    }
    if config.fallback_on_low_trust && evidence.field.entropy < config.trust.min_field_entropy {
        return Some(LowFieldEntropy);
    }
    let stats = &evidence.stats;
    if stats.contributors == 0 {
        return Some(NoContact);
    }
    if !config.fallback_on_low_trust {
        return None;
    }

    let trust = &config.trust;
    let low =
        stats.maximum < trust.min_geo_score && stats.strong_evidence < trust.min_strong_evidence;
    let coverage = stats.contributors as f64 / evidence.items().len() as f64;
    if low && coverage < trust.min_candidate_coverage {
        Some(InsufficientCoverage)
    } else if low && stats.contributors > 1 && stats.maximum - stats.minimum < trust.min_geo_spread
    {
        Some(InsufficientSpread)
    } else {
        None
    }
}
