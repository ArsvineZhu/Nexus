// SPDX-License-Identifier: Apache-2.0

use super::{DtscEvidenceValues, DtscItem, DtscReadoutConfig};
use nexus_core::{Diagnostics, ExecutionContext, Operator, VcpResult};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtscOrder {
    Score,
    Input,
}

pub struct DtscReadoutInput<'call> {
    pub evidence: &'call DtscEvidenceValues,
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

impl Operator for DtscReadoutOperator {
    type Config = DtscReadoutConfig;
    type Input<'call> = DtscReadoutInput<'call>;
    type Output = DtscOutput;

    fn name(&self) -> &'static str {
        "dtsc_readout"
    }

    fn execute<'call>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'call>,
    ) -> VcpResult<Self::Output> {
        let start = Instant::now();
        ctx.check_cancelled(self.name())?;
        config.validate()?;
        let evidence = input.evidence;
        let field_trusted = evidence.field_trusted(config.fallback_on_low_trust);
        let reason = fallback(evidence, field_trusted, config);
        let available = evidence.items().len();
        let limit = if config.top_k == 0 {
            available
        } else {
            config.top_k
        };

        let (results, order) = if reason.is_some() {
            let results = evidence
                .items()
                .iter()
                .take(limit)
                .cloned()
                .map(|mut item| {
                    item.geo_bonus = 0.0;
                    item.geo_base_bonus = 0.0;
                    item.geo_aux_bonus = 0.0;
                    item.score = item.original_cosine;
                    item.geo_effect = "neutral";
                    item
                })
                .collect();
            (results, DtscOrder::Input)
        } else {
            let mut results = evidence.items().to_vec();
            results.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| b.original_cosine.total_cmp(&a.original_cosine))
            });
            results.truncate(limit);
            (results, DtscOrder::Score)
        };

        Ok(DtscOutput {
            results,
            field_trusted,
            field_entropy: evidence.field_entropy(),
            order,
            available,
            fallback_used: reason.is_some(),
            fallback_reason: reason,
            diagnostics: Diagnostics {
                elapsed_ms: start.elapsed().as_secs_f64() * 1_000.0,
                ..Diagnostics::default()
            },
        })
    }
}

fn fallback(
    evidence: &DtscEvidenceValues,
    field_trusted: bool,
    config: &DtscReadoutConfig,
) -> Option<&'static str> {
    if !field_trusted {
        return Some("query-energy-field-low-trust");
    }
    let stats = &evidence.stats;
    if stats.contributors == 0 {
        return Some("no-candidate-curve-contacted-query-field");
    }
    if !config.fallback_on_low_trust {
        return None;
    }

    let trust = &config.trust;
    let low =
        stats.maximum < trust.min_geo_score && stats.strong_evidence < trust.min_strong_evidence;
    let coverage = stats.contributors as f64 / evidence.items().len() as f64;
    if low && coverage < trust.min_candidate_coverage {
        Some("candidate-curve-readout-jointly-low-trust")
    } else if low && stats.contributors > 1 && stats.maximum - stats.minimum < trust.min_geo_spread
    {
        Some("candidate-curve-scores-lack-discrimination")
    } else {
        None
    }
}
