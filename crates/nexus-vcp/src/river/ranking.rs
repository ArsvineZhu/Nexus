// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{ReadoutItem, RiverReadoutOutput};
use crate::observation::invalid;
use nexus_core::{CandidateId, Diagnostics, ExecutionContext, Operator, ScratchPad, VcpResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FusedRanking {
    pub items: Vec<ReadoutItem>,
    pub rrf_active: bool,
    pub diagnostics: Diagnostics,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OmegaRrfFuseConfig {
    pub rrf_k: f64,
}

impl Default for OmegaRrfFuseConfig {
    fn default() -> Self {
        Self { rrf_k: 60.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OmegaRrfFuseInput<'asset, 'query> {
    pub baseline_readout: &'asset RiverReadoutOutput,
    pub external_ranked: &'query [(CandidateId, f64)],
}

pub struct OmegaRrfFuseOperator;

impl Operator for OmegaRrfFuseOperator {
    type Config = OmegaRrfFuseConfig;
    type Input<'asset, 'query> = OmegaRrfFuseInput<'asset, 'query>;
    type Output = FusedRanking;

    fn name(&self) -> &'static str {
        "ranking.omega_rrf"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        _: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        if !config.rrf_k.is_finite() || config.rrf_k <= 0.0 {
            return Err(invalid(stage, "rrf_k must be positive and finite"));
        }
        let baseline = input.baseline_readout;
        let mut result = FusedRanking {
            items: baseline.items.clone(),
            rrf_active: false,
            diagnostics: baseline.diagnostics.clone(),
        };
        let mut external = input.external_ranked.to_vec();
        if external.iter().any(|&(_, score)| !score.is_finite()) {
            return Err(invalid(
                stage,
                "external ranking contains a non-finite score",
            ));
        }
        external.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut ranks = HashMap::new();
        for (candidate, _) in external {
            let next_rank = ranks.len() + 1;
            ranks.entry(candidate).or_insert(next_rank);
        }
        if !result
            .items
            .iter()
            .any(|item| ranks.contains_key(&item.candidate))
        {
            return Ok(result);
        }
        let alpha = if (0.0..=1.0).contains(&baseline.omega.omega) {
            1.0 - (0.5 + 0.8 * (baseline.omega.omega - 0.65)).clamp(0.3, 0.7)
        } else {
            0.5
        };
        for (index, item) in result.items.iter_mut().enumerate() {
            ctx.check_cancelled(stage)?;
            let base = (1.0 - alpha) / (config.rrf_k + (index + 1) as f64);
            item.ranking_score = base
                + ranks
                    .get(&item.candidate)
                    .map_or(0.0, |&rank| alpha / (config.rrf_k + rank as f64));
        }
        result
            .items
            .sort_by(|a, b| b.ranking_score.total_cmp(&a.ranking_score));
        result.rrf_active = true;
        Ok(result)
    }
}
