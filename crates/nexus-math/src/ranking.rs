// SPDX-License-Identifier: Apache-2.0

use nexus_core::{ExecutionContext, VcpError, VcpResult};
use std::collections::{HashMap, HashSet};

/// An ordered ranking with an explicit, unnormalized contribution weight.
#[derive(Clone, Copy, Debug)]
pub struct WeightedRanking<'call> {
    pub ids: &'call [u64],
    pub weight: f64,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RrfConfig {
    pub k: f64,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RrfItem {
    pub id: u64,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RrfOutput {
    pub items: Vec<RrfItem>,
}

pub fn rrf<'call>(
    rankings: &[WeightedRanking<'call>],
    config: &RrfConfig,
    ctx: &ExecutionContext<'_>,
) -> VcpResult<RrfOutput> {
    let stage = "rrf";
    let invalid = |detail: String| VcpError::InvalidInput { stage, detail };
    ctx.check_cancelled(stage)?;
    if !config.k.is_finite() || config.k <= 0.0 {
        return Err(invalid("k must be finite and positive".into()));
    }
    if rankings
        .iter()
        .any(|ranking| !ranking.weight.is_finite() || ranking.weight < 0.0)
    {
        return Err(invalid("weights must be finite and nonnegative".into()));
    }
    if !rankings.is_empty() && rankings.iter().all(|ranking| ranking.weight == 0.0) {
        return Err(invalid(
            "at least one ranking weight must be positive".into(),
        ));
    }

    let mut items = Vec::<RrfItem>::new();
    let mut indices = HashMap::new();
    let mut seen = HashSet::new();
    for (list, ranking) in rankings.iter().enumerate() {
        ctx.check_cancelled(stage)?;
        seen.clear();
        for (row, &id) in ranking.ids.iter().enumerate() {
            if row % 256 == 0 {
                ctx.check_cancelled(stage)?;
            }
            if !seen.insert(id) {
                return Err(invalid(format!(
                    "ranking {list} contains duplicate ID {id}"
                )));
            }
            let index = *indices.entry(id).or_insert_with(|| {
                let index = items.len();
                items.push(RrfItem { id, score: 0.0 });
                index
            });
            items[index].score += ranking.weight / (config.k + (row + 1) as f64);
        }
    }
    items.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(RrfOutput { items })
}
