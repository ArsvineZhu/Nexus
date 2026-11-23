// SPDX-License-Identifier: Apache-2.0

use crate::kernel::{PrecisionMode, cosine_f32_with_norms, norm_sq_f32};
use crate::{Error, Result, RunOptions, VectorRows};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreRow {
    pub id: u64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scores(Vec<ScoreRow>);

impl Scores {
    pub fn new(values: impl IntoIterator<Item = (u64, f64)>) -> Result<Self> {
        let mut seen = HashSet::new();
        let rows = values
            .into_iter()
            .map(|(id, value)| {
                if !value.is_finite() || !seen.insert(id) {
                    return Err(Error::Input {
                        field: "scores",
                        id: Some(id),
                        message: "score must be finite and ID unique".into(),
                    });
                }
                Ok(ScoreRow { id, value })
            })
            .collect::<Result<_>>()?;
        Ok(Self(rows))
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = (u64, f64)> + '_ {
        self.0.iter().map(|row| (row.id, row.value))
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ScoreRow> + '_ {
        self.0.iter()
    }

    pub fn top(&self, limit: Option<usize>) -> Ranking {
        let order = rank_order(self.0.len(), limit, |a, b| {
            self.0[b]
                .value
                .partial_cmp(&self.0[a].value)
                .expect("finite scores")
        });
        let rows: Vec<_> = order.into_iter().map(|index| self.0[index]).collect();
        let ids = rows.iter().map(|row| row.id).collect();
        Ranking { rows, ids }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ranking {
    rows: Vec<ScoreRow>,
    ids: Vec<u64>,
}

impl Ranking {
    pub fn rows(&self) -> &[ScoreRow] {
        &self.rows
    }

    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    pub fn into_rows(self) -> Vec<ScoreRow> {
        self.rows
    }
}

/// Select positions for a stable ranking. The comparator orders best positions first.
#[doc(hidden)]
pub fn rank_order(
    len: usize,
    limit: Option<usize>,
    compare: impl Fn(usize, usize) -> Ordering,
) -> Vec<usize> {
    let take = limit.unwrap_or(len).min(len);
    if take == 0 {
        return Vec::new();
    }
    let mut order: Vec<_> = (0..len).collect();
    let stable_compare = |&a: &usize, &b: &usize| compare(a, b).then_with(|| a.cmp(&b));
    if take < len {
        order.select_nth_unstable_by(take, stable_compare);
        order.truncate(take);
    }
    order.sort_unstable_by(stable_compare);
    order
}

#[derive(Clone, Copy, Debug)]
pub struct RankedList<'a> {
    pub ids: &'a [u64],
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

#[derive(Clone, Debug, Default)]
pub struct RrfOptions {
    pub config: RrfConfig,
    pub run: RunOptions,
}

pub fn reciprocal_rank_fusion(lists: &[RankedList<'_>], options: &RrfOptions) -> Result<Scores> {
    let run = &options.run;
    let k = options.config.k;
    run.check_cancelled()?;
    if !k.is_finite() || k <= 0.0 {
        return Err(Error::config("k", "k must be finite and positive"));
    }
    if lists
        .iter()
        .any(|list| !list.weight.is_finite() || list.weight < 0.0)
    {
        return Err(Error::config(
            "weight",
            "weights must be finite and nonnegative",
        ));
    }
    if lists.iter().any(|list| !list.ids.is_empty())
        && !lists
            .iter()
            .any(|list| !list.ids.is_empty() && list.weight > 0.0)
    {
        return Err(Error::config(
            "weight",
            "a nonempty list must have positive weight",
        ));
    }

    let mut scores = Vec::<(u64, f64)>::new();
    let mut indices = HashMap::new();
    let mut seen = HashSet::new();
    for list in lists {
        run.check_cancelled()?;
        seen.clear();
        for (index, &id) in list.ids.iter().enumerate() {
            if index.is_multiple_of(256) {
                run.check_cancelled()?;
            }
            if !seen.insert(id) {
                continue;
            }
            let slot = *indices.entry(id).or_insert_with(|| {
                scores.push((id, 0.0));
                scores.len() - 1
            });
            scores[slot].1 += list.weight / (k + seen.len() as f64);
            if !scores[slot].1.is_finite() {
                return Err(Error::numeric("reciprocal_rank_fusion", "score overflowed"));
            }
        }
    }
    run.check_cancelled()?;
    Scores::new(scores)
}

pub fn merge_ranked_ids(lists: &[&[u64]]) -> Vec<u64> {
    let mut seen = HashSet::new();
    lists
        .iter()
        .flat_map(|ids| ids.iter().copied())
        .filter(|id| seen.insert(*id))
        .collect()
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DedupConfig {
    pub similarity_threshold: f64,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DedupOptions {
    pub config: DedupConfig,
    pub run: RunOptions,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Folded {
    pub candidate: u64,
    pub representative: u64,
    pub cosine: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DedupResult {
    pub kept: Vec<u64>,
    pub folded: Vec<Folded>,
}

const DEDUP_BLOCK: usize = 256;
const DEDUP_PRUNING_REPRESENTATIVES: usize = 64;

struct Representative<'a> {
    id: u64,
    vector: &'a [f32],
    squared_norm: f64,
    tail_norms: Option<Vec<f64>>,
}

fn block_tail_norms(vector: &[f32]) -> Vec<f64> {
    let mut remaining = 0.0;
    let mut tails = vec![0.0; vector.len().div_ceil(DEDUP_BLOCK)];
    for (index, block) in vector.chunks(DEDUP_BLOCK).enumerate().rev() {
        tails[index] = f64::sqrt(remaining);
        remaining += norm_sq_f32(block, PrecisionMode::StrictLeftFold);
    }
    tails
}

/// Bounds the unexamined dot product with Cauchy–Schwarz, including FP64 roundoff.
fn dedup_bounded_cosine(
    vector: &[f32],
    tail: &[f64],
    prior: &Representative<'_>,
    norm: f64,
    threshold: f64,
) -> Option<f64> {
    let prior_tail = prior
        .tail_norms
        .as_ref()
        .expect("representative has tail norms");
    let denominator = norm.sqrt() * prior.squared_norm.sqrt();
    let target = threshold * denominator;
    let roundoff = (16.0 * f64::EPSILON * (vector.len() as f64 + 1.0)) * denominator;
    let mut dot = -0.0;
    for (block, (left, right)) in vector
        .chunks(DEDUP_BLOCK)
        .zip(prior.vector.chunks(DEDUP_BLOCK))
        .enumerate()
    {
        dot = left
            .iter()
            .zip(right)
            .fold(dot, |sum, (&a, &b)| sum + f64::from(a) * f64::from(b));
        if dot + tail[block] * prior_tail[block] + roundoff < target {
            return None;
        }
    }
    Some((dot / denominator).clamp(-1.0, 1.0))
}

pub fn semantic_dedup(rows: VectorRows<'_>, options: &DedupOptions) -> Result<DedupResult> {
    let run = &options.run;
    run.check_cancelled()?;
    let threshold = options.config.similarity_threshold;
    if !(-1.0..=1.0).contains(&threshold) {
        return Err(Error::config(
            "similarity_threshold",
            "threshold must be in [-1, 1]",
        ));
    }

    let mut result = DedupResult::default();
    let mut representatives = Vec::<Representative>::new();
    let mode = PrecisionMode::StrictLeftFold;
    let prune = rows.dimension() > DEDUP_BLOCK;
    for (index, &id) in rows.ids().iter().enumerate() {
        run.check_cancelled()?;
        let vector = rows.row(index).expect("validated row index");
        let norm = norm_sq_f32(vector, mode);
        let tail = prune.then(|| block_tail_norms(vector));
        let mut hit = None;
        for (representative_index, prior) in representatives.iter().enumerate() {
            if representative_index.is_multiple_of(256) {
                run.check_cancelled()?;
            }
            let cosine = if let Some(tail) = &tail
                && prior.tail_norms.is_some()
            {
                dedup_bounded_cosine(vector, tail, prior, norm, threshold)
            } else {
                Some(
                    cosine_f32_with_norms(
                        vector,
                        prior.vector,
                        norm,
                        prior.squared_norm,
                        mode,
                        0.0,
                    )
                    .clamp(-1.0, 1.0),
                )
            };
            if let Some(cosine) = cosine.filter(|&cosine| cosine >= threshold) {
                hit = Some(Folded {
                    candidate: id,
                    representative: prior.id,
                    cosine,
                });
                break;
            }
        }
        if let Some(folded) = hit {
            result.folded.push(folded);
        } else {
            result.kept.push(id);
            let tail_norms = if representatives.len() < DEDUP_PRUNING_REPRESENTATIVES {
                tail
            } else {
                None
            };
            representatives.push(Representative {
                id,
                vector,
                squared_norm: norm,
                tail_norms,
            });
        }
    }
    run.check_cancelled()?;
    Ok(result)
}
