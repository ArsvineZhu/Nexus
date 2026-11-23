// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{FusionOptions, Seeds, SenseResult, direction};
use crate::{
    AnchorRole, ConceptId, ConceptView, Error, Query, Result, RunOptions, config,
    input as bad_input, invalid,
};
use nexus_math::kernel::{PrecisionMode, cosine_f32, normalize_f64};
use nexus_math::{EvidenceSource, UnitVector};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FusionConfig {
    pub deduplication_threshold: f64,
    pub duplicate_weight_transfer: f64,
    pub max_fusion_tags: usize,
    pub max_emergent_nodes: Option<usize>,
    pub core_boost_factor: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            deduplication_threshold: 0.88,
            duplicate_weight_transfer: 0.20,
            max_fusion_tags: 128,
            max_emergent_nodes: Some(50),
            core_boost_factor: 1.33,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FusionInput<'a> {
    pub query: Query<'a>,
    pub concepts: ConceptView<'a>,
    pub seeds: &'a Seeds,
    pub sense: &'a SenseResult,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionMember {
    pub concept: Option<ConceptId>,
    pub weight: f64,
    pub is_core: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusionResult {
    pub enhanced: UnitVector,
    pub selected: Vec<FusionMember>,
}

pub fn fuse_query(input: FusionInput<'_>, options: &FusionOptions) -> Result<FusionResult> {
    options.run.check_cancelled()?;
    let original = direction(
        input.query.vector,
        input.concepts.dimension(),
        "query.vector",
    )?;
    let query = Query {
        vector: original.as_slice(),
        ..input.query
    };
    fuse(
        FusionInput { query, ..input },
        &original,
        &options.config,
        &options.run,
    )
}

pub(super) fn fuse(
    input: FusionInput<'_>,
    original: &UnitVector,
    cfg: &FusionConfig,
    run: &RunOptions,
) -> Result<FusionResult> {
    run.check_cancelled()?;
    if !(-1.0..=1.0).contains(&cfg.deduplication_threshold)
        || !(0.0..=1.0).contains(&cfg.duplicate_weight_transfer)
        || !cfg.core_boost_factor.is_finite()
        || cfg.core_boost_factor <= 0.0
        || cfg.max_fusion_tags == 0
    {
        return Err(config("fusion", "invalid fusion coefficients or tag limit"));
    }
    if !(0.0..=1.0).contains(&input.seeds.fusion_alpha)
        || !input.seeds.dynamic_core_boost.is_finite()
        || input.seeds.dynamic_core_boost < 0.0
    {
        return Err(invalid(
            "fusion.seeds",
            "invalid fusion alpha or dynamic core boost",
        ));
    }
    let mut merged = BTreeMap::<ConceptId, FusionMember>::new();
    for seed in &input.seeds.items {
        run.check_cancelled()?;
        input.concepts.node(seed.concept)?;
        if !seed.weight.is_finite() || seed.weight < 0.0 || seed.kind == EvidenceSource::Emergent {
            return Err(bad_input(
                "fusion.seeds",
                Some(seed.concept),
                "invalid seed weight or kind",
            ));
        }
        if seed.weight == 0.0 {
            continue;
        }
        let tag = merged.entry(seed.concept).or_insert(FusionMember {
            concept: Some(seed.concept),
            weight: 0.0,
            is_core: false,
        });
        tag.weight = tag.weight.max(seed.weight);
        tag.is_core |= seed.kind == EvidenceSource::Core;
    }
    let mut emergent = Vec::new();
    for node in &input.sense.nodes {
        run.check_cancelled()?;
        input.concepts.node(node.concept)?;
        if !node.energy.is_finite() || node.energy < 0.0 {
            return Err(bad_input(
                "fusion.sense",
                Some(node.concept),
                "energy must be finite and non-negative",
            ));
        }
        if node.energy == 0.0 {
            continue;
        }
        let member = FusionMember {
            concept: Some(node.concept),
            weight: node.energy,
            is_core: node.kind == EvidenceSource::Core,
        };
        if let Some(tag) = merged.get_mut(&node.concept) {
            tag.weight = tag.weight.max(node.energy);
            tag.is_core |= member.is_core;
        } else if node.kind == EvidenceSource::Emergent {
            emergent.push((node.concept, member));
        } else {
            merged.insert(node.concept, member);
        }
    }
    emergent.sort_unstable_by(|(i, a), (j, b)| b.weight.total_cmp(&a.weight).then(i.cmp(j)));
    if let Some(limit) = cfg.max_emergent_nodes {
        emergent.truncate(limit);
    }
    merged.extend(emergent);
    let base = if merged.is_empty() {
        1.0
    } else {
        merged
            .values()
            .map(|item| item.weight / cfg.core_boost_factor)
            .fold(0.0, f64::max)
    };
    if !base.is_finite() {
        return Err(Error::numeric("fusion", "restored fusion mass overflowed"));
    }
    for &concept in input.query.core_concepts {
        run.check_cancelled()?;
        input
            .concepts
            .node(concept)
            .map_err(|_| bad_input("query.core_concepts", Some(concept), "unknown concept"))?;
        merged
            .entry(concept)
            .and_modify(|item| item.is_core = true)
            .or_insert(FusionMember {
                concept: Some(concept),
                weight: base * input.seeds.dynamic_core_boost,
                is_core: true,
            });
    }

    let anchors = input
        .query
        .anchors
        .iter()
        .map(|anchor| {
            run.check_cancelled()?;
            direction(
                anchor.vector,
                input.concepts.dimension(),
                "query.anchors.vector",
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut vectors = Vec::new();
    for (concept, tag) in merged {
        let slot = input.concepts.node(concept)?;
        let vector = input.concepts.row(slot);
        if tag.weight > 0.0 {
            vectors.push((tag, 0, vector));
        }
    }
    for (index, (anchor, unit)) in input.query.anchors.iter().zip(&anchors).enumerate() {
        let is_core = anchor.role == AnchorRole::Core;
        let weight = base
            * if is_core {
                input.seeds.dynamic_core_boost
            } else {
                1.0
            };
        if weight > 0.0 {
            vectors.push((
                FusionMember {
                    concept: None,
                    weight,
                    is_core,
                },
                index,
                unit.as_slice(),
            ));
        }
    }
    vectors.sort_unstable_by(|(a, ai, _), (b, bi, _)| {
        b.weight
            .total_cmp(&a.weight)
            .then(a.concept.cmp(&b.concept))
            .then(ai.cmp(bi))
    });
    let mut selected = Vec::<(FusionMember, &[f32])>::new();
    for (candidate, _, vector) in vectors {
        run.check_cancelled()?;
        if let Some((tag, _)) = selected.iter_mut().find(|(_, prior)| {
            cosine_f32(prior, vector, PrecisionMode::FastSimd) > cfg.deduplication_threshold
        }) {
            tag.weight += candidate.weight * cfg.duplicate_weight_transfer;
            tag.is_core |= candidate.is_core;
        } else if selected.len() < cfg.max_fusion_tags {
            selected.push((candidate, vector));
        }
    }

    let mass = selected.iter().map(|(tag, _)| tag.weight).sum::<f64>();
    if !mass.is_finite() {
        return Err(Error::numeric("fusion", "combined mass overflowed"));
    }
    let mut enhanced = None;
    if mass > 0.0 {
        let mut context = vec![0.0; input.concepts.dimension()];
        for (tag, vector) in &selected {
            for (out, value) in context.iter_mut().zip(*vector) {
                *out += f64::from(*value) * (tag.weight / mass);
            }
        }
        if normalize_f64(&mut context, 1e-12, PrecisionMode::FastSimd) {
            let alpha = input.seeds.fusion_alpha;
            for (memo, &source) in context.iter_mut().zip(input.query.vector) {
                *memo = (1.0 - alpha) * f64::from(source) + alpha * *memo;
            }
            if normalize_f64(&mut context, 0.0, PrecisionMode::FastSimd) && alpha > 0.0 {
                let vector: Vec<_> = context.into_iter().map(|value| value as f32).collect();
                enhanced = Some(nexus_math::normalize(&vector)?);
            }
        }
    }
    run.check_cancelled()?;
    Ok(FusionResult {
        enhanced: enhanced.unwrap_or_else(|| original.clone()),
        selected: selected.into_iter().map(|(tag, _)| tag).collect(),
    })
}
