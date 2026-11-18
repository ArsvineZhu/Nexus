// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{GatedSeeds, SenseOutput, finite_vector, invalid, numeric_failure};
use nexus_core::kernel::{cosine_f32, norm_sq_f32, normalize_f64};
use nexus_core::{
    Diagnostics, ExecutionContext, NodeId, Operator, ScratchPad, VcpResult, VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservationFuseConfig {
    pub deduplication_threshold: f64,
    pub duplicate_weight_transfer: f64,
    pub max_fusion_tags: usize,
    pub max_emergent_nodes: usize,
    pub core_boost_factor: f64,
}

impl Default for ObservationFuseConfig {
    fn default() -> Self {
        Self {
            deduplication_threshold: 0.88,
            duplicate_weight_transfer: 0.20,
            max_fusion_tags: 128,
            max_emergent_nodes: 50,
            core_boost_factor: 1.33,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ObservationFuseInput<'asset, 'query> {
    pub query: &'query [f32],
    pub gated: &'query GatedSeeds,
    pub sense: &'query SenseOutput,
    pub core_tags: &'query [NodeId],
    pub ghost_vectors: Option<VectorMatrixView<'query, f32>>,
    pub ghost_is_core: &'query [bool],
    pub tag_vectors: VectorMatrixView<'asset, f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionTag {
    pub node: Option<NodeId>,
    pub weight: f64,
    pub is_core: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FusedObservation {
    pub enhanced_query: Vec<f32>,
    pub selected_tags: Vec<FusionTag>,
    pub source_field: Vec<(NodeId, f64)>,
    pub diagnostics: Diagnostics,
}

pub struct ObservationFuseOperator;

impl Operator for ObservationFuseOperator {
    type Config = ObservationFuseConfig;
    type Input<'asset, 'query> = ObservationFuseInput<'asset, 'query>;
    type Output = FusedObservation;

    fn name(&self) -> &'static str {
        "observation.fuse"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
        scratch: &mut ScratchPad,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        input.tag_vectors.validate(stage)?;
        let dim = input.tag_vectors.dim;
        finite_vector(stage, input.query, dim)?;
        if !(-1.0..=1.0).contains(&config.deduplication_threshold)
            || !(0.0..=1.0).contains(&config.duplicate_weight_transfer)
            || !config.core_boost_factor.is_finite()
            || config.core_boost_factor <= 0.0
            || !(0.0..=1.0).contains(&input.gated.effective_tag_boost)
            || !input.gated.dynamic_core_boost.is_finite()
            || input.gated.dynamic_core_boost < 0.0
        {
            return Err(invalid(stage, "invalid fusion coefficients"));
        }
        if let Some(ghosts) = input.ghost_vectors {
            ghosts.validate(stage)?;
            if ghosts.dim != dim || ghosts.rows != input.ghost_is_core.len() {
                return Err(invalid(
                    stage,
                    "ghost vectors and flags must match the query and each other",
                ));
            }
        } else if !input.ghost_is_core.is_empty() {
            return Err(invalid(stage, "ghost flags require ghost vectors"));
        }
        let mut merged = BTreeMap::<NodeId, FusionTag>::new();
        for seed in &input.gated.seeds {
            if !seed.weight.is_finite() || seed.weight < 0.0 {
                return Err(invalid(
                    stage,
                    "seed weight must be finite and non-negative",
                ));
            }
            let tag = merged.entry(seed.node).or_insert(FusionTag {
                node: Some(seed.node),
                weight: 0.0,
                is_core: false,
            });
            tag.weight = tag.weight.max(seed.weight);
            tag.is_core |= seed.is_core;
        }
        let mut emergent = Vec::new();
        for node in &input.sense.nodes {
            if !node.energy.is_finite() || node.energy < 0.0 {
                return Err(invalid(
                    stage,
                    "observed energy must be finite and non-negative",
                ));
            }
            if let Some(tag) = merged.get_mut(&node.node) {
                tag.weight = tag.weight.max(node.energy);
            } else {
                emergent.push(node);
            }
        }
        emergent.sort_unstable_by(|a, b| b.energy.total_cmp(&a.energy).then(a.node.cmp(&b.node)));
        if config.max_emergent_nodes > 0 {
            emergent.truncate(config.max_emergent_nodes);
        }
        for node in emergent {
            merged.insert(
                node.node,
                FusionTag {
                    node: Some(node.node),
                    weight: node.energy,
                    is_core: false,
                },
            );
        }
        let base = if merged.is_empty() {
            1.0
        } else {
            merged
                .values()
                .map(|tag| tag.weight / config.core_boost_factor)
                .fold(0.0, f64::max)
        };
        for &node in input.core_tags {
            merged
                .entry(node)
                .and_modify(|tag| tag.is_core = true)
                .or_insert(FusionTag {
                    node: Some(node),
                    weight: base * input.gated.dynamic_core_boost,
                    is_core: true,
                });
        }
        // The second tuple field preserves caller order as the tie-break for ghosts.
        let mut vectors = Vec::with_capacity(merged.len() + input.ghost_is_core.len());
        for (node, tag) in merged {
            let vector = input
                .tag_vectors
                .row(node.index())
                .ok_or_else(|| invalid(stage, "fusion node is outside the tag matrix"))?;
            finite_vector(stage, vector, dim)?;
            if tag.weight > 0.0 && norm_sq_f32(vector, ctx.precision) > 1e-24 {
                vectors.push((tag, 0, vector));
            }
        }
        if let Some(ghosts) = input.ghost_vectors {
            for (index, &is_core) in input.ghost_is_core.iter().enumerate() {
                let vector = ghosts.row(index).unwrap();
                finite_vector(stage, vector, dim)?;
                let weight = base
                    * if is_core {
                        input.gated.dynamic_core_boost
                    } else {
                        1.0
                    };
                if weight > 0.0 && norm_sq_f32(vector, ctx.precision) > 1e-24 {
                    vectors.push((
                        FusionTag {
                            node: None,
                            weight,
                            is_core,
                        },
                        index,
                        vector,
                    ));
                }
            }
        }
        vectors.sort_unstable_by(|(a, ai, _), (b, bi, _)| {
            b.weight
                .total_cmp(&a.weight)
                .then(a.node.cmp(&b.node))
                .then(ai.cmp(bi))
        });
        vectors.truncate(config.max_fusion_tags.clamp(1, 1024));
        let mut selected = Vec::<(FusionTag, &[f32])>::new();
        for (candidate, _, vector) in vectors {
            ctx.check_cancelled(stage)?;
            if let Some((tag, _)) = selected.iter_mut().find(|(_, prior)| {
                cosine_f32(prior, vector, ctx.precision) > config.deduplication_threshold
            }) {
                tag.weight += candidate.weight * config.duplicate_weight_transfer;
                tag.is_core |= candidate.is_core;
            } else {
                selected.push((candidate, vector));
            }
        }

        let mass = selected.iter().map(|(tag, _)| tag.weight).sum::<f64>();
        if !mass.is_finite() {
            return Err(numeric_failure(stage, "combined fusion mass is not finite"));
        }
        let mut enhanced_query = input.query.to_vec();
        if mass > 0.0 {
            let context = ScratchPad::prepare_f64(&mut scratch.f64_buf_a, dim, 0.0);
            for (tag, vector) in &selected {
                for (out, value) in context.iter_mut().zip(*vector) {
                    *out += f64::from(*value) * (tag.weight / mass);
                }
            }
            normalize_f64(context, 1e-12, ctx.precision);
            let alpha = input.gated.effective_tag_boost;
            for (memo, &source) in context.iter_mut().zip(input.query) {
                *memo = (1.0 - alpha) * f64::from(source) + alpha * *memo;
            }
            normalize_f64(context, 1e-12, ctx.precision);
            for (out, &value) in enhanced_query.iter_mut().zip(context.iter()) {
                *out = value as f32;
            }
        }
        Ok(FusedObservation {
            enhanced_query,
            selected_tags: selected.into_iter().map(|(tag, _)| tag).collect(),
            source_field: input.sense.source_field.clone(),
            diagnostics: Diagnostics::default(),
        })
    }
}
