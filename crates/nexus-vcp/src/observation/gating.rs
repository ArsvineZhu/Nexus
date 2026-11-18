// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{EpaAnalysis, PyramidResidual, PyramidTagEntry, finite_vector, invalid};
use nexus_core::kernel::{cosine_f32, dot_f64, norm_sq_f64, normalize_f64};
use nexus_core::{
    ExecutionContext, NodeId, Operator, PrecisionMode, ScratchPad, TagLexicalMetaView, VcpResult,
    VectorMatrixView,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn lexical_tokens(text: &str) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let mut ascii = String::new();
    let mut cjk = Vec::new();
    let flush = |ascii: &mut String, cjk: &mut Vec<char>, tokens: &mut HashSet<String>| {
        if !ascii.is_empty() {
            tokens.insert(std::mem::take(ascii));
        }
        if cjk.len() == 1 {
            tokens.insert(cjk[0].to_string());
        }
        for pair in cjk.windows(2) {
            tokens.insert(pair.iter().collect());
        }
        cjk.clear();
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            if !cjk.is_empty() {
                flush(&mut ascii, &mut cjk, &mut tokens);
            }
            ascii.push(ch.to_ascii_lowercase());
        } else if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            if !ascii.is_empty() {
                flush(&mut ascii, &mut cjk, &mut tokens);
            }
            cjk.push(ch);
        } else {
            flush(&mut ascii, &mut cjk, &mut tokens);
        }
    }
    flush(&mut ascii, &mut cjk, &mut tokens);
    tokens
}

fn lexical_grounding(query: &HashSet<String>, text: &str) -> bool {
    let tag = lexical_tokens(text);
    !tag.is_empty() && tag.intersection(query).count() * 2 >= tag.len()
}

fn level_threshold(tags: &[PyramidTagEntry], floor: f64, ratio: f64) -> (f64, f64) {
    let max = tags.iter().map(|tag| tag.contribution).fold(0.0, f64::max);
    (max, floor.max(max * ratio))
}

fn validate_metadata(
    stage: &'static str,
    meta: TagLexicalMetaView<'_>,
    count: usize,
) -> VcpResult<()> {
    meta.validate(stage)?;
    if count != meta.node_count {
        return Err(invalid(stage, "lexical metadata must cover the tag matrix"));
    }
    Ok(())
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HypernymConfig {
    pub enabled: bool,
    pub peer_max_cosine: f64,
    pub query_parent_min_cosine: f64,
    pub differentia_confidence_scale: f64,
}

impl Default for HypernymConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            peer_max_cosine: 0.90,
            query_parent_min_cosine: 0.65,
            differentia_confidence_scale: 0.1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HypernymPruneInput<'asset, 'query> {
    pub query: &'query [f32],
    pub pyramid: &'query PyramidResidual,
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub lexical_meta: TagLexicalMetaView<'asset>,
    pub core_tags: &'query [NodeId],
    pub seed_contribution_floor: f64,
    pub seed_contribution_ratio: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisambiguatedTags {
    pub retained_levels: Vec<Vec<PyramidTagEntry>>,
    pub pruned_hypernyms: Vec<NodeId>,
    pub differentia_q_perp: Option<Vec<f32>>,
    pub differentia_confidence: f64,
}

fn remove_span(vector: &mut [f64], axes: &[Vec<f64>]) {
    for _ in 0..2 {
        for axis in axes {
            let coefficient = dot_f64(vector, axis, PrecisionMode::StrictLeftFold);
            for (value, unit) in vector.iter_mut().zip(axis) {
                *value -= coefficient * unit;
            }
        }
    }
}

pub struct HypernymPruneOperator;

impl Operator for HypernymPruneOperator {
    type Config = HypernymConfig;
    type Input<'asset, 'query> = HypernymPruneInput<'asset, 'query>;
    type Output = DisambiguatedTags;

    fn name(&self) -> &'static str {
        "hypernym.prune"
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
        input.tag_vectors.validate(stage)?;
        finite_vector(stage, input.query, input.tag_vectors.dim)?;
        validate_metadata(stage, input.lexical_meta, input.tag_vectors.rows)?;
        if ![config.peer_max_cosine, config.query_parent_min_cosine]
            .into_iter()
            .all(|v| (-1.0..=1.0).contains(&v))
            || !config.differentia_confidence_scale.is_finite()
            || config.differentia_confidence_scale <= 0.0
            || ![input.seed_contribution_floor, input.seed_contribution_ratio]
                .into_iter()
                .all(|v| (0.0..=1.0).contains(&v))
        {
            return Err(invalid(stage, "invalid disambiguation thresholds"));
        }

        let mut result = DisambiguatedTags::default();
        for level in &input.pyramid.levels {
            ctx.check_cancelled(stage)?;
            let (_, threshold) = level_threshold(
                &level.tags,
                input.seed_contribution_floor,
                input.seed_contribution_ratio,
            );
            let mut retained = Vec::with_capacity(level.tags.len());
            let mut vectors = Vec::with_capacity(level.tags.len());
            let mut tokens = Vec::with_capacity(level.tags.len());
            for tag in &level.tags {
                let vector = input
                    .tag_vectors
                    .row(tag.node.index())
                    .ok_or_else(|| invalid(stage, "pyramid node is outside the tag matrix"))?;
                vectors.push(vector);
                tokens.push(lexical_tokens(input.lexical_meta.name(tag.node)));
            }
            for (index, tag) in level.tags.iter().enumerate() {
                ctx.check_cancelled(stage)?;
                let vector = vectors[index];
                let parent = &tokens[index];
                let is_parent = config.enabled
                    && !input.core_tags.contains(&tag.node)
                    && tag.contribution < threshold
                    && !parent.is_empty()
                    && cosine_f32(input.query, vector, ctx.precision)
                        >= config.query_parent_min_cosine
                    && level.tags.iter().enumerate().any(|(peer, entry)| {
                        entry.node != tag.node
                            && entry.contribution >= threshold
                            && entry.similarity > 0.0
                            && parent.len() < tokens[peer].len()
                            && parent.is_subset(&tokens[peer])
                            && cosine_f32(vector, vectors[peer], ctx.precision)
                                < config.peer_max_cosine
                    });
                if is_parent {
                    result.pruned_hypernyms.push(tag.node);
                } else {
                    retained.push(*tag);
                }
            }
            result.retained_levels.push(retained);
        }

        result.pruned_hypernyms.sort_unstable();
        result.pruned_hypernyms.dedup();
        if result.pruned_hypernyms.is_empty() {
            return Ok(result);
        }
        let mut axes = Vec::new();
        let precision = PrecisionMode::StrictLeftFold;
        for &node in &result.pruned_hypernyms {
            let mut axis: Vec<_> = input
                .tag_vectors
                .row(node.index())
                .unwrap()
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let minimum = 1e-5 * norm_sq_f64(&axis, precision).sqrt();
            remove_span(&mut axis, &axes);
            if normalize_f64(&mut axis, minimum.max(1e-12), precision) {
                axes.push(axis);
            }
        }
        if axes.is_empty() {
            return Ok(result);
        }
        let mut residual: Vec<_> = input.query.iter().map(|&v| f64::from(v)).collect();
        if normalize_f64(&mut residual, 1e-12, precision) {
            remove_span(&mut residual, &axes);
            let norm = norm_sq_f64(&residual, precision).sqrt();
            if norm >= 1e-4 {
                result.differentia_confidence =
                    (norm / config.differentia_confidence_scale).min(1.0);
                result.differentia_q_perp =
                    Some(residual.into_iter().map(|v| (v / norm) as f32).collect());
            }
        }
        Ok(result)
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TagGatingConfig {
    pub base_tag_boost: f64,
    pub layer_decay: f64,
    pub seed_contribution_floor: f64,
    pub seed_contribution_ratio: f64,
    pub implicit_max_ratio_of_explicit: f64,
    pub lang_confidence_enabled: bool,
    pub lang_penalty_unknown: f64,
    pub activation_multiplier: [f64; 2],
    pub dynamic_boost_range: [f64; 2],
    pub core_boost_range: [f64; 2],
}

impl Default for TagGatingConfig {
    fn default() -> Self {
        Self {
            base_tag_boost: 0.6,
            layer_decay: 0.7,
            seed_contribution_floor: 0.05,
            seed_contribution_ratio: 0.1,
            implicit_max_ratio_of_explicit: 0.35,
            lang_confidence_enabled: false,
            lang_penalty_unknown: 0.05,
            activation_multiplier: [0.5, 1.5],
            dynamic_boost_range: [0.3, 2.0],
            core_boost_range: [1.2, 1.4],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TagGatingInput<'asset, 'query> {
    pub epa: &'query EpaAnalysis,
    pub pyramid: &'query PyramidResidual,
    pub disambiguated: &'query DisambiguatedTags,
    pub query_text: &'query str,
    pub core_tags: &'query [NodeId],
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub lexical_meta: TagLexicalMetaView<'asset>,
    pub graph_node_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatedSeed {
    pub node: NodeId,
    pub weight: f64,
    pub is_core: bool,
    pub lexical: bool,
    pub residual_level: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GatedSeeds {
    pub seeds: Vec<GatedSeed>,
    pub effective_tag_boost: f64,
    pub dynamic_core_boost: f64,
    /// An exact canonical first-layer focus with positive NNLS support.
    pub exact_focus: Option<NodeId>,
}

pub struct TagGatingOperator;

impl Operator for TagGatingOperator {
    type Config = TagGatingConfig;
    type Input<'asset, 'query> = TagGatingInput<'asset, 'query>;
    type Output = GatedSeeds;

    fn name(&self) -> &'static str {
        "tag.gating"
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
        validate_metadata(stage, input.lexical_meta, input.graph_node_count)?;
        input.tag_vectors.validate(stage)?;
        if input.tag_vectors.rows != input.graph_node_count {
            return Err(invalid(stage, "tag matrix must cover the graph"));
        }
        let [activation_min, activation_max] = config.activation_multiplier;
        let [boost_min, boost_max] = config.dynamic_boost_range;
        let [core_min, core_max] = config.core_boost_range;
        if ![
            config.layer_decay,
            config.seed_contribution_floor,
            config.seed_contribution_ratio,
            config.implicit_max_ratio_of_explicit,
            config.lang_penalty_unknown,
        ]
        .into_iter()
        .all(|v| (0.0..=1.0).contains(&v))
            || ![
                config.base_tag_boost,
                activation_min,
                activation_max,
                boost_min,
                boost_max,
                core_min,
                core_max,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
            || activation_min > activation_max
            || boost_min > boost_max
            || core_min > core_max
        {
            return Err(invalid(stage, "invalid seed gating configuration"));
        }
        let activation = activation_min
            + input.pyramid.activation.clamp(0.0, 1.0) * (activation_max - activation_min);
        let dynamic_factor = input.epa.logic_depth * (1.0 + input.epa.resonance.ln_1p())
            / (1.0 + input.epa.entropy * 0.5)
            * activation;
        let core_metric =
            (input.epa.logic_depth + 1.0 - input.pyramid.coverage).clamp(0.0, 2.0) * 0.5;
        let core_boost = core_min + core_metric * (core_max - core_min);
        let query_name = input.query_text.trim().to_lowercase();
        let exact_focus = input.pyramid.levels.first().and_then(|level| {
            level
                .tags
                .iter()
                .filter(|tag| {
                    !query_name.is_empty()
                        && tag.similarity > 0.0
                        && tag.contribution > 0.0
                        && input.lexical_meta.name(tag.node).trim().to_lowercase() == query_name
                })
                .max_by(|a, b| {
                    a.contribution
                        .total_cmp(&b.contribution)
                        .then(b.node.cmp(&a.node))
                })
                .map(|tag| tag.node)
        });
        let mut result = GatedSeeds {
            effective_tag_boost: (config.base_tag_boost
                * dynamic_factor.clamp(boost_min, boost_max))
            .clamp(0.0, 1.0),
            dynamic_core_boost: core_boost,
            exact_focus,
            ..GatedSeeds::default()
        };
        let query_tokens = lexical_tokens(input.query_text);
        let mut seen = HashSet::new();
        for (level, tags) in input.disambiguated.retained_levels.iter().enumerate() {
            ctx.check_cancelled(stage)?;
            let (max, threshold) = level_threshold(
                tags,
                config.seed_contribution_floor,
                config.seed_contribution_ratio,
            );
            let decay = config.layer_decay.powf(level as f64);
            let dominant = tags
                .first()
                .filter(|tag| tag.contribution.powi(2) > 0.85 && tag.similarity > 0.0);
            let mut redundancy_left = dominant.map_or(0.0, |tag| 0.3 * tag.contribution);
            let mut redundancy_slots = 2;
            for tag in tags {
                let name = input.lexical_meta.name(tag.node);
                let vector = input
                    .tag_vectors
                    .row(tag.node.index())
                    .ok_or_else(|| invalid(stage, "pyramid node is outside the graph"))?;
                if seen.contains(&tag.node) {
                    continue;
                }
                let is_core = input.core_tags.contains(&tag.node);
                let lexical = std::iter::once(name)
                    .chain(input.lexical_meta.aliases(tag.node).iter().copied())
                    .any(|name| lexical_grounding(&query_tokens, name));
                let sponsor = tags
                    .iter()
                    .filter(|peer| {
                        peer.node != tag.node
                            && peer.contribution >= threshold
                            && peer.similarity > 0.0
                    })
                    .filter_map(|peer| {
                        let peer_vector = input.tag_vectors.row(peer.node.index())?;
                        let tightness =
                            cosine_f32(vector, peer_vector, ctx.precision).clamp(0.0, 1.0);
                        (tightness >= 0.8).then_some(peer.contribution * tightness)
                    })
                    .fold(0.0, f64::max);
                let alias = tag.similarity >= 0.2 && sponsor > 0.0;
                let mut redundancy_floor = 0.0;
                if !is_core
                    && !lexical
                    && tag.contribution < 1e-6
                    && tag.similarity >= 0.75
                    && let Some(first) = dominant.filter(|first| first.node != tag.node)
                    && let Some(first_vector) = input.tag_vectors.row(first.node.index())
                {
                    let tightness = cosine_f32(vector, first_vector, ctx.precision).clamp(0.0, 1.0);
                    if tightness >= 0.90 {
                        if redundancy_slots == 0 {
                            continue;
                        }
                        redundancy_floor = (0.15
                            * first.contribution
                            * (tag.similarity / first.similarity).clamp(0.0, 1.0)
                            * tightness)
                            .min(redundancy_left);
                        redundancy_left -= redundancy_floor;
                        redundancy_slots -= 1;
                    }
                }
                if !is_core
                    && !lexical
                    && !alias
                    && (tag.contribution < threshold || tag.similarity <= 0.0)
                {
                    continue;
                }
                let contribution = if lexical {
                    tag.contribution.max(0.6 * sponsor).max(0.6 * max)
                } else if redundancy_floor > 0.0 {
                    tag.contribution.max(redundancy_floor)
                } else if alias {
                    tag.contribution
                        .max(0.6 * sponsor * tag.similarity.clamp(0.2, 1.0))
                } else {
                    tag.contribution
                };
                let layer_gain = if lexical {
                    1.0
                } else {
                    decay + (1.0 - decay) * (contribution / max.max(1e-12)).clamp(0.0, 1.0)
                };
                let core_gain = if is_core {
                    core_boost * (0.95 + tag.similarity * 0.1)
                } else {
                    1.0
                };
                let technical = name.is_ascii() && name.chars().any(|ch| ch.is_ascii_alphabetic());
                let language_gain = if !lexical && config.lang_confidence_enabled && technical {
                    config.lang_penalty_unknown
                } else {
                    1.0
                };
                let mut weight = contribution * layer_gain * core_gain * language_gain;
                if lexical && !is_core {
                    weight *= 1.6;
                }
                if !is_core {
                    weight = weight.min(1.0);
                }
                if weight > 0.0 {
                    seen.insert(tag.node);
                    result.seeds.push(GatedSeed {
                        node: tag.node,
                        weight,
                        is_core,
                        lexical,
                        residual_level: level,
                    });
                }
            }
        }

        for &node in input.core_tags {
            if node.index() >= input.graph_node_count {
                return Err(invalid(stage, "explicit core node is outside the graph"));
            }
            if let Some(seed) = result.seeds.iter_mut().find(|seed| seed.node == node) {
                seed.is_core = true;
                seed.weight = seed.weight.max(core_boost);
            } else if core_boost > 0.0 {
                result.seeds.push(GatedSeed {
                    node,
                    weight: core_boost,
                    is_core: true,
                    lexical: false,
                    residual_level: usize::MAX,
                });
            }
        }

        let explicit = result
            .seeds
            .iter()
            .filter(|seed| seed.is_core || seed.lexical)
            .map(|seed| seed.weight)
            .fold(0.0, f64::max);
        let implicit = result
            .seeds
            .iter()
            .filter(|seed| !seed.is_core && !seed.lexical)
            .map(|seed| seed.weight)
            .fold(0.0, f64::max);
        let cap = explicit * config.implicit_max_ratio_of_explicit;
        if explicit > 0.0 && implicit > cap {
            for seed in result
                .seeds
                .iter_mut()
                .filter(|seed| !seed.is_core && !seed.lexical)
            {
                seed.weight *= cap / implicit;
            }
        }
        Ok(result)
    }
}
