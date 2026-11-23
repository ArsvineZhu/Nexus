// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{EpaAnalysis, LanguageContext, PyramidResult, PyramidTag, SeedOptions, direction};
use crate::{
    ConceptId, ConceptView, Query, Result, RunOptions, config, input as bad_input, invalid,
};
use nexus_math::kernel::{PrecisionMode, cosine_f32, dot_f64, norm_sq_f64, normalize_f64};
use nexus_math::{EvidenceSource, NodeId, UnitVector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const LEXICAL_GAIN: f64 = 1.6;
const LEXICAL_RESCUE: f64 = 0.6;
const ALIAS_ALIGNMENT: f64 = 0.80;
const ALIAS_MIN_SIMILARITY: f64 = 0.20;
const REDUNDANT_TAGS: usize = 2;
const REDUNDANT_TOTAL_BUDGET: f64 = 0.30;
const REDUNDANT_TAG_BUDGET: f64 = 0.15;

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

#[derive(Default)]
pub(crate) struct Lexicon {
    tokens: HashMap<String, usize>,
    canonical: Vec<HashSet<usize>>,
    forms: Vec<(NodeId, usize)>,
    postings: Vec<Vec<usize>>,
    named: usize,
}

impl Lexicon {
    pub(crate) fn insert(
        &mut self,
        slot: NodeId,
        label: Option<&str>,
        aliases: &[String],
        run: &RunOptions,
    ) -> Result<()> {
        let label = label.unwrap_or_default();
        let named =
            !label.trim().is_empty() || aliases.iter().any(|alias| !alias.trim().is_empty());
        self.named += usize::from(named);
        let canonical = self.add_form(slot, label);
        self.canonical.push(canonical);
        for alias in aliases {
            run.check_cancelled()?;
            self.add_form(slot, alias);
        }
        Ok(())
    }

    fn add_form(&mut self, slot: NodeId, value: &str) -> HashSet<usize> {
        let tokens = lexical_tokens(value);
        let form = self.forms.len();
        if tokens.is_empty() {
            return HashSet::new();
        }
        self.forms.push((slot, tokens.len().div_ceil(2)));
        tokens
            .into_iter()
            .map(|token| {
                let key = *self.tokens.entry(token).or_insert_with(|| {
                    let key = self.postings.len();
                    self.postings.push(Vec::new());
                    key
                });
                self.postings[key].push(form);
                key
            })
            .collect()
    }

    fn matches(&self, query: &str, run: &RunOptions) -> Result<HashSet<NodeId>> {
        let mut hits = HashSet::new();
        let mut counts = HashMap::<usize, usize>::new();
        for token in lexical_tokens(query) {
            run.check_cancelled()?;
            if let Some(&key) = self.tokens.get(&token) {
                for (index, &form) in self.postings[key].iter().enumerate() {
                    if index.is_multiple_of(256) {
                        run.check_cancelled()?;
                    }
                    let count = counts.entry(form).or_default();
                    *count += 1;
                    let (slot, minimum) = self.forms[form];
                    if *count == minimum {
                        hits.insert(slot);
                    }
                }
            }
        }
        Ok(hits)
    }

    fn canonical(&self, slot: NodeId) -> &HashSet<usize> {
        &self.canonical[slot.index()]
    }
}

fn level_threshold(tags: &[PyramidTag], floor: f64, ratio: f64) -> (f64, f64) {
    let max = tags.iter().map(|tag| tag.contribution).fold(0.0, f64::max);
    (max, floor.max(max * ratio))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HypernymConfig {
    pub peer_max_cosine: f64,
    pub query_parent_min_cosine: f64,
    pub differentia_confidence_scale: f64,
}

impl Default for HypernymConfig {
    fn default() -> Self {
        Self {
            peer_max_cosine: 0.90,
            query_parent_min_cosine: 0.65,
            differentia_confidence_scale: 0.1,
        }
    }
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SeedConfig {
    pub hypernym: Option<HypernymConfig>,
    pub base_tag_boost: f64,
    pub layer_decay: f64,
    pub seed_contribution_floor: f64,
    pub seed_contribution_ratio: f64,
    pub implicit_max_ratio_of_explicit: f64,
    pub lang_confidence_enabled: bool,
    pub lang_penalty_unknown: f64,
    pub lang_penalty_cross_domain: f64,
    pub activation_multiplier: [f64; 2],
    pub dynamic_boost_range: [f64; 2],
    pub core_boost_range: [f64; 2],
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            hypernym: Some(HypernymConfig::default()),
            base_tag_boost: 0.6,
            layer_decay: 0.7,
            seed_contribution_floor: 0.05,
            seed_contribution_ratio: 0.1,
            implicit_max_ratio_of_explicit: 0.35,
            lang_confidence_enabled: true,
            lang_penalty_unknown: 0.05,
            lang_penalty_cross_domain: 0.1,
            activation_multiplier: [0.5, 1.5],
            dynamic_boost_range: [0.3, 2.0],
            core_boost_range: [1.2, 1.4],
        }
    }
}

#[derive(Clone, Copy)]
pub struct SeedInput<'a> {
    pub query: Query<'a>,
    pub concepts: ConceptView<'a>,
    pub epa: Option<&'a EpaAnalysis>,
    pub pyramid: Option<&'a PyramidResult>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seed {
    pub concept: ConceptId,
    pub weight: f64,
    pub kind: EvidenceSource,
    pub residual_level: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Seeds {
    pub items: Vec<Seed>,
    pub fusion_alpha: f64,
    pub dynamic_core_boost: f64,
    pub exact_focus: Option<ConceptId>,
    pub pruned_hypernyms: Vec<ConceptId>,
    pub differentia: Option<UnitVector>,
    pub differentia_confidence: f64,
}

fn technical_name(value: &str, allow_space: bool) -> bool {
    value.len() > 3
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '-' | '_' | '.')
                || (allow_space && character.is_ascii_whitespace())
        })
}

fn language_context(
    input: SeedInput<'_>,
    cfg: &SeedConfig,
    named: usize,
) -> Result<(LanguageContext, f64)> {
    let Some(primary) = input.epa.and_then(|epa| epa.axes.first()) else {
        return Ok(if named == 0 {
            (LanguageContext::Unavailable, 1.0)
        } else {
            (LanguageContext::Unknown, cfg.lang_penalty_unknown)
        });
    };
    input.concepts.node(primary.concept)?;
    let Some(name) = input
        .concepts
        .label(primary.concept)
        .map(str::trim)
        .filter(|name| !name.is_empty())
    else {
        return Ok((LanguageContext::Unavailable, 1.0));
    };
    let lower = name.to_ascii_lowercase();
    let social = ["politics", "society", "history", "economics", "culture"]
        .iter()
        .any(|token| lower.contains(token));
    Ok(if social {
        (
            LanguageContext::Social,
            cfg.lang_penalty_cross_domain.sqrt(),
        )
    } else if lower == "unknown" {
        (LanguageContext::Unknown, cfg.lang_penalty_unknown)
    } else if technical_name(name, false) {
        (LanguageContext::Technical, 1.0)
    } else {
        (LanguageContext::Unknown, cfg.lang_penalty_cross_domain)
    })
}

fn differentia(
    input: SeedInput<'_>,
    cfg: &HypernymConfig,
    pruned: &[ConceptId],
    run: &RunOptions,
) -> Result<(Option<UnitVector>, f64)> {
    let mut axes = Vec::new();
    let precision = PrecisionMode::StrictLeftFold;
    for &concept in pruned {
        run.check_cancelled()?;
        let node = input.concepts.node(concept)?;
        let mut axis: Vec<_> = input
            .concepts
            .row(node)
            .iter()
            .copied()
            .map(f64::from)
            .collect();
        let minimum = 1e-3 * norm_sq_f64(&axis, precision).sqrt();
        remove_span(&mut axis, &axes);
        if normalize_f64(&mut axis, minimum.max(1e-12), precision) {
            axes.push(axis);
        }
    }
    if axes.is_empty() {
        return Ok((None, 0.0));
    }
    let mut residual: Vec<_> = input.query.vector.iter().copied().map(f64::from).collect();
    normalize_f64(&mut residual, 0.0, precision);
    remove_span(&mut residual, &axes);
    let norm = norm_sq_f64(&residual, precision).sqrt();
    if norm < 1e-4 {
        return Ok((None, 0.0));
    }
    let vector: Vec<_> = residual
        .into_iter()
        .map(|value| (value / norm) as f32)
        .collect();
    let confidence = (norm / cfg.differentia_confidence_scale).min(1.0);
    Ok((Some(nexus_math::normalize(&vector)?), confidence))
}

pub fn select_seeds(input: SeedInput<'_>, options: &SeedOptions) -> Result<Seeds> {
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
    seeds(SeedInput { query, ..input }, &options.config, &options.run).map(|(seeds, _, _)| seeds)
}

pub(super) fn seeds(
    input: SeedInput<'_>,
    cfg: &SeedConfig,
    run: &RunOptions,
) -> Result<(Seeds, LanguageContext, usize)> {
    run.check_cancelled()?;
    let [activation_min, activation_max] = cfg.activation_multiplier;
    let [boost_min, boost_max] = cfg.dynamic_boost_range;
    let [core_min, core_max] = cfg.core_boost_range;
    for (field, value) in [
        ("seeds.layer_decay", cfg.layer_decay),
        (
            "seeds.implicit_max_ratio_of_explicit",
            cfg.implicit_max_ratio_of_explicit,
        ),
        ("seeds.lang_penalty_unknown", cfg.lang_penalty_unknown),
        (
            "seeds.lang_penalty_cross_domain",
            cfg.lang_penalty_cross_domain,
        ),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(config(field, "must be in [0, 1]"));
        }
    }
    for (field, value) in [
        ("seeds.base_tag_boost", cfg.base_tag_boost),
        ("seeds.seed_contribution_floor", cfg.seed_contribution_floor),
        ("seeds.seed_contribution_ratio", cfg.seed_contribution_ratio),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(config(field, "must be finite and non-negative"));
        }
    }
    for (field, [low, high]) in [
        ("seeds.activation_multiplier", cfg.activation_multiplier),
        ("seeds.dynamic_boost_range", cfg.dynamic_boost_range),
        ("seeds.core_boost_range", cfg.core_boost_range),
    ] {
        if !high.is_finite() || !(0.0..=high).contains(&low) {
            return Err(config(
                field,
                "must have finite ordered non-negative bounds",
            ));
        }
    }
    if let Some(hypernym) = &cfg.hypernym
        && (![hypernym.peer_max_cosine, hypernym.query_parent_min_cosine]
            .into_iter()
            .all(|v| (-1.0..=1.0).contains(&v))
            || !hypernym.differentia_confidence_scale.is_finite()
            || hypernym.differentia_confidence_scale <= 0.0)
    {
        return Err(config(
            "seeds.hypernym",
            "cosines must be in [-1, 1] and confidence scale must be finite and positive",
        ));
    }
    if let Some(epa) = input.epa
        && (!(0.0..=1.0).contains(&epa.concentration)
            || !(0.0..=1.0).contains(&epa.entropy)
            || !epa.resonance.is_finite()
            || epa.resonance < 0.0)
    {
        return Err(invalid(
            "seeds.epa",
            "invalid EPA concentration, entropy or resonance",
        ));
    }
    if let Some(pyramid) = input.pyramid
        && (!(0.0..=1.0).contains(&pyramid.activation) || !(0.0..=1.0).contains(&pyramid.coverage))
    {
        return Err(invalid(
            "seeds.pyramid",
            "activation and coverage must be in [0, 1]",
        ));
    }

    let activation = activation_min
        + input.pyramid.map_or(0.0, |pyramid| pyramid.activation)
            * (activation_max - activation_min);
    let dynamic_factor = input.epa.map_or(0.0, |epa| {
        epa.concentration * (1.0 + epa.resonance.ln_1p()) / (1.0 + epa.entropy * 0.5) * activation
    });
    let core_metric = 0.5 * input.epa.map_or(0.0, |epa| epa.concentration)
        + 0.5 * input.pyramid.map_or(0.0, |pyramid| 1.0 - pyramid.coverage);
    let core_boost = core_min + core_metric * (core_max - core_min);
    let mut result = Seeds {
        fusion_alpha: (cfg.base_tag_boost * dynamic_factor.clamp(boost_min, boost_max)).min(1.0),
        dynamic_core_boost: core_boost,
        ..Default::default()
    };
    let query_text = input.query.text.unwrap_or_default();
    let query_name = query_text.trim().to_lowercase();
    let lexicon = input.concepts.lexicon();
    let matches = lexicon.matches(query_text, run)?;
    let named = lexicon.named;
    let (language, penalty) = language_context(input, cfg, named)?;
    let mut core = HashSet::new();
    for &id in input.query.core_concepts {
        run.check_cancelled()?;
        input
            .concepts
            .node(id)
            .map_err(|_| bad_input("query.core_concepts", Some(id), "unknown concept"))?;
        core.insert(id);
    }
    let mut seen = HashSet::new();
    let levels = input
        .pyramid
        .map_or(&[][..], |pyramid| pyramid.levels.as_slice());
    for pyramid_level in levels {
        run.check_cancelled()?;
        let level = pyramid_level.level;
        let tags = &pyramid_level.tags;
        let (max, threshold) = level_threshold(
            tags,
            cfg.seed_contribution_floor,
            cfg.seed_contribution_ratio,
        );
        let facts: Vec<_> = tags
            .iter()
            .map(|tag| {
                if !tag.contribution.is_finite()
                    || tag.contribution < 0.0
                    || !(-1.0..=1.0).contains(&tag.similarity)
                {
                    return Err(bad_input(
                        "seeds.pyramid.tags",
                        Some(tag.concept),
                        "invalid contribution or similarity",
                    ));
                }
                let slot = input.concepts.node(tag.concept)?;
                let vector = input.concepts.row(slot);
                let name = input.concepts.label(tag.concept).unwrap_or_default();
                let tokens = lexicon.canonical(slot);
                let lexical = matches.contains(&slot);
                Ok((
                    tag,
                    name,
                    vector,
                    tokens,
                    core.contains(&tag.concept),
                    lexical,
                ))
            })
            .collect::<Result<_>>()?;
        let peers: Vec<_> = facts
            .iter()
            .filter(|(tag, ..)| {
                tag.contribution >= threshold && tag.contribution > 0.0 && tag.similarity > 0.0
            })
            .collect();
        if level == 0 && !query_name.is_empty() {
            result.exact_focus = facts
                .iter()
                .filter(|(tag, name, ..)| {
                    tag.contribution > 0.0
                        && tag.similarity > 0.0
                        && name.trim().to_lowercase() == query_name
                })
                .max_by(|a, b| {
                    a.0.contribution
                        .total_cmp(&b.0.contribution)
                        .then(b.0.concept.cmp(&a.0.concept))
                })
                .map(|(tag, ..)| tag.concept);
        }
        let decay = cfg.layer_decay.powf(level as f64);
        let mut redundancy = None;
        let mut redundancy_left = 0.0;
        let mut redundancy_slots = REDUNDANT_TAGS;
        for (tag, name, vector, tokens, is_core, lexical) in &facts {
            run.check_cancelled()?;
            let (is_core, lexical) = (*is_core, *lexical);
            let can_be_parent = cfg.hypernym.as_ref().filter(|hypernym| {
                !is_core
                    && tag.contribution < threshold
                    && !tokens.is_empty()
                    && cosine_f32(input.query.vector, vector, PrecisionMode::FastSimd)
                        >= hypernym.query_parent_min_cosine
            });
            let mut parent = false;
            let mut sponsor = 0.0_f64;
            for &&(peer, _, peer_vector, peer_tokens, _, _) in &peers {
                if peer.concept == tag.concept {
                    continue;
                }
                let cosine = cosine_f32(vector, peer_vector, PrecisionMode::FastSimd);
                if can_be_parent.is_some_and(|hypernym| {
                    tokens.len() < peer_tokens.len()
                        && tokens.is_subset(peer_tokens)
                        && cosine < hypernym.peer_max_cosine
                }) {
                    parent = true;
                    break;
                }
                if cosine >= ALIAS_ALIGNMENT {
                    sponsor = sponsor.max(peer.contribution * cosine.min(1.0));
                }
            }
            if parent {
                result.pruned_hypernyms.push(tag.concept);
                continue;
            }
            if redundancy.is_none() && tag.contribution.powi(2) > 0.85 && tag.similarity > 0.0 {
                redundancy = Some((*tag, *vector));
                redundancy_left = REDUNDANT_TOTAL_BUDGET * tag.contribution;
            }
            if seen.contains(&tag.concept) {
                continue;
            }
            let alias = tag.similarity >= ALIAS_MIN_SIMILARITY && sponsor > 0.0;
            let mut redundancy_floor = 0.0;
            if !is_core
                && !lexical
                && tag.contribution < 1e-6
                && tag.similarity >= 0.75
                && let Some((dominant, direction)) =
                    redundancy.filter(|(first, _)| first.concept != tag.concept)
            {
                let tightness =
                    cosine_f32(vector, direction, PrecisionMode::FastSimd).clamp(0.0, 1.0);
                if tightness >= 0.90 {
                    if redundancy_slots == 0 {
                        continue;
                    }
                    redundancy_floor = (REDUNDANT_TAG_BUDGET
                        * dominant.contribution
                        * (tag.similarity / dominant.similarity).clamp(0.0, 1.0)
                        * tightness)
                        .min(redundancy_left);
                    redundancy_slots -= 1;
                    redundancy_left -= redundancy_floor;
                }
            }
            if !is_core
                && !lexical
                && !alias
                && redundancy_floor == 0.0
                && (tag.contribution < threshold || tag.similarity <= 0.0)
            {
                continue;
            }
            let contribution = if lexical {
                tag.contribution
                    .max(LEXICAL_RESCUE * sponsor)
                    .max(LEXICAL_RESCUE * max)
            } else if redundancy_floor > 0.0 {
                tag.contribution.max(redundancy_floor)
            } else if alias {
                tag.contribution
                    .max(LEXICAL_RESCUE * sponsor * tag.similarity.clamp(ALIAS_MIN_SIMILARITY, 1.0))
            } else {
                tag.contribution
            };
            let layer_gain = if lexical {
                1.0
            } else {
                decay + (1.0 - decay) * (contribution / max.max(1e-12)).clamp(0.0, 1.0)
            };
            let language_gain =
                if !lexical && cfg.lang_confidence_enabled && technical_name(name.trim(), true) {
                    penalty
                } else {
                    1.0
                };
            let raw = contribution * layer_gain * language_gain;
            let (kind, weight) = if is_core {
                (
                    EvidenceSource::Core,
                    core_boost.max(raw * core_boost * (0.95 + tag.similarity * 0.1)),
                )
            } else if lexical {
                (EvidenceSource::Lexical, (raw * LEXICAL_GAIN).min(1.0))
            } else {
                (EvidenceSource::Inferred, raw.min(1.0))
            };
            if !weight.is_finite() {
                return Err(crate::Error::numeric("seeds", "non-finite seed weight"));
            }
            if weight > 0.0 {
                seen.insert(tag.concept);
                result.items.push(Seed {
                    concept: tag.concept,
                    weight,
                    kind,
                    residual_level: if is_core { 0 } else { level },
                });
            }
        }
    }

    result.pruned_hypernyms.sort_unstable();
    result.pruned_hypernyms.dedup();
    for &concept in input.query.core_concepts {
        if seen.insert(concept) && core_boost > 0.0 {
            result.items.push(Seed {
                concept,
                weight: core_boost,
                kind: EvidenceSource::Core,
                residual_level: 0,
            });
        }
    }
    // The explicit term match supplies a seed even when no pyramid layer accepted this concept.
    let mut literal_slots: Vec<_> = matches.into_iter().collect();
    literal_slots.sort_unstable();
    for slot in literal_slots {
        run.check_cancelled()?;
        let concept = input.concepts.ids()[slot.index()];
        if seen.contains(&concept)
            || core.contains(&concept)
            || result.pruned_hypernyms.binary_search(&concept).is_ok()
        {
            continue;
        }
        let vector = input.concepts.row(slot);
        let similarity =
            cosine_f32(input.query.vector, vector, PrecisionMode::FastSimd).clamp(0.0, 1.0);
        let weight =
            (LEXICAL_GAIN * cfg.seed_contribution_floor.max(LEXICAL_RESCUE * similarity)).min(1.0);
        if weight > 0.0 {
            result.items.push(Seed {
                concept,
                weight,
                kind: EvidenceSource::Lexical,
                residual_level: 0,
            });
        }
    }
    let explicit = result
        .items
        .iter()
        .filter(|seed| seed.kind != EvidenceSource::Inferred)
        .map(|seed| seed.weight)
        .fold(0.0, f64::max);
    let implicit = result
        .items
        .iter()
        .filter(|seed| seed.kind == EvidenceSource::Inferred)
        .map(|seed| seed.weight)
        .fold(0.0, f64::max);
    let cap = explicit * cfg.implicit_max_ratio_of_explicit;
    if explicit > 0.0 && implicit > cap {
        for seed in result
            .items
            .iter_mut()
            .filter(|seed| seed.kind == EvidenceSource::Inferred)
        {
            seed.weight *= cap / implicit;
        }
    }
    result.items.retain(|seed| seed.weight > 0.0);
    if let Some(hypernym) = &cfg.hypernym {
        (result.differentia, result.differentia_confidence) =
            differentia(input, hypernym, &result.pruned_hypernyms, run)?;
    }
    Ok((result, language, named))
}
