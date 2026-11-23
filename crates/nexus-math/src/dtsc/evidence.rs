// SPDX-License-Identifier: Apache-2.0

use super::curve::{CurveBody, CurveScratch, DtscCurveContext, TagBodyCosine, eval_path};
use super::field::{GraphView, Sampler, dtsc_cosine};
use super::{
    CurveMetrics, DtscEvidenceConfig, Field, FieldStats, GeometryState, SampleInput, unit,
};
use crate::kernel::{PrecisionMode, norm_sq_f32};
use crate::matrix::validate_vector;
use crate::{EffectiveTransport, Error, Result, RunOptions, TransportGraph, VectorRows};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct Path<'a> {
    pub candidate: u64,
    pub ordinal: usize,
    pub concepts: &'a [u64],
}

pub struct EvidenceInput<'a> {
    pub field: &'a Field,
    pub concepts: VectorRows<'a>,
    pub graph: Option<&'a TransportGraph>,
    pub query: &'a [f32],
    pub enhanced_query: Option<&'a [f32]>,
    pub candidates: VectorRows<'a>,
    pub paths: &'a [Path<'a>],
    pub geometry: Option<GeometryState>,
}

#[derive(Clone, Debug, Default)]
pub struct EvidenceOptions {
    pub config: DtscEvidenceConfig,
    pub run: RunOptions,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurveEvidence {
    pub candidate: u64,
    pub path: Option<usize>,
    pub metrics: CurveMetrics,
}

/// All candidate geometry and the shared statistics used by the trust guard.
#[derive(Clone, Debug)]
pub struct Evidence {
    items: Vec<CurveEvidence>,
    slots: HashMap<u64, usize>,
    pub(super) field: FieldStats,
    min_field_tags: usize,
    pub(super) stats: CandidateStats,
}

impl Evidence {
    pub fn items(&self) -> &[CurveEvidence] {
        &self.items
    }

    pub fn field_stats(&self) -> &FieldStats {
        &self.field
    }

    pub(super) fn item(&self, candidate: u64) -> Option<&CurveEvidence> {
        self.slots.get(&candidate).map(|&slot| &self.items[slot])
    }

    pub(super) fn field_has_support(&self) -> bool {
        self.field.positive_nodes >= self.min_field_tags
            && self.field.retained_nodes >= self.min_field_tags
            && self.field.total_mass > 0.0
    }
}

#[derive(Clone, Debug)]
pub(super) struct CandidateStats {
    pub contributors: usize,
    pub maximum: f64,
    pub minimum: f64,
    pub strong_evidence: f64,
}

impl CandidateStats {
    fn new(items: &[CurveEvidence]) -> Self {
        let mut stats = Self {
            contributors: 0,
            maximum: 0.0,
            minimum: f64::INFINITY,
            strong_evidence: 0.0,
        };
        for item in items {
            let metrics = &item.metrics;
            stats.strong_evidence += metrics.strong_hits as f64 + 0.75 * metrics.exact_hits as f64;
            if metrics.geo_score > 0.0 {
                stats.contributors += 1;
                stats.maximum = stats.maximum.max(metrics.geo_score);
                stats.minimum = stats.minimum.min(metrics.geo_score);
            }
        }
        stats
    }
}

pub fn evidence(input: EvidenceInput<'_>, options: &EvidenceOptions) -> Result<Evidence> {
    evidence_with_shared(input, options, SharedEvidence::default())
}

#[doc(hidden)]
#[derive(Default)]
pub struct SharedEvidence<'a> {
    pub effective: Option<&'a EffectiveTransport>,
    pub cosines: Option<[&'a [f64]; 2]>,
    pub tag_body: Option<&'a TagBodyCosine<'a>>,
}

#[doc(hidden)]
pub fn evidence_with_shared<'a>(
    input: EvidenceInput<'a>,
    options: &EvidenceOptions,
    shared: SharedEvidence<'a>,
) -> Result<Evidence> {
    let config = &options.config;
    let run = &options.run;
    run.check_cancelled()?;
    config.validate()?;
    let dimension = input.concepts.dimension();
    for (field, actual) in [
        ("field.dimension", input.field.dimension()),
        ("query", input.query.len()),
        ("candidates.dimension", input.candidates.dimension()),
    ] {
        if actual != dimension {
            return Err(Error::dimension(field, dimension, actual));
        }
    }
    validate_vector(input.query, "query", None)?;
    if let Some(enhanced) = input.enhanced_query {
        if enhanced.len() != dimension {
            return Err(Error::dimension(
                "enhanced_query",
                dimension,
                enhanced.len(),
            ));
        }
        validate_vector(enhanced, "enhanced_query", None)?;
    }
    let geometry = match config.auxiliary.as_ref().and_then(|a| a.geometry.as_ref()) {
        Some(_) if !input.candidates.is_empty() => {
            let state = input.geometry.ok_or_else(|| {
                Error::input(
                    "geometry",
                    "enabled geometry auxiliary requires EPA and Pyramid measurements",
                )
            })?;
            for (field, value) in [
                ("geometry.epa_concentration", state.epa_concentration),
                ("geometry.epa_entropy", state.epa_entropy),
                ("geometry.epa_resonance", state.epa_resonance),
                ("geometry.pyramid_coverage", state.pyramid_coverage),
                ("geometry.pyramid_novelty", state.pyramid_novelty),
                ("geometry.pyramid_depth", state.pyramid_depth),
            ] {
                if !value.is_finite() {
                    return Err(Error::input(field, "measurement must be finite"));
                }
            }
            if let Some(columns) = shared.cosines {
                for column in columns {
                    if column.len() != input.candidates.len() {
                        return Err(Error::dimension(
                            "candidate_cosines",
                            input.candidates.len(),
                            column.len(),
                        ));
                    }
                }
            }
            Some(state)
        }
        _ => None,
    };
    let graph = input.graph.map(|graph| {
        let base = graph.view();
        let view = shared
            .effective
            .map_or(base, |effective| effective.view(base));
        GraphView::new(graph, view)
    });

    let concept_rows: HashMap<_, _> = input
        .concepts
        .ids()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    let candidate_rows: HashMap<_, _> = input
        .candidates
        .ids()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    let sampler = Sampler::new(input.field, graph);
    let mut field_samples = HashMap::new();
    let mut candidate_paths = vec![Vec::new(); input.candidates.len()];
    let mut neighbor_buffer = Vec::new();
    let mut path_ids = HashSet::with_capacity(input.paths.len());
    let mut path_concepts = HashSet::new();
    for path in input.paths {
        run.check_cancelled()?;
        let index = *candidate_rows
            .get(&path.candidate)
            .ok_or_else(|| Error::Input {
                field: "paths.candidate",
                id: Some(path.candidate),
                message: "unknown candidate".into(),
            })?;
        if !path_ids.insert((path.candidate, path.ordinal)) {
            return Err(Error::Input {
                field: "paths.ordinal",
                id: Some(path.candidate),
                message: format!("duplicate ordinal {}", path.ordinal),
            });
        }
        path_concepts.clear();
        let mut concepts = Vec::with_capacity(path.concepts.len());
        for (position, &id) in path.concepts.iter().enumerate() {
            if position.is_multiple_of(64) {
                run.check_cancelled()?;
            }
            if path_concepts.insert(id) {
                concepts.push(id);
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = field_samples.entry(id) {
                let row = *concept_rows
                    .get(&id)
                    .filter(|_| input.field.knows(id))
                    .ok_or_else(|| Error::Input {
                        field: "paths.concepts",
                        id: Some(id),
                        message: "unknown path concept".into(),
                    })?;
                let vector = input.concepts.row(row).expect("validated concept index");
                let norm = norm_sq_f32(vector, PrecisionMode::StrictLeftFold);
                let sample = sampler.sample(
                    SampleInput {
                        concept: Some(id),
                        vector,
                    },
                    norm,
                    &mut neighbor_buffer,
                );
                entry.insert((row, sample, norm));
            }
        }
        candidate_paths[index].push((path.ordinal, concepts));
    }
    let curve = DtscCurveContext {
        config,
        run,
        field_samples: &field_samples,
        graph,
        tag_vectors: input.concepts,
        geometry,
    };
    let score = |scratch: &mut CurveScratch, index: usize| {
        run.check_cancelled()?;
        let candidate = input.candidates.ids()[index];
        let mut item = CurveEvidence {
            candidate,
            path: None,
            metrics: CurveMetrics::default(),
        };
        if candidate_paths[index].is_empty() {
            return Ok(item);
        }
        let vector = input
            .candidates
            .row(index)
            .expect("validated candidate index");
        let queries = if geometry.is_some() {
            let as_closure = |cosine| unit((cosine + 1.0) / 2.0);
            let original = shared.cosines.map_or_else(
                || dtsc_cosine(input.query, vector),
                |values| values[0][index],
            );
            let enhanced = input.enhanced_query.map(|query| {
                shared
                    .cosines
                    .map_or_else(|| dtsc_cosine(query, vector), |values| values[1][index])
            });
            (as_closure(original), enhanced.map(as_closure))
        } else {
            (0.0, None)
        };
        let body = CurveBody {
            candidate,
            vector,
            norm: norm_sq_f32(vector, PrecisionMode::StrictLeftFold),
            query_closures: queries,
            cached_cosines: shared.tag_body,
        };

        for (ordinal, tags) in &candidate_paths[index] {
            run.check_cancelled()?;
            let metrics = eval_path(&body, tags, &curve, scratch)?;
            let order = metrics
                .geo_bonus
                .total_cmp(&item.metrics.geo_bonus)
                .then_with(|| metrics.geo_score.total_cmp(&item.metrics.geo_score));
            if item.path.is_none() || order.is_gt() {
                item.path = Some(*ordinal);
                item.metrics = metrics;
            }
        }
        Ok(item)
    };
    let items = if run.parallelism.is_parallel() {
        (0..input.candidates.len())
            .into_par_iter()
            .map_init(CurveScratch::default, score)
            .collect::<Result<Vec<_>>>()?
    } else {
        let mut scratch = CurveScratch::default();
        (0..input.candidates.len())
            .map(|index| score(&mut scratch, index))
            .collect::<Result<Vec<_>>>()?
    };
    run.check_cancelled()?;
    Ok(Evidence {
        stats: CandidateStats::new(&items),
        items,
        slots: candidate_rows,
        field: input.field.stats,
        min_field_tags: input.field.config.min_field_tags,
    })
}
