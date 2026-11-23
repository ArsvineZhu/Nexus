// SPDX-License-Identifier: Apache-2.0

use super::{DtscFieldConfig, unit};
use crate::kernel::{PrecisionMode, cosine_f32_with_norms, norm_sq_f32};
use crate::matrix::validate_vector;
use crate::{Error, Result, RunOptions, TransportGraph, TransportGraphView, VectorRows};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

const FP64: PrecisionMode = PrecisionMode::StrictLeftFold;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    Core,
    Lexical,
    Inferred,
    Emergent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyNode {
    pub concept: u64,
    pub energy: f64,
    pub kind: EvidenceSource,
    pub hop: usize,
    pub seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldStats {
    pub positive_nodes: usize,
    pub retained_nodes: usize,
    pub entropy: f64,
    pub total_mass: f64,
}

#[derive(Clone, Debug, Default)]
pub struct FieldOptions {
    pub config: DtscFieldConfig,
    pub run: RunOptions,
}

#[derive(Clone, Copy, Debug)]
pub struct SampleInput<'a> {
    pub concept: Option<u64>,
    pub vector: &'a [f32],
}

#[derive(Clone, Debug, Default)]
pub struct SampleOptions {
    pub run: RunOptions,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub concept: Option<u64>,
    pub potential: f64,
    pub exact: bool,
    pub kind: Option<EvidenceSource>,
    pub specificity: f64,
}

#[derive(Clone, Debug)]
struct FieldNode {
    concept: u64,
    potential: f64,
    vector: Vec<f32>,
    vector_norm: f64,
    kind: EvidenceSource,
}

/// Positive potentials cover all input identities. Only interpolation nodes own vectors.
#[derive(Clone, Debug)]
pub struct Field {
    concepts: HashSet<u64>,
    dimension: usize,
    exact: HashMap<u64, (f64, EvidenceSource)>,
    retained: Vec<FieldNode>,
    pub(super) stats: FieldStats,
    pub(super) config: DtscFieldConfig,
}

impl Field {
    pub(super) fn validate_sample(&self, input: SampleInput<'_>) -> Result<f64> {
        if input.vector.len() != self.dimension {
            return Err(Error::dimension(
                "sample.vector",
                self.dimension,
                input.vector.len(),
            ));
        }
        if let Some(concept) = input.concept
            && !self.concepts.contains(&concept)
        {
            return Err(Error::Input {
                field: "sample.concept",
                id: Some(concept),
                message: "unknown concept".into(),
            });
        }
        validate_vector(input.vector, "sample.vector", input.concept)
    }

    pub(super) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(super) fn knows(&self, concept: u64) -> bool {
        self.concepts.contains(&concept)
    }
}

pub fn build_field(
    concepts: VectorRows<'_>,
    nodes: &[EnergyNode],
    options: &FieldOptions,
) -> Result<Field> {
    let run = &options.run;
    let config = &options.config;
    run.check_cancelled()?;
    config.validate()?;

    let lookup: HashMap<_, _> = concepts
        .ids()
        .iter()
        .copied()
        .enumerate()
        .map(|(row, id)| (id, row))
        .collect();
    let mut seen = HashSet::with_capacity(nodes.len());
    let mut positive = Vec::new();
    for &node in nodes {
        run.check_cancelled()?;
        let row = lookup
            .get(&node.concept)
            .copied()
            .ok_or_else(|| Error::Input {
                field: "nodes.concept",
                id: Some(node.concept),
                message: "unknown field concept".into(),
            })?;
        if !seen.insert(node.concept) {
            return Err(Error::Input {
                field: "nodes.concept",
                id: Some(node.concept),
                message: "duplicate field concept".into(),
            });
        }
        if let Some(seed) = node.seed
            && !lookup.contains_key(&seed)
        {
            return Err(Error::Input {
                field: "nodes.seed",
                id: Some(seed),
                message: "unknown seed concept".into(),
            });
        }
        if !node.energy.is_finite() || node.energy < 0.0 {
            return Err(Error::Input {
                field: "nodes.energy",
                id: Some(node.concept),
                message: "energy must be finite and nonnegative".into(),
            });
        }
        if node.energy > 0.0 {
            positive.push((node, row));
        }
    }

    positive.sort_unstable_by(|(a, _), (b, _)| {
        b.energy
            .total_cmp(&a.energy)
            .then_with(|| a.concept.cmp(&b.concept))
    });
    let total_mass: f64 = positive.iter().map(|(node, _)| node.energy).sum();
    if !total_mass.is_finite() {
        return Err(Error::numeric("build_field", "total field mass overflowed"));
    }
    let entropy = if positive.len() > 1 {
        let sum = positive
            .iter()
            .map(|(node, _)| {
                let p = node.energy / total_mass;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum::<f64>();
        unit(sum / (positive.len() as f64).ln())
    } else {
        0.0
    };

    let maximum = positive.first().map_or(0.0, |(node, _)| node.energy);
    let mut exact = HashMap::with_capacity(positive.len());
    let mut retained = Vec::new();
    let mut retained_mass = 0.0;
    for (node, row) in positive {
        run.check_cancelled()?;
        let potential = node.energy / maximum;
        exact.insert(node.concept, (potential, node.kind));
        if retained.len() < config.max_field_nodes
            && (retained.len() < config.min_field_tags
                || retained_mass / total_mass < config.field_energy_mass_ratio)
        {
            let vector = concepts.row(row).expect("validated concept index");
            retained.push(FieldNode {
                concept: node.concept,
                potential,
                vector: vector.to_vec(),
                vector_norm: norm_sq_f32(vector, FP64),
                kind: node.kind,
            });
            retained_mass += node.energy;
        }
    }
    Ok(Field {
        concepts: lookup.into_keys().collect(),
        dimension: concepts.dimension(),
        stats: FieldStats {
            positive_nodes: exact.len(),
            retained_nodes: retained.len(),
            entropy,
            total_mass,
        },
        exact,
        retained,
        config: config.clone(),
    })
}

#[derive(Clone, Copy)]
pub(super) struct GraphView<'a> {
    graph: &'a TransportGraph,
    view: TransportGraphView<'a>,
}

impl<'a> GraphView<'a> {
    pub fn new(graph: &'a TransportGraph, view: TransportGraphView<'a>) -> Self {
        Self { graph, view }
    }

    pub fn specificity(self, id: Option<u64>, floor: f64) -> f64 {
        if let Some(node) = id.and_then(|id| self.graph.slot(id))
            && self.view.max_inbound > 0.0
        {
            return floor
                .max(1.0 - unit(self.view.inbound_mass(node) / self.view.max_inbound).sqrt());
        }
        1.0
    }

    pub fn anchor_gain(self, id: u64) -> f64 {
        self.graph
            .slot(id)
            .map_or(1.0, |node| self.view.anchor_gain(node))
            .clamp(0.5, 2.0)
    }

    pub fn edge(self, from: u64, to: u64) -> f64 {
        let (Some(from), Some(to)) = (self.graph.slot(from), self.graph.slot(to)) else {
            return 0.0;
        };
        self.view.edge_weight(from, to)
    }
}

pub(super) struct Sampler<'a> {
    field: &'a Field,
    graph: Option<GraphView<'a>>,
    specificity: Vec<f64>,
}

impl<'a> Sampler<'a> {
    pub fn new(field: &'a Field, graph: Option<GraphView<'a>>) -> Self {
        let specificity = field
            .retained
            .iter()
            .map(|node| {
                graph.map_or(1.0, |g| {
                    g.specificity(Some(node.concept), field.config.public_hub_floor)
                })
            })
            .collect();
        Self {
            field,
            graph,
            specificity,
        }
    }

    pub fn sample(
        &self,
        input: SampleInput<'_>,
        vector_norm: f64,
        buffer: &mut Vec<(f64, EvidenceSource)>,
    ) -> Sample {
        let config = &self.field.config;
        let exact = input.concept.and_then(|id| self.field.exact.get(&id));
        let mut kind = exact.map(|(_, kind)| *kind);
        buffer.clear();
        for (node, &specificity) in self.field.retained.iter().zip(&self.specificity) {
            if Some(node.concept) == input.concept {
                continue;
            }
            let similarity = cosine_f32_with_norms(
                input.vector,
                &node.vector,
                vector_norm,
                node.vector_norm,
                FP64,
                0.0,
            );
            let scaled = unit(
                (similarity - config.field_similarity_threshold)
                    / (1.0 - config.field_similarity_threshold),
            );
            let contribution =
                node.potential * scaled.powf(config.field_kernel_exponent) * specificity;
            if contribution > 0.0 {
                buffer.push((contribution, node.kind));
            }
        }
        buffer.sort_by(|a, b| b.0.total_cmp(&a.0));
        buffer.truncate(config.max_field_neighbors);
        let interpolated = if buffer.is_empty() {
            0.0
        } else {
            kind = kind.or_else(|| buffer.first().map(|(_, kind)| *kind));
            buffer.iter().map(|(value, _)| *value).sum::<f64>() / (buffer.len() as f64).sqrt()
        };
        Sample {
            concept: input.concept,
            potential: unit(
                exact
                    .map_or(0.0, |(potential, _)| *potential)
                    .max(interpolated),
            ),
            exact: exact.is_some(),
            kind,
            specificity: self.graph.map_or(1.0, |g| {
                g.specificity(input.concept, config.public_hub_floor)
            }),
        }
    }
}

pub fn sample_field(
    field: &Field,
    graph: Option<&TransportGraph>,
    inputs: &[SampleInput<'_>],
    options: &SampleOptions,
) -> Result<Vec<Sample>> {
    let run = &options.run;
    run.check_cancelled()?;
    let sampler = Sampler::new(field, graph.map(|g| GraphView::new(g, g.view())));
    let sample = |buffer: &mut Vec<_>, &input: &SampleInput<'_>| {
        run.check_cancelled()?;
        let norm = field.validate_sample(input)?;
        Ok(sampler.sample(input, norm, buffer))
    };
    let samples = if run.parallelism.is_parallel() {
        inputs
            .par_iter()
            .map_init(Vec::new, sample)
            .collect::<Result<Vec<_>>>()?
    } else {
        let mut buffer = Vec::new();
        inputs
            .iter()
            .map(|input| sample(&mut buffer, input))
            .collect::<Result<Vec<_>>>()?
    };
    run.check_cancelled()?;
    Ok(samples)
}

pub(super) fn dtsc_cosine(a: &[f32], b: &[f32]) -> f64 {
    let aa = norm_sq_f32(a, FP64);
    let bb = norm_sq_f32(b, FP64);
    cosine_f32_with_norms(a, b, aa, bb, FP64, 0.0)
}
