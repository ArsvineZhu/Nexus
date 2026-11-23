// SPDX-License-Identifier: CC-BY-NC-SA-4.0

mod analysis;
mod epoch;
mod fields;
mod fusion;
mod gating;
mod projection;
mod sensing;

pub use analysis::*;
pub use fields::*;
pub use fusion::*;
pub(crate) use gating::Lexicon;
pub use gating::{HypernymConfig, Seed, SeedConfig, SeedInput, Seeds, select_seeds};
pub use projection::*;
pub use sensing::*;

use crate::{ConceptId, ConceptSearch, ConceptView, Error, Model, Query, Result, RunOptions};
use nexus_math::{
    EnergyNode, NodeId, TransportGraph, TransportGraphView, UnitVector, csr::EffectiveTransport,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

macro_rules! options {
    ($($name:ident: $config:ident),+ $(,)?) => {$(
        #[derive(Clone, Default)]
        pub struct $name {
            pub config: $config,
            pub run: RunOptions,
        }
    )+};
}

options! {
    EpaOptions: EpaConfig,
    PyramidOptions: PyramidConfig,
    SeedOptions: SeedConfig,
    SenseOptions: SenseConfig,
    FusionOptions: FusionConfig,
    FieldSolveOptions: FieldSolveConfig,
    FieldProjectionOptions: ProjectionConfig,
}

#[derive(Clone, Default)]
pub struct ObserveOptions<'a> {
    pub config: ObserveConfig,
    pub run: RunOptions,
    pub concept_search: Option<&'a dyn ConceptSearch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObserveConfig {
    pub epa: EpaConfig,
    pub pyramid: Option<PyramidConfig>,
    pub seeds: SeedConfig,
    pub sense: SenseConfig,
    pub fusion: FusionConfig,
    pub fields: FieldsConfig,
}

impl Default for ObserveConfig {
    fn default() -> Self {
        Self {
            epa: EpaConfig::default(),
            pyramid: Some(PyramidConfig::default()),
            seeds: SeedConfig::default(),
            sense: SenseConfig::default(),
            fusion: FusionConfig::default(),
            fields: FieldsConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldsConfig {
    pub solve: FieldSolveConfig,
    pub projection: ProjectionConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryAnalysis {
    pub epa: Option<EpaAnalysis>,
    pub pyramid: Option<PyramidResult>,
    pub seeds: Seeds,
    pub sense: SenseResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fields {
    pub values: DualFields,
    pub projection: FieldProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecallKind {
    Original,
    Enhanced,
    LocalField,
    TransferField,
}

#[derive(Clone, Copy, Debug)]
pub struct RecallVector<'a> {
    pub kind: RecallKind,
    pub vector: &'a [f32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationMetadata {
    pub has_text: bool,
    pub named_concepts: usize,
    pub language_context: LanguageContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageContext {
    Unavailable,
    Unknown,
    Technical,
    Social,
}

pub struct Observation {
    pub(crate) model: Model,
    pub(crate) effective: Option<EffectiveTransport>,
    pub(crate) transition_index: HashMap<crate::ConceptId, Vec<usize>>,
    pub(crate) wormholes: Vec<WormholeFlow>,
    pub(crate) ordinary_first_hops: Vec<(ConceptId, ConceptId)>,
    pub(crate) peak_transition_flow: f64,
    original: UnitVector,
    enhanced: UnitVector,
    analysis: QueryAnalysis,
    fields: Fields,
    metadata: ObservationMetadata,
}

impl Model {
    pub fn observe(&self, query: Query<'_>, options: &ObserveOptions<'_>) -> Result<Observation> {
        self.observe_with(query, &options.config, &options.run, options.concept_search)
    }

    pub(crate) fn observe_with(
        &self,
        query: Query<'_>,
        config: &ObserveConfig,
        run: &RunOptions,
        concept_search: Option<&dyn ConceptSearch>,
    ) -> Result<Observation> {
        run.check_cancelled()?;
        let concepts = self.concepts();
        let original = direction(query.vector, concepts.dimension(), "query.vector")?;
        let query = Query {
            vector: original.as_slice(),
            ..query
        };
        let graph = self.transport().view();
        let mut effective = None;
        if !query.edge_gates.is_empty() {
            let mut gates = vec![1.0_f64; graph.nnz()];
            for gate in query.edge_gates {
                run.check_cancelled()?;
                if !(0.0..=1.0).contains(&gate.factor) {
                    return Err(crate::input(
                        "query.edge_gates.factor",
                        Some(gate.from),
                        "factor must be in [0, 1]",
                    ));
                }
                let from = concepts.node(gate.from)?;
                let to = concepts.node(gate.to)?;
                let edge = graph.find_edge(from, to).ok_or_else(|| {
                    crate::input(
                        "query.edge_gates",
                        Some(gate.from),
                        format!("transport edge to {} does not exist", gate.to),
                    )
                })?;
                gates[edge] = gates[edge].min(gate.factor);
            }
            effective = Some(graph.attenuate(&gates, run)?);
        }
        let graph = effective
            .as_ref()
            .map_or(graph, |effective| effective.view(graph));
        let epa = analysis::epa(query.vector, self.span(), &config.epa, run)?;
        let pyramid = config
            .pyramid
            .as_ref()
            .map(|config| {
                analysis::pyramid(
                    PyramidInput {
                        query: query.vector,
                        concepts,
                        epa: epa.as_ref(),
                        index: self.concept_index(),
                        concept_search,
                    },
                    config,
                    run,
                )
            })
            .transpose()?;
        let (seeds, language_context, named_concepts) = gating::seeds(
            SeedInput {
                query,
                concepts,
                epa: epa.as_ref(),
                pyramid: pyramid.as_ref(),
            },
            &config.seeds,
            run,
        )?;
        let residuals = pyramid
            .as_ref()
            .map_or(&[][..], |pyramid| pyramid.residuals.as_slice());
        let space = ConceptGraph::aligned(self.transport(), graph, concepts);
        let sensing::Propagation {
            sense,
            wormholes,
            ordinary_first_hops,
        } = sensing::propagate(space, &seeds, residuals, &config.sense, run)?;
        let FusionResult { enhanced, .. } = fusion::fuse(
            FusionInput {
                query,
                concepts,
                seeds: &seeds,
                sense: &sense,
            },
            &original,
            &config.fusion,
            run,
        )?;
        let source: Vec<_> = sense
            .nodes
            .iter()
            .map(|node| (node.concept, node.energy))
            .collect();
        let values = fields::solve(graph, self.transport(), &source, &config.fields.solve, run)?;
        let projection =
            projection::project(space, &sense, &values, &config.fields.projection, run)?;
        let has_text = query.text.is_some_and(|text| !text.trim().is_empty());
        let mut transition_index: HashMap<_, Vec<_>> = HashMap::new();
        let mut peak_transition_flow = 0.0_f64;
        for (index, transition) in sense.transitions.iter().enumerate() {
            run.check_cancelled()?;
            transition_index
                .entry(transition.from)
                .or_default()
                .push(index);
            peak_transition_flow = peak_transition_flow.max(transition.flow);
        }
        Ok(Observation {
            model: self.clone(),
            effective,
            transition_index,
            wormholes,
            ordinary_first_hops,
            peak_transition_flow,
            original,
            enhanced,
            analysis: QueryAnalysis {
                epa,
                pyramid,
                seeds,
                sense,
            },
            fields: Fields { values, projection },
            metadata: ObservationMetadata {
                has_text,
                named_concepts,
                language_context,
            },
        })
    }
}

impl Observation {
    pub(crate) fn graph(&self) -> TransportGraphView<'_> {
        let graph = self.model.transport().view();
        self.effective
            .as_ref()
            .map_or(graph, |effective| effective.view(graph))
    }

    pub fn vector(&self, kind: RecallKind) -> Option<&[f32]> {
        match kind {
            RecallKind::Original => Some(self.original.as_slice()),
            RecallKind::Enhanced => Some(self.enhanced.as_slice()),
            RecallKind::LocalField => self
                .fields
                .projection
                .local
                .direction
                .as_ref()
                .map(UnitVector::as_slice),
            RecallKind::TransferField => self
                .fields
                .projection
                .transfer
                .direction
                .as_ref()
                .map(UnitVector::as_slice),
        }
    }

    pub fn recall_vectors(&self) -> impl Iterator<Item = RecallVector<'_>> {
        [
            RecallKind::Original,
            RecallKind::Enhanced,
            RecallKind::LocalField,
            RecallKind::TransferField,
        ]
        .into_iter()
        .filter_map(|kind| {
            self.vector(kind)
                .map(|vector| RecallVector { kind, vector })
        })
    }

    pub fn active_concepts(&self) -> &[EnergyNode] {
        &self.analysis.sense.nodes
    }
    pub fn analysis(&self) -> &QueryAnalysis {
        &self.analysis
    }
    pub fn fields(&self) -> &Fields {
        &self.fields
    }
    pub fn transitions(&self) -> &[Transition] {
        &self.analysis.sense.transitions
    }
    pub fn metadata(&self) -> &ObservationMetadata {
        &self.metadata
    }
}

fn direction(vector: &[f32], dim: usize, field: &'static str) -> Result<UnitVector> {
    if vector.len() != dim {
        return Err(crate::invalid(
            field,
            format!("expected dimension {dim}, got {}", vector.len()),
        ));
    }
    nexus_math::normalize(vector).map_err(|err| match err {
        Error::Input { message, .. } => crate::invalid(field, message),
        other => other,
    })
}

/// Resolves dense transport slots against the shared concept IDs of query-stage inputs.
#[derive(Clone, Copy)]
struct ConceptGraph<'a> {
    view: TransportGraphView<'a>,
    transport: &'a TransportGraph,
    concepts: ConceptView<'a>,
    aligned: bool,
}

impl<'a> ConceptGraph<'a> {
    fn aligned(
        transport: &'a TransportGraph,
        view: TransportGraphView<'a>,
        concepts: ConceptView<'a>,
    ) -> Self {
        Self {
            transport,
            view,
            concepts,
            aligned: true,
        }
    }

    fn new(
        transport: &'a TransportGraph,
        concepts: ConceptView<'a>,
        run: &RunOptions,
    ) -> Result<Self> {
        let mut space = Self::aligned(transport, transport.view(), concepts);
        let ids = transport.nodes();
        space.aligned = ids == concepts.ids();
        if !space.aligned {
            if ids.len() != concepts.ids().len() {
                return Err(crate::invalid(
                    "concepts",
                    "graph and concepts must contain the same IDs",
                ));
            }
            for &concept in ids {
                run.check_cancelled()?;
                concepts.node(concept)?;
            }
        }
        Ok(space)
    }

    fn ids(self) -> &'a [ConceptId] {
        self.transport.nodes()
    }

    fn id(self, slot: NodeId) -> ConceptId {
        self.ids()[slot.index()]
    }

    fn slot(self, concept: ConceptId) -> Result<NodeId> {
        self.transport
            .slot(concept)
            .ok_or_else(|| crate::input("concept", Some(concept), "unknown graph concept ID"))
    }

    fn row(self, slot: NodeId) -> &'a [f32] {
        let concept = if self.aligned {
            slot
        } else {
            self.concepts
                .node(self.id(slot))
                .expect("validated graph concept")
        };
        self.concepts.row(concept)
    }
}
