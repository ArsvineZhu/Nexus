// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::HashMap;
use std::sync::Arc;

use nexus_math::{NodeId, OwnedUnitVectors, UnitVectors};

use crate::{ConceptId, Error, Result, SourceId, input, observation::Lexicon};

pub struct RelationData {
    pub dimension: usize,
    pub concepts: Vec<Concept>,
    pub sources: Vec<Source>,
    pub relations: Vec<Relation>,
}

pub struct Concept {
    pub id: ConceptId,
    pub vector: Vec<f32>,
    pub label: Option<String>,
    pub aliases: Vec<String>,
}

impl Concept {
    pub fn new(id: ConceptId, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            label: None,
            aliases: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases = aliases;
        self
    }
}

pub struct Source {
    pub id: SourceId,
    pub paths: Vec<Vec<ConceptId>>,
    pub origin: SourceOrigin,
}

pub enum SourceOrigin {
    Independent,
    DerivedFrom(Vec<SourceId>),
    Unknown,
}

impl Source {
    pub fn independent(id: SourceId, paths: Vec<Vec<ConceptId>>) -> Self {
        Self {
            id,
            paths,
            origin: SourceOrigin::Independent,
        }
    }

    pub fn derived(id: SourceId, parents: Vec<SourceId>, paths: Vec<Vec<ConceptId>>) -> Self {
        Self {
            id,
            paths,
            origin: SourceOrigin::DerivedFrom(parents),
        }
    }

    pub fn unknown_origin(id: SourceId, paths: Vec<Vec<ConceptId>>) -> Self {
        Self {
            id,
            paths,
            origin: SourceOrigin::Unknown,
        }
    }
}

pub struct Relation {
    pub from: ConceptId,
    pub to: ConceptId,
    pub support: f64,
    pub source: SourceId,
}

pub struct ConceptData {
    vectors: OwnedUnitVectors,
    slots: HashMap<ConceptId, NodeId>,
    names: Vec<(Option<String>, Vec<String>)>,
    lexicon: Lexicon,
}

impl ConceptData {
    pub fn new(dimension: usize, concepts: Vec<Concept>) -> Result<Self> {
        Self::prepare(dimension, concepts, &Default::default())
    }

    pub(super) fn prepare(
        dimension: usize,
        concepts: Vec<Concept>,
        run: &crate::RunOptions,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(input("dimension", None, "must be positive"));
        }
        if std::alloc::Layout::array::<f64>(dimension).is_err() {
            return Err(input("dimension", None, "geometry axis size overflows"));
        }
        let len = concepts
            .len()
            .checked_mul(dimension)
            .filter(|_| u32::try_from(concepts.len()).is_ok())
            .filter(|&len| std::alloc::Layout::array::<f64>(len).is_ok())
            .ok_or_else(|| input("concepts", None, "matrix or slot count overflows"))?;
        let count = concepts.len();
        let mut ids = Vec::with_capacity(count);
        let mut values = Vec::with_capacity(len);
        let mut slots = HashMap::with_capacity(count);
        let mut names = Vec::with_capacity(count);
        let mut lexicon = Lexicon::default();
        for Concept {
            id,
            vector,
            label,
            aliases,
        } in concepts
        {
            run.check_cancelled()?;
            let node = NodeId(ids.len() as u32);
            if slots.insert(id, node).is_some() {
                return Err(input("concept.id", Some(id), "duplicate concept ID"));
            }
            if vector.len() != dimension {
                return Err(input(
                    "concept.vector",
                    Some(id),
                    format!("expected dimension {dimension}, got {}", vector.len()),
                ));
            }
            values.extend(vector);
            ids.push(id);
            lexicon.insert(node, label.as_deref(), &aliases, run)?;
            names.push((label, aliases));
        }
        let vectors = OwnedUnitVectors::new(ids.into(), values, dimension, run).map_err(
            |error| match error {
                Error::Input { id, message, .. } => input("concept.vector", id, message),
                other => other,
            },
        )?;
        Ok(Self {
            vectors,
            slots,
            names,
            lexicon,
        })
    }

    pub fn view(&self) -> ConceptView<'_> {
        ConceptView(self)
    }
}

#[derive(Clone, Copy)]
pub struct ConceptView<'a>(pub(super) &'a ConceptData);

impl<'a> ConceptView<'a> {
    pub fn ids(&self) -> &'a [ConceptId] {
        self.0.vectors.ids()
    }

    pub fn vectors(&self) -> UnitVectors<'a> {
        self.0.vectors.view()
    }

    pub fn label(&self, id: ConceptId) -> Option<&'a str> {
        self.0
            .names
            .get(self.0.slots.get(&id)?.index())?
            .0
            .as_deref()
    }

    pub fn aliases(&self, id: ConceptId) -> Option<&'a [String]> {
        Some(&self.0.names.get(self.0.slots.get(&id)?.index())?.1)
    }

    pub(crate) fn lexicon(&self) -> &'a Lexicon {
        &self.0.lexicon
    }

    pub(crate) fn node(&self, id: ConceptId) -> Result<NodeId> {
        self.0
            .slots
            .get(&id)
            .copied()
            .ok_or_else(|| input("concept", Some(id), "unknown concept ID"))
    }

    pub(crate) fn row(&self, node: NodeId) -> &'a [f32] {
        let dimension = self.dimension();
        let start = node.index() * dimension;
        &self.values()[start..start + dimension]
    }

    pub(crate) fn dimension(&self) -> usize {
        self.0.vectors.dimension()
    }
    pub(super) fn values(&self) -> &'a [f32] {
        self.0.vectors.values()
    }

    pub(crate) fn identity(&self) -> &'a Arc<[ConceptId]> {
        self.0.vectors.ids()
    }
}
