// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use nexus_math::{NodeId, OwnedUnitVectors, UnitVectors, VectorRows};

use crate::{CandidateId, DocumentId, Model, Result, SelectedPath, SourceId, input};

pub type CandidateData = Vec<Candidate>;

pub struct Candidate {
    pub id: CandidateId,
    pub vector: Vec<f32>,
    pub document_id: Option<DocumentId>,
    pub sources: Vec<SourceId>,
    pub paths: Option<Vec<PathRef>>,
}

impl Candidate {
    pub fn new(id: CandidateId, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            document_id: None,
            sources: Vec::new(),
            paths: None,
        }
    }

    pub fn with_sources(mut self, sources: Vec<SourceId>) -> Self {
        self.sources = sources;
        self
    }

    pub fn with_paths(mut self, paths: Vec<PathRef>) -> Self {
        self.paths = Some(paths);
        self
    }

    pub fn with_document_id(mut self, document: DocumentId) -> Self {
        self.document_id = Some(document);
        self
    }

    fn into_data(self) -> (Vec<f32>, CandidateMetadata) {
        (
            self.vector,
            CandidateMetadata {
                id: self.id,
                document_id: self.document_id,
                sources: self.sources,
                paths: self.paths,
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PathRef {
    pub source: SourceId,
    pub path: usize,
    pub positions: Option<Vec<usize>>,
}

pub struct CandidateMetadata {
    pub id: CandidateId,
    pub document_id: Option<DocumentId>,
    pub sources: Vec<SourceId>,
    pub paths: Option<Vec<PathRef>>,
}

pub struct CandidateView<'a> {
    vectors: UnitVectors<'a>,
    metadata: HashMap<CandidateId, CandidateMetadata>,
}

impl<'a> CandidateView<'a> {
    pub fn new(vectors: UnitVectors<'a>) -> Self {
        Self {
            vectors,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: Vec<CandidateMetadata>) -> Result<Self> {
        let rows = self.vectors.as_rows();
        let ids: HashSet<_> = rows.ids().iter().copied().collect();
        for item in metadata {
            let id = item.id;
            if !ids.contains(&id) || self.metadata.insert(id, item).is_some() {
                return Err(input(
                    "candidate.metadata",
                    Some(id),
                    "expected unique metadata for an existing vector ID",
                ));
            }
        }
        Ok(self)
    }
}

pub(crate) struct BoundPath {
    pub(crate) source: SourceId,
    pub(crate) ordinal: usize,
    pub(crate) nodes: Vec<NodeId>,
    pub(crate) positions: Vec<usize>,
}

impl BoundPath {
    pub(crate) fn selected(&self, model: &Model) -> SelectedPath {
        let ids = model.concepts().ids();
        SelectedPath {
            source: self.source,
            path: self.ordinal,
            positions: self.positions.clone(),
            concepts: self.nodes.iter().map(|node| ids[node.index()]).collect(),
        }
    }
}

pub(crate) struct CandidateInfo {
    pub(crate) document_id: Option<DocumentId>,
    pub(crate) roots: Option<BTreeSet<SourceId>>,
    pub(crate) paths: Vec<BoundPath>,
}

struct Catalog<'a> {
    identity: Arc<()>,
    vectors: Storage<'a>,
    info: Vec<CandidateInfo>,
    slots: HashMap<CandidateId, usize>,
}

enum Storage<'a> {
    Owned(OwnedUnitVectors),
    Borrowed(UnitVectors<'a>),
}

impl Storage<'_> {
    fn rows(&self) -> VectorRows<'_> {
        match self {
            Self::Owned(vectors) => vectors.as_rows(),
            Self::Borrowed(vectors) => vectors.as_rows(),
        }
    }
}

struct Selection {
    ids: Vec<CandidateId>,
    slots: Vec<usize>,
}

pub struct Candidates<'data> {
    catalog: Arc<Catalog<'data>>,
    selection: Option<Selection>,
}

impl<'data> Candidates<'data> {
    pub fn select(&self, ids: impl IntoIterator<Item = CandidateId>) -> Result<Self> {
        let mut selection = Selection {
            ids: Vec::new(),
            slots: Vec::new(),
        };
        let allowed = self
            .selection
            .as_ref()
            .map(|subset| subset.ids.iter().copied().collect::<HashSet<_>>());
        let mut seen = HashSet::new();
        for id in ids {
            let slot = self
                .catalog
                .slots
                .get(&id)
                .filter(|_| allowed.as_ref().is_none_or(|set| set.contains(&id)))
                .ok_or_else(|| input("candidates.select", Some(id), "unknown candidate ID"))?;
            if !seen.insert(id) {
                return Err(input(
                    "candidates.select",
                    Some(id),
                    "duplicate candidate ID",
                ));
            }
            selection.ids.push(id);
            selection.slots.push(*slot);
        }
        Ok(Self {
            catalog: self.catalog.clone(),
            selection: Some(selection),
        })
    }

    pub fn ids(&self) -> &[CandidateId] {
        self.selection.as_ref().map_or_else(
            || self.catalog.vectors.rows().ids(),
            |selection| &selection.ids,
        )
    }
    pub fn len(&self) -> usize {
        self.ids().len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids().is_empty()
    }

    pub fn vectors(&self) -> VectorRows<'_> {
        let base = self.catalog.vectors.rows();
        self.selection.as_ref().map_or(base, |s| {
            base.select_rows(&s.ids, &s.slots)
                .expect("validated selection")
        })
    }

    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.catalog.identity
    }

    pub(crate) fn info(&self, ordinal: usize) -> &CandidateInfo {
        let slot = self
            .selection
            .as_ref()
            .map_or(ordinal, |s| s.slots[ordinal]);
        &self.catalog.info[slot]
    }
}

impl Model {
    pub fn candidates(&self, data: CandidateData) -> Result<Candidates<'static>> {
        let dimension = self.concepts().dimension();
        let length = data
            .len()
            .checked_mul(dimension)
            .ok_or_else(|| input("candidates", None, "matrix size overflows"))?;
        let mut ids = Vec::with_capacity(data.len());
        let mut values = Vec::with_capacity(length);
        let mut info = Vec::with_capacity(data.len());
        let mut slots = HashMap::with_capacity(data.len());
        for candidate in data {
            let (vector, metadata) = candidate.into_data();
            let id = metadata.id;
            if slots.insert(id, ids.len()).is_some() {
                return Err(input("candidate.id", Some(id), "duplicate candidate ID"));
            }
            if vector.len() != dimension {
                return Err(input(
                    "candidate.vector",
                    Some(id),
                    format!("expected dimension {dimension}, got {}", vector.len()),
                ));
            }
            ids.push(id);
            values.extend(vector);
            info.push(bind(self, metadata)?);
        }
        let vectors = OwnedUnitVectors::new(ids.into(), values, dimension, &Default::default())
            .map_err(|error| match error {
                crate::Error::Input { id, message, .. } => input("candidate.vector", id, message),
                error => error,
            })?;
        Ok(Candidates {
            catalog: Arc::new(Catalog {
                identity: self.identity().clone(),
                vectors: Storage::Owned(vectors),
                info,
                slots,
            }),
            selection: None,
        })
    }

    pub fn candidate_view<'a>(&self, mut data: CandidateView<'a>) -> Result<Candidates<'a>> {
        let rows = data.vectors.as_rows();
        let dimension = self.concepts().dimension();
        if dimension != rows.dimension() {
            return Err(input(
                "candidate.vectors",
                None,
                format!("expected dimension {dimension}, got {}", rows.dimension()),
            ));
        }
        let mut info = Vec::with_capacity(rows.len());
        let mut slots = HashMap::with_capacity(rows.len());
        for (slot, &id) in rows.ids().iter().enumerate() {
            slots.insert(id, slot);
            let metadata = data.metadata.remove(&id).unwrap_or(CandidateMetadata {
                id,
                document_id: None,
                sources: Vec::new(),
                paths: None,
            });
            info.push(bind(self, metadata)?);
        }
        Ok(Candidates {
            catalog: Arc::new(Catalog {
                identity: self.identity().clone(),
                vectors: Storage::Borrowed(data.vectors),
                info,
                slots,
            }),
            selection: None,
        })
    }
}

fn bind(model: &Model, metadata: CandidateMetadata) -> Result<CandidateInfo> {
    let mut sources = BTreeSet::new();
    let mut roots = Some(BTreeSet::new());
    let book = &model.graph_assets().sources;
    let candidate = Some(metadata.id);
    let source_lookup = |source| {
        book.get(source).ok_or_else(|| {
            input(
                "candidate.sources",
                candidate,
                format!("unknown source {source}"),
            )
        })
    };
    let mut paths = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut add_path = |path: PathRef| -> Result<()> {
        if !seen_paths.insert(path.clone()) {
            return Ok(());
        }
        let source = source_lookup(path.source)?;
        let original = source.paths.get(path.path).ok_or_else(|| {
            input(
                "candidate.paths",
                candidate,
                format!("unknown path {} of source {}", path.path, path.source),
            )
        })?;
        let (positions, nodes) = if let Some(positions) = path.positions {
            if positions.windows(2).any(|p| p[0] >= p[1])
                || positions
                    .last()
                    .is_some_and(|&i| i >= original.original.len())
            {
                return Err(input(
                    "candidate.paths.positions",
                    candidate,
                    format!(
                        "invalid source {} path {} positions {positions:?}",
                        path.source, path.path
                    ),
                ));
            }
            let mut seen = HashSet::new();
            positions
                .into_iter()
                .map(|p| (p, original.original[p]))
                .filter(|&(_, node)| seen.insert(node))
                .unzip()
        } else {
            original.unique.iter().copied().unzip()
        };
        paths.push(BoundPath {
            source: path.source,
            ordinal: path.path,
            nodes,
            positions,
        });
        Ok(())
    };

    if let Some(explicit) = metadata.paths {
        for path in explicit {
            sources.insert(path.source);
            add_path(path)?;
        }
    } else {
        for &id in &metadata.sources {
            if sources.insert(id) {
                for path in 0..source_lookup(id)?.paths.len() {
                    add_path(PathRef {
                        source: id,
                        path,
                        positions: None,
                    })?;
                }
            }
        }
    }
    sources.extend(metadata.sources);
    for &source in &sources {
        let known = &source_lookup(source)?.roots;
        match (&mut roots, known) {
            (Some(roots), Some(known)) => roots.extend(known),
            (_, None) => roots = None,
            (None, Some(_)) => {}
        }
    }
    if sources.is_empty() {
        roots = None;
    }
    Ok(CandidateInfo {
        document_id: metadata.document_id,
        roots,
        paths,
    })
}
