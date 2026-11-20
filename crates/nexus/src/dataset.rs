// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::kernel::{norm_sq_f32, normalize_direction};
use nexus_core::*;
use nexus_vcp::*;
use std::collections::{BTreeSet, HashMap};

const STAGE: &str = "prepare";

pub(crate) fn invalid(stage: &'static str, detail: impl Into<String>) -> VcpError {
    VcpError::InvalidInput {
        stage,
        detail: detail.into(),
    }
}

pub struct Concept {
    pub id: u64,
    pub name: String,
    pub aliases: Vec<String>,
    pub vector: Vec<f32>,
}

impl Concept {
    pub fn new(id: u64, name: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id,
            name: name.into(),
            aliases: Vec::new(),
            vector,
        }
    }
}

/// A retrievable row. Rows from the same logical document share a document ID.
pub struct Document {
    pub id: u64,
    pub document_id: Option<u64>,
    pub vector: Vec<f32>,
    pub paths: Vec<Sequence>,
}

impl Document {
    pub fn new(id: u64, vector: Vec<f32>) -> Self {
        Self {
            id,
            document_id: None,
            vector,
            paths: Vec::new(),
        }
    }
}

pub struct Sequence {
    pub source: u64,
    pub tags: Vec<(u64, u32)>,
}

pub struct SourceFact {
    pub id: u64,
    pub checksum: u64,
    pub tags: Vec<u64>,
    pub required_sources: Vec<u64>,
}

pub struct Relation {
    pub from: u64,
    pub to: u64,
    pub source: u64,
    pub evidence: RelationEvidence,
}

pub enum RelationEvidence {
    Positive {
        mass: f64,
    },
    Negative {
        kind: u16,
        strength: f64,
        confidence: f64,
    },
}

#[derive(Default)]
pub struct Dataset {
    pub concepts: Vec<Concept>,
    pub documents: Vec<Document>,
    pub sources: Vec<SourceFact>,
    pub relations: Vec<Relation>,
}

#[derive(Clone, Debug, Default)]
pub struct AssetConfig {
    pub local: LocalConeAgsConfig,
    pub global: GlobalEpaConfig,
    pub facts: FactMatrixConfig,
    pub transport: TransportConfig,
    pub provenance: ProvenanceConfig,
}

/// Long-lived candidate rows and their document, path and source facts.
#[derive(Clone, Copy)]
pub struct Catalog<'a> {
    pub ids: &'a [u64],
    pub document_ids: Option<&'a [u64]>,
    pub vectors: VectorMatrixView<'a>,
    pub root_offsets: &'a [u32],
    pub roots: &'a [RootId],
    pub path_offsets: &'a [u32],
    /// Source root for each path, when the catalog records that attribution.
    pub path_sources: Option<&'a [RootId]>,
    pub tag_offsets: &'a [u32],
    pub tags: &'a [NodeId],
    pub positions: &'a [u32],
}

/// Borrowed assets. The caller can share them across multiple threads and queries.
#[derive(Clone, Copy)]
pub struct PreparedDataset<'a> {
    /// External concept IDs in vector row order.
    pub concept_ids: &'a [u64],
    pub source_ids: &'a [u64],
    pub tags: VectorMatrixView<'a>,
    pub lexical: TagLexicalMetaView<'a>,
    pub graph: TransportGraphView<'a>,
    pub basis: Option<EpaBasisView<'a>>,
    /// Per-concept residuals in vector row order, required by the optional concept index.
    pub tag_residuals: Option<&'a [IntrinsicResidual]>,
    pub provenance: ProvenanceView<'a>,
    pub groups: SupportGroupView<'a>,
    pub sources: RootFactView<'a>,
    pub counterevidence: CounterevidenceAdjView<'a>,
    pub documents: Option<Catalog<'a>>,
}

pub struct Nexus<'a> {
    assets: Assets<'a>,
    pub(crate) concept_slots: HashMap<u64, usize>,
    pub(crate) document_slots: HashMap<u64, usize>,
    pub(crate) source_slots: HashMap<u64, usize>,
}

enum Assets<'a> {
    Owned(Box<OwnedDataset>),
    Borrowed(Box<PreparedDataset<'a>>),
}

struct OwnedDataset {
    dim: usize,
    concept_ids: Vec<u64>,
    tags: Vec<f32>,
    names: Vec<String>,
    aliases: Vec<String>,
    alias_offsets: Vec<u32>,
    graph: TransportGraph,
    basis: EpaBasis,
    provenance: ProvenanceData,
    source_ids: Vec<u64>,
    source_checksums: Vec<u64>,
    source_tag_offsets: Vec<u32>,
    source_tags: Vec<NodeId>,
    source_req_offsets: Vec<u32>,
    source_requirements: Vec<RootId>,
    group_offsets: Vec<u32>,
    group_kinds: Vec<SupportKind>,
    group_roots: Vec<RootId>,
    counterevidence: CounterevidenceAdj,
    tag_residuals: Vec<IntrinsicResidual>,
    documents: Option<OwnedCatalog>,
}

struct OwnedCatalog {
    ids: Vec<u64>,
    document_ids: Vec<u64>,
    vectors: Vec<f32>,
    root_offsets: Vec<u32>,
    roots: Vec<RootId>,
    path_offsets: Vec<u32>,
    path_sources: Vec<RootId>,
    tag_offsets: Vec<u32>,
    tags: Vec<NodeId>,
    positions: Vec<u32>,
}

pub(crate) fn push_offset(offsets: &mut Vec<u32>, len: usize) -> VcpResult<()> {
    offsets.push(u32::try_from(len).map_err(|_| invalid(STAGE, "asset offsets exceed u32"))?);
    Ok(())
}

pub(crate) fn ids(keys: impl IntoIterator<Item = u64>) -> VcpResult<HashMap<u64, usize>> {
    let mut slots = HashMap::new();
    for (slot, key) in keys.into_iter().enumerate() {
        if slots.insert(key, slot).is_some() {
            return Err(invalid(STAGE, format!("duplicate ID {key}")));
        }
    }
    Ok(slots)
}

pub(crate) fn slot(slots: &HashMap<u64, usize>, id: u64) -> VcpResult<usize> {
    slots
        .get(&id)
        .copied()
        .ok_or_else(|| invalid(STAGE, format!("unknown ID {id}")))
}

fn node(slots: &HashMap<u64, usize>, id: u64) -> VcpResult<NodeId> {
    NodeId::from_index(slot(slots, id)?, STAGE)
}

pub(crate) fn validate_unit_matrix(
    matrix: VectorMatrixView<'_>,
    stage: &'static str,
) -> VcpResult<()> {
    matrix.validate(stage)?;
    if matrix.dim == 0 {
        return Err(invalid(stage, "unit matrix must have a positive dimension"));
    }
    for (index, row) in matrix.data.chunks_exact(matrix.dim).enumerate() {
        let squared = norm_sq_f32(row, PrecisionMode::StrictLeftFold);
        if !squared.is_finite() || (squared - 1.0).abs() > 1e-6 {
            return Err(invalid(
                stage,
                format!("row {index} is not a finite unit direction (squared norm {squared})"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn path_steps(
    slots: &HashMap<u64, usize>,
    path: &Sequence,
) -> VcpResult<Vec<(u32, NodeId)>> {
    let mut steps = path
        .tags
        .iter()
        .map(|&(tag, position)| Ok((position, node(slots, tag)?)))
        .collect::<VcpResult<Vec<_>>>()?;
    steps.sort_by_key(|step| step.0);
    Ok(steps)
}

impl Nexus<'static> {
    pub fn prepare(
        data: Dataset,
        config: &AssetConfig,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<Self> {
        let dim = data
            .concepts
            .first()
            .map_or(0, |concept| concept.vector.len());
        if dim == 0 {
            return Err(invalid(
                STAGE,
                "provide a nonempty set of nonempty concept vectors",
            ));
        }
        let concept_slots = ids(data.concepts.iter().map(|tag| tag.id))?;
        let document_slots = ids(data.documents.iter().map(|doc| doc.id))?;
        let source_ids: Vec<_> = data.sources.iter().map(|source| source.id).collect();
        let source_slots = ids(source_ids.iter().copied())?;
        let root = |id| RootId::from_index(slot(&source_slots, id)?, STAGE);

        let mut concept_ids = Vec::new();
        let mut tags = Vec::new();
        let mut names = Vec::new();
        let mut aliases = Vec::new();
        let mut alias_offsets = vec![0];
        for concept in data.concepts {
            ctx.check_cancelled(STAGE)?;

            concept_ids.push(concept.id);
            tags.extend(normalize_direction(&concept.vector, dim, STAGE)?);
            names.push(concept.name);
            aliases.extend(concept.aliases);
            push_offset(&mut alias_offsets, aliases.len())?;
        }
        let tag_view = VectorMatrixView::new(&tags, concept_ids.len(), dim)?;
        let count = source_ids.len();
        let mut source_tags = vec![BTreeSet::new(); count];
        let mut source_checksums = vec![0; count];
        let mut required = vec![Vec::new(); count];
        for fact in data.sources {
            let index = root(fact.id)?.index();
            source_checksums[index] = fact.checksum;
            for tag in fact.tags {
                source_tags[index].insert(node(&concept_slots, tag)?);
            }
            for requirement in fact.required_sources {
                required[index].push(root(requirement)?);
            }
            required[index].sort_unstable();
            required[index].dedup();
        }

        let mut doc_ids = Vec::new();
        let mut doc_vectors = Vec::new();
        let mut doc_document_ids = Vec::new();
        let mut root_offsets = vec![0];
        let mut doc_roots = Vec::new();
        let mut path_offsets = vec![0];
        let mut tag_offsets = vec![0];
        let mut seq_roots = Vec::new();
        let mut seq_tags = Vec::new();
        let mut positions = Vec::new();
        for doc in data.documents {
            ctx.check_cancelled(STAGE)?;

            doc_ids.push(doc.id);
            doc_document_ids.push(doc.document_id.unwrap_or(doc.id));
            doc_vectors.extend(normalize_direction(&doc.vector, dim, STAGE)?);
            let mut roots = BTreeSet::new();
            for path in doc.paths {
                let source = root(path.source)?;
                roots.insert(source);
                roots.extend(required[source.index()].iter().copied());
                seq_roots.push(source);
                for (position, tag) in path_steps(&concept_slots, &path)? {
                    seq_tags.push(tag);
                    positions.push(position);
                    source_tags[source.index()].insert(tag);
                }
                push_offset(&mut tag_offsets, seq_tags.len())?;
            }
            doc_roots.extend(roots);
            push_offset(&mut root_offsets, doc_roots.len())?;
            push_offset(&mut path_offsets, seq_roots.len())?;
        }
        let sequences = SequenceSliceView {
            sequence_count: seq_roots.len(),
            seq_offsets: &tag_offsets,
            seq_source_roots: &seq_roots,
            tag_ids: &seq_tags,
            positions: &positions,
        };
        let mut positive = PositiveRows::default();
        let mut negative = CounterevidenceAdj::default();
        let mut group_roots = Vec::new();
        let mut group_offsets = vec![0];
        let mut group_kinds = Vec::new();
        for (index, requirements) in required.iter().enumerate() {
            let source = RootId::from_index(index, STAGE)?;
            group_roots.push(source);
            group_roots.extend(requirements.iter().copied().filter(|root| *root != source));
            group_kinds.push(SupportKind::Source);
            push_offset(&mut group_offsets, group_roots.len())?;
        }
        for relation in data.relations {
            let from = node(&concept_slots, relation.from)?;
            let to = node(&concept_slots, relation.to)?;
            let source = root(relation.source)?;
            match relation.evidence {
                RelationEvidence::Positive { mass } => {
                    if mass > 0.0 {
                        source_tags[source.index()].extend([from, to]);
                    }
                    positive.sources.push(from);
                    positive.targets.push(to);
                    positive.roots.push(source);
                    positive.groups.push(GroupId(source.0));
                    positive.masses.push(mass);
                }
                RelationEvidence::Negative {
                    kind,
                    strength,
                    confidence,
                } => {
                    negative.sources.push(from);
                    negative.targets.push(to);
                    negative.roots.push(source);
                    negative.relation_kinds.push(kind);
                    negative.strengths.push(strength);
                    negative.confidences.push(confidence);
                }
            }
        }
        let group_view = SupportGroupView {
            group_count: group_kinds.len(),
            group_kinds: &group_kinds,
            group_root_offsets: &group_offsets,
            group_roots: &group_roots,
        };
        let pos = positive.view();
        let neg = NegativeRelationView {
            sources: &negative.sources,
            targets: &negative.targets,
            roots: &negative.roots,
            relation_kinds: &negative.relation_kinds,
            strengths: &negative.strengths,
            confidences: &negative.confidences,
        };
        let explicit_edges: Vec<_> = pos
            .sources
            .iter()
            .zip(pos.targets)
            .zip(pos.masses)
            .map(|((&source, &target), &mass)| (source, target, mass))
            .collect();
        let geometry = local_cone(
            LocalConeAgsInput {
                tag_vectors: tag_view,
                sequences,
                explicit_edges: Some(&explicit_edges),
            },
            &config.local,
            ctx,
        )?;
        let basis = global_epa(
            GlobalEpaInput {
                tag_vectors: tag_view,
                geometry: &geometry,
            },
            &config.global,
            ctx,
        )?;
        let facts = build_fact_matrix(
            FactMatrixInput {
                tag_vectors: tag_view,
                sequences,
                anchor_gains: &geometry.anchor_gains,
                positive_relations: Some(pos),
                negative_relations: Some(neg),
            },
            &config.facts,
            ctx,
        )?;
        let graph = calibrate_transport(
            TransportCalibrateInput {
                fact_matrix: &facts.fact_matrix,
                anchor_gains: &geometry.anchor_gains,
            },
            &config.transport,
            ctx,
        )?;
        let provenance = build_provenance(
            ProvenanceBuildInput {
                transport_row_offsets: &graph.graph.row_offsets,
                transport_targets: &graph.graph.targets,
                sequences,
                positive_relations: Some(pos),
                support_groups: Some(group_view),
            },
            &config.provenance,
            ctx,
        )?;
        let tag_residuals = geometry.intrinsics;
        let mut all_tags = Vec::new();
        let mut source_tag_offsets = vec![0];
        let mut source_requirements = Vec::new();
        let mut source_req_offsets = vec![0];
        for (tags, requirements) in source_tags.into_iter().zip(required) {
            all_tags.extend(tags);
            source_requirements.extend(requirements);
            push_offset(&mut source_tag_offsets, all_tags.len())?;
            push_offset(&mut source_req_offsets, source_requirements.len())?;
        }
        let documents = if doc_ids.is_empty() {
            None
        } else {
            Some(OwnedCatalog {
                ids: doc_ids,
                document_ids: doc_document_ids,
                vectors: doc_vectors,
                root_offsets,
                roots: doc_roots,
                path_offsets,
                path_sources: seq_roots,
                tag_offsets,
                tags: seq_tags,
                positions,
            })
        };
        Ok(Self {
            assets: Assets::Owned(Box::new(OwnedDataset {
                dim,
                concept_ids,
                tags,
                names,
                aliases,
                alias_offsets,
                graph: graph.graph,
                basis,
                provenance,
                source_ids,
                source_checksums,
                source_tag_offsets,
                source_tags: all_tags,
                source_req_offsets,
                source_requirements,
                group_offsets,
                group_kinds,
                group_roots,
                counterevidence: facts.counterevidence,
                tag_residuals,
                documents,
            })),
            concept_slots,
            document_slots,
            source_slots,
        })
    }
}

#[derive(Default)]
struct PositiveRows {
    sources: Vec<NodeId>,
    targets: Vec<NodeId>,
    roots: Vec<RootId>,
    groups: Vec<GroupId>,
    masses: Vec<f64>,
}

impl PositiveRows {
    fn view(&self) -> PositiveRelationView<'_> {
        PositiveRelationView {
            sources: &self.sources,
            targets: &self.targets,
            roots: &self.roots,
            groups: &self.groups,
            masses: &self.masses,
        }
    }
}

impl<'a> Nexus<'a> {
    pub fn from_prepared(data: PreparedDataset<'a>) -> VcpResult<Self> {
        validate_unit_matrix(data.tags, STAGE)?;
        if data.graph.edge_gates.is_some() {
            return Err(invalid(
                STAGE,
                "prepared assets require an ungated transport graph",
            ));
        }
        if let Some(residuals) = data.tag_residuals
            && (residuals.len() != data.tags.rows
                || residuals.iter().enumerate().any(|(i, residual)| {
                    residual.node.index() != i || !(0.0..=1.0).contains(&residual.residual_ratio)
                }))
        {
            return Err(invalid(
                STAGE,
                "tag residuals must align with concept rows and lie in [0, 1]",
            ));
        }
        data.lexical.validate(STAGE)?;
        data.provenance.validate(data.graph.nnz(), STAGE)?;
        data.groups.validate(STAGE)?;
        data.sources.validate(STAGE)?;
        data.counterevidence.validate(data.tags.rows, STAGE)?;
        if data.tags.rows == 0
            || data.graph.node_count != data.tags.rows
            || data.lexical.node_count != data.tags.rows
            || data.concept_ids.len() != data.tags.rows
            || data.source_ids.len() != data.sources.root_count
            || data.basis.is_some_and(|basis| basis.dim != data.tags.dim)
            || data
                .groups
                .group_roots
                .iter()
                .any(|id| id.index() >= data.sources.root_count)
            || data
                .provenance
                .prov_groups
                .iter()
                .any(|id| id.index() >= data.groups.group_count)
            || data
                .sources
                .root_tags
                .iter()
                .any(|id| id.index() >= data.tags.rows)
            || data
                .sources
                .root_required_roots
                .iter()
                .any(|id| id.index() >= data.sources.root_count)
            || data
                .counterevidence
                .roots
                .iter()
                .any(|id| id.index() >= data.sources.root_count)
        {
            return Err(invalid(
                STAGE,
                "prepared concept, graph, provenance and source columns must share slot spaces",
            ));
        }
        if let Some(basis) = data.basis {
            basis.validate(STAGE)?;
        }
        let concept_slots = ids(data.concept_ids.iter().copied())?;
        let source_slots = ids(data.source_ids.iter().copied())?;
        let document_slots = if let Some(catalog) = data.documents {
            catalog.validate(data.tags.dim, data.tags.rows, data.sources.root_count)?;
            ids(catalog.ids.iter().copied())?
        } else {
            HashMap::new()
        };
        Ok(Self {
            assets: Assets::Borrowed(Box::new(data)),
            concept_slots,
            document_slots,
            source_slots,
        })
    }

    /// Borrows the same assets that are used for queries.
    pub fn prepared(&self) -> PreparedDataset<'_> {
        match &self.assets {
            Assets::Borrowed(data) => **data,
            Assets::Owned(data) => PreparedDataset {
                concept_ids: &data.concept_ids,
                source_ids: &data.source_ids,
                tags: VectorMatrixView::new(&data.tags, data.concept_ids.len(), data.dim)
                    .expect("prepared matrix dimensions"),
                lexical: TagLexicalMetaView {
                    names: TextRows::Strings(&data.names),
                    aliases: TextRows::Strings(&data.aliases),
                    alias_offsets: &data.alias_offsets,
                    node_count: data.concept_ids.len(),
                },
                graph: data.graph.view(),
                basis: Some(data.basis.as_view()),
                tag_residuals: Some(&data.tag_residuals),
                provenance: data.provenance.as_view(),
                groups: SupportGroupView {
                    group_count: data.group_kinds.len(),
                    group_kinds: &data.group_kinds,
                    group_root_offsets: &data.group_offsets,
                    group_roots: &data.group_roots,
                },
                sources: RootFactView {
                    root_count: data.source_checksums.len(),
                    checksum_hashes: &data.source_checksums,
                    root_tag_offsets: &data.source_tag_offsets,
                    root_tags: &data.source_tags,
                    root_req_offsets: &data.source_req_offsets,
                    root_required_roots: &data.source_requirements,
                },
                counterevidence: data.counterevidence.as_view(),
                documents: data.documents.as_ref().map(|catalog| Catalog {
                    ids: &catalog.ids,
                    document_ids: Some(&catalog.document_ids),
                    vectors: VectorMatrixView::new(&catalog.vectors, catalog.ids.len(), data.dim)
                        .expect("prepared catalog dimensions"),
                    root_offsets: &catalog.root_offsets,
                    roots: &catalog.roots,
                    path_offsets: &catalog.path_offsets,
                    path_sources: Some(&catalog.path_sources),
                    tag_offsets: &catalog.tag_offsets,
                    tags: &catalog.tags,
                    positions: &catalog.positions,
                }),
            },
        }
    }
}

impl Catalog<'_> {
    fn validate(self, dim: usize, tags: usize, roots: usize) -> VcpResult<()> {
        validate_unit_matrix(self.vectors, STAGE)?;
        let n = self.ids.len();
        let path_count = self.tag_offsets.len().saturating_sub(1);
        let offsets_match = |offsets: &[u32], rows: usize, len: usize| {
            offsets.len() == rows + 1
                && offsets.first() == Some(&0)
                && offsets.last().is_some_and(|&x| x as usize == len)
                && offsets.windows(2).all(|w| w[0] <= w[1])
        };
        if self.vectors.dim != dim
            || self.vectors.rows != n
            || self.document_ids.is_some_and(|ids| ids.len() != n)
            || self.roots.iter().any(|root| root.index() >= roots)
            || self.tags.len() != self.positions.len()
            || self.tags.iter().any(|tag| tag.index() >= tags)
            || !offsets_match(self.root_offsets, n, self.roots.len())
            || !offsets_match(self.path_offsets, n, path_count)
            || !offsets_match(self.tag_offsets, path_count, self.tags.len())
        {
            return Err(invalid(
                STAGE,
                "prepared document catalog columns do not align",
            ));
        }
        if let Some(sources) = self.path_sources {
            if sources.len() != path_count {
                return Err(invalid(
                    STAGE,
                    "prepared path sources must align with paths",
                ));
            }
            for row in 0..n {
                let row_roots = &self.roots
                    [self.root_offsets[row] as usize..self.root_offsets[row + 1] as usize];
                let row_sources =
                    &sources[self.path_offsets[row] as usize..self.path_offsets[row + 1] as usize];
                if row_sources.iter().any(|root| !row_roots.contains(root)) {
                    return Err(invalid(
                        STAGE,
                        "prepared path source must be a root of its candidate",
                    ));
                }
            }
        }
        Ok(())
    }
}
