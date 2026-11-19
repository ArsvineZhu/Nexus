// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::{invalid, path_steps, push_offset, slot, validate_unit_matrix};
use crate::{Nexus, Sequence};
use nexus_core::kernel::normalize_direction;
use nexus_core::{CandidateRowsView, DocumentId, NodeId, RootId, VcpResult, VectorMatrixView};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};

const STAGE: &str = "candidates";

pub struct Candidate {
    pub id: u64,
    pub document_id: Option<u64>,
    pub vector: Vec<f32>,
    pub paths: Vec<Sequence>,
}

impl Candidate {
    pub fn new(id: u64, vector: Vec<f32>) -> Self {
        Self {
            id,
            document_id: None,
            vector,
            paths: Vec::new(),
        }
    }
}

pub enum CandidateInput<'rows> {
    Ids(&'rows [u64]),
    Owned(Vec<Candidate>),
    Borrowed(BorrowedCandidates<'rows>),
}

pub struct BorrowedCandidates<'rows> {
    ids: &'rows [u64],
    vectors: VectorMatrixView<'rows>,
    document_ids: Option<&'rows [u64]>,
    paths: Option<&'rows [&'rows [Sequence]]>,
}

impl<'rows> BorrowedCandidates<'rows> {
    pub fn new(ids: &'rows [u64], vectors: VectorMatrixView<'rows>) -> Self {
        Self {
            ids,
            vectors,
            document_ids: None,
            paths: None,
        }
    }

    pub fn with_document_ids(mut self, ids: &'rows [u64]) -> Self {
        self.document_ids = Some(ids);
        self
    }

    pub fn with_paths(mut self, paths: &'rows [&'rows [Sequence]]) -> Self {
        self.paths = Some(paths);
        self
    }
}

/// Static ordered candidate facts that may be evaluated by multiple observations.
pub struct CandidateSet<'data> {
    pub(crate) nexus: &'data Nexus<'data>,
    ids: Cow<'data, [u64]>,
    document_ids: Cow<'data, [u64]>,
    vectors: CandidateVectors<'data>,
    columns: CandidateColumns,
    pub(crate) slots: HashMap<u64, usize>,
}

enum CandidateVectors<'data> {
    Owned { data: Vec<f32>, dim: usize },
    Borrowed(VectorMatrixView<'data>),
}

struct CandidateColumns {
    documents: Vec<DocumentId>,
    root_offsets: Vec<u32>,
    roots: Vec<RootId>,
    path_offsets: Vec<u32>,
    path_tag_offsets: Vec<u32>,
    path_tags: Vec<NodeId>,
    path_positions: Vec<u32>,
}

impl CandidateColumns {
    fn new() -> Self {
        Self {
            documents: Vec::new(),
            root_offsets: vec![0],
            roots: Vec::new(),
            path_offsets: vec![0],
            path_tag_offsets: vec![0],
            path_tags: Vec::new(),
            path_positions: Vec::new(),
        }
    }

    fn add_path(&mut self, tags: &[NodeId], positions: &[u32]) -> VcpResult<()> {
        self.path_tags.extend_from_slice(tags);
        self.path_positions.extend_from_slice(positions);
        push_offset(&mut self.path_tag_offsets, self.path_tags.len())
    }

    fn finish_row(&mut self, roots: impl IntoIterator<Item = RootId>) -> VcpResult<()> {
        self.roots.extend(roots);
        push_offset(&mut self.root_offsets, self.roots.len())?;
        push_offset(&mut self.path_offsets, self.path_tag_offsets.len() - 1)
    }

    fn add_host_row(&mut self, nexus: &Nexus<'_>, paths: &[Sequence]) -> VcpResult<()> {
        let mut roots = BTreeSet::new();
        let sources = nexus.prepared().sources;
        for path in paths {
            let root = RootId::from_index(slot(&nexus.source_slots, path.source)?, STAGE)?;
            roots.insert(root);
            roots.extend(sources.required_roots(root));
            for (position, tag) in path_steps(&nexus.concept_slots, path)? {
                self.path_positions.push(position);
                self.path_tags.push(tag);
            }
            push_offset(&mut self.path_tag_offsets, self.path_tags.len())?;
        }
        self.finish_row(roots)
    }
}

fn offsets(values: &[u32], row: usize) -> std::ops::Range<usize> {
    values[row] as usize..values[row + 1] as usize
}

impl<'data> CandidateSet<'data> {
    fn new(
        nexus: &'data Nexus<'data>,
        ids: Cow<'data, [u64]>,
        document_ids: Cow<'data, [u64]>,
        vectors: CandidateVectors<'data>,
        mut columns: CandidateColumns,
    ) -> VcpResult<Self> {
        let slots = crate::dataset::ids(ids.iter().copied())?;
        let mut documents = HashMap::new();
        for &id in document_ids.iter() {
            let next = DocumentId::from_index(documents.len(), STAGE)?;
            columns.documents.push(*documents.entry(id).or_insert(next));
        }
        Ok(Self {
            nexus,
            ids,
            document_ids,
            vectors,
            columns,
            slots,
        })
    }

    pub(crate) fn from_catalog_slots(
        nexus: &'data Nexus<'data>,
        slots: impl IntoIterator<Item = usize>,
    ) -> VcpResult<Self> {
        let assets = nexus.prepared();
        let Some(catalog) = assets.documents else {
            return Err(invalid(STAGE, "candidate IDs require a prepared catalog"));
        };
        let mut ids = Vec::new();
        let mut document_ids = Vec::new();
        let mut vectors = Vec::new();
        let mut columns = CandidateColumns::new();
        for row in slots {
            let id = catalog.ids[row];
            ids.push(id);
            document_ids.push(catalog.document_ids.map_or(id, |values| values[row]));
            vectors.extend_from_slice(catalog.vectors.row(row).expect("validated catalog"));
            for path in offsets(catalog.path_offsets, row) {
                let steps = offsets(catalog.tag_offsets, path);
                columns.add_path(&catalog.tags[steps.clone()], &catalog.positions[steps])?;
            }
            columns.finish_row(
                catalog.roots[offsets(catalog.root_offsets, row)]
                    .iter()
                    .copied(),
            )?;
        }
        Self::new(
            nexus,
            Cow::Owned(ids),
            Cow::Owned(document_ids),
            CandidateVectors::Owned {
                data: vectors,
                dim: assets.tags.dim,
            },
            columns,
        )
    }

    pub fn ids(&self) -> &[u64] {
        &self.ids
    }
    pub fn document_ids(&self) -> &[u64] {
        &self.document_ids
    }

    pub(crate) fn rows(&self) -> CandidateRowsView<'_> {
        let columns = &self.columns;
        CandidateRowsView {
            candidate_count: self.ids.len(),
            candidate_vectors: match &self.vectors {
                CandidateVectors::Owned { data, dim } => {
                    VectorMatrixView::new(data, self.ids.len(), *dim)
                        .expect("prepared candidate dimensions")
                }
                CandidateVectors::Borrowed(view) => *view,
            },
            document_ids: &columns.documents,
            candidate_root_offsets: &columns.root_offsets,
            candidate_roots: &columns.roots,
            candidate_path_offsets: &columns.path_offsets,
            path_tag_offsets: &columns.path_tag_offsets,
            path_tags: &columns.path_tags,
            path_positions: &columns.path_positions,
        }
    }
}

impl Nexus<'_> {
    pub fn candidates<'data>(
        &'data self,
        input: CandidateInput<'data>,
    ) -> VcpResult<CandidateSet<'data>> {
        let dim = self.prepared().tags.dim;
        let mut columns = CandidateColumns::new();
        match input {
            CandidateInput::Ids(keys) => {
                let slots = keys
                    .iter()
                    .map(|&id| slot(&self.document_slots, id))
                    .collect::<VcpResult<Vec<_>>>()?;
                CandidateSet::from_catalog_slots(self, slots)
            }
            CandidateInput::Owned(rows) => {
                let mut ids = Vec::with_capacity(rows.len());
                let mut document_ids = Vec::with_capacity(rows.len());
                let mut vectors = Vec::with_capacity(rows.len().saturating_mul(dim));
                for row in rows {
                    ids.push(row.id);
                    document_ids.push(row.document_id.unwrap_or(row.id));
                    vectors.extend(normalize_direction(&row.vector, dim, STAGE)?);
                    columns.add_host_row(self, &row.paths)?;
                }
                CandidateSet::new(
                    self,
                    Cow::Owned(ids),
                    Cow::Owned(document_ids),
                    CandidateVectors::Owned { data: vectors, dim },
                    columns,
                )
            }
            CandidateInput::Borrowed(rows) => {
                if rows.vectors.dim != dim
                    || rows.vectors.rows != rows.ids.len()
                    || rows
                        .document_ids
                        .is_some_and(|ids| ids.len() != rows.ids.len())
                    || rows
                        .paths
                        .is_some_and(|paths| paths.len() != rows.ids.len())
                {
                    return Err(invalid(
                        STAGE,
                        "borrowed candidate columns must align with candidate IDs and asset dimension",
                    ));
                }
                let mut vectors = rows.vectors;
                vectors.sq_norms_f64 = None;
                validate_unit_matrix(vectors, STAGE)?;
                for row in 0..rows.ids.len() {
                    columns.add_host_row(self, rows.paths.map_or(&[], |paths| paths[row]))?;
                }
                CandidateSet::new(
                    self,
                    Cow::Borrowed(rows.ids),
                    Cow::Borrowed(rows.document_ids.unwrap_or(rows.ids)),
                    CandidateVectors::Borrowed(vectors),
                    columns,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Concept, Dataset, ExecutionContext};

    #[test]
    fn borrowed_candidates_validate_directions_without_copying() -> VcpResult<()> {
        let nexus = Nexus::prepare(
            Dataset {
                concepts: vec![Concept {
                    id: 10,
                    name: "example".into(),
                    aliases: vec![],
                    vector: vec![1., 0.],
                }],
                ..Default::default()
            },
            &Default::default(),
            &ExecutionContext::default(),
        )?;
        let inside = [f32::from_bits(1.0_f32.to_bits() + 2), 0.];
        let rows = BorrowedCandidates::new(&[100], VectorMatrixView::new(&inside, 1, 2)?);
        let set = nexus.candidates(CandidateInput::Borrowed(rows))?;
        assert!(std::ptr::eq(
            set.rows().candidate_vectors.data,
            inside.as_slice()
        ));
        let outside = [f32::from_bits(1.0_f32.to_bits() + 8), 0.];
        let rows = BorrowedCandidates::new(&[100], VectorMatrixView::new(&outside, 1, 2)?);
        assert!(nexus.candidates(CandidateInput::Borrowed(rows)).is_err());
        Ok(())
    }
}
