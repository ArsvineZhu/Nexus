// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::error::VcpError;
use crate::id::{CandidateId, GroupId, NodeId, PathId, RootId, SequenceId};

fn check_csr_u32_offsets(
    offsets: &[u32],
    count: usize,
    total_len: usize,
    stage: &'static str,
    name: &'static str,
) -> Result<(), VcpError> {
    if offsets.len() != count + 1 {
        return Err(VcpError::DimensionMismatch {
            stage,
            expected: count + 1,
            actual: offsets.len(),
        });
    }
    if offsets.first().copied() != Some(0) || offsets.last().map(|&v| v as usize) != Some(total_len)
    {
        return Err(VcpError::InvalidInput {
            stage,
            detail: format!("{name} must start at 0 and end at {total_len}"),
        });
    }
    if offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(VcpError::InvalidInput {
            stage,
            detail: format!("{name} must be monotonically non-decreasing"),
        });
    }
    Ok(())
}

/// Row-major contiguous vector matrix view; optional precomputed f64 squared norms.
#[derive(Clone, Copy, Debug)]
pub struct VectorMatrixView<'a, T = f32> {
    pub data: &'a [T],
    pub rows: usize,
    pub dim: usize,
    pub sq_norms_f64: Option<&'a [f64]>,
}

impl<'a, T> VectorMatrixView<'a, T> {
    pub const fn empty(dim: usize) -> Self {
        Self {
            data: &[],
            rows: 0,
            dim,
            sq_norms_f64: None,
        }
    }

    pub fn new(data: &'a [T], rows: usize, dim: usize) -> Result<Self, VcpError> {
        let expected = rows
            .checked_mul(dim)
            .ok_or_else(|| VcpError::InvalidInput {
                stage: "VectorMatrixView::new",
                detail: "rows * dim overflows usize".into(),
            })?;
        if data.len() != expected {
            return Err(VcpError::DimensionMismatch {
                stage: "VectorMatrixView::new",
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            rows,
            dim,
            sq_norms_f64: None,
        })
    }

    pub fn with_sq_norms(mut self, sq_norms_f64: &'a [f64]) -> Result<Self, VcpError> {
        if sq_norms_f64.len() != self.rows {
            return Err(VcpError::DimensionMismatch {
                stage: "VectorMatrixView::with_sq_norms",
                expected: self.rows,
                actual: sq_norms_f64.len(),
            });
        }
        if sq_norms_f64.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(VcpError::InvalidInput {
                stage: "VectorMatrixView::with_sq_norms",
                detail: "sq_norms_f64 must be finite and non-negative".into(),
            });
        }
        self.sq_norms_f64 = Some(sq_norms_f64);
        Ok(self)
    }

    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        let expected = self
            .rows
            .checked_mul(self.dim)
            .ok_or_else(|| VcpError::InvalidInput {
                stage,
                detail: "rows * dim overflows usize".into(),
            })?;
        if self.data.len() != expected {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected,
                actual: self.data.len(),
            });
        }
        if let Some(norms) = self.sq_norms_f64 {
            if norms.len() != self.rows {
                return Err(VcpError::DimensionMismatch {
                    stage,
                    expected: self.rows,
                    actual: norms.len(),
                });
            }
            if norms.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(VcpError::InvalidInput {
                    stage,
                    detail: "sq_norms_f64 must be finite and non-negative".into(),
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn row(&self, idx: usize) -> Option<&'a [T]> {
        if idx >= self.rows {
            return None;
        }
        let start = idx * self.dim;
        Some(&self.data[start..start + self.dim])
    }

    #[inline]
    pub fn sq_norm_f64(&self, idx: usize) -> Option<f64> {
        self.sq_norms_f64.and_then(|s| s.get(idx).copied())
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum SupportKind {
    Source,
    Interpretation,
    Hypothesis,
    Imagination,
}

/// Multi-root support groups. Excluding any required root excludes the whole group.
/// A source group's first root identifies the source; the remaining roots are its requirements.
#[derive(Clone, Copy, Debug)]
pub struct SupportGroupView<'a> {
    pub group_count: usize,
    pub group_root_offsets: &'a [u32],
    pub group_roots: &'a [RootId],
    pub group_kinds: &'a [SupportKind],
}

impl<'a> SupportGroupView<'a> {
    pub const fn empty() -> Self {
        Self {
            group_count: 0,
            group_root_offsets: &[0],
            group_roots: &[],
            group_kinds: &[],
        }
    }

    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        check_csr_u32_offsets(
            self.group_root_offsets,
            self.group_count,
            self.group_roots.len(),
            stage,
            "group_root_offsets",
        )?;
        if self.group_kinds.len() != self.group_count {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.group_count,
                actual: self.group_kinds.len(),
            });
        }
        Ok(())
    }

    #[inline]
    pub fn required_roots(&self, group: GroupId) -> &'a [RootId] {
        let g = group.index();
        if g >= self.group_count {
            return &[];
        }
        let start = self.group_root_offsets[g] as usize;
        let end = self.group_root_offsets[g + 1] as usize;
        &self.group_roots[start..end]
    }

    #[inline]
    pub fn kind(&self, group: GroupId) -> Option<SupportKind> {
        self.group_kinds.get(group.index()).copied()
    }
}

/// Edge-level independent source provenance CSR view (indexed by global CSR edge index `0..nnz`).
#[derive(Clone, Copy, Debug)]
pub struct ProvenanceView<'a> {
    pub edge_count: usize,
    pub edge_prov_offsets: &'a [u32],
    pub prov_groups: &'a [GroupId],
    pub prov_masses: &'a [f64],
}

impl<'a> ProvenanceView<'a> {
    pub const fn empty() -> Self {
        Self {
            edge_count: 0,
            edge_prov_offsets: &[0],
            prov_groups: &[],
            prov_masses: &[],
        }
    }

    pub fn validate(&self, expected_edges: usize, stage: &'static str) -> Result<(), VcpError> {
        if self.edge_count != expected_edges {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: expected_edges,
                actual: self.edge_count,
            });
        }
        if self.prov_masses.len() != self.prov_groups.len() {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.prov_groups.len(),
                actual: self.prov_masses.len(),
            });
        }
        check_csr_u32_offsets(
            self.edge_prov_offsets,
            self.edge_count,
            self.prov_groups.len(),
            stage,
            "edge_prov_offsets",
        )?;
        if self.prov_masses.iter().any(|m| !m.is_finite() || *m < 0.0) {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "prov_masses must be finite and non-negative".into(),
            });
        }
        Ok(())
    }

    #[inline]
    pub fn edge_contributions(&self, edge_idx: usize) -> (&'a [GroupId], &'a [f64]) {
        if edge_idx >= self.edge_count {
            return (&[], &[]);
        }
        let start = self.edge_prov_offsets[edge_idx] as usize;
        let end = self.edge_prov_offsets[edge_idx + 1] as usize;
        (&self.prov_groups[start..end], &self.prov_masses[start..end])
    }

    /// Raw independent evidence fraction in `[0.0, 1.0]`, returning `None` if edge has no positive provenance.
    /// Matches `Nous/native/vcp/src/graph.rs:257-279` 1:1.
    pub fn independent_support_fraction(
        &self,
        edge_idx: usize,
        excluded_roots: &[RootId],
        groups: &SupportGroupView<'_>,
    ) -> Option<f64> {
        let (group_ids, masses) = self.edge_contributions(edge_idx);
        if group_ids.is_empty() {
            return None;
        }
        let total: f64 = masses.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let excluded: f64 = group_ids
            .iter()
            .zip(masses)
            .filter(|(group_id, _)| {
                let g_idx = group_id.index();
                if g_idx < groups.group_count {
                    let req = groups.required_roots(**group_id);
                    req.is_empty() || req.iter().any(|r| excluded_roots.contains(r))
                } else {
                    excluded_roots.contains(&RootId(group_id.0))
                }
            })
            .map(|(_, mass)| *mass)
            .sum();
        let raw = 1.0 - excluded / total;
        Some(if raw.is_finite() {
            raw.clamp(0.0, 1.0)
        } else {
            0.0
        })
    }

    /// Independent support fraction with the `0.15` attenuation floor (`Nous/native/vcp/src/graph.rs:251-255`).
    pub fn independent_fraction_excluding(
        &self,
        edge_idx: usize,
        excluded_roots: &[RootId],
        groups: &SupportGroupView<'_>,
    ) -> f64 {
        self.independent_support_fraction(edge_idx, excluded_roots, groups)
            .unwrap_or(0.0)
            .max(0.15)
    }
}

/// Root facts for cross-file witness and bridge checks.
#[derive(Clone, Copy, Debug)]
pub struct RootFactView<'a> {
    pub root_count: usize,
    pub checksum_hashes: &'a [u64],
    pub root_tag_offsets: &'a [u32],
    pub root_tags: &'a [NodeId],
    pub root_req_offsets: &'a [u32],
    pub root_required_roots: &'a [RootId],
}

impl<'a> RootFactView<'a> {
    pub const fn empty() -> Self {
        Self {
            root_count: 0,
            checksum_hashes: &[],
            root_tag_offsets: &[0],
            root_tags: &[],
            root_req_offsets: &[0],
            root_required_roots: &[],
        }
    }

    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        if self.checksum_hashes.len() != self.root_count {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.root_count,
                actual: self.checksum_hashes.len(),
            });
        }
        check_csr_u32_offsets(
            self.root_tag_offsets,
            self.root_count,
            self.root_tags.len(),
            stage,
            "root_tag_offsets",
        )?;
        check_csr_u32_offsets(
            self.root_req_offsets,
            self.root_count,
            self.root_required_roots.len(),
            stage,
            "root_req_offsets",
        )?;
        for r in 0..self.root_count {
            let start = self.root_tag_offsets[r] as usize;
            let end = self.root_tag_offsets[r + 1] as usize;
            if self.root_tags[start..end].windows(2).any(|w| w[0] > w[1]) {
                return Err(VcpError::InvalidInput {
                    stage,
                    detail: format!("root_tags for root {r} must be sorted ascending"),
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn checksum(&self, root: RootId) -> Option<u64> {
        let h = self.checksum_hashes.get(root.index()).copied()?;
        if h == 0 { None } else { Some(h) }
    }

    #[inline]
    pub fn tags(&self, root: RootId) -> &'a [NodeId] {
        let r = root.index();
        if r >= self.root_count {
            return &[];
        }
        let start = self.root_tag_offsets[r] as usize;
        let end = self.root_tag_offsets[r + 1] as usize;
        &self.root_tags[start..end]
    }

    #[inline]
    pub fn has_tag(&self, root: RootId, tag: NodeId) -> bool {
        self.tags(root).binary_search(&tag).is_ok()
    }

    #[inline]
    pub fn required_roots(&self, root: RootId) -> &'a [RootId] {
        let r = root.index();
        if r >= self.root_count {
            return &[];
        }
        let start = self.root_req_offsets[r] as usize;
        let end = self.root_req_offsets[r + 1] as usize;
        &self.root_required_roots[start..end]
    }
}

/// Global EPA topic basis read-only view.
#[derive(Clone, Copy, Debug)]
pub struct EpaBasisView<'a> {
    pub dim: usize,
    pub k_sem: usize,
    pub mean: &'a [f64],
    pub basis: &'a [f64],
    pub basis_count: usize,
    pub coverage: &'a [f64],
}

impl<'a> EpaBasisView<'a> {
    pub const fn empty(dim: usize) -> Self {
        Self {
            dim,
            k_sem: 0,
            mean: &[],
            basis: &[],
            basis_count: 0,
            coverage: &[],
        }
    }

    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        if self.is_empty() {
            return Ok(());
        }
        if self.mean.len() != self.dim {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.dim,
                actual: self.mean.len(),
            });
        }
        let expected_basis =
            self.basis_count
                .checked_mul(self.dim)
                .ok_or_else(|| VcpError::InvalidInput {
                    stage,
                    detail: "basis_count * dim overflows usize".into(),
                })?;
        if self.basis.len() != expected_basis {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: expected_basis,
                actual: self.basis.len(),
            });
        }
        if self.coverage.len() != self.basis_count {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.basis_count,
                actual: self.coverage.len(),
            });
        }
        if self.mean.iter().any(|v| !v.is_finite())
            || self.basis.iter().any(|v| !v.is_finite())
            || self.coverage.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "EPA basis arrays contain non-finite or negative coverage values".into(),
            });
        }
        Ok(())
    }

    #[inline]
    pub fn basis_row(&self, idx: usize) -> Option<&'a [f64]> {
        if idx >= self.basis_count || self.dim == 0 {
            return None;
        }
        let start = idx * self.dim;
        Some(&self.basis[start..start + self.dim])
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.basis_count == 0 || self.dim == 0
    }
}

/// Ordered sequence slices for build-time Geometry / FactMatrix / Provenance without 100-item truncation.
#[derive(Clone, Copy, Debug)]
pub struct SequenceSliceView<'a> {
    pub sequence_count: usize,
    pub seq_offsets: &'a [u32],
    pub seq_source_roots: &'a [RootId],
    pub tag_ids: &'a [NodeId],
    pub positions: &'a [u32],
}

impl<'a> SequenceSliceView<'a> {
    pub const fn empty() -> Self {
        Self {
            sequence_count: 0,
            seq_offsets: &[0],
            seq_source_roots: &[],
            tag_ids: &[],
            positions: &[],
        }
    }

    pub fn validate(&self, node_count: usize, stage: &'static str) -> Result<(), VcpError> {
        if self.seq_source_roots.len() != self.sequence_count {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.sequence_count,
                actual: self.seq_source_roots.len(),
            });
        }
        if self.positions.len() != self.tag_ids.len() {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.tag_ids.len(),
                actual: self.positions.len(),
            });
        }
        check_csr_u32_offsets(
            self.seq_offsets,
            self.sequence_count,
            self.tag_ids.len(),
            stage,
            "seq_offsets",
        )?;
        if self.tag_ids.iter().any(|n| n.index() >= node_count) {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "sequence tag_ids contains out-of-bounds NodeId".into(),
            });
        }
        Ok(())
    }

    #[inline]
    pub fn sequence(&self, seq: SequenceId) -> (&'a [NodeId], &'a [u32]) {
        let s = seq.index();
        if s >= self.sequence_count {
            return (&[], &[]);
        }
        let start = self.seq_offsets[s] as usize;
        let end = self.seq_offsets[s + 1] as usize;
        (&self.tag_ids[start..end], &self.positions[start..end])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum RelationPolarity {
    Positive,
    Negative,
}

#[derive(Clone, Copy, Debug)]
pub struct PositiveRelationView<'a> {
    pub sources: &'a [NodeId],
    pub targets: &'a [NodeId],
    pub roots: &'a [RootId],
    pub groups: &'a [GroupId],
    pub masses: &'a [f64],
}

impl<'a> PositiveRelationView<'a> {
    pub fn validate(&self, node_count: usize, stage: &'static str) -> Result<(), VcpError> {
        let n = self.sources.len();
        if self.targets.len() != n
            || self.roots.len() != n
            || self.groups.len() != n
            || self.masses.len() != n
        {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n,
                actual: self
                    .targets
                    .len()
                    .min(self.roots.len())
                    .min(self.groups.len())
                    .min(self.masses.len()),
            });
        }
        if self
            .sources
            .iter()
            .chain(self.targets.iter())
            .any(|id| id.index() >= node_count)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "PositiveRelationView contains out-of-bounds NodeId".into(),
            });
        }
        if self.masses.iter().any(|m| !m.is_finite() || *m < 0.0) {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "PositiveRelationView masses must be finite and non-negative".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NegativeRelationView<'a> {
    pub sources: &'a [NodeId],
    pub targets: &'a [NodeId],
    pub roots: &'a [RootId],
    pub relation_kinds: &'a [u16],
    pub strengths: &'a [f64],
    pub confidences: &'a [f64],
}

impl<'a> NegativeRelationView<'a> {
    pub fn validate(&self, node_count: usize, stage: &'static str) -> Result<(), VcpError> {
        let n = self.sources.len();
        if self.targets.len() != n
            || self.roots.len() != n
            || self.relation_kinds.len() != n
            || self.strengths.len() != n
            || self.confidences.len() != n
        {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n,
                actual: self
                    .targets
                    .len()
                    .min(self.roots.len())
                    .min(self.relation_kinds.len())
                    .min(self.strengths.len())
                    .min(self.confidences.len()),
            });
        }
        if self
            .sources
            .iter()
            .chain(self.targets.iter())
            .any(|id| id.index() >= node_count)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "NegativeRelationView contains out-of-bounds NodeId".into(),
            });
        }
        if self
            .strengths
            .iter()
            .chain(self.confidences.iter())
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail:
                    "NegativeRelationView strengths/confidences must be finite and non-negative"
                        .into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CounterevidenceAdjView<'a> {
    pub edge_count: usize,
    pub sources: &'a [NodeId],
    pub targets: &'a [NodeId],
    pub roots: &'a [RootId],
    pub relation_kinds: &'a [u16],
    pub strengths: &'a [f64],
    pub confidences: &'a [f64],
}

impl<'a> CounterevidenceAdjView<'a> {
    pub const fn empty() -> Self {
        Self {
            edge_count: 0,
            sources: &[],
            targets: &[],
            roots: &[],
            relation_kinds: &[],
            strengths: &[],
            confidences: &[],
        }
    }

    pub fn validate(&self, node_count: usize, stage: &'static str) -> Result<(), VcpError> {
        let n = self.edge_count;
        if self.sources.len() != n
            || self.targets.len() != n
            || self.roots.len() != n
            || self.relation_kinds.len() != n
            || self.strengths.len() != n
            || self.confidences.len() != n
        {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n,
                actual: self
                    .sources
                    .len()
                    .min(self.targets.len())
                    .min(self.roots.len())
                    .min(self.relation_kinds.len())
                    .min(self.strengths.len())
                    .min(self.confidences.len()),
            });
        }
        if self
            .sources
            .iter()
            .chain(self.targets.iter())
            .any(|id| id.index() >= node_count)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "CounterevidenceAdjView contains out-of-bounds NodeId".into(),
            });
        }
        if self
            .strengths
            .iter()
            .chain(self.confidences.iter())
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail:
                    "CounterevidenceAdjView strengths/confidences must be finite and non-negative"
                        .into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TextRows<'a> {
    Strings(&'a [String]),
    Borrowed(&'a [&'a str]),
}

impl<'a> TextRows<'a> {
    fn len(self) -> usize {
        match self {
            Self::Strings(rows) => rows.len(),
            Self::Borrowed(rows) => rows.len(),
        }
    }

    fn get(self, index: usize) -> Option<&'a str> {
        match self {
            Self::Strings(rows) => rows.get(index).map(String::as_str),
            Self::Borrowed(rows) => rows.get(index).copied(),
        }
    }
}

/// Tag lexical metadata view (canonical names and aliases for HypernymPrune and TagGating).
#[derive(Clone, Copy, Debug)]
pub struct TagLexicalMetaView<'a> {
    pub node_count: usize,
    pub names: TextRows<'a>,
    pub alias_offsets: &'a [u32],
    pub aliases: TextRows<'a>,
}

impl<'a> TagLexicalMetaView<'a> {
    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        if self.names.len() != self.node_count {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.node_count,
                actual: self.names.len(),
            });
        }
        check_csr_u32_offsets(
            self.alias_offsets,
            self.node_count,
            self.aliases.len(),
            stage,
            "alias_offsets",
        )
    }

    #[inline]
    pub fn name(&self, node: NodeId) -> &'a str {
        self.names.get(node.index()).unwrap_or("")
    }

    #[inline]
    pub fn aliases(&self, node: NodeId) -> impl Iterator<Item = &'a str> {
        let index = node.index();
        let range = if index < self.node_count {
            self.alias_offsets[index] as usize..self.alias_offsets[index + 1] as usize
        } else {
            0..0
        };
        let aliases = self.aliases;
        range.filter_map(move |index| aliases.get(index))
    }
}

/// Static candidate rows with multi-root and multi-path CSR projections.
#[derive(Clone, Copy, Debug)]
pub struct CandidateRowsView<'a> {
    pub candidate_count: usize,
    pub document_ids: &'a [u64],
    pub candidate_vectors: VectorMatrixView<'a, f32>,
    pub candidate_root_offsets: &'a [u32],
    pub candidate_roots: &'a [RootId],
    pub candidate_path_offsets: &'a [u32],
    pub path_tag_offsets: &'a [u32],
    pub path_tags: &'a [NodeId],
    pub path_positions: &'a [u32],
}

impl<'a> CandidateRowsView<'a> {
    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        let n = self.candidate_count;
        for (name, len) in [
            ("document_ids", self.document_ids.len()),
            ("candidate_vectors.rows", self.candidate_vectors.rows),
        ] {
            if len != n {
                return Err(VcpError::InvalidInput {
                    stage,
                    detail: format!("{name} length {len} does not match candidate_count {n}"),
                });
            }
        }
        self.candidate_vectors.validate(stage)?;
        check_csr_u32_offsets(
            self.candidate_root_offsets,
            n,
            self.candidate_roots.len(),
            stage,
            "candidate_root_offsets",
        )?;
        if self.candidate_path_offsets.len() != n + 1 {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n + 1,
                actual: self.candidate_path_offsets.len(),
            });
        }
        let path_count = *self.candidate_path_offsets.last().unwrap_or(&0) as usize;
        if self.candidate_path_offsets.first().copied() != Some(0)
            || self.candidate_path_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "candidate_path_offsets must start at 0 and be non-decreasing".into(),
            });
        }
        if self.path_positions.len() != self.path_tags.len() {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: self.path_tags.len(),
                actual: self.path_positions.len(),
            });
        }
        check_csr_u32_offsets(
            self.path_tag_offsets,
            path_count,
            self.path_tags.len(),
            stage,
            "path_tag_offsets",
        )
    }

    #[inline]
    pub fn candidate_roots(&self, cand: CandidateId) -> &'a [RootId] {
        let c = cand.index();
        if c >= self.candidate_count {
            return &[];
        }
        let start = self.candidate_root_offsets[c] as usize;
        let end = self.candidate_root_offsets[c + 1] as usize;
        &self.candidate_roots[start..end]
    }

    #[inline]
    pub fn candidate_paths(&self, cand: CandidateId) -> std::ops::Range<usize> {
        let c = cand.index();
        if c >= self.candidate_count {
            return 0..0;
        }
        let start = self.candidate_path_offsets[c] as usize;
        let end = self.candidate_path_offsets[c + 1] as usize;
        start..end
    }

    #[inline]
    pub fn path_steps(&self, path: PathId) -> (&'a [NodeId], &'a [u32]) {
        let p = path.index();
        if p + 1 >= self.path_tag_offsets.len() {
            return (&[], &[]);
        }
        let start = self.path_tag_offsets[p] as usize;
        let end = self.path_tag_offsets[p + 1] as usize;
        (
            &self.path_tags[start..end],
            &self.path_positions[start..end],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ThinkingStageInputView<'a> {
    pub stage_label: &'a str,
    pub k: usize,
    pub candidates: &'a [CandidateId],
}

/// Borrowed input view for `aggregate`.
#[derive(Clone, Copy, Debug)]
pub struct ContextFragmentsView<'a> {
    pub count: usize,
    pub keys: &'a [&'a str],
    pub roles: &'a [&'a str],
    pub stages: &'a [&'a str],
    pub parent_keys: &'a [Option<&'a str>],
    pub sequences: &'a [i64],
    pub weights: &'a [f64],
    pub relevances: &'a [f64],
    pub validities: &'a [f64],
    pub distances: &'a [f64],
    pub half_lives: &'a [Option<f64>],
    pub source_keys: &'a [Option<&'a str>],
    pub source_ranges: &'a [Option<(u64, u64)>],
}

impl<'a> ContextFragmentsView<'a> {
    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        let n = self.count;
        for (name, len) in [
            ("keys", self.keys.len()),
            ("roles", self.roles.len()),
            ("stages", self.stages.len()),
            ("parent_keys", self.parent_keys.len()),
            ("sequences", self.sequences.len()),
            ("weights", self.weights.len()),
            ("relevances", self.relevances.len()),
            ("validities", self.validities.len()),
            ("distances", self.distances.len()),
            ("half_lives", self.half_lives.len()),
            ("source_keys", self.source_keys.len()),
            ("source_ranges", self.source_ranges.len()),
        ] {
            if len != n {
                return Err(VcpError::InvalidInput {
                    stage,
                    detail: format!("{name} length {len} does not match count {n}"),
                });
            }
        }
        if self
            .weights
            .iter()
            .chain(self.relevances.iter())
            .chain(self.validities.iter())
            .chain(self.distances.iter())
            .any(|v| !v.is_finite())
            || self
                .half_lives
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "ContextFragmentsView contains non-finite or negative half_life values"
                    .into(),
            });
        }
        Ok(())
    }
}
