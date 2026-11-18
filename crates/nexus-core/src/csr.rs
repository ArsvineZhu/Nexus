// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::error::VcpError;
use crate::id::NodeId;

/// Dense CSR transport graph view.
/// Primary row edges are sorted descending by weight (for top-K neighbor pruning);
/// optional `target_sorted_indices` stores global `0..nnz` edge indices sorted ascending by `targets[edge_idx]`
/// within each row to enable O(log deg) lookup; optional `edge_gates` enables request-time temporal gating `P^q_ij`.
#[derive(Clone, Copy, Debug)]
pub struct CsrGraphView<'a> {
    pub node_count: usize,
    pub row_offsets: &'a [usize],
    pub targets: &'a [NodeId],
    pub weights: &'a [f64],
    pub is_static_wormhole: &'a [bool],
    pub inbound: &'a [f64],
    pub max_inbound: f64,
    pub anchor_gains: &'a [f64],
    pub target_sorted_indices: Option<&'a [u32]>,
    pub edge_gates: Option<&'a [f64]>,
}

impl<'a> CsrGraphView<'a> {
    /// Binds request-level temporal edge gates `[0.0, 1.0]` aligned 1:1 with global `0..nnz` edges.
    pub fn with_edge_gates(mut self, gates: &'a [f64]) -> Result<Self, VcpError> {
        if gates.len() != self.weights.len() {
            return Err(VcpError::DimensionMismatch {
                stage: "CsrGraphView::with_edge_gates",
                expected: self.weights.len(),
                actual: gates.len(),
            });
        }
        if gates
            .iter()
            .any(|g| !g.is_finite() || !(0.0..=1.0).contains(g))
        {
            return Err(VcpError::InvalidInput {
                stage: "CsrGraphView::with_edge_gates",
                detail: "edge_gates must be finite and in [0.0, 1.0]".into(),
            });
        }
        self.edge_gates = Some(gates);
        Ok(self)
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.targets.len()
    }

    #[inline]
    pub fn effective_weight(&self, edge_idx: usize) -> f64 {
        let base = self.weights[edge_idx];
        match self.edge_gates {
            Some(gates) => base * gates[edge_idx],
            None => base,
        }
    }

    /// Locates global CSR edge index `e` in `0..nnz` for `(src -> dst)`.
    /// Uses O(log deg) binary search when `target_sorted_indices` is present, or linear row scan otherwise.
    #[inline]
    pub fn find_edge(&self, src: NodeId, dst: NodeId) -> Option<usize> {
        let u = src.index();
        if u >= self.node_count {
            return None;
        }
        let start = self.row_offsets[u];
        let end = self.row_offsets[u + 1];
        if let Some(sorted) = self.target_sorted_indices {
            let row = &sorted[start..end];
            let pos = row.partition_point(|&edge_idx| self.targets[edge_idx as usize] < dst);
            if pos < row.len() {
                let edge_idx = row[pos] as usize;
                if self.targets[edge_idx] == dst {
                    return Some(edge_idx);
                }
            }
            None
        } else {
            (start..end).find(|&e| self.targets[e] == dst)
        }
    }

    /// Returns effective edge weight for `(src -> dst)`, or `0.0` if absent.
    #[inline]
    pub fn edge_weight(&self, src: NodeId, dst: NodeId) -> f64 {
        self.find_edge(src, dst)
            .map_or(0.0, |e| self.effective_weight(e))
    }

    #[inline]
    pub fn row_range(&self, node: NodeId) -> std::ops::Range<usize> {
        let u = node.index();
        if u >= self.node_count {
            return 0..0;
        }
        self.row_offsets[u]..self.row_offsets[u + 1]
    }

    #[inline]
    pub fn row_targets(&self, node: NodeId) -> &'a [NodeId] {
        &self.targets[self.row_range(node)]
    }

    #[inline]
    pub fn anchor_gain(&self, node: NodeId) -> f64 {
        self.anchor_gains.get(node.index()).copied().unwrap_or(1.0)
    }

    #[inline]
    pub fn inbound_mass(&self, node: NodeId) -> f64 {
        self.inbound.get(node.index()).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn is_wormhole_edge(&self, edge_idx: usize) -> bool {
        self.is_static_wormhole
            .get(edge_idx)
            .copied()
            .unwrap_or(false)
    }

    pub fn validate(&self, stage: &'static str) -> Result<(), VcpError> {
        let n = self.node_count;
        let nnz = self.targets.len();

        if self.row_offsets.len() != n + 1 {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n + 1,
                actual: self.row_offsets.len(),
            });
        }
        if self.row_offsets.first().copied() != Some(0)
            || self.row_offsets.last().copied() != Some(nnz)
            || self.row_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "row_offsets must start at 0, be non-decreasing, and end at nnz".into(),
            });
        }
        if self.weights.len() != nnz {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: nnz,
                actual: self.weights.len(),
            });
        }
        if self.is_static_wormhole.len() != nnz {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: nnz,
                actual: self.is_static_wormhole.len(),
            });
        }
        if self.inbound.len() != n {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n,
                actual: self.inbound.len(),
            });
        }
        if self.anchor_gains.len() != n {
            return Err(VcpError::DimensionMismatch {
                stage,
                expected: n,
                actual: self.anchor_gains.len(),
            });
        }
        if !self.max_inbound.is_finite() || self.max_inbound < 0.0 {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "max_inbound must be finite and non-negative".into(),
            });
        }
        if self.targets.iter().any(|t| t.index() >= n) {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "CSR targets contains out-of-bounds NodeId".into(),
            });
        }
        if self.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "CSR weights must be finite and non-negative".into(),
            });
        }
        if self.inbound.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.anchor_gains.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage,
                detail: "CSR inbound and anchor_gains must be finite and non-negative".into(),
            });
        }

        if let Some(sorted) = self.target_sorted_indices {
            if sorted.len() != nnz {
                return Err(VcpError::DimensionMismatch {
                    stage,
                    expected: nnz,
                    actual: sorted.len(),
                });
            }
            for u in 0..n {
                let start = self.row_offsets[u];
                let end = self.row_offsets[u + 1];
                let row = &sorted[start..end];
                for &edge_idx in row {
                    let e = edge_idx as usize;
                    if e < start || e >= end {
                        return Err(VcpError::InvalidInput {
                            stage,
                            detail: format!(
                                "target_sorted_indices entry {e} out of row range {start}..{end} for node {u}"
                            ),
                        });
                    }
                }
                if row.windows(2).any(|w| {
                    let e0 = w[0] as usize;
                    let e1 = w[1] as usize;
                    (self.targets[e0], e0) >= (self.targets[e1], e1)
                }) {
                    return Err(VcpError::InvalidInput {
                        stage,
                        detail: format!(
                            "target_sorted_indices for node {u} must be strictly ascending by (target, edge_idx)"
                        ),
                    });
                }
            }
        }

        if let Some(gates) = self.edge_gates {
            if gates.len() != nnz {
                return Err(VcpError::DimensionMismatch {
                    stage,
                    expected: nnz,
                    actual: gates.len(),
                });
            }
            if gates
                .iter()
                .any(|g| !g.is_finite() || !(0.0..=1.0).contains(g))
            {
                return Err(VcpError::InvalidInput {
                    stage,
                    detail: "edge_gates must be finite and in [0.0, 1.0]".into(),
                });
            }
        }

        Ok(())
    }
}
