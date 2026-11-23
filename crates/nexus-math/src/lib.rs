// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

#[doc(hidden)]
pub mod csr;
pub mod dtsc;
mod error;
mod execution;
mod id;
#[doc(hidden)]
pub mod kernel;
mod matrix;
#[doc(hidden)]
pub mod nnls;
mod ranking;

pub use csr::{GraphInputOptions, TransportGraph, WeightedEdge};
pub use dtsc::{
    DtscConfig, DtscDecision, DtscEvidenceConfig, DtscFallbackReason, DtscFieldConfig,
    DtscReadoutConfig, EnergyNode, EvidenceSource,
};
pub use error::{Error, Result};
pub use execution::{CancelToken, Parallelism, RunOptions};
pub use matrix::{CosineOptions, UnitVector, UnitVectors, VectorRows, cosine_scores, normalize};
pub use nnls::{NnlsOptions, NnlsSolution, NumericColumns, solve_nnls};
pub use ranking::{
    DedupConfig, DedupOptions, DedupResult, Folded, RankedList, Ranking, RrfConfig, RrfOptions,
    ScoreRow, Scores, merge_ranked_ids, reciprocal_rank_fusion, semantic_dedup,
};

#[doc(hidden)]
pub use csr::{EffectiveTransport, TransportGraphView};
#[doc(hidden)]
pub use id::NodeId;
#[doc(hidden)]
pub use matrix::{OwnedUnitVectors, cosine_columns};
#[doc(hidden)]
pub use nnls::NnlsWorkspace;
#[doc(hidden)]
pub use ranking::rank_order;
