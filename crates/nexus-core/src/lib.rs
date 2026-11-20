// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

pub mod csr;
mod epoch;
pub mod error;
pub mod execution;
pub mod id;
pub mod kernel;
pub mod nnls;
mod query_cosines;
pub mod view;

pub use csr::{CsrGraph, CsrGraphView, CsrParts, CsrStorage, TransportGraph, TransportGraphView};
pub use epoch::EpochTable;
pub use error::{VcpError, VcpResult};
pub use execution::{CancellationToken, ExecutionContext, ParallelPolicy, PrecisionMode};
pub use id::{CandidateId, GroupId, NodeId, PathId, RootId, SequenceId};
pub use nnls::NnlsWorkspace;
pub use nnls::{CHECK_TOL, DUAL_TOL, NnlsSolution, RANK_TOL};
pub use query_cosines::QueryCosines;
pub use view::{
    CandidateRowsView, ContextFragmentsView, CounterevidenceAdjView, EpaBasisView,
    NegativeRelationView, PositiveRelationView, ProvenanceView, RelationPolarity, RootFactView,
    SequenceSliceView, SupportGroupView, SupportKind, TagLexicalMetaView, TextRows,
    ThinkingStageInputView, VectorMatrixView,
};
