// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

pub mod csr;
mod epoch;
pub mod error;
pub mod id;
pub mod kernel;
pub mod nnls;
pub mod operator;
pub mod plan;
pub mod view;

pub use csr::{CsrGraph, CsrGraphView, CsrParts, CsrStorage};
pub use epoch::EpochTable;
pub use error::{Diagnostics, VcpError, VcpResult};
pub use id::{CandidateId, DocumentId, GroupId, NodeId, PathId, RootId, SequenceId};
pub use nnls::NnlsWorkspace;
pub use nnls::{CHECK_TOL, DUAL_TOL, NnlsSolution, RANK_TOL};
pub use operator::{
    CancellationToken, ExecutionContext, Operator, ParallelPolicy, PrecisionMode, ResourceLimits,
};
pub use plan::{Handle, Plan};
pub use view::{
    CandidatePoolView, ContextFragmentsView, CounterevidenceAdjView, EpaBasisView,
    NegativeRelationView, PositiveRelationView, ProvenanceView, RelationPolarity, RootFactView,
    SequenceSliceView, SupportGroupView, SupportKind, TagLexicalMetaView, TextRows,
    ThinkingStageInputView, VectorMatrixView,
};
