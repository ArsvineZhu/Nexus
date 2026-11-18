// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

pub mod csr;
pub mod error;
pub mod id;
pub mod kernel;
pub mod nnls;
pub mod operator;
pub mod scratch;
pub mod view;

pub use csr::CsrGraphView;
pub use error::{Diagnostics, VcpError, VcpResult};
pub use id::{CandidateId, DocumentId, GroupId, NodeId, PathId, RootId, SequenceId};
pub use nnls::{CHECK_TOL, DUAL_TOL, NnlsSolution, RANK_TOL};
pub use operator::{
    CancellationToken, ExecutionContext, Operator, ParallelPolicy, PrecisionMode, ResourceLimits,
};
pub use scratch::{EpochTable, ScratchPad};
pub use view::{
    CandidatePoolView, ContextFragmentsView, CounterevidenceAdjView, EpaBasisView,
    NegativeRelationView, PositiveRelationView, ProvenanceView, RelationPolarity, RootFactView,
    SequenceSliceView, SupportGroupView, SupportKind, TagLexicalMetaView, ThinkingStageInputView,
    VectorMatrixView,
};
