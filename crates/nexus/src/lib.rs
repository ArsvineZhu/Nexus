// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]

mod candidates;
mod dataset;
mod evaluation;
mod observation;
mod query;
mod readout;
mod recall;

pub use candidates::{BorrowedCandidates, Candidate, CandidateInput, CandidateSet};
pub use dataset::{
    AssetConfig, Catalog, Concept, Dataset, Document, Nexus, PreparedDataset, Relation,
    RelationEvidence, Sequence, SourceFact,
};
pub use evaluation::{
    DtscEvidence, DtscQueryConfig, Evaluation, PreparedRiver, RiverConfig, RiverQueryConfig,
};
pub use nexus_core as core;
pub use nexus_core::{CancellationToken, ExecutionContext, Handle, Operator, ParallelPolicy, Plan};
pub use nexus_math as math;
pub use nexus_vcp as vcp;
pub use observation::{DtscField, FieldState, Observation};
pub use query::{Query, TagRecall, TagRecallRequest};
pub use readout::{
    CandidatePathEvidence, ConceptTransition, DtscItem, DtscReadout, PathRef, RiverItem,
    RiverPathEvidence, RiverReadout, ScoreItem, Stage, TagMemoReadout, ThinkingItem, ThinkingPlan,
    ThinkingStage,
};
pub use recall::{RecallConfig, RecallIndexes, RecallRequest};
