// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]
//! Prepare retrieval assets once, then observe, recall and rank queries over them.
//!
//! Build optional indexes with [Nexus::prepare_recall], or supply concept recall with
//! [Query::with_tag_recall] and preselected document candidates through [CandidateInput].
//! [Evaluation] shares one observation and candidate pool across ranking and Thinking.

mod candidates;
mod dataset;
mod evaluation;
mod observation;
mod query;
mod readout;
mod recall;

pub use candidates::{BorrowedCandidates, CandidateInput, CandidateSet};
pub use dataset::{
    AssetConfig, Catalog, Concept, Dataset, Document, Nexus, PreparedDataset, Relation,
    RelationEvidence, Sequence, SourceFact,
};
pub use evaluation::{
    DtscEvidence, DtscQueryConfig, Evaluation, RiverConfig, RiverEvidence, RiverQueryConfig,
};
pub use nexus_core as core;
pub use nexus_core::{CancellationToken, ExecutionContext, ParallelPolicy};
pub use nexus_math as math;
pub use nexus_vcp as vcp;
pub use observation::{DtscField, FieldState, Observation};
pub use query::{Query, TagRecall, TagRecallRequest};
pub use readout::{
    ConceptTransition, DtscItem, DtscReadout, PathRef, RiverItem, RiverReadout, ScoreItem, Stage,
    TagMemoReadout, ThinkingItem, ThinkingPlan, ThinkingStage,
};
pub use recall::{DocumentHit, RecallConfig, RecallIndexes, RecallRequest};
