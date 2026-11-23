// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]
//! Build a relation model, observe a query, and evaluate retrievable candidates.
mod candidates;
pub mod context;
mod evaluation;
mod model;
mod observation;
mod query;
mod readout;
mod requests;
mod river;
mod thinking;
mod warped;

pub type ConceptId = u64;
pub type SourceId = u64;
pub type CandidateId = u64;
pub type DocumentId = u64;

pub use candidates::{
    Candidate, CandidateData, CandidateMetadata, CandidateView, Candidates, PathRef,
};
pub use evaluation::{
    DtscEvidence, DtscEvidenceOptions, DtscOptions, DtscReadoutOptions, EvaluateOptions, Evaluation,
};
pub use model::{
    BuildConfig, BuildOptions, Concept, ConceptData, ConceptView, FactsConfig, GeometryConfig,
    GlobalCandidateMode, GlobalPriorMode, GlobalSpanConfig, GraphConfig, LocalConeConfig, Model,
    Relation, RelationData, Source, SourceOrigin, TransportConfig,
};
pub use nexus_math::dtsc::{
    DtscAuxConfig, DtscConfig, DtscCurveConfig, DtscEvidenceConfig, DtscFieldConfig,
    DtscGeometryAuxConfig, DtscIdentityConfig, DtscReadoutConfig, DtscRewardConfig,
    DtscSparseConfig, DtscTrustConfig,
};
pub use nexus_math::{
    CancelToken, EnergyNode, Error, EvidenceSource, Parallelism, Result, RunOptions,
};
pub use observation::{
    Basin, ConceptSearchConfig, ConceptSearchMode, DualFields, DynamicWormholeConfig, EpaAnalysis,
    EpaAxis, EpaBridge, EpaConfig, EpaGateMode, FieldProjection, FieldSolution, FieldSolveConfig,
    Fields, FieldsConfig, FusionConfig, FusionMember, FusionResult, HypernymConfig,
    LanguageContext, Observation, ObservationMetadata, ObserveConfig, ObserveOptions, ObservedEdge,
    ProjectedField, ProjectionConfig, PyramidConfig, PyramidLevel, PyramidResult, PyramidTag,
    QueryAnalysis, RecallKind, RecallVector, Residual, Seed, SeedConfig, Seeds, SenseConfig,
    SenseResult, SenseStats, Transition,
};
pub use query::{AnchorRole, ConceptSearch, ConceptSearchRequest, EdgeGate, Query, VectorAnchor};
pub use readout::{
    BridgeWitness, DtscCandidateEvidence, DtscDecision, DtscFallbackReason, Ranking,
    ReadoutSummary, RiverCandidateEvidence, ScoreEvidence, ScoreRow, Scores, SelectedPath,
};
pub use requests::{Baseline, DtscRequestOptions, RiverRequestOptions, TagMemoRequestOptions};
pub use river::{
    CrossSourceConfig, RiverAnalysisConfig, RiverConfig, RiverOptions, RiverScoringConfig,
};
pub use thinking::{
    ThinkingConfig, ThinkingOptions, ThinkingRelation, ThinkingResult, ThinkingSelection,
    ThinkingStage, ThinkingStageResult,
};
pub use warped::WarpedGeometryConfig;

/// Independent numeric operations and in-memory vector search.
pub mod math {
    pub use crate::warped::{
        DistanceHit, SearchEffort, SearchHits, SearchRequest, WarpedBuildOptions,
        WarpedGeometryConfig, WarpedIndex,
    };
    pub use nexus_math::{
        CancelToken, CosineOptions, DedupConfig, DedupOptions, DedupResult, EnergyNode, Error,
        EvidenceSource, Folded, GraphInputOptions, NnlsOptions, NnlsSolution, NumericColumns,
        Parallelism, RankedList, Ranking, Result, RrfConfig, RrfOptions, RunOptions, ScoreRow,
        Scores, TransportGraph, UnitVector, UnitVectors, VectorRows, WeightedEdge, cosine_scores,
        dtsc, merge_ranked_ids, normalize, reciprocal_rank_fusion, semantic_dedup, solve_nnls,
    };
}

/// Build and query stages for callers who need individual computations.
pub mod operators {
    pub use crate::model::{
        ConceptData, ConceptView, GlobalSpan, GlobalSpanOptions, GraphAssets, GraphOptions,
        LocalConeOptions, LocalGeometry, LocalResidual, compile_graph, global_span, local_cone,
    };
    pub use crate::observation::{
        EpaOptions, FieldProjectionInput, FieldProjectionOptions, FieldSolveOptions, FusionInput,
        FusionOptions, PyramidInput, PyramidOptions, SeedInput, SeedOptions, SenseInput,
        SenseOptions, analyze_epa, analyze_pyramid, fuse_query, project_fields, select_seeds,
        sense, solve_fields,
    };
    pub use crate::river::{
        CrossSourceOptions, OmegaOutput, PathEvidence, QueryMode, QueryMorphology,
        RiverAnalysisOptions, RiverEvidence, RiverScoringOptions, bridge_witnesses, river_evidence,
    };
    pub use crate::thinking::plan_thinking;
    pub use crate::warped::ConceptIndex;
}

fn input(field: &'static str, id: Option<u64>, message: impl Into<String>) -> Error {
    Error::Input {
        field,
        id,
        message: message.into(),
    }
}

fn invalid(field: &'static str, message: impl Into<String>) -> Error {
    input(field, None, message)
}

fn config(field: &'static str, message: impl Into<String>) -> Error {
    Error::config(field, message)
}

fn numeric(operator: &'static str, message: impl Into<String>) -> Error {
    Error::numeric(operator, message)
}

fn parameter(
    operator: &'static str,
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        return Err(config(
            field,
            format!("{operator} requires a finite value in [{min}, {max}]"),
        ));
    }
    Ok(())
}
