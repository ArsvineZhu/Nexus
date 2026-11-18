// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]

pub use nexus_core;

pub mod artifact;
pub mod geometry;

pub use artifact::{
    CounterevidenceAdj, FactMatrix, FactMatrixConfig, FactMatrixInput, FactMatrixOperator,
    FactMatrixOutput, ProvenanceBuildInput, ProvenanceBuildOperator, ProvenanceConfig,
    ProvenanceData, TransportCalibrateInput, TransportCalibrateOperator, TransportConfig,
    TransportGraph,
};
pub use geometry::{
    EpaBasis, GeometryState, GlobalEpaCandidateMode, GlobalEpaConfig, GlobalEpaInput,
    GlobalEpaOperator, GlobalEpaPriorMode, IntrinsicResidual, LocalConeAgsConfig,
    LocalConeAgsInput, LocalConeAgsOperator,
};
