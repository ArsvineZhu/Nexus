// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]

pub use nexus_core;

pub mod artifact;
pub mod context;
pub mod dedup;
pub mod geometry;
pub mod hyperbolic;
pub mod observation;
pub mod river;
pub mod thinking;
mod warped_geometry;

pub use context::*;
pub use dedup::*;
pub use observation::*;
pub use river::*;
pub use thinking::*;

pub use artifact::{
    CounterevidenceAdj, FactMatrix, FactMatrixConfig, FactMatrixInput, FactMatrixOperator,
    FactMatrixOutput, ProvenanceBuildInput, ProvenanceBuildOperator, ProvenanceConfig,
    ProvenanceData, TransportCalibrateInput, TransportCalibrateOperator, TransportCalibration,
    TransportConfig,
};
pub use geometry::{
    EpaBasis, GeometryState, GlobalEpaCandidateMode, GlobalEpaConfig, GlobalEpaInput,
    GlobalEpaOperator, GlobalEpaPriorMode, IntrinsicResidual, LocalConeAgsConfig,
    LocalConeAgsInput, LocalConeAgsOperator,
};

fn parameter(
    stage: &'static str,
    name: &str,
    value: f64,
    min: f64,
    max: f64,
) -> nexus_core::VcpResult<()> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(nexus_core::VcpError::InvalidInput {
            stage,
            detail: format!("{name} must be finite and in [{min:e}, {max:e}]"),
        })
    }
}
