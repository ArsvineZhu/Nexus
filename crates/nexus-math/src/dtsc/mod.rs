// SPDX-License-Identifier: Apache-2.0

mod config;
mod curve;
mod evidence;
mod field;
mod path_scoring;
mod readout;

pub use config::*;
pub use curve::{AuxiliaryEvidence, CurveMetrics, EvidenceClass, GeometryState};
pub use evidence::*;
pub use field::*;
pub use readout::*;

#[inline]
fn unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[inline]
fn positive(value: f64) -> f64 {
    value.max(0.0)
}
