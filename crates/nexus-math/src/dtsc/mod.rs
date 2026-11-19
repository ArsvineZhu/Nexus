// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod curve;
pub mod evidence;
pub mod field;
pub(crate) mod path_scoring;
pub mod readout;

pub use config::*;
pub use curve::{AuxiliaryEvidence, DtscGeometryState};
pub use evidence::*;
pub use field::*;
pub use readout::*;

#[inline]
pub(crate) fn unit(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn positive(v: f64) -> f64 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}
