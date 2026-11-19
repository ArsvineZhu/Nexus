// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::observation::invalid;
use nexus_core::VcpResult;

/// Returns the external ranking weight for reciprocal rank fusion.
pub fn omega_rrf_weight(omega: f64) -> VcpResult<f64> {
    if !(0.0..=1.0).contains(&omega) {
        return Err(invalid("ranking.omega_weight", "omega must be in [0, 1]"));
    }
    Ok(1.0 - (0.5 + 0.8 * (omega - 0.65)).clamp(0.3, 0.7))
}
