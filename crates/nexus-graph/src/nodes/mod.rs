// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "math-nodes")]
mod math;
#[cfg(feature = "vcp-nodes")]
mod vcp;

#[cfg(feature = "math-nodes")]
pub use math::*;
#[cfg(feature = "vcp-nodes")]
pub use vcp::*;

fn required<T>(value: Option<T>, name: &str) -> nexus_core::VcpResult<T> {
    value.ok_or_else(|| crate::dag::graph_error(format!("execution environment requires {name}")))
}
