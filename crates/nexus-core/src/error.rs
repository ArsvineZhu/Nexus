// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct Diagnostics {
    pub truncated: bool,
    pub truncation_reason: Option<String>,
    pub elapsed_ms: f64,
}

impl Diagnostics {
    pub fn mark_truncated(&mut self, reason: impl Into<String>) {
        if !self.truncated {
            self.truncated = true;
            self.truncation_reason = Some(reason.into());
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VcpError {
    #[error("CANCELLED at stage `{stage}`")]
    Cancelled { stage: &'static str },
    #[error("INVALID_INPUT at `{stage}`: {detail}")]
    InvalidInput { stage: &'static str, detail: String },
    #[error("DIMENSION_MISMATCH at `{stage}`: expected {expected}, got {actual}")]
    DimensionMismatch {
        stage: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("RESOURCE_LIMIT_EXCEEDED at `{stage}`: {limit_name} ({actual} > {limit})")]
    ResourceLimitExceeded {
        stage: &'static str,
        limit_name: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("THINKING_UNAVAILABLE: {reason}")]
    ThinkingUnavailable { reason: &'static str },
    #[error(
        "NOT_CONVERGED at {stage}: residual {residual} after {iterations} iterations (tolerance {tolerance})"
    )]
    NotConverged {
        stage: &'static str,
        iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error("NUMERIC_DIVERGENCE at `{stage}`: {detail}")]
    NumericDivergence { stage: &'static str, detail: String },
}

pub type VcpResult<T> = std::result::Result<T, VcpError>;
