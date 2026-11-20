// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::error::VcpError;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum PrecisionMode {
    /// Strict left-associative f64 sequential multiply-add (no FMA/SIMD reassociation).
    StrictLeftFold,
    /// Portable safe SIMD via `wide` (`f64x4`/`f32x8`), max delta <= 2.61e-7 vs StrictLeftFold.
    #[default]
    FastSimd,
}

#[derive(Clone, Copy)]
pub struct CancellationToken<'a> {
    flag: Option<&'a AtomicBool>,
    callback: Option<&'a (dyn Fn() -> bool + Sync)>,
}

impl<'a> CancellationToken<'a> {
    pub const fn none() -> Self {
        Self {
            flag: None,
            callback: None,
        }
    }

    pub const fn from_atomic(flag: &'a AtomicBool) -> Self {
        Self {
            flag: Some(flag),
            callback: None,
        }
    }

    pub const fn from_fn(cb: &'a (dyn Fn() -> bool + Sync)) -> Self {
        Self {
            flag: None,
            callback: Some(cb),
        }
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.is_some_and(|f| f.load(Ordering::Relaxed)) || self.callback.is_some_and(|cb| cb())
    }

    #[inline]
    pub fn check(&self, stage: &'static str) -> Result<(), VcpError> {
        if self.is_cancelled() {
            Err(VcpError::Cancelled { stage })
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
pub enum ParallelPolicy<'a> {
    Sequential,
    CustomPool(&'a rayon::ThreadPool),
    GlobalRayon,
}

impl<'a> ParallelPolicy<'a> {
    #[inline]
    pub fn is_parallel(&self) -> bool {
        !matches!(self, Self::Sequential)
    }

    /// Executes `op` on the configured thread pool (or inline if `Sequential`).
    #[inline]
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match self {
            Self::Sequential | Self::GlobalRayon => op(),
            Self::CustomPool(pool) => pool.install(op),
        }
    }
}

#[derive(Clone, Copy)]
pub struct ExecutionContext<'a> {
    pub parallel: ParallelPolicy<'a>,
    pub cancel: CancellationToken<'a>,
}

impl Default for ExecutionContext<'_> {
    fn default() -> Self {
        Self {
            parallel: ParallelPolicy::Sequential,
            cancel: CancellationToken::none(),
        }
    }
}

impl<'a> ExecutionContext<'a> {
    #[inline]
    pub fn with_parallel(mut self, parallel: ParallelPolicy<'a>) -> Self {
        self.parallel = parallel;
        self
    }

    #[inline]
    pub fn check_cancelled(&self, stage: &'static str) -> Result<(), VcpError> {
        self.cancel.check(stage)
    }
}
