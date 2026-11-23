// SPDX-License-Identifier: Apache-2.0

use crate::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    pub parallelism: Parallelism,
    pub cancel: Option<CancelToken>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Parallelism {
    #[default]
    Sequential,
    Rayon,
}

impl Parallelism {
    #[doc(hidden)]
    pub fn is_parallel(self) -> bool {
        self == Self::Rayon
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl RunOptions {
    #[doc(hidden)]
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}
