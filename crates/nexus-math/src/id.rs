// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::error::Error;

macro_rules! define_slot_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(
            Clone,
            Copy,
            Debug,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
        )]
        #[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
        pub struct $name(pub u32);

        impl $name {
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            #[inline]
            pub fn from_index(index: usize, stage: &'static str) -> Result<Self, Error> {
                u32::try_from(index).map(Self).map_err(|_| Error::Input {
                    field: stage,
                    id: None,
                    message: format!("{} index {} exceeds u32::MAX", stringify!($name), index),
                })
            }
        }

        impl From<u32> for $name {
            #[inline]
            fn from(value: u32) -> Self {
                Self(value)
            }
        }
    };
}

define_slot_id!(NodeId, "Dense 0-based concept/tag node slot index.");
