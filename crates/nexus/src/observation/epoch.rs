// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

/// Reusable O(1) epoch-stamped membership table (avoids HashMap/HashSet clearing & heap allocs).
#[derive(Default, Debug)]
pub struct EpochTable {
    epoch: u32,
    stamps: Vec<u32>,
}

impl EpochTable {
    /// Advances epoch and ensures `stamps.len() >= len`.
    /// Returns `(&mut stamps[..len], active_epoch)`.
    #[inline]
    pub fn next_epoch(&mut self, len: usize) -> (&mut [u32], u32) {
        if self.stamps.len() < len {
            self.stamps.resize(len, 0);
        }
        if self.epoch == u32::MAX {
            self.stamps.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
        (&mut self.stamps[..len], self.epoch)
    }

    /// Marks a slot within the range prepared by the current epoch.
    #[inline]
    pub fn mark(&mut self, index: usize) {
        self.stamps[index] = self.epoch;
    }

    #[inline]
    pub fn is_stamped(&self, index: usize) -> bool {
        self.epoch != 0 && self.stamps.get(index).copied() == Some(self.epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::EpochTable;

    #[test]
    fn epoch_table_u32_max_wraparound() {
        let mut table = EpochTable {
            epoch: u32::MAX,
            stamps: vec![0; 4],
        };
        table.stamps[0] = u32::MAX;
        table.stamps[2] = u32::MAX - 1;
        assert!(table.is_stamped(0));

        let (stamps, epoch) = table.next_epoch(4);
        assert_eq!(epoch, 1);
        assert_eq!(stamps, &[0, 0, 0, 0]);
        stamps[3] = epoch;
        assert!(!table.is_stamped(0));
        assert!(table.is_stamped(3));
    }
}
