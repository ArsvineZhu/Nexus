// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

/// Reusable O(1) epoch-stamped membership table (avoids HashMap/HashSet clearing & heap allocs).
#[derive(Default, Debug)]
pub struct EpochTable {
    epoch: u32,
    stamps: Vec<u32>,
}

impl EpochTable {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            epoch: 0,
            stamps: vec![0; cap],
        }
    }

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

    #[inline]
    pub const fn current_epoch(&self) -> u32 {
        self.epoch
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

    pub fn reset(&mut self) {
        self.epoch = 0;
        self.stamps.fill(0);
    }
}

/// Single-threaded reusable workspace shared across operators/requests to eliminate hot-path heap allocs.
/// Public fields + split `EpochTable`s enable simultaneous disjoint mutable borrows in `nexus-vcp` and `nexus-math`.
#[derive(Default, Debug)]
pub struct ScratchPad {
    pub epochs: EpochTable,
    pub epochs_aux: EpochTable,
    pub f64_buf_a: Vec<f64>,
    pub f64_buf_b: Vec<f64>,
    pub f64_buf_c: Vec<f64>,
    pub f64_buf_d: Vec<f64>,
    pub f64_buf_e: Vec<f64>,
    pub f64_buf_f: Vec<f64>,
    pub node_accum: Vec<f64>,
    pub nnls_qr_work: Vec<f64>,
    pub ags_basis_work: Vec<f64>,
    pub u32_scratch: Vec<u32>,
    pub u32_aux: Vec<u32>,
    pub f32_scratch: Vec<f32>,
}

impl ScratchPad {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-warms all buffers to eliminate heap allocations in steady-state hot paths.
    pub fn with_capacities(node_capacity: usize, dim: usize, nnls_max_cols: usize) -> Self {
        let mut pad = Self {
            epochs: EpochTable::with_capacity(node_capacity),
            epochs_aux: EpochTable::with_capacity(node_capacity),
            f64_buf_a: Vec::with_capacity(node_capacity.max(dim)),
            f64_buf_b: Vec::with_capacity(node_capacity.max(dim)),
            f64_buf_c: Vec::with_capacity(node_capacity.max(dim)),
            f64_buf_d: Vec::with_capacity(node_capacity.max(dim)),
            f64_buf_e: Vec::with_capacity(node_capacity),
            f64_buf_f: Vec::with_capacity(node_capacity),
            node_accum: Vec::with_capacity(node_capacity),
            nnls_qr_work: Vec::with_capacity((nnls_max_cols + 1) * (nnls_max_cols + 1)),
            ags_basis_work: Vec::with_capacity(32 * dim),
            u32_scratch: Vec::with_capacity(node_capacity),
            u32_aux: Vec::with_capacity(node_capacity),
            f32_scratch: Vec::with_capacity(dim),
        };
        pad.epochs.stamps.resize(node_capacity, 0);
        pad.epochs_aux.stamps.resize(node_capacity, 0);
        pad
    }

    #[inline]
    pub fn prepare_f64(buf: &mut Vec<f64>, len: usize, value: f64) -> &mut [f64] {
        if buf.len() < len {
            buf.resize(len, value);
        }
        let slice = &mut buf[..len];
        slice.fill(value);
        slice
    }

    #[inline]
    pub fn prepare_f32(buf: &mut Vec<f32>, len: usize, value: f32) -> &mut [f32] {
        if buf.len() < len {
            buf.resize(len, value);
        }
        let slice = &mut buf[..len];
        slice.fill(value);
        slice
    }

    #[inline]
    pub fn prepare_u32(buf: &mut Vec<u32>, len: usize, value: u32) -> &mut [u32] {
        if buf.len() < len {
            buf.resize(len, value);
        }
        let slice = &mut buf[..len];
        slice.fill(value);
        slice
    }
}

#[cfg(test)]
mod tests {
    use super::EpochTable;

    #[test]
    fn epoch_table_u32_max_wraparound() {
        let mut table = EpochTable::with_capacity(4);
        table.epoch = u32::MAX;
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
