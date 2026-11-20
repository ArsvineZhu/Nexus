// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::{FieldValues, observation::finite_vector};
use nexus_core::kernel::batch_cosines_f32;
use nexus_core::{ExecutionContext, PrecisionMode, QueryCosines, VcpResult};

/// The field cosines for one set of query features and one field solution.
#[derive(Debug)]
pub(super) struct RiverFeatures {
    local_cosines: Vec<f64>,
    transfer_cosines: Vec<f64>,
}

impl RiverFeatures {
    pub fn prepare(
        ctx: &ExecutionContext<'_>,
        query: &QueryCosines<'_, '_>,
        fields: &FieldValues,
    ) -> VcpResult<Self> {
        const STAGE: &str = "river.features";
        ctx.check_cancelled(STAGE)?;
        let rows = query.rows();
        let projections = [
            fields.basins.local_direction.as_deref(),
            fields.basins.transfer_direction.as_deref(),
        ];
        for projection in projections.into_iter().flatten() {
            finite_vector(STAGE, projection, rows.candidate_vectors.dim)?;
        }
        let mut local_cosines = vec![0.0; rows.candidate_count];
        let mut transfer_cosines = vec![0.0; rows.candidate_count];
        let (directions, mut columns): (Vec<_>, Vec<_>) = projections
            .into_iter()
            .zip([
                local_cosines.as_mut_slice(),
                transfer_cosines.as_mut_slice(),
            ])
            .filter_map(|(direction, column)| direction.map(|direction| (direction, column)))
            .unzip();
        if !directions.is_empty() {
            batch_cosines_f32(
                &rows.candidate_vectors,
                &directions,
                &mut columns,
                PrecisionMode::StrictLeftFold,
                1e-12,
                ctx.parallel,
            )?;
        }
        ctx.check_cancelled(STAGE)?;
        Ok(Self {
            local_cosines,
            transfer_cosines,
        })
    }

    pub fn local_cosines(&self) -> &[f64] {
        &self.local_cosines
    }

    pub fn transfer_cosines(&self) -> &[f64] {
        &self.transfer_cosines
    }
}
