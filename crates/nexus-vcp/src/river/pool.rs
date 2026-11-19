// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::{FieldValues, observation::finite_vector};
use nexus_core::kernel::batch_cosines_f32;
use nexus_core::{ExecutionContext, PrecisionMode, QueryCosines, VcpResult};

/// The field cosines for one set of query features and one field solution.
#[derive(Debug)]
pub struct RiverFeatures<'call> {
    query: &'call QueryCosines<'call, 'call>,
    fields: &'call FieldValues,
    local_cosines: Vec<f64>,
    transfer_cosines: Vec<f64>,
}

impl<'call> RiverFeatures<'call> {
    pub fn prepare(
        ctx: &ExecutionContext<'_>,
        query: &'call QueryCosines<'call, 'call>,
        fields: &'call FieldValues,
    ) -> VcpResult<Self> {
        const STAGE: &str = "river.features";
        ctx.check_cancelled(STAGE)?;
        let rows = query.rows();
        let projections = [
            fields.basins.local_vector.as_slice(),
            fields.basins.transfer_vector.as_slice(),
        ];
        for projection in projections {
            finite_vector(STAGE, projection, rows.candidate_vectors.dim)?;
        }
        let mut local_cosines = vec![0.0; rows.candidate_count];
        let mut transfer_cosines = vec![0.0; rows.candidate_count];
        batch_cosines_f32(
            &rows.candidate_vectors,
            &projections,
            &mut [&mut local_cosines, &mut transfer_cosines],
            PrecisionMode::StrictLeftFold,
            1e-12,
            ctx.parallel,
        )?;
        ctx.check_cancelled(STAGE)?;
        Ok(Self {
            query,
            fields,
            local_cosines,
            transfer_cosines,
        })
    }

    pub fn query(&self) -> &QueryCosines<'_, '_> {
        self.query
    }

    pub fn fields(&self) -> &FieldValues {
        self.fields
    }

    pub fn local_cosines(&self) -> &[f64] {
        &self.local_cosines
    }

    pub fn transfer_cosines(&self) -> &[f64] {
        &self.transfer_cosines
    }
}
