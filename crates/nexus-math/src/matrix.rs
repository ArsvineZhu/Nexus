// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use crate::error::Error;
use crate::kernel::{PrecisionMode, all_finite_f32, cosine_from_dot, norm_sq_f32};
use crate::{Result as MathResult, RunOptions, Scores};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;

const FP64: PrecisionMode = PrecisionMode::StrictLeftFold;

/// A validated, borrowed row-major matrix or an indexed view into one.
#[derive(Clone, Copy, Debug)]
pub struct VectorRows<'a> {
    ids: &'a [u64],
    values: &'a [f32],
    dimension: usize,
    indices: Option<&'a [usize]>,
}

impl<'a> VectorRows<'a> {
    pub fn new(ids: &'a [u64], values: &'a [f32], dimension: usize) -> MathResult<Self> {
        validate_layout(ids, values.len(), dimension)?;
        for (&id, vector) in ids.iter().zip(values.chunks_exact(dimension)) {
            validate_vector(vector, "values", Some(id))?;
        }
        Ok(Self::contiguous(ids, values, dimension))
    }

    fn contiguous(ids: &'a [u64], values: &'a [f32], dimension: usize) -> Self {
        Self {
            ids,
            values,
            dimension,
            indices: None,
        }
    }

    pub fn ids(&self) -> &'a [u64] {
        self.ids
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn row(&self, index: usize) -> Option<&'a [f32]> {
        self.ids.get(index)?;
        let index = self.indices.map_or(index, |indices| indices[index]);
        let start = index * self.dimension;
        Some(&self.values[start..start + self.dimension])
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Projects unique slots from validated storage while keeping the matrix borrowed.
    #[doc(hidden)]
    pub fn select_rows(self, ids: &'a [u64], slots: &'a [usize]) -> MathResult<Self> {
        if self.indices.is_some() {
            return Err(Error::input(
                "slots",
                "selection slots must reference the original matrix",
            ));
        }
        if ids.len() != slots.len() {
            return Err(Error::dimension("slots", ids.len(), slots.len()));
        }
        let mut seen = HashSet::with_capacity(ids.len());
        for (&id, &slot) in ids.iter().zip(slots) {
            if self.ids.get(slot) != Some(&id) || !seen.insert(id) {
                return Err(Error::Input {
                    field: "slots",
                    id: Some(id),
                    message: "duplicate ID or incorrect source slot".into(),
                });
            }
        }
        Ok(Self {
            ids,
            indices: Some(slots),
            ..self
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnitVectors<'a>(VectorRows<'a>);

impl<'a> UnitVectors<'a> {
    pub fn new(ids: &'a [u64], values: &'a [f32], dimension: usize) -> MathResult<Self> {
        validate_layout(ids, values.len(), dimension)?;
        for (&id, vector) in ids.iter().zip(values.chunks_exact(dimension)) {
            let norm = validate_vector(vector, "values", Some(id))?;
            if (norm - 1.0).abs() > 8.0 * f64::from(f32::EPSILON) {
                return Err(Error::Input {
                    field: "values",
                    id: Some(id),
                    message: "row must have unit norm".into(),
                });
            }
        }
        Ok(Self(VectorRows::contiguous(ids, values, dimension)))
    }

    pub fn as_rows(&self) -> VectorRows<'a> {
        self.0
    }
}

/// Owns one contiguous matrix after validating and normalizing its rows.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct OwnedUnitVectors {
    ids: Arc<[u64]>,
    values: Vec<f32>,
    dimension: usize,
}

impl OwnedUnitVectors {
    pub fn new(
        ids: Arc<[u64]>,
        mut values: Vec<f32>,
        dimension: usize,
        run: &RunOptions,
    ) -> MathResult<Self> {
        run.check_cancelled()?;
        validate_layout(&ids, values.len(), dimension)?;
        let normalize = |(&id, row): (&u64, &mut [f32])| -> MathResult<()> {
            run.check_cancelled()?;
            let length = validate_vector(row, "values", Some(id))?.sqrt();
            for value in row {
                *value = (f64::from(*value) / length) as f32;
            }
            Ok(())
        };
        if run.parallelism.is_parallel() {
            ids.par_iter()
                .zip(values.par_chunks_mut(dimension))
                .try_for_each(normalize)?;
        } else {
            ids.iter()
                .zip(values.chunks_exact_mut(dimension))
                .try_for_each(normalize)?;
        }
        run.check_cancelled()?;
        Ok(Self {
            ids,
            values,
            dimension,
        })
    }

    pub fn view(&self) -> UnitVectors<'_> {
        UnitVectors(self.as_rows())
    }

    pub fn as_rows(&self) -> VectorRows<'_> {
        VectorRows::contiguous(&self.ids, &self.values, self.dimension)
    }

    pub fn ids(&self) -> &Arc<[u64]> {
        &self.ids
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

fn validate_layout(ids: &[u64], values: usize, dimension: usize) -> MathResult<()> {
    if dimension == 0 {
        return Err(Error::input("dimension", "dimension must be positive"));
    }
    let expected = ids
        .len()
        .checked_mul(dimension)
        .ok_or_else(|| Error::input("values", "matrix dimensions overflow"))?;
    if values != expected {
        return Err(Error::dimension("values", expected, values));
    }
    let mut seen = HashSet::with_capacity(ids.len());
    for &id in ids {
        if !seen.insert(id) {
            return Err(Error::Input {
                field: "ids",
                id: Some(id),
                message: "duplicate ID".into(),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnitVector(Vec<f32>);

impl UnitVector {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

impl AsRef<[f32]> for UnitVector {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

pub fn normalize(vector: &[f32]) -> MathResult<UnitVector> {
    let length = validate_vector(vector, "vector", None)?.sqrt();
    Ok(UnitVector(
        vector
            .iter()
            .map(|&v| (f64::from(v) / length) as f32)
            .collect(),
    ))
}

pub(crate) fn validate_vector(
    vector: &[f32],
    field: &'static str,
    id: Option<u64>,
) -> MathResult<f64> {
    if all_finite_f32(vector) {
        let norm = norm_sq_f32(vector, FP64);
        if norm > 0.0 {
            return Ok(norm);
        }
    }
    Err(Error::Input {
        field,
        id,
        message: "vector must be finite and nonzero".into(),
    })
}

#[derive(Clone, Debug, Default)]
pub struct CosineOptions {
    pub run: RunOptions,
}

pub fn cosine_scores(
    query: &[f32],
    rows: VectorRows<'_>,
    options: &CosineOptions,
) -> MathResult<Scores> {
    let values = cosine_columns([Some(query)], rows, &options.run)?;
    Scores::new(
        rows.ids()
            .iter()
            .copied()
            .zip(values.into_iter().map(|[value]| value)),
    )
}

/// Computes cosine columns in row order with one norm and row traversal per candidate.
#[doc(hidden)]
pub fn cosine_columns<const N: usize>(
    queries: [Option<&[f32]>; N],
    rows: VectorRows<'_>,
    run: &RunOptions,
) -> MathResult<Vec<[f64; N]>> {
    run.check_cancelled()?;
    let mut query_norms = [0.0; N];
    for (norm, query) in query_norms.iter_mut().zip(queries) {
        if let Some(query) = query {
            if query.len() != rows.dimension() {
                return Err(Error::dimension("query", rows.dimension(), query.len()));
            }
            *norm = validate_vector(query, "query", None)?;
        }
    }
    let packed: Vec<[f64; N]> = (0..rows.dimension())
        .map(|axis| queries.map(|query| query.map_or(0.0, |query| f64::from(query[axis]))))
        .collect();
    let score = |index: usize| {
        if index.is_multiple_of(32) {
            run.check_cancelled()?;
        }
        let row = rows.row(index).expect("validated row index");
        let mut dots = [0.0; N];
        let mut norm = 0.0;
        for (&component, lanes) in row.iter().zip(&packed) {
            let value = f64::from(component);
            norm += value * value;
            for (dot, &query) in dots.iter_mut().zip(lanes) {
                *dot += value * query;
            }
        }
        Ok(std::array::from_fn(|lane| {
            cosine_from_dot(dots[lane], norm, query_norms[lane], 0.0).clamp(-1.0, 1.0)
        }))
    };
    let columns = if run.parallelism.is_parallel() {
        (0..rows.len())
            .into_par_iter()
            .map(score)
            .collect::<MathResult<Vec<_>>>()?
    } else {
        (0..rows.len()).map(score).collect::<MathResult<Vec<_>>>()?
    };
    run.check_cancelled()?;
    Ok(columns)
}
