// SPDX-License-Identifier: CC-BY-NC-SA-4.0
//! Combine or segment already-vectorized material.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use nexus_math::UnitVector;
use nexus_math::kernel::{PrecisionMode, cosine_f32};

use crate::{Result, RunOptions, input, numeric, parameter};

#[derive(Clone, Default)]
pub struct WeightedProjectionOptions {
    pub run: RunOptions,
}

pub struct WeightedVector<'a> {
    pub vector: &'a [f32],
    pub weight: f64,
}

pub fn weighted_projection(
    input: &[WeightedVector<'_>],
    options: &WeightedProjectionOptions,
) -> Result<Option<UnitVector>> {
    let run = &options.run;
    run.check_cancelled()?;
    let mut sum = Sum::new(input.first().map_or(0, |v| v.vector.len()));
    for item in input {
        run.check_cancelled()?;
        if !item.weight.is_finite() || item.weight < 0.0 {
            return Err(crate::invalid(
                "projection.weight",
                "must be finite and nonnegative",
            ));
        }
        let vector = unit(item.vector, sum.len(), "projection.vector", None)?;
        sum.add(vector.as_slice(), item.weight)?;
    }
    sum.direction()
}

pub struct ContextGroup {
    pub id: u64,
    pub budget: f64,
    pub null_mass: f64,
    pub parent_limit: Option<usize>,
}

impl ContextGroup {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            budget: 0.2,
            null_mass: 1.0,
            parent_limit: Some(10),
        }
    }
}

pub struct ContextFragment<'a> {
    pub id: u64,
    pub vector: &'a [f32],
    pub group: u64,
    pub parent: Option<u64>,
    pub source: Option<u64>,
    pub range: Option<Range<u64>>,
    pub sequence: u64,
    pub weight: f64,
    pub relevance: f64,
    pub validity: f64,
    pub distance: f64,
    pub half_life: Option<f64>,
}

impl<'a> ContextFragment<'a> {
    pub fn new(id: u64, vector: &'a [f32], group: u64) -> Self {
        Self {
            id,
            vector,
            group,
            parent: None,
            source: None,
            range: None,
            sequence: 0,
            weight: 1.0,
            relevance: 1.0,
            validity: 1.0,
            distance: 0.0,
            half_life: None,
        }
    }
}

pub struct ContextInput<'a> {
    pub query: Option<&'a [f32]>,
    pub fragments: &'a [ContextFragment<'a>],
    pub groups: &'a [ContextGroup],
}

#[derive(Clone, Default)]
pub struct ContextOptions {
    pub run: RunOptions,
}

pub struct ContextResult {
    pub direction: Option<UnitVector>,
    pub allocations: Vec<ContextAllocation>,
}
pub struct ContextAllocation {
    pub fragment: u64,
    pub mass: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Key {
    Explicit(u64),
    Fragment(u64),
}

impl ContextFragment<'_> {
    fn parent_key(&self) -> Key {
        self.parent.map_or(Key::Fragment(self.id), Key::Explicit)
    }
    fn source_key(&self) -> Key {
        self.source.map_or(Key::Fragment(self.id), Key::Explicit)
    }
}

struct Row {
    unit: UnitVector,
    value: f64,
}
enum Coverage {
    Whole(bool),
    Ranges(Vec<Range<u64>>),
}
impl Coverage {
    fn add(&mut self, range: Option<&Range<u64>>) -> f64 {
        match (self, range) {
            (Self::Whole(allocated), None) => f64::from(!std::mem::replace(allocated, true)),
            (Self::Ranges(union), Some(range)) => {
                let from = union.partition_point(|p| p.end < range.start);
                let to = from + union[from..].partition_point(|p| p.start <= range.end);
                let joined = &union[from..to];
                let covered: u64 = joined
                    .iter()
                    .map(|p| {
                        p.end
                            .min(range.end)
                            .saturating_sub(p.start.max(range.start))
                    })
                    .sum();
                let start = joined
                    .first()
                    .map_or(range.start, |p| p.start.min(range.start));
                let end = joined.last().map_or(range.end, |p| p.end.max(range.end));
                union.splice(from..to, std::iter::once(start..end));
                (range.end - range.start - covered) as f64
            }
            _ => unreachable!("validated coverage mode"),
        }
    }
}

#[derive(Default)]
struct Parent {
    sequence: u64,
    coverage: f64,
    weighted: f64,
}
impl Parent {
    fn utility(&self) -> f64 {
        if self.coverage > 0.0 {
            self.weighted / self.coverage
        } else {
            0.0
        }
    }
}

pub fn aggregate(request: ContextInput<'_>, options: &ContextOptions) -> Result<ContextResult> {
    let run = &options.run;
    run.check_cancelled()?;
    let fragments = request.fragments;
    let dim = request
        .query
        .map(<[f32]>::len)
        .or_else(|| fragments.first().map(|fragment| fragment.vector.len()))
        .unwrap_or(0);
    let query = request
        .query
        .map(|v| unit(v, dim, "context.query", None))
        .transpose()?;
    let mut group_slots = HashMap::new();
    for (i, group) in request.groups.iter().enumerate() {
        run.check_cancelled()?;
        if group_slots.insert(group.id, i).is_some() {
            return Err(input(
                "context.groups",
                Some(group.id),
                "duplicate group ID",
            ));
        }
        if !group.budget.is_finite()
            || group.budget < 0.0
            || !group.null_mass.is_finite()
            || group.null_mass <= 0.0
        {
            return Err(input(
                "context.groups",
                Some(group.id),
                "budget must be finite and nonnegative and null_mass positive",
            ));
        }
    }
    let mut by_group = vec![Vec::new(); request.groups.len()];
    let mut rows = Vec::with_capacity(fragments.len());
    let mut ids = HashSet::new();
    let mut modes = HashMap::new();
    for (i, fragment) in fragments.iter().enumerate() {
        run.check_cancelled()?;
        let id = Some(fragment.id);
        if !ids.insert(fragment.id) {
            return Err(input("context.fragments", id, "duplicate fragment ID"));
        }
        let group = *group_slots
            .get(&fragment.group)
            .ok_or_else(|| input("context.group", id, "unknown group ID"))?;
        if !fragment.weight.is_finite()
            || fragment.weight < 0.0
            || !fragment.distance.is_finite()
            || fragment.distance < 0.0
        {
            return Err(input(
                "context.fragment",
                id,
                "weight and distance must be finite and nonnegative",
            ));
        }
        if !(0.0..=1.0).contains(&fragment.relevance) || !(0.0..=1.0).contains(&fragment.validity) {
            return Err(input(
                "context.fragment",
                id,
                "relevance and validity must be in [0,1]",
            ));
        }
        if fragment
            .half_life
            .is_some_and(|t| !t.is_finite() || t <= 0.0)
        {
            return Err(input(
                "context.half_life",
                id,
                "must be positive and finite",
            ));
        }
        if fragment.range.as_ref().is_some_and(|r| r.start >= r.end) {
            return Err(input(
                "context.range",
                id,
                "must be a nonempty half-open range",
            ));
        }
        let ranged = fragment.range.is_some();
        if modes
            .insert((group, fragment.source_key()), ranged)
            .is_some_and(|mode| mode != ranged)
        {
            return Err(input(
                "context.range",
                id,
                "source uses both whole and ranged coverage within the group",
            ));
        }
        let decay = fragment
            .half_life
            .map_or(1.0, |t| (-fragment.distance / t).exp2());
        let value = fragment.weight * fragment.relevance * fragment.validity * decay;
        rows.push(Row {
            unit: unit(fragment.vector, dim, "context.vector", id)?,
            value,
        });
        by_group[group].push(i);
    }

    let mut sum = Sum::new(dim);
    if let Some(query) = &query {
        sum.add(query.as_slice(), 1.0)?;
    }
    let mut allocations = Vec::new();
    for (group, mut indices) in request.groups.iter().zip(by_group) {
        run.check_cancelled()?;
        let mut parents: BTreeMap<Key, Parent> = BTreeMap::new();
        for &i in &indices {
            let fragment = &fragments[i];
            let parent = parents.entry(fragment.parent_key()).or_default();
            parent.sequence = parent.sequence.max(fragment.sequence);
        }
        let mut ordered: Vec<_> = parents
            .iter()
            .map(|(&key, parent)| (key, parent.sequence))
            .collect();
        ordered.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ordered.truncate(group.parent_limit.unwrap_or(ordered.len()));
        let retained: HashSet<_> = ordered.into_iter().map(|(key, _)| key).collect();
        indices.retain(|&i| retained.contains(&fragments[i].parent_key()) && rows[i].value > 0.0);
        indices.sort_unstable_by(|&a, &b| {
            fragments[b]
                .sequence
                .cmp(&fragments[a].sequence)
                .then(fragments[a].id.cmp(&fragments[b].id))
        });

        let mut sources = HashMap::new();
        let mut covered = Vec::with_capacity(indices.len());
        for i in indices {
            run.check_cancelled()?;
            let fragment = &fragments[i];
            let coverage = sources
                .entry(fragment.source_key())
                .or_insert_with(|| {
                    if fragment.range.is_some() {
                        Coverage::Ranges(Vec::new())
                    } else {
                        Coverage::Whole(false)
                    }
                })
                .add(fragment.range.as_ref());
            let weighted = coverage * rows[i].value;
            let parent = parents
                .get_mut(&fragment.parent_key())
                .expect("selected parent");
            parent.coverage += coverage;
            parent.weighted += weighted;
            if !parent.weighted.is_finite() {
                return Err(numeric("context", "parent coverage mass overflowed"));
            }
            if weighted > 0.0 {
                covered.push((i, weighted));
            }
        }
        let denominator = group.null_mass + parents.values().map(Parent::utility).sum::<f64>();
        if !denominator.is_finite() {
            return Err(numeric("context", "group utility overflowed"));
        }
        for (i, weighted) in covered {
            run.check_cancelled()?;
            let fragment = &fragments[i];
            let parent = &parents[&fragment.parent_key()];
            let mass =
                group.budget * (parent.utility() / denominator) * (weighted / parent.weighted);
            sum.add(rows[i].unit.as_slice(), mass)?;
            if mass > 0.0 {
                allocations.push(ContextAllocation {
                    fragment: fragment.id,
                    mass,
                });
            }
        }
    }
    Ok(ContextResult {
        direction: sum.direction()?.or(query),
        allocations,
    })
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SegmentConfig {
    pub adjacent_similarity_threshold: f64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            adjacent_similarity_threshold: 0.7,
        }
    }
}

#[derive(Clone, Default)]
pub struct SegmentOptions {
    pub config: SegmentConfig,
    pub run: RunOptions,
}
pub struct Segment {
    pub range: Range<usize>,
    pub direction: Option<UnitVector>,
}

pub fn segment(vectors: &[&[f32]], options: &SegmentOptions) -> Result<Vec<Segment>> {
    let threshold = options.config.adjacent_similarity_threshold;
    let run = &options.run;
    run.check_cancelled()?;
    parameter(
        "segment",
        "adjacent_similarity_threshold",
        threshold,
        -1.0,
        1.0,
    )?;
    let Some((first, tail)) = vectors.split_first() else {
        return Ok(Vec::new());
    };
    let dim = first.len();
    let mut previous = unit(first, dim, "segment.vector", Some(0))?;
    let mut sum = Sum::new(dim);
    sum.add(previous.as_slice(), 1.0)?;
    let mut start = 0;
    let mut segments = Vec::new();
    for (i, vector) in tail.iter().enumerate() {
        run.check_cancelled()?;
        let i = i + 1;
        let direction = unit(vector, dim, "segment.vector", Some(i as u64))?;
        let similarity = cosine_f32(
            previous.as_slice(),
            direction.as_slice(),
            PrecisionMode::StrictLeftFold,
        )
        .clamp(-1.0, 1.0);
        if similarity < threshold {
            segments.push(Segment {
                range: start..i,
                direction: sum.direction()?,
            });
            sum = Sum::new(dim);
            start = i;
        }
        sum.add(direction.as_slice(), 1.0)?;
        previous = direction;
    }
    segments.push(Segment {
        range: start..vectors.len(),
        direction: sum.direction()?,
    });
    Ok(segments)
}

fn unit(vector: &[f32], dim: usize, field: &'static str, id: Option<u64>) -> Result<UnitVector> {
    if vector.len() != dim {
        return Err(input(
            field,
            id,
            format!("expected dimension {dim}, got {}", vector.len()),
        ));
    }
    nexus_math::normalize(vector).map_err(|e| input(field, id, e.to_string()))
}

// Compensated component sums retain low-mass contributions and exact antipodal cancellation.
struct Sum(Vec<(f64, f64)>);

impl Sum {
    fn new(dim: usize) -> Self {
        Self(vec![(0.0, 0.0); dim])
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn add(&mut self, row: &[f32], weight: f64) -> Result<()> {
        for ((sum, error), &component) in self.0.iter_mut().zip(row) {
            let value = f64::from(component) * weight;
            let next = *sum + value;
            if !next.is_finite() {
                return Err(numeric("weighted_projection", "vector sum overflowed"));
            }
            *error += if sum.abs() >= value.abs() {
                (*sum - next) + value
            } else {
                (value - next) + *sum
            };
            *sum = next;
        }
        Ok(())
    }

    fn direction(self) -> Result<Option<UnitVector>> {
        let values: Vec<_> = self.0.into_iter().map(|(sum, error)| sum + error).collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(numeric(
                "weighted_projection",
                "projected direction overflowed",
            ));
        }
        let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if scale == 0.0 {
            return Ok(None);
        }
        let relative: Vec<_> = values
            .into_iter()
            .map(|value| (value / scale) as f32)
            .collect();
        nexus_math::normalize(&relative).map(Some)
    }
}
