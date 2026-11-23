// SPDX-License-Identifier: Apache-2.0

use super::curve::{DtscCurveContext, DtscEval, GeometryState, Sample, is_direct};
use super::field::dtsc_cosine;
use super::{positive, unit};
use crate::Result;

fn contacting(sample: &Sample, threshold: f64) -> bool {
    sample.field.potential > 0.0 && sample.field.potential >= threshold
}

pub(super) fn contacts(
    samples: &[Sample],
    ctx: &DtscCurveContext<'_>,
    score: &mut DtscEval,
) -> f64 {
    let mut contacted_mass = 0.0;
    let mut potential_mass = 0.0;
    let mut closure_mass = 0.0;
    let mut semantic_mass = 0.0;

    for s in samples {
        closure_mass += s.closure * s.mass;
        if contacting(s, ctx.config.curve.weak_contact_threshold) {
            score.metrics.hit_count += 1;
            contacted_mass += s.mass;
            potential_mass += s.mass * s.field.potential;
        }
        score.metrics.strong_hits +=
            usize::from(contacting(s, ctx.config.curve.strong_contact_threshold));
        if s.field.exact {
            score.metrics.exact_hits += 1;
            if is_direct(s) {
                score.geo_direct_exact_hits += 1;
            } else {
                score.geo_indirect_exact_hits += 1;
            }
        } else if is_direct(s)
            && s.field.potential >= ctx.config.reward.direct_semantic_min_potential
        {
            score.geo_direct_semantic_hits += 1;
            semantic_mass += s.field.potential;
        }
        score.geo_max_potential = score.geo_max_potential.max(s.field.potential);
    }

    let total: f64 = samples.iter().map(|s| s.mass).sum();
    score.geo_weighted_coverage = unit(contacted_mass / total);
    score.geo_mean_potential = unit(potential_mass / contacted_mass.max(1e-12));
    score.geo_closure_quality = unit(closure_mass / total);

    let r = &ctx.config.reward;
    if score.geo_direct_semantic_hits >= r.direct_semantic_min_contacts {
        score.geo_direct_semantic_strength = unit(
            (semantic_mass / score.geo_direct_semantic_hits as f64
                - r.direct_semantic_min_potential)
                / (r.direct_semantic_saturation - r.direct_semantic_min_potential),
        );
    }

    potential_mass
}

#[inline]
fn edge(ctx: &DtscCurveContext<'_>, a: u64, b: u64) -> f64 {
    ctx.graph.map_or(0.0, |graph| graph.edge(a, b))
}

fn similarity(ctx: &DtscCurveContext<'_>, a: &Sample, b: &Sample) -> f64 {
    let left = ctx.tag_vectors.row(a.row).expect("validated concept row");
    let right = ctx.tag_vectors.row(b.row).expect("validated concept row");
    dtsc_cosine(left, right)
}

fn sparse(
    samples: &[Sample],
    isolated: &[usize],
    isolated_mass: f64,
    ctx: &DtscCurveContext<'_>,
    connected_flags: &mut Vec<bool>,
) -> Result<f64> {
    let cfg = &ctx.config.sparse;
    if !cfg.enabled || ctx.graph.is_none() || isolated.len() < cfg.min_contacts {
        return Ok(0.0);
    }

    connected_flags.clear();
    connected_flags.resize(samples.len(), false);

    let mut pairs = 0usize;
    let mut quality_sum = 0.0;

    for (offset, &a) in isolated.iter().enumerate() {
        ctx.run.check_cancelled()?;
        let left = &samples[a];
        if left.field.potential < cfg.min_potential || left.closure < cfg.min_closure {
            continue;
        }
        for (index, &b) in isolated.iter().skip(offset + 1).enumerate() {
            if index.is_multiple_of(64) {
                ctx.run.check_cancelled()?;
            }
            let right = &samples[b];
            if a.abs_diff(b) <= 1
                || right.field.potential < cfg.min_potential
                || right.closure < cfg.min_closure
            {
                continue;
            }
            let conductance =
                edge(ctx, left.node, right.node).max(edge(ctx, right.node, left.node));
            let similarity = similarity(ctx, left, right);
            if conductance <= 0.0
                || conductance < cfg.min_conductance
                || similarity < cfg.min_similarity
            {
                continue;
            }
            let tq =
                unit((conductance - cfg.min_conductance) / (1.0 - cfg.min_conductance).max(1e-6));
            let sq = unit((similarity - cfg.min_similarity) / (1.0 - cfg.min_similarity).max(1e-6));
            let quality = positive(
                tq * sq
                    * (left.field.potential * right.field.potential).sqrt()
                    * (left.closure * right.closure).sqrt(),
            )
            .powf(0.25);
            if quality > 0.0 {
                pairs += 1;
                quality_sum += quality;
                connected_flags[a] = true;
                connected_flags[b] = true;
            }
        }
    }

    let connected_mass: f64 = connected_flags
        .iter()
        .enumerate()
        .filter(|&(_, &used)| used)
        .map(|(i, _)| samples[i].mass * samples[i].field.potential)
        .sum();

    let relief = if pairs > 0 && isolated_mass > 0.0 {
        unit(connected_mass / isolated_mass)
            * unit(pairs as f64 / cfg.pair_saturation as f64)
            * unit(quality_sum / pairs as f64)
    } else {
        0.0
    };
    Ok(relief)
}

pub(super) fn path(
    samples: &[Sample],
    weighted_potential: f64,
    ctx: &DtscCurveContext<'_>,
    score: &mut DtscEval,
    isolated_buf: &mut Vec<usize>,
    connected_flags: &mut Vec<bool>,
) -> Result<()> {
    let weak = ctx.config.curve.weak_contact_threshold;
    isolated_buf.clear();
    for (i, s) in samples.iter().enumerate() {
        if i.is_multiple_of(64) {
            ctx.run.check_cancelled()?;
        }
        if contacting(s, weak)
            && (i == 0 || !contacting(&samples[i - 1], weak))
            && (i + 1 == samples.len() || !contacting(&samples[i + 1], weak))
        {
            isolated_buf.push(i);
        }
    }

    let isolated_mass: f64 = isolated_buf
        .iter()
        .map(|&i| samples[i].mass * samples[i].field.potential)
        .sum();

    let mut sums = [0.0; 6];
    for (index, pair) in samples.windows(2).enumerate() {
        if index.is_multiple_of(64) {
            ctx.run.check_cancelled()?;
        }
        let a = &pair[0];
        let b = &pair[1];
        let arc = similarity(ctx, a, b).clamp(-1.0, 1.0).acos() / std::f64::consts::PI;
        let forward = edge(ctx, a.node, b.node);
        let reverse = edge(ctx, b.node, a.node);
        let topology = unit(forward.max(reverse).sqrt());
        let potential = (a.field.potential * b.field.potential).sqrt();
        let mass = (a.mass * b.mass).sqrt();

        sums[0] += mass;
        sums[1] += mass * potential * (0.65 + 0.35 * topology);
        sums[2] += arc * mass;
        sums[3] += arc * mass / (potential + 0.25 * topology).max(0.08);
        if forward + reverse > 0.0 {
            let w = mass * potential.max(0.05);
            sums[4] += w;
            sums[5] += w * forward / (forward + reverse);
        }
    }

    score.metrics.continuity = if sums[0] > 0.0 {
        unit(sums[1] / sums[0])
    } else {
        unit(score.geo_max_potential * 0.35)
    };
    let isolated_ratio = unit(isolated_mass / weighted_potential.max(1e-12));
    let confidence = sparse(samples, isolated_buf, isolated_mass, ctx, connected_flags)?;
    score.metrics.isolated_ratio =
        unit(isolated_ratio * (1.0 - ctx.config.sparse.max_relief * confidence));

    score.geo_action_quality = if sums[2] > 0.0 {
        unit((-sums[3] / sums[2].max(0.15)).exp())
    } else {
        unit(score.geo_max_potential * 0.5)
    };

    score.geo_direction_consistency = if sums[4] > 0.0 {
        unit(sums[5] / sums[4])
    } else {
        0.0
    };
    Ok(())
}

pub(super) fn shadow(
    (a, b): (f64, Option<f64>),
    target: usize,
    ctx: &DtscCurveContext<'_>,
    g: GeometryState,
    score: &mut DtscEval,
) {
    let enhanced = b.map_or(0.0, |b| 0.35 * b + 0.10 * unit(0.5 + (b - a) * 5.0));
    score.geo_closure_score = unit(0.25 * a + 0.30 * score.geo_closure_quality + enhanced);

    let saturation = unit(
        (score.geo_direct_exact_hits + score.geo_direct_semantic_hits) as f64
            / (ctx.config.reward.direct_semantic_min_contacts as f64 + 1.0),
    );
    score.geo_direct_score = unit(
        0.45 * score
            .geo_direct_semantic_strength
            .max(if score.geo_direct_exact_hits > 0 {
                1.0
            } else {
                0.0
            })
            + 0.25 * score.geo_max_potential
            + 0.20 * score.geo_mean_potential
            + 0.1 * saturation,
    );

    let contact = unit(
        (score.geo_indirect_exact_hits + score.metrics.strong_hits.min(target)) as f64
            / (target as f64 + 1.0),
    );
    score.geo_structural_score = unit(
        0.25 * score.metrics.continuity
            + 0.2 * score.geo_action_quality
            + 0.15 * score.geo_closure_quality
            + 0.15 * (1.0 - score.metrics.isolated_ratio)
            + 0.15 * score.geo_direction_consistency
            + 0.1 * contact,
    );

    score.geo_thematic_score = unit(
        0.35 * score.geo_weighted_coverage
            + 0.3 * score.geo_mean_potential
            + 0.2 * (1.0 - score.metrics.isolated_ratio)
            + 0.15 * score.geo_closure_quality,
    );

    let weights = [
        0.45 + 0.35 * unit(g.epa_concentration) + 0.2 * unit(g.pyramid_coverage),
        0.35 + 0.3 * unit(g.pyramid_depth)
            + 0.25 * unit(g.pyramid_novelty)
            + 0.1 * unit(g.epa_resonance),
        0.2 + 0.35 * unit(g.epa_entropy)
            + 0.3 * unit(g.epa_resonance)
            + 0.15 * (1.0 - unit(g.epa_concentration)),
    ];
    let total = weights.iter().sum::<f64>().max(1e-12);

    score.geo_fused_shadow_score = unit(
        (1.0 - (1.0 - weights[0] / total * score.geo_direct_score)
            * (1.0 - weights[1] / total * score.geo_structural_score)
            * (1.0 - weights[2] / total * score.geo_thematic_score))
            * score.geo_closure_score,
    );
}
