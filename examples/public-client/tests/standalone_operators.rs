// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::context::{
    ContextFragment, ContextGroup, ContextInput, SegmentOptions, WeightedVector, aggregate,
    segment, weighted_projection,
};
use nexus::math::{
    DedupConfig, DedupOptions, NumericColumns, RankedList, Scores, SearchEffort, SearchRequest,
    VectorRows, WarpedIndex, dtsc, merge_ranked_ids, reciprocal_rank_fusion, semantic_dedup,
    solve_nnls,
};
use nexus::{CancelToken, DtscDecision, EnergyNode, Error, EvidenceSource, RunOptions};

const EAST: &[f32] = &[1.0, 0.0];
const WEST: &[f32] = &[-1.0, 0.0];
const NORTH: &[f32] = &[0.0, 1.0];

#[track_caller]
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
}

#[test]
fn weighted_projection_handles_relative_strength_and_cancellation() -> nexus::Result<()> {
    let options = Default::default();
    let projected = weighted_projection(
        &[
            WeightedVector {
                vector: &[10.0, 0.0],
                weight: 2.0,
            },
            WeightedVector {
                vector: NORTH,
                weight: 1.0,
            },
        ],
        &options,
    )?
    .expect("net direction");
    close(f64::from(projected.as_slice()[0]), 2.0 / 5.0_f64.sqrt());
    close(f64::from(projected.as_slice()[1]), 1.0 / 5.0_f64.sqrt());
    assert!(weighted_projection(&[], &options)?.is_none());
    assert!(
        weighted_projection(
            &[
                WeightedVector {
                    vector: EAST,
                    weight: 1.0
                },
                WeightedVector {
                    vector: WEST,
                    weight: 1.0
                },
            ],
            &options,
        )?
        .is_none()
    );

    let mut group = ContextGroup::new(10);
    group.budget = 2.0;
    let west = ContextFragment::new(2, WEST, 10);
    let aggregate = aggregate(
        ContextInput {
            query: Some(EAST),
            fragments: &[west],
            groups: &[group],
        },
        &Default::default(),
    )?;
    assert_eq!(aggregate.allocations[0].mass, 1.0);
    assert_eq!(
        aggregate.direction.expect("original direction").as_slice(),
        EAST
    );
    Ok(())
}

#[test]
fn context_allocates_each_source_range_once_per_group_across_parents() -> nexus::Result<()> {
    let mut group_a = ContextGroup::new(10);
    group_a.budget = 0.6;
    group_a.parent_limit = None;
    let mut group_b = ContextGroup::new(20);
    group_b.budget = 0.5;
    let groups = [group_a, group_b];

    let mut first = ContextFragment::new(1, EAST, 10);
    first.parent = Some(100);
    first.source = Some(900);
    first.range = Some(0..10);
    first.sequence = 3;

    let mut later = ContextFragment::new(2, NORTH, 10);
    later.parent = Some(200);
    later.source = Some(900);
    later.range = Some(5..15);
    later.sequence = 2;
    later.distance = 2.0;
    later.half_life = Some(2.0);

    let mut zero = ContextFragment::new(3, WEST, 10);
    zero.parent = Some(100);
    zero.source = Some(900);
    zero.range = Some(5..15);
    zero.sequence = 10;
    zero.weight = 0.0;

    let mut repeated = ContextFragment::new(4, WEST, 10);
    repeated.parent = Some(300);
    repeated.source = Some(900);
    repeated.range = Some(2..7);
    repeated.sequence = 1;

    let source_free = ContextFragment::new(5, NORTH, 10);
    let mut other_group = ContextFragment::new(6, NORTH, 20);
    other_group.source = Some(900);
    other_group.range = Some(0..10);

    let mut fragments = vec![first, later, zero, repeated, source_free, other_group];
    let result = aggregate(
        ContextInput {
            query: Some(&[5.0, 0.0]),
            fragments: &fragments,
            groups: &groups,
        },
        &Default::default(),
    )?;
    let allocations: Vec<_> = result.allocations.iter().map(|row| row.fragment).collect();
    assert_eq!(allocations, [1, 2, 5, 6]);
    let expected = [0.6 / 3.5, 0.6 * 0.5 / 3.5, 0.6 / 3.5, 0.5 / 2.0];
    for (actual, expected) in result.allocations.iter().zip(expected) {
        close(actual.mass, expected);
    }
    let x = 1.0 + expected[0];
    let y = expected[1..].iter().sum::<f64>();
    let direction = result.direction.expect("net direction");
    close(f64::from(direction.as_slice()[0]), x / x.hypot(y));
    close(f64::from(direction.as_slice()[1]), y / x.hypot(y));

    let mut whole = ContextFragment::new(7, EAST, 10);
    whole.source = Some(900);
    fragments.push(whole);
    let result = aggregate(
        ContextInput {
            query: None,
            fragments: &fragments,
            groups: &groups,
        },
        &Default::default(),
    );
    assert!(matches!(
        result,
        Err(Error::Input {
            field: "context.range",
            id: Some(7),
            ..
        })
    ));
    Ok(())
}

#[test]
fn contiguous_segmentation_keeps_a_range_when_its_unit_directions_cancel() -> nexus::Result<()> {
    let a = (3.0_f32).sqrt() / 2.0;
    let first_half = [
        [1.0, 0.0],
        [a, 0.5],
        [0.5, a],
        [0.0, 1.0],
        [-0.5, a],
        [-a, 0.5],
    ];
    let circle: Vec<_> = first_half
        .into_iter()
        .chain(first_half.map(|[x, y]| [-x, -y]))
        .collect();
    let rows: Vec<&[f32]> = circle.iter().map(<[f32; 2]>::as_slice).collect();
    let segments = segment(&rows, &Default::default())?;
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].range, 0..12);
    assert!(segments[0].direction.is_none());

    let rows: &[&[f32]] = &[EAST, &[2.0, 1.0], NORTH, &[0.0, 3.0]];
    let segments = segment(rows, &Default::default())?;
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].range, 0..2);
    assert_eq!(segments[1].range, 2..4);
    assert_eq!(segments[1].direction.as_ref().unwrap().as_slice(), NORTH);
    let cancel = CancelToken::default();
    cancel.cancel();
    let options = SegmentOptions {
        run: RunOptions {
            cancel: Some(cancel),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(segment(rows, &options), Err(Error::Cancelled)));
    Ok(())
}

#[test]
fn rank_fusion_and_dedup_use_stable_business_order() -> nexus::Result<()> {
    let lists = [
        RankedList {
            ids: &[12, 12, 19],
            weight: 1.0,
        },
        RankedList {
            ids: &[19, 27],
            weight: 2.0,
        },
    ];
    let fusion = reciprocal_rank_fusion(&lists, &Default::default())?;
    let ranked = fusion.top(None);
    assert_eq!(ranked.ids(), &[19, 27, 12]);
    close(ranked.rows()[0].value, 1.0 / 62.0 + 2.0 / 61.0);
    close(ranked.rows()[1].value, 2.0 / 62.0);
    close(ranked.rows()[2].value, 1.0 / 61.0);
    assert_eq!(
        merge_ranked_ids(&[lists[0].ids, lists[1].ids]),
        [12, 19, 27]
    );

    let rows = VectorRows::new(&[12, 19, 27], &[1.0, 0.0, 0.0, 1.0, 0.6, 0.8], 2)?;
    let options = DedupOptions {
        config: DedupConfig {
            similarity_threshold: 0.5,
        },
        ..Default::default()
    };
    let result = semantic_dedup(rows, &options)?;
    assert_eq!(result.kept, [12, 19]);
    assert_eq!(result.folded.len(), 1);
    assert_eq!(
        (result.folded[0].candidate, result.folded[0].representative),
        (27, 12)
    );
    close(result.folded[0].cosine, 0.6);
    Ok(())
}

#[test]
fn standalone_nnls_and_dtsc_have_no_model_dependency() -> nexus::Result<()> {
    let solution = solve_nnls(
        NumericColumns {
            dimension: 2,
            column_count: 2,
            values: &[1.0, 1.0, 0.0, -1.0],
        },
        &[1.0, 0.1],
        &Default::default(),
    )?;
    close(solution.coefficients[0], 1.0);
    close(solution.coefficients[1], 0.9);
    close(solution.residual_ratio, 0.0);

    let concepts = VectorRows::new(&[41, 52, 63], &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], 2)?;
    let nodes = [
        EnergyNode {
            concept: 41,
            energy: 1.0,
            kind: EvidenceSource::Core,
            hop: 0,
            seed: None,
        },
        EnergyNode {
            concept: 52,
            energy: 0.8,
            kind: EvidenceSource::Inferred,
            hop: 0,
            seed: None,
        },
        EnergyNode {
            concept: 63,
            energy: 0.6,
            kind: EvidenceSource::Emergent,
            hop: 1,
            seed: Some(41),
        },
    ];
    let field = dtsc::build_field(concepts, &nodes, &Default::default())?;
    let inputs = [
        dtsc::SampleInput {
            concept: Some(52),
            vector: NORTH,
        },
        dtsc::SampleInput {
            concept: None,
            vector: EAST,
        },
    ];
    let samples = dtsc::sample_field(&field, None, &inputs, &Default::default())?;
    assert!(samples[0].exact);
    assert_eq!(samples[0].kind, Some(EvidenceSource::Inferred));
    close(samples[0].potential, 0.8);
    assert!(!samples[1].exact);
    close(samples[1].potential, 1.0);

    let paths = [dtsc::Path {
        candidate: 500,
        ordinal: 1,
        concepts: &[41],
    }];
    let candidates = VectorRows::new(&[500, 600], &[1.0, 0.0, 0.0, 1.0], 2)?;
    let evidence = dtsc::evidence(
        dtsc::EvidenceInput {
            field: &field,
            concepts,
            graph: None,
            query: EAST,
            enhanced_query: None,
            candidates,
            paths: &paths,
            geometry: None,
        },
        &Default::default(),
    )?;
    assert_eq!(evidence.items()[0].path, Some(1));
    let bonus = evidence.items()[0].metrics.geo_bonus;
    assert!(bonus > 0.0);
    let baseline = Scores::new([(600, 2.5), (500, -0.5)])?;
    let result = evidence.apply(&baseline, &Default::default())?;
    assert_eq!(result.decision, DtscDecision::Applied);
    assert_eq!(
        result.scores.values().collect::<Vec<_>>(),
        [(600, 2.5), (500, -0.5 + bonus)]
    );
    let zero_gain = dtsc::ReadoutOptions {
        config: dtsc::DtscReadoutConfig {
            correction_gain: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        evidence
            .apply(&baseline, &zero_gain)?
            .scores
            .values()
            .collect::<Vec<_>>(),
        [(600, 2.5), (500, -0.5)]
    );
    close(evidence.items()[0].metrics.geo_bonus, bonus);
    Ok(())
}

#[test]
fn warped_search_uses_one_index_for_full_and_scoped_searches() -> nexus::Result<()> {
    let rows = VectorRows::new(&[7, 18, 211], &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], 2)?;
    let index = WarpedIndex::build(rows, &Default::default())?;
    let result = index.search(
        EAST,
        &SearchRequest {
            limit: 3,
            ..Default::default()
        },
    )?;
    assert_eq!(result.ids().first(), Some(&7));
    let restricted = SearchRequest {
        allowed: Some(&[18, 211]),
        limit: 2,
        effort: SearchEffort::Exact,
        ..Default::default()
    };
    let scoped = index.search_many(&[EAST, NORTH], &restricted)?;
    assert_eq!(scoped[0].ids().len(), 2);
    assert_eq!(scoped[1].ids().first(), Some(&18));
    for row in scoped[0].rows() {
        close(
            row.distance,
            result
                .rows()
                .iter()
                .find(|hit| hit.id == row.id)
                .unwrap()
                .distance,
        );
    }
    assert!(
        index
            .search(
                EAST,
                &SearchRequest {
                    limit: 0,
                    ..Default::default()
                }
            )?
            .rows()
            .is_empty()
    );
    Ok(())
}
