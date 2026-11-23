// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::math::{
    DistanceHit, SearchEffort, SearchRequest, VectorRows, WarpedBuildOptions, WarpedGeometryConfig,
    WarpedIndex,
};
use nexus::{CancelToken, Error, Parallelism, RunOptions};

const IDS: [u64; 4] = [91, 22, 73_000, u64::MAX - 5];
const DOCUMENTS: [f32; 12] = [0.2, 0.9, 0.1, 1.0, 0.0, 0.0, 0.7, 0.35, 0.2, -0.3, 0.8, 0.3];
const QUERY: [f32; 3] = [0.8, 0.4, 0.1];

fn fixture() -> nexus::Result<WarpedIndex> {
    WarpedIndex::build(VectorRows::new(&IDS, &DOCUMENTS, 3)?, &Default::default())
}

#[test]
fn exact_search_replays_vcp_chunk_distances_and_preserves_scope() -> nexus::Result<()> {
    // VCPToolBox 5adb12bb55e616e22bd0e0dbfdc13032c069941a, Chunk quantiles
    // and exact_distance_at_radius evaluated on the captured input vectors.
    let reference = [
        (IDS[2], 1.014_206_084_546_737_6),
        (IDS[1], 1.292_017_025_608_602_3),
        (IDS[0], 6.266_815_000_432_454),
        (IDS[3], 7.742_266_993_289_002),
    ];
    let index = fixture()?;
    let exhaustive = index.search(&QUERY, &Default::default())?;
    assert_eq!(exhaustive.ids(), &reference.map(|(id, _)| id));
    for (&DistanceHit { id, distance }, (expected_id, expected)) in
        exhaustive.rows().iter().zip(reference)
    {
        assert_eq!(id, expected_id);
        assert!(
            (distance - expected).abs() < 1e-12,
            "ID {id}: {distance} != {expected}"
        );
    }
    for limit in 1..reference.len() {
        let hits = index.search(
            &QUERY,
            &SearchRequest {
                limit,
                ..Default::default()
            },
        )?;
        assert_eq!(hits.rows(), &exhaustive.rows()[..limit]);
    }

    let allowed = [IDS[3], IDS[1]];
    let request = SearchRequest {
        limit: 4,
        allowed: Some(&allowed),
        run: RunOptions {
            parallelism: Parallelism::Rayon,
            ..Default::default()
        },
        ..Default::default()
    };
    let scoped = index.search(&QUERY, &request)?;
    assert_eq!(scoped.rows(), [exhaustive.rows()[1], exhaustive.rows()[3]]);
    let second_query = [0.1, 0.9, 0.2];
    let multiple = index.search_many(&[&QUERY, &second_query], &request)?;
    assert_eq!(multiple.len(), 2);
    assert_eq!(multiple[0].rows(), scoped.rows());
    assert_eq!(
        multiple[1].rows(),
        index.search(&second_query, &request)?.rows()
    );
    Ok(())
}

#[test]
fn exact_shortlist_matches_exhaustive_distances_on_near_ties_and_spread() -> nexus::Result<()> {
    let ids: Vec<_> = (0..48).map(|slot| 800 - 13 * slot).collect();
    let mut matrix = vec![1.0, 0.001, 0.0, 1.0, 0.001, 0.0, 1.0, 0.0011, 0.00001];
    for slot in 3..ids.len() {
        let angle = 0.27 * slot as f32;
        matrix.extend([angle.cos(), angle.sin(), 0.35 * (3.1 * angle).sin()]);
    }
    let vectors = VectorRows::new(&ids, &matrix, 3)?;
    let profiles = [
        WarpedGeometryConfig::default(),
        WarpedGeometryConfig {
            throat: 0.0,
            angular_scale: 1.0,
            ..Default::default()
        },
    ];

    for geometry in profiles {
        let index = WarpedIndex::build(
            vectors,
            &WarpedBuildOptions {
                geometry,
                ..Default::default()
            },
        )?;
        for query in [[1.0, 0.001, 0.0], [-1.0, 0.0, 0.0], [0.05, 0.2, 1.0]] {
            let all = index.search(
                &query,
                &SearchRequest {
                    limit: usize::MAX,
                    ..Default::default()
                },
            )?;
            assert_eq!(all.rows().len(), ids.len());
            assert_eq!(
                index.search(&query, &Default::default())?.rows(),
                all.rows()
            );
            let duplicates: Vec<_> = all
                .rows()
                .iter()
                .filter(|row| ids[..2].contains(&row.id))
                .collect();
            assert_eq!([duplicates[0].id, duplicates[1].id], [ids[0], ids[1]]);
            assert_eq!(duplicates[0].distance, duplicates[1].distance);
            for limit in [1, 2, 7, 31] {
                let top = index.search(
                    &query,
                    &SearchRequest {
                        limit,
                        ..Default::default()
                    },
                )?;
                assert_eq!(top.rows(), &all.rows()[..limit]);
            }
        }
    }
    Ok(())
}

#[test]
fn approximate_search_returns_warped_distances_within_its_scope_and_budget() -> nexus::Result<()> {
    let index = fixture()?;
    let full = index.search(&QUERY, &Default::default())?;
    let complete_visit = SearchRequest {
        limit: 2,
        effort: SearchEffort::Approximate {
            visit_budget: 4,
            rerank: None,
        },
        ..Default::default()
    };
    assert_eq!(
        index.search(&QUERY, &complete_visit)?.rows(),
        &full.rows()[..2]
    );
    let allowed = [IDS[0], IDS[3]];
    let scoped = SearchRequest {
        allowed: Some(&allowed),
        effort: SearchEffort::Approximate {
            visit_budget: 4,
            rerank: Some(2),
        },
        ..complete_visit.clone()
    };
    assert_eq!(index.search(&QUERY, &scoped)?.rows(), &full.rows()[2..]);
    let budgeted = SearchRequest {
        limit: 3,
        effort: SearchEffort::Approximate {
            visit_budget: 1,
            rerank: None,
        },
        ..Default::default()
    };
    let hits = index.search(&QUERY, &budgeted)?;
    assert_eq!(hits.rows().len(), 1);
    assert!(full.rows().contains(&hits.rows()[0]));
    Ok(())
}

#[test]
fn empty_scopes_and_invalid_searches_keep_their_distinct_results() -> nexus::Result<()> {
    let empty = WarpedIndex::build(VectorRows::new(&[], &[], 3)?, &Default::default())?;
    assert!(empty.search(&QUERY, &Default::default())?.rows().is_empty());
    assert!(empty.search_many(&[], &Default::default())?.is_empty());
    let index = fixture()?;
    for request in [
        SearchRequest {
            limit: 0,
            ..Default::default()
        },
        SearchRequest {
            allowed: Some(&[]),
            ..Default::default()
        },
    ] {
        assert!(index.search(&QUERY, &request)?.ids().is_empty());
    }

    let unknown = SearchRequest {
        allowed: Some(&[111_111]),
        ..Default::default()
    };
    assert!(matches!(
        index.search(&QUERY, &unknown),
        Err(Error::Input {
            field: "search.allowed",
            id: Some(111_111),
            ..
        })
    ));
    for query in [&[0.0, 1.0][..], &[0.0, 0.0, 0.0], &[0.0, 1.0, f32::NAN]] {
        assert!(matches!(
            index.search(query, &Default::default()),
            Err(Error::Input { .. })
        ));
    }
    for (field, effort) in [
        (
            "visit_budget",
            SearchEffort::Approximate {
                visit_budget: 0,
                rerank: None,
            },
        ),
        (
            "rerank",
            SearchEffort::Approximate {
                visit_budget: 4,
                rerank: Some(1),
            },
        ),
    ] {
        let request = SearchRequest {
            limit: 2,
            effort,
            ..Default::default()
        };
        assert!(
            matches!(index.search(&QUERY, &request), Err(Error::Config { field: actual, .. }) if actual == field)
        );
    }

    let cancel = CancelToken::default();
    cancel.cancel();
    let cancelled = SearchRequest {
        run: RunOptions {
            cancel: Some(cancel),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        index.search(&QUERY, &cancelled),
        Err(Error::Cancelled)
    ));
    Ok(())
}
