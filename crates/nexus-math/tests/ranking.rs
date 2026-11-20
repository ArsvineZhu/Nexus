// SPDX-License-Identifier: Apache-2.0

use nexus_math::nexus_core::{ExecutionContext, VcpError};
use nexus_math::{RrfConfig, WeightedRanking, rrf};

#[test]
fn rrf_sums_reciprocal_ranks_over_the_id_union() {
    let rankings = [
        WeightedRanking {
            ids: &[10, 20],
            weight: 1.0,
        },
        WeightedRanking {
            ids: &[20, 30],
            weight: 1.0,
        },
        WeightedRanking {
            ids: &[90, 80],
            weight: 0.0,
        },
        WeightedRanking {
            ids: &[],
            weight: 0.5,
        },
    ];
    let result = rrf(
        &rankings,
        &RrfConfig::default(),
        &ExecutionContext::default(),
    )
    .unwrap();
    let first = 1.0 / 61.0;
    let second = 1.0 / 62.0;
    let expected = [
        (20, first + second),
        (10, first),
        (30, second),
        (90, 0.0),
        (80, 0.0),
    ];
    assert_eq!(result.items.len(), expected.len());
    for (item, (id, score)) in result.items.iter().zip(expected) {
        assert_eq!(item.id, id);
        assert!((item.score - score).abs() < 1e-14);
    }
}

#[test]
fn rrf_distinguishes_empty_input_from_invalid_rankings() {
    let ctx = ExecutionContext::default();
    let config = RrfConfig::default();
    let empty = rrf(&[], &config, &ctx);
    assert!(empty.unwrap().items.is_empty());

    for rankings in [
        [WeightedRanking {
            ids: &[10, 10],
            weight: 1.0,
        }],
        [WeightedRanking {
            ids: &[10],
            weight: 0.0,
        }],
        [WeightedRanking {
            ids: &[10],
            weight: -1.0,
        }],
        [WeightedRanking {
            ids: &[10],
            weight: f64::INFINITY,
        }],
    ] {
        assert!(matches!(
            rrf(&rankings, &config, &ctx),
            Err(VcpError::InvalidInput { .. })
        ));
    }
    let rankings = [WeightedRanking {
        ids: &[10],
        weight: 1.0,
    }];
    assert!(matches!(
        rrf(&rankings, &RrfConfig { k: 0.0 }, &ctx),
        Err(VcpError::InvalidInput { .. })
    ));
}

#[test]
fn rrf_keeps_first_appearance_on_equal_positive_scores() {
    let rankings = [
        WeightedRanking {
            ids: &[20, 10],
            weight: 1.0,
        },
        WeightedRanking {
            ids: &[10, 20],
            weight: 1.0,
        },
    ];
    let result = rrf(
        &rankings,
        &RrfConfig::default(),
        &ExecutionContext::default(),
    )
    .unwrap();
    assert_eq!(result.items[0].score, result.items[1].score);
    assert_eq!(
        result.items.iter().map(|item| item.id).collect::<Vec<_>>(),
        [20, 10]
    );
}
