// SPDX-License-Identifier: Apache-2.0

use nexus_math::{
    DedupConfig, DedupOptions, Error, RankedList, RrfConfig, RrfOptions, VectorRows,
    merge_ranked_ids, reciprocal_rank_fusion, semantic_dedup,
};

#[test]
fn rrf_deduplicates_each_list_and_sums_compact_ranks() {
    let lists = [
        RankedList {
            ids: &[10, 10, 20],
            weight: 1.0,
        },
        RankedList {
            ids: &[20, 30, 30],
            weight: 1.0,
        },
        RankedList {
            ids: &[90, 80],
            weight: 0.0,
        },
        RankedList {
            ids: &[],
            weight: 0.5,
        },
    ];
    let result = reciprocal_rank_fusion(&lists, &Default::default()).unwrap();
    let first = 1.0 / 61.0;
    let second = 1.0 / 62.0;
    let expected = [
        (20, first + second),
        (10, first),
        (30, second),
        (90, 0.0),
        (80, 0.0),
    ];
    let ranking = result.top(None);
    assert_eq!(
        ranking
            .rows()
            .iter()
            .map(|r| (r.id, r.value))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        merge_ranked_ids(&[lists[0].ids, lists[1].ids, lists[2].ids]),
        [10, 20, 30, 90, 80]
    );

    let ties = [
        RankedList {
            ids: &[20, 10],
            weight: 1.0,
        },
        RankedList {
            ids: &[10, 20],
            weight: 1.0,
        },
    ];
    assert_eq!(
        reciprocal_rank_fusion(&ties, &Default::default())
            .unwrap()
            .top(Some(1))
            .ids(),
        &[20]
    );
}

#[test]
fn distinguishes_empty_lists_from_invalid_effective_weights() {
    let empty = reciprocal_rank_fusion(
        &[RankedList {
            ids: &[],
            weight: 0.0,
        }],
        &Default::default(),
    )
    .unwrap();
    assert_eq!(empty.iter().len(), 0);
    for weight in [0.0, -1.0, f64::INFINITY] {
        let rankings = [
            RankedList { ids: &[10], weight },
            RankedList {
                ids: &[],
                weight: 1.0,
            },
        ];
        assert!(matches!(
            reciprocal_rank_fusion(&rankings, &Default::default()),
            Err(Error::Config {
                field: "weight",
                ..
            })
        ));
    }
    let options = RrfOptions {
        config: RrfConfig { k: 0.0 },
        ..Default::default()
    };
    assert!(matches!(
        reciprocal_rank_fusion(&[], &options),
        Err(Error::Config { field: "k", .. })
    ));
}

#[test]
fn dedup_uses_first_matching_representative_with_and_without_pruning() {
    for dimension in [2, 600] {
        let mut vectors = vec![0.0; 4 * dimension];
        vectors[0] = 1.0;
        vectors[dimension + 1] = 1.0;
        vectors[2 * dimension] = 0.6;
        vectors[2 * dimension + 1] = 0.8;
        vectors[3 * dimension + 1] = -1.0;
        let rows = VectorRows::new(&[8, 12, 9, 11], &vectors, dimension).unwrap();
        let options = DedupOptions {
            config: DedupConfig {
                similarity_threshold: 0.5,
            },
            ..Default::default()
        };
        let result = semantic_dedup(rows, &options).unwrap();
        assert_eq!(result.kept, [8, 12, 11]);
        assert_eq!(result.folded.len(), 1);
        assert_eq!(
            (result.folded[0].candidate, result.folded[0].representative),
            (9, 8)
        );
        assert!((result.folded[0].cosine - 0.6).abs() < 1e-7);
    }
}
