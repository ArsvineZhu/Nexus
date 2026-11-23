// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus::math::UnitVectors;
use nexus::{
    Candidate, CandidateMetadata, CandidateView, Error, Model, Query, ReadoutSummary, RelationData,
    ScoreEvidence,
};

fn empty_model() -> Model {
    Model::build(
        RelationData {
            dimension: 2,
            concepts: vec![],
            sources: vec![],
            relations: vec![],
        },
        &Default::default(),
    )
    .unwrap()
}

#[test]
fn candidate_selection_shares_owned_or_borrowed_rows_and_tracks_business_identity() {
    let model = empty_model();
    let ids = [300, 0, u64::MAX];
    let matrix = [1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
    let selected = [u64::MAX, 300];
    let owned = model
        .candidates(
            ids.into_iter()
                .zip(matrix.as_chunks::<2>().0)
                .map(|(id, row)| {
                    Candidate::new(id, row.iter().map(|component| 8.0 * component).collect())
                        .with_document_id(17)
                })
                .collect(),
        )
        .unwrap();
    let units = UnitVectors::new(&ids, &matrix, 2).unwrap();
    let metadata = CandidateMetadata {
        id: 300,
        document_id: Some(17),
        sources: vec![],
        paths: None,
    };
    let view = CandidateView::new(units)
        .with_metadata(vec![metadata])
        .unwrap();
    let borrowed = model.candidate_view(view).unwrap();
    assert_eq!(
        borrowed.vectors().row(2).unwrap().as_ptr(),
        matrix[4..].as_ptr()
    );
    for candidates in [&owned, &borrowed] {
        let subset = candidates.select(selected).unwrap();
        assert_eq!(subset.ids(), selected);
        assert_eq!(
            subset.vectors().row(0).unwrap().as_ptr(),
            candidates.vectors().row(2).unwrap().as_ptr()
        );
        assert!(matches!(
            subset.select([0]),
            Err(Error::Input { id: Some(0), .. })
        ));
        assert!(matches!(
            candidates.select([300, 300]),
            Err(Error::Input { id: Some(300), .. })
        ));
        assert!(candidates.select([]).unwrap().is_empty());
        let observe = model
            .clone()
            .observe(Query::new(&[4.0, 0.0]), &Default::default())
            .unwrap();
        let evaluation = observe.evaluate(&subset, &Default::default()).unwrap();
        assert_eq!(
            evaluation.cosine().values().collect::<Vec<_>>(),
            [(u64::MAX, 0.0), (300, 1.0)]
        );
        let ranking = evaluation.cosine().top(None);
        assert_eq!(ranking.ids(), [300, u64::MAX]);
        assert_eq!(ranking.rows()[0].document_id, Some(17));
        let other = empty_model()
            .observe(Query::new(&[4.0, 0.0]), &Default::default())
            .unwrap();
        assert!(matches!(
            other.evaluate(&subset, &Default::default()),
            Err(Error::Association { .. })
        ));
    }
}

#[test]
fn reusable_dtsc_geometry_applies_only_to_baselines_from_its_evaluation() {
    let (base, reversed, evidence) = {
        let model = empty_model();
        let candidates = model
            .candidates(vec![
                Candidate::new(1, vec![9.0, 0.0]),
                Candidate::new(2, vec![0.0, 8.0]),
            ])
            .unwrap();
        let observation = model
            .observe(Query::new(&[1.0, 0.0]), &Default::default())
            .unwrap();
        let eval = observation
            .evaluate(&candidates, &Default::default())
            .unwrap();
        let alternate_eval = observation
            .evaluate(&candidates, &Default::default())
            .unwrap();
        let evidence = eval.dtsc_evidence(&Default::default()).unwrap();
        let base = eval.baseline([(2, 15.0), (1, -3.0)]).unwrap();
        let reversed = eval.baseline([(2, 7.0), (1, 7.0)]).unwrap();
        assert!(matches!(
            alternate_eval.dtsc(&base, &Default::default()),
            Err(Error::Association { .. })
        ));
        assert!(matches!(
            evidence.apply(&alternate_eval.cosine(), &Default::default()),
            Err(Error::Association { .. })
        ));
        for values in [
            vec![(1, 1.0)],
            vec![(1, 1.0), (1, 1.0)],
            vec![(1, f64::NAN), (2, 1.0)],
        ] {
            assert!(matches!(eval.baseline(values), Err(Error::Input { .. })));
        }
        assert!(matches!(base.summary(), ReadoutSummary::External));
        (base, reversed, evidence)
    };
    for base in [base, reversed] {
        let result = evidence.apply(&base, &Default::default()).unwrap();
        assert_eq!(
            result.values().collect::<Vec<_>>(),
            base.values().collect::<Vec<_>>()
        );
        assert_eq!(result.top(None).ids(), [2, 1]);
        assert!(result.top(Some(0)).rows().is_empty());
        for (row, baseline) in result.iter().zip(base.iter()) {
            let ScoreEvidence::Dtsc(detail) = &row.evidence else {
                panic!("DTSC evidence")
            };
            assert_eq!(detail.baseline_value, baseline.value);
            assert_eq!(detail.correction, 0.0);
        }
    }
}
