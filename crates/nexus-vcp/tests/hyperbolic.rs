// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::{ExecutionContext, VectorMatrixView};
use nexus_vcp::hyperbolic::{ChunkField, SearchConfig, WarpedConfig};

#[test]
fn chunk_recall_matches_pinned_exact_distance_and_preserves_scope() {
    // VCPToolBox 5adb12b: Chunk quantiles, WarpedFieldIndex transform,
    // and exact_distance_at_radius, evaluated without the storage layer.
    let documents = [0.2, 0.9, 0.1, 1.0, 0.0, 0.0, 0.7, 0.35, 0.2, -0.3, 0.8, 0.3];
    let query = [0.8, 0.4, 0.1];
    let ctx = ExecutionContext::default();
    let matrix = VectorMatrixView::new(&documents, 4, 3).unwrap();
    let field = ChunkField::build(matrix, WarpedConfig::default(), &ctx).unwrap();
    assert_eq!(field.radius(&query).unwrap(), f64::from(0.4419725_f32));
    let exact = field
        .search(&query, 4, None, SearchConfig::default(), &ctx)
        .unwrap();
    let reference = [
        (2, 1.014_206_084_546_737_6),
        (1, 1.292_017_025_608_602_3),
        (0, 6.266_815_000_432_454),
        (3, 7.742_266_993_289_002),
    ];
    for (hit, (slot, distance)) in exact.iter().zip(reference) {
        assert_eq!(hit.slot, slot);
        assert!((hit.distance - distance).abs() < 1e-12);
    }

    let scoped = field
        .search(
            &query,
            4,
            Some(&[false, true, false, true]),
            SearchConfig {
                exact_scope: 0,
                visit_budget: 4,
                rerank: Some(4),
            },
            &ctx,
        )
        .unwrap();
    assert_eq!(scoped, [exact[1], exact[3]]);
}

#[test]
fn tag_radii_and_new_residual_radius_match_pinned_geometry() {
    use nexus_core::NodeId;
    use nexus_vcp::{
        EpaAnalysis, IntrinsicResidual,
        hyperbolic::{residual_radius, tag_field},
    };

    let matrix = VectorMatrixView::new(&[1., 0., 0., 1., -1., 0., 0., -1.], 4, 2).unwrap();
    let residuals: Vec<_> = [0.3, 0.5, 0.2, 0.9]
        .into_iter()
        .enumerate()
        .map(|(slot, residual_ratio)| IntrinsicResidual {
            node: NodeId(slot as u32),
            residual_ratio,
            neighbor_count: 4,
            basis_used: 2,
            mgs2_fallback: false,
        })
        .collect();
    let field = tag_field(
        matrix,
        &[],
        &[0.0, 1.0, 1.0, 3.0],
        &residuals,
        WarpedConfig::default(),
        &ExecutionContext::default(),
    )
    .unwrap();
    let radii: Vec<_> = (0..4).map(|slot| field.radius_of(slot).unwrap()).collect();
    assert_eq!(
        radii,
        [0.56715024_f32, 0.48518756, 0.45308796, 0.42946547].map(f64::from)
    );

    let epa = EpaAnalysis {
        cache_available: true,
        in_domain_gate: 0.6,
        logic_depth: 0.4,
        entropy: 0.2,
        ..Default::default()
    };
    let radii: Vec<_> = [1.0, 0.25, 0.04]
        .into_iter()
        .enumerate()
        .map(|(level, ratio)| residual_radius(Some(&epa), level, ratio))
        .collect();
    assert_eq!(radii, [0.4652_f32, 0.5184817, 0.5588754].map(f64::from));
}
