// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::*;
use nexus_vcp::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
fn warmed_propagation_reuses_allocation_and_recovers_from_cancel() {
    let config = SenseConfig {
        max_safe_hops: 2,
        max_neighbors_per_node: 1,
        max_transition_records: 1,
        minimum_injected_current: 0.001,
        dynamic_wormhole_slots: 1,
        ..Default::default()
    };
    let graph = TransportGraphView::new(
        CsrGraphView::new(CsrParts {
            node_count: 3,
            row_offsets: &[0, 2, 2, 3],
            targets: &[NodeId(1), NodeId(2), NodeId(0)],
            weights: &[0.65, 0.3, 0.85],
            is_static_wormhole: &[false; 3],
            inbound: &[0.85, 0.65, 0.3],
            max_inbound: 0.85,
            anchor_gains: &[0.0; 3],
            target_sorted_indices: None,
            edge_gates: None,
        })
        .unwrap(),
    )
    .unwrap();
    let tag_vectors = VectorMatrixView::new(&[1., 0., 1., 0., 0., 1.], 3, 2).unwrap();
    let seeds = GatedSeeds {
        seeds: vec![GatedSeed {
            node: NodeId(0),
            weight: 1.0,
            is_core: true,
            lexical: true,
            residual_level: 0,
        }],
        ..Default::default()
    };
    let residuals = [vec![1., 0.], vec![0., 1.]];
    let input = SenseInput {
        graph,
        tag_vectors,
        seeds: &seeds,
        residuals: Some(&residuals),
    };
    let ctx = ExecutionContext::default();
    let mut expected = SpikeSenseOperator.execute(&ctx, &config, input).unwrap();
    assert!(expected.diagnostics.elapsed_ms > 0.0);
    expected.diagnostics.elapsed_ms = 0.0;
    assert!(expected.transitions_truncated);
    assert!(
        expected
            .edges
            .iter()
            .any(|edge| edge.target == NodeId(2) && edge.wormhole)
    );
    assert!(expected.edges.iter().all(|edge| edge.target != NodeId(1)));

    let mut work = SenseWorkspace::default();
    let mut output = SenseOutput::default();
    SpikeSenseOperator
        .execute_into(&ctx, &config, input, &mut work, &mut output)
        .unwrap();
    let profile = dhat::Profiler::builder().testing().build();
    let before = dhat::HeapStats::get();
    for _ in 0..32 {
        SpikeSenseOperator
            .execute_into(&ctx, &config, input, &mut work, &mut output)
            .unwrap();
    }
    let after = dhat::HeapStats::get();
    drop(profile);
    assert_eq!(before.total_blocks, after.total_blocks);
    assert!(output.diagnostics.elapsed_ms > 0.0);
    output.diagnostics.elapsed_ms = 0.0;
    assert_eq!(output, expected);

    let count = AtomicUsize::new(0);
    let cancel = || count.fetch_add(1, Ordering::Relaxed) > 3;
    let stop = ExecutionContext {
        cancel: CancellationToken::from_fn(&cancel),
        ..ctx
    };
    let result = SpikeSenseOperator.execute_into(&stop, &config, input, &mut work, &mut output);
    assert!(matches!(result, Err(VcpError::Cancelled { .. })));
    assert_eq!(output, SenseOutput::default());
    SpikeSenseOperator
        .execute_into(&ctx, &config, input, &mut work, &mut output)
        .unwrap();
    assert!(output.diagnostics.elapsed_ms > 0.0);
    output.diagnostics.elapsed_ms = 0.0;
    assert_eq!(output, expected);

    let gates = [1.0, 0.0, 1.0];
    let input = SenseInput {
        graph: graph.with_edge_gates(&gates).unwrap(),
        ..input
    };
    SpikeSenseOperator
        .execute_into(&ctx, &config, input, &mut work, &mut output)
        .unwrap();
    assert!(output.edges.iter().any(|edge| edge.target == NodeId(1)));
    assert!(output.edges.iter().all(|edge| edge.target != NodeId(2)));
}
