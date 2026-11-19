# Code and algorithm map

| Responsibility | Rust entry point | Contract |
|---|---|---|
| Host integration | [nexus](crates/nexus/src/lib.rs) | Prepare or import assets, attach business-ID recall and evaluate an ordered candidate set. |
| Typed composition | [Plan](crates/nexus-core/src/plan.rs) | Typed handles connect owned results. Steps borrow dependencies while the closure runs. |
| Vector and CSR primitives | [kernel](crates/nexus-core/src/kernel.rs), [CSR](crates/nexus-core/src/csr.rs) | Direction preparation, strict shared cosines and checked effective-weight transport. |
| Nonnegative fit | [NNLS](crates/nexus-core/src/nnls.rs) | Active-set least squares with QR compression and SVD subproblems. Outputs coefficients, projection and residual. |
| Background geometry and facts | [geometry](crates/nexus-vcp/src/geometry.rs), [artifact](crates/nexus-vcp/src/artifact.rs) | Local Cone-AGS and global Span, weighted ordered facts, CSR calibration, provenance and counterevidence. |
| Warped fields | [hyperbolic](crates/nexus-vcp/src/hyperbolic.rs) | Chunk quantile radii, Tag residual and inbound radii, EPA-led query radii and Clairaut distance. |
| Query observations | [observation](crates/nexus-vcp/src/observation/mod.rs) | EPA, residual pyramid, species difference, gated seeds, propagation and query fusion. Each operation has its own input and config. |
| Double response fields | [fields](crates/nexus-vcp/src/observation/fields.rs), [basins](crates/nexus-vcp/src/observation/projection.rs) | Local and transfer fixed points use propagated source mass. Field preparation projects only converged fields. |
| River readout | [river](crates/nexus-vcp/src/river/mod.rs) | Path evidence, morphology weights, Ω and maximum complete-path score. Cross-file transmission shares the topology budget. |
| Alternative readouts | [TagMemo](crates/nexus/src/evaluation.rs), [DTSC](crates/nexus-math/src/dtsc/mod.rs), [RRF](crates/nexus-math/src/ranking.rs) | Enhanced-query cosine, DTSC field/evidence/pool decisions, and weighted fusion over ordered business IDs. |
| Association and other operations | [Thinking](crates/nexus-vcp/src/thinking.rs), [context](crates/nexus-vcp/src/context.rs), [dedup](crates/nexus-vcp/src/dedup.rs) | Grounded ordered routes, context aggregation and segmentation, and ordered cosine deduplication. |

The [numerical sources](docs/upstream.md) link formulas, source revisions and license terms. The live example demonstrates the public API, while the [frozen benchmark](docs/benchmarks.md) measures host-style readout on a larger corpus.
