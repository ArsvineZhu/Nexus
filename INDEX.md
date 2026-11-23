# Code and algorithm map

| Responsibility | Entry point | Computation |
| --- | --- | --- |
| Host integration | [nexus](crates/nexus/src/lib.rs), [guide](docs/host-integration.md) | Build a model, bind candidates, observe a query, then score one candidate set. |
| Concept and source assets | [model](crates/nexus/src/model/) | Compile canonical source facts, root support, local Cone-AGS, Global Span and transport. |
| Candidate data | [candidates](crates/nexus/src/candidates.rs) | Normalize owned candidates or borrow unit rows, resolve original path positions and share subsets. |
| Query observation | [observation](crates/nexus/src/observation/) | EPA, Pyramid, lexical/core/inferred seeds, Sense, TagMemo fusion, dual fields and basin projection. |
| Vector search | [warped](crates/nexus/src/warped/) | Independent Chunk search on row matrices and internal Tag search on residuals. Both use Clairaut distance. |
| Shared candidate computation | [evaluation](crates/nexus/src/evaluation.rs) | Cache cosine columns and tag/body affinities for one observation and candidate set. |
| River | [river](crates/nexus/src/river/) | Complete-path evidence, independent source witnesses and a shared path/graph reward budget. |
| Thinking | [thinking](crates/nexus/src/thinking.rs) | Ordered stage selection over retained transitions and document identities. |
| Pure numeric operations | [nexus-math](crates/nexus-math/src/lib.rs) | Validated matrices, normalization, cosine, NNLS, transport, ranked scores, RRF and semantic dedup. |
| DTSC | [dtsc](crates/nexus-math/src/dtsc/) | Field, sampling, curve evidence and trust-aware correction of explicit baselines. |
| Context | [context](crates/nexus/src/context.rs) | Weighted projection, source-coverage aggregation and continuous vector segmentation. |

[Public examples](examples/public-client/) demonstrate direct use, host recall and shared readouts. The [retrieval study](docs/benchmarks.md) replays judged model vectors, and the [reference checks](docs/vcp-conformance.md) document pinned numerical inputs.
