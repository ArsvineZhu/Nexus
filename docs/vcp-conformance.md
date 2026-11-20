# VCPToolBox conformance

The [golden vectors](../crates/nexus-vcp/tests/fixtures/vcp_5adb12b.json) come from running the Rust implementation at [VCPToolBox `5adb12b`](https://github.com/lioensky/VCPToolBox/commit/5adb12bb55e616e22bd0e0dbfdc13032c069941a). The reference harness lives beside the Nexus tests. It calls upstream functions in a disposable checkout and uses in-memory upstream SQLite assets for complete River and wormhole readout.

| Area | Cases compared |
| --- | --- |
| NNLS | Opposed rays, joint fit, duplicate and zero columns, near collinearity, zero query |
| Cone-AGS | Opposed rays, reconsidering conditional gain, tied rays, a cone with zero positive fit |
| Hypernym and differentia | Redundant, explicit and independent parents, tight aliases, weak projected direction and low parent similarity |
| River | Exact focus, field rescaling, minimum observed flow, candidate order and score components |
| Cross-file wormholes | Independent bridge, missing third tag, transport with no source attribution |
| Deduplication | Matching rank, different host and query ranks, zero vectors |

The Nexus NNLS solver uses `faer` and retains FP64 projection and residual vectors. The oracle captures the upstream FP32 projection boundary. Comparisons allow that rounding and separately check unit coefficients, basis counts and residual ratios.

Nexus consumes host-resolved dense tags and candidate paths. Its hypernym output lists parents admitted to differentia projection, which requires query-to-parent cosine of at least 0.65. Upstream also exposes a lexical containment test before that geometric check. The exact-focus fixture provides the parent vector and witnessed first-hop edge to both implementations.

Candidate order is a host input to Nexus. Deduplication preserves that order when choosing representatives. Upstream deduplication chooses representatives by query similarity and then restores score order. The distinct-rank fixture asserts each result.

Nexus accepts unattributed edges at full weight by default. The discount policy applies a 0.15 multiplier to unattributed edges. With provenance, either policy excludes any group with a required root belonging to the candidate and preserves the 0.15 floor. Upstream assumes complete independence for an unattributed edge and has one file identifier per candidate. The attributable River and wormhole cases compare the common scoring path. The unattributed case verifies that Nexus attenuates the bonus while preserving the semantic and path scores. Negative relations stay in the counterevidence view and the host can apply temporal edge gates.

The shared behavior includes zero-target NNLS, peak-normalized River fields and the minimum observed flow for active edges. River peak-normalizes the fields when preparing path evidence. The dual-field solver uses propagated source mass and exposes the actual fixed point and residuals. River chooses the path with the highest complete score, with path and graph contributions from that same path inside the shared topology budget. The conformance test selects the upstream budget of 0.08. The ordinary readout defaults to 0.02.

## Reproduce

From Nexus, run the capture script against a disposable VCPToolBox checkout pinned to the revision above. The script verifies the revision, builds the upstream test crate, and restores its source files after capture.

```sh
python scripts/capture-vcp-golden.py temp/reference/VCPToolBox crates/nexus-vcp/tests/fixtures/vcp_5adb12b.json
cargo test -p nexus-vcp --test upstream_golden --all-features --locked
```

## DTSC field and curve references

The [field](../crates/nexus-math/tests/fixtures/vcp-dtsc-field.json) and [curve](../crates/nexus-math/tests/fixtures/vcp-dtsc.json) fixtures are the original numerical corpus in Nous-Wave at `412b062`. Their embedded capture revision is `e03b891d42055cdf3cab5dc961cc71dd5facd65a`. The standalone DTSC tests exercise field sampling, trusted scoring and low-trust input order with full geometric weight. Nexus starts from strict original-query cosine. The default readout applies 15% of the prepared bonus and, when geometry falls back, orders by cosine. Input-order fallback remains available for pools already ordered by a host ranker. The host sends lexically grounded and inferred seeds to DTSC as distinct evidence types. An inferred contact contributes to structure and direct reward requires a caller-supplied or lexically grounded seed.

## Warped Chunk reference

[The exact Chunk test](../crates/nexus-vcp/tests/hyperbolic.rs) captures a query radius and four distances from the pinned VCPToolBox quantile, direction transform and Clairaut routines. It also exercises scoped navigation followed by exact reranking.
