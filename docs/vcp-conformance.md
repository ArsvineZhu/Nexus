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

Nexus follows the Nous evidence contract. Missing provenance gets the 0.15 numerical floor, and a group loses independent support when any required root belongs to the candidate. Upstream assumes complete independence for an unattributed edge and has one file identifier per candidate. The attributable River and wormhole cases compare the common scoring path. The unattributed case verifies that Nexus attenuates the bonus while preserving the semantic and path scores. Negative relations stay in the counterevidence view and the host can apply temporal edge gates.

The shared behavior includes zero-target NNLS, peak-normalized River fields and the minimum observed flow for active edges. River applies field normalization at readout, so the dual-field solver can expose its actual fixed point.

## Reproduce

From Nexus, run the capture script against a disposable VCPToolBox checkout pinned to the revision above. The script verifies the revision, builds the upstream test crate, and restores its source files after capture.

```sh
python scripts/capture-vcp-golden.py temp/reference/VCPToolBox crates/nexus-vcp/tests/fixtures/vcp_5adb12b.json
cargo test -p nexus-vcp --test upstream_golden --all-features --locked
```
