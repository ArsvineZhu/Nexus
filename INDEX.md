# Code map

| Responsibility | Entry point |
| --- | --- |
| Typed operator contract, cancellation and resource limits | [`nexus-core/operator.rs`](crates/nexus-core/src/operator.rs) |
| Dense asset views and CSR transport | [`nexus-core/view.rs`](crates/nexus-core/src/view.rs), [`csr.rs`](crates/nexus-core/src/csr.rs) |
| SIMD, strict arithmetic and QR/NNLS | [`nexus-core/kernel.rs`](crates/nexus-core/src/kernel.rs), [`nnls.rs`](crates/nexus-core/src/nnls.rs) |
| VCP geometry, facts, transport and provenance | [`nexus-vcp`](crates/nexus-vcp/src/lib.rs) |
| EPA, residual pyramid, seed gating, propagation, dual fields and basin projection | [`nexus-vcp/observation`](crates/nexus-vcp/src/observation/mod.rs) |
| Candidate calibration, RiverMemo, wormhole witnesses and RRF | [`nexus-vcp/river`](crates/nexus-vcp/src/river/mod.rs) |
| Thinking, cosine folding and context | [`thinking.rs`](crates/nexus-vcp/src/thinking.rs), [`dedup.rs`](crates/nexus-vcp/src/dedup.rs), [`context.rs`](crates/nexus-vcp/src/context.rs) |
| DTSC field construction, sampling and readout | [`nexus-math/dtsc`](crates/nexus-math/src/dtsc/mod.rs) |
| Graph ports, validation and execution | [`dag.rs`](crates/nexus-graph/src/dag.rs), [built-in nodes](crates/nexus-graph/src/nodes/mod.rs) |

Public contract tests live under `crates/<crate>/tests/`. Workspace CI is in [`.github/workflows/check.yml`](.github/workflows/check.yml).
