# Code map

| Responsibility | Entry point |
| --- | --- |
| Typed operator contract, cancellation and resource limits | [`nexus-core/operator.rs`](crates/nexus-core/src/operator.rs) |
| Dense asset views and CSR transport | [`nexus-core/view.rs`](crates/nexus-core/src/view.rs), [`csr.rs`](crates/nexus-core/src/csr.rs) |
| SIMD, strict arithmetic and QR/NNLS | [`nexus-core/kernel.rs`](crates/nexus-core/src/kernel.rs), [`nnls.rs`](crates/nexus-core/src/nnls.rs) |
| VCP geometry, facts, transport and provenance | [`nexus-vcp`](crates/nexus-vcp/src/lib.rs) |
| DTSC field construction, sampling and readout | [`nexus-math/dtsc`](crates/nexus-math/src/dtsc/mod.rs) |
| Feature-gated graph crate | [`nexus-graph`](crates/nexus-graph/src/lib.rs) |

Public contract tests live under `crates/<crate>/tests/`. Workspace CI is in [`.github/workflows/check.yml`](.github/workflows/check.yml).
