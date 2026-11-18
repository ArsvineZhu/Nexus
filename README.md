# Nexus

Pure Rust, `#![forbid(unsafe_code)]`, composable VCP & DTSC numerical compute foundation for `Nous`.

## Workspace Architecture (4-Crate Physical License Isolation)

| Crate | License | Role |
|---|---|---|
| `crates/nexus-core` | `Apache-2.0` | Dense `u32` slot IDs, zero-copy read-only views (`CsrGraphView`, `ProvenanceView`, `CandidatePoolView`, etc.), `wide` `FastSimd` & `StrictLeftFold` vector/CSR kernels, `faer` Householder QR + SVD Lawson–Hanson NNLS, `Operator` trait, `ExecutionContext`, `ScratchPad`, `VcpError`. |
| `crates/nexus-math` | `Apache-2.0` | Standalone numerical operators (`DTSC` from `Nous-Wave`), zero `CC-BY-NC-SA-4.0` dependencies. |
| `crates/nexus-vcp` | `CC-BY-NC-SA-4.0` | VCPToolBox V5.2 and Nous VCP, RiverMemo, Wormhole, Thinking, Dedup, and Context operators. |
| `crates/nexus-graph` | `Apache-2.0` | Strongly typed port-DAG orchestrator (`AssetBundle`, `ExecutionEnv`, `ComputeGraph`, `GraphValidator`, `GraphExecutor`). |

## Feature Flags

- `nexus-graph`:
  - `math-nodes` (default): enables `nexus-math` (`Apache-2.0`) node adapters.
  - `vcp-nodes` (default): enables `nexus-vcp` (`CC-BY-NC-SA-4.0`) node adapters. Disable via `--no-default-features --features math-nodes` for an `Apache-2.0`-only build with zero `CC-BY-NC-SA-4.0` transitive dependencies.
  - `schemars`: propagates `schemars::JsonSchema` derives across enabled crates.
- `nexus-core` / `nexus-math` / `nexus-vcp`:
  - `schemars`: optional JSON Schema derivation for IDs, configs, and diagnostics.

## Development

The workspace pins its Rust toolchain in `rust-toolchain.toml`. Run `just check` to format-check, lint, and test the workspace and the standalone math configuration. `just --list` lists the development commands. The [code map](INDEX.md) and [agent guide](AGENTS.md) describe the Rust entry points and local indexing. See [numerical sources](docs/upstream.md) for provenance and license terms.
