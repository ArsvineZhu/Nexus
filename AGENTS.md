# Developing Nexus

Nexus is the in-memory Rust computation library for Nous. The host owns indexes, external IDs, permission and time views, persistence, and model calls. Nexus consumes borrowed dense assets and returns typed numerical results. See [INDEX.md](INDEX.md) for code entry points and [upstream sources](docs/upstream.md) for numerical references.

- Use the Rust toolchain pinned at the workspace root. Run `just check` for the same checks as CI. For a focused change, run `cargo test -p <crate> --all-features --locked` and the applicable Clippy and format checks. `just --list` shows the available shortcuts.
- Keep nightly-only APIs and unsafe code out of authored crates. Use the shared `ExecutionContext`, `ScratchPad`, dense slot IDs, and kernels. Define algorithm rules in their owning operator. Graph nodes call those operators using typed inputs.
- Keep VCP-derived code in `nexus-vcp` (CC BY-NC-SA 4.0). The math-only graph must build without it. Follow the pinned upstream for numerical behavior and use the Nous sources for its independently documented extensions.
- Use `codegraph init --yes` to index the repository locally and `codegraph explore` for relationships. With an existing index, use `codegraph sync` after changes. Keep the generated index, reference checkouts and experimental outputs local under `.codegraph/` or `temp/`.
- Put behavioral regressions in the owning crate's `tests/`. Prefer a complete public Rust call for integration checks. Reference-output fixtures should identify their source and revision. Use the host's real frozen data for performance measurements and keep local absolute paths out of source and configuration.
