# Developing Nexus

Nexus is a Rust library for mathematical and retrieval algorithms. See [INDEX.md](INDEX.md) for entry points and [numerical sources](docs/upstream.md) for formula definitions and attribution.

- Run `just check` for the workspace checks. Keep VCP-derived work in `nexus-vcp` and DTSC available through the independent Apache-2.0 `nexus-math` crate.
- Use the owning operator's typed input and config in both direct calls and `Plan`. Prepare and validate immutable data at the asset boundary. Keep reusable workspaces with the algorithms that own them.
- Use CodeGraph for indexed relationships. Keep reference checkouts, exported corpus data and experimental outputs in ignored local directories.
- Use the public Rust entry point for integration checks. Record the source revision of numerical reference outputs and profile genuine hotspots on a realistic load.
