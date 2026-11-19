# Numerical sources

- [VCPToolBox RiverMemo V5.2 at 5adb12b](https://github.com/lioensky/VCPToolBox/blob/5adb12bb55e616e22bd0e0dbfdc13032c069941a/docs/RIVERMEMO_TOPOLOGY_V3.md) defines background geometry, query observation, dual fields and continuous readout. The VCP-derived Rust operators retain [CC BY-NC-SA 4.0](../LICENSES/CC-BY-NC-SA-4.0.txt).
- The warped Chunk and Tag fields use the pinned radial mappings, mean-axis projection and Clairaut distance. Nexus uses an in-memory navigation graph over radial and directional orderings followed by exact distance reranking. Small scopes use exhaustive distance evaluation.
- [Nous-Wave at 412b062](https://github.com/Heptalogos-Devs/Nous-Wave/tree/412b062082afb75ccfae4c3f69fc606d3e3cc862/crates/retrieval/src/reference/dtsc) provides DTSC field and curve scoring, including optional shadow, auxiliary and identity scores. DTSC uses FP64 strict left folds. On low trust it returns the candidate order supplied by the host.
- `Nous/native/vcp/UPSTREAM.md` describes the optional multi-root provenance policy and the independent Thinking, context and dedup adaptations. The shared numerical infrastructure and DTSC are [Apache 2.0](../LICENSES/Apache-2.0.txt).

See the [golden comparisons](vcp-conformance.md) for the covered numerical behavior.

## Research definitions

The [upstream TagMemo archive](https://github.com/lioensky/VCPToolBox/tree/5adb12bb55e616e22bd0e0dbfdc13032c069941a/docs/archive/tagmemoV10) preserves direct-anchor readout, relative topology matching and alternative path scoring equations for research. TagMemo reads the shared enhanced-query cosine column. DTSC exposes field, evidence and pool stages with optional auxiliary and identity terms. The Ω weight function can feed independent business-ID RRF; Wormhole route ranking keeps its maximum-route contribution.

For River, paths expose quality, field support, node and edge alignment, direct contacts and the three continuous scoring components. The `AnalyzeRiverPaths` operator returns this evidence for all paths. `analyze_river_network` returns Ω and the morphology weights without requiring candidate rows.


Semantic inputs at the host boundary are finite unit directions after preparation. Domain transport uses effective nonnegative CSR weights with each row summing to at most one. Dual fields use propagated SpikeSense source mass, and field preparation requires both fixed points to converge before projection. EPA's spectral concentration is axis-energy concentration discounted by the domain gate. Without a basis, 0.5 is a neutral initialization and cache availability is false.
