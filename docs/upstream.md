# Numerical sources

The VCP reference is [lioensky/VCPToolBox at 5adb12b](https://github.com/lioensky/VCPToolBox/tree/5adb12bb55e616e22bd0e0dbfdc13032c069941a/rust-vexus-lite/src). The geometry, observation and RiverMemo algorithms are adapted under [CC BY-NC-SA 4.0](../LICENSES/CC-BY-NC-SA-4.0.txt). Nexus adaptations use borrowed dense arrays, a host-supplied candidate search and reusable workspaces in place of the upstream index, persistence and runtime. Nous extensions for independent-root provenance, counterevidence and other behavior are referenced from `Nous/native/vcp/UPSTREAM.md` and its adjacent source manifest.

DTSC adapts [`Nous-Wave` at `412b062`](https://github.com/Heptalogos-Devs/Nous-Wave/tree/412b062082afb75ccfae4c3f69fc606d3e3cc862) under [Apache 2.0](../LICENSES/Apache-2.0.txt). Its field sampler and curve scoring live in `nexus-math`. `nexus-core` and the orchestration code in `nexus-graph` are also Apache 2.0.
