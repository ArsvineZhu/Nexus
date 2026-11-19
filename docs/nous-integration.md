# Calling Nexus from Nous

## Prepare and share data

Use `Nexus::prepare(Dataset, &AssetConfig)` when you have raw concept vectors, passages, positions and source relations. Concept, passage and source IDs each keep their own slot order. A passage can have multiple ordered concept paths and cite multiple sources. Negative relations are available through the counterevidence view. The prepared basis, graphs and indexes can be shared across concurrent requests.

For existing assets, assemble a `PreparedDataset` and call `Nexus::from_prepared`. Create a `CsrGraphView` from raw parts once or obtain a view from an owned `CsrGraph`. Supply edge provenance in the same CSR order. `with_edge_gates` checks only the request's gate column and multiplies effective conductance in propagation, field solving and readout. Request-specific candidate matrices may be passed with `Candidates::Borrowed`. Built-in recall requires the prepared Tag field and, for Chunk recall, a document catalog.

The host handles permissions, time views, model calls and persistence. Set the visible Chunk scope in `Candidates::Recall`, or pass an already selected candidate batch. Keep the tag graph, source facts and evidence appropriate to the request's visibility. `Readout::candidate_ids` maps result slots back to host IDs.

## Observe, read out and combine

Call `nexus.observe` to retain query geometry, residual layers, seed decisions, the river network and the enhanced vector. Call `nexus.fields` when you also need the dual response fields and projection supports. `nexus.river` reads those results and `nexus.dtsc` reads the observation. `nexus.thinking` takes a complete retained trace and named candidate stages. Use `ObservationConfig::sense.max_transition_records` to configure the retention limit.

For host-provided Tag recall, call `observe_with_recall`. The callback receives the layer, current residual, limit, empty candidate buffer, initial EPA analysis and previously accepted Tag IDs. Fill the buffer in retrieval order. An empty buffer ends decomposition.

Use `Plan::add` with an operator, its config, and typed handles from earlier calls. Its binder takes those results and fills the same Rust input used by a direct call. `Plan::run` accepts one handle or a tuple of result handles. The example builds one observation and feeds it to River, DTSC and Thinking. [Example](../crates/nexus/examples/live_retrieval.rs)

Use an `ExecutionContext` with the host's cancellation source and optional Rayon pool for request-scoped controls. Sense has an optional caller-owned `SenseWorkspace` for allocation reuse. Budgets that leave propagation incomplete are surfaced in its diagnostics. DTSC exposes the number of available results and the actual ranking basis separately from diagnostic truncation.
