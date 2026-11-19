# Calling Nexus from Nous

## Prepare and share data

Use `Nexus::prepare(Dataset, &AssetConfig)` when you have raw concept vectors, passages, positions and source relations. Concept, passage and source IDs each keep their own slot order. A passage can have multiple ordered concept paths and cite multiple sources. Negative relations are available through the counterevidence view. The prepared basis, graphs and indexes can be shared across concurrent requests.

For existing assets, assemble a `PreparedDataset` and call `Nexus::from_prepared`. Create a `CsrGraphView` from raw parts once or obtain a view from an owned `CsrGraph`. Supply edge provenance in the same CSR order. `with_edge_gates` checks only the request's gate column and multiplies effective conductance in propagation, field solving and readout. Request-specific candidate matrices may be passed with `Candidates::Borrowed`. Built-in recall requires the prepared Tag field and, for Chunk recall, a document catalog.

The host handles permissions, time views, model calls and persistence. Set the visible Chunk scope in `Candidates::Recall`, or pass an already selected candidate batch. Keep the tag graph, source facts and evidence appropriate to the request's visibility. `Readout::candidate_ids` maps result slots back to host IDs.

## Observe, read out and combine

For a preset with host-provided Tag recall, put the callback on the query. The same query works with `observe` and `ObserveQuery` in a custom plan.

~~~rust
let nexus = Nexus::from_prepared(assets)?;
let query = Query::new(vector, text).with_tag_recall(&recall);
let candidates = Candidates::Borrowed { ids, pool: &pool };
let result = nexus.run(&query, &candidates, Pipeline::river(), &Default::default())?;
~~~

The callback receives the layer, current residual, limit, reusable candidate buffer, initial EPA analysis and previously accepted Tag IDs. Fill the buffer with `(NodeId, score)` pairs in retrieval order. An empty buffer ends decomposition. An error ends the query and is returned to the caller.

When you want intermediate results or multiple readouts, prepare the candidate selection once.

~~~rust
let batch = nexus.candidates(query.vector, &candidates)?;
let observation = nexus.observe(&query, &Default::default())?;
let fields = nexus.fields(&observation, &Default::default())?;
let river = nexus.river(&fields, &batch, &Default::default())?;
let dtsc = nexus.dtsc(&observation, &batch, &Default::default())?;
let thinking = nexus.thinking(&observation, &batch, &stages, &Default::default())?;
~~~

The field result retains the observation it was computed from. `fields.observation()` gives access to its geometry, residuals, seeds, river network and enhanced vector. Thinking takes stages named by the caller, with document IDs from the batch. Set `ObservationConfig::sense.max_transition_records` to retain the complete propagation trace used by Thinking.

## Compose operators

`Plan::add` takes an operator, its configuration and handles from previous steps. Use `inputs.get()` in the binder to fill the same typed input you would pass to `Operator::execute`. For example, a `PyramidAnalyzeOperator` step can take the handle of `EpaAnalyzeOperator`, then combine that EPA result with the original query vector and the host callback. `Plan::run` accepts one output handle or a tuple of handles.

For River and DTSC together, the [Rust example](../crates/nexus/examples/live_retrieval.rs) shares one observation, candidate batch and `PoolCalibrateOperator` result. Set `PoolCalibrateConfig::precision` to `StrictLeftFold` to share the two query cosine columns with DTSC. The calibration configuration belongs to the calibration step. `RiverConfig` controls scoring, evidence and Wormhole.

Use an `ExecutionContext` with the host's cancellation source and optional Rayon pool for request-scoped controls. Sense has an optional caller-owned `SenseWorkspace` for allocation reuse. Budgets that leave propagation incomplete are surfaced in its diagnostics. DTSC exposes the number of available results and the actual ranking basis separately from diagnostic truncation.
