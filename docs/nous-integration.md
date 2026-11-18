# Calling Nexus from Nous

Create dense slot maps for concept, document, candidate and root IDs when opening a Nous snapshot. Keep the business IDs and their reverse maps in the host. The `nexus-core` views borrow the snapshot matrices and CSR arrays, and query input borrows the visibility-filtered candidate pool.

Build geometry and transport using `LocalConeAgsOperator`, `GlobalEpaOperator`, `FactMatrixOperator`, `TransportCalibrateOperator` and `ProvenanceBuildOperator`. Build provenance against the calibrated graph's edge order. Borrow an ungated `TransportGraph` through `as_csr_view()` and attach request gates with `with_edge_gates()`. Gates multiply the calibrated weights in propagation, field solving and readout. A graph calibrated with baked gates already contains attenuated weights.

## Observe and read out

Call `nexus_vcp::observe` with an `ObservationInput`, an `ObservationConfig`, and a caller-owned `ScratchPad`. The candidate provider receives the current residual and the layer's Top-K limit. Fill its empty buffer with authorized, graph-local concept IDs and finite scores in recall order. Returning an empty buffer ends the decomposition. Map recall failures to a `VcpError` at the host boundary.

Pass the returned `ObservationState` to candidate calibration and River readout. River scores complete paths and uses one shared topology budget of at most 0.08. `CrossFileWormholeOperator` takes independent root facts and combines their evidence inside that budget. Use an RRF operator when a caller explicitly requests a fused ranking. RRF updates ranking scores while retaining the readout quality scores.

Enable `ObservationConfig::sense.max_transition_records` when the graph feeds Thinking. Thinking requires complete retained transitions and a host-provided list of stages. Each selected candidate uses evidence from a single candidate path. The graph validator checks that the upstream node records transitions and the operator checks their completeness for the actual query.

For explicit memory reuse in a hot loop, keep a `SenseWorkspace` and a `SenseOutput` with the caller and invoke `SpikeSenseOperator::execute_into`. Candidate calibration also has an `execute_into` entry for retaining its output buffers. Each worker should own its execution workspace.

## Share the observation through a graph

Build a `ComputeGraph` with an `ObserveNode`. Connect its output to `PoolCalibrateNode` and to port 0 of `RiverReadoutNode`. Connect the calibrated pool to port 1 of River. Connect the same observation to `ObservedDtscReadoutNode` and `ThinkingNode`. Select result ports before calling `GraphValidator::validate` and reuse its execution plan across queries. Each call to `GraphExecutor::execute` takes a fresh borrowed `ExecutionEnv` and the caller's workspace. The shared observation is released after its last consumer unless requested as an output.

For a DTSC-only consumer, enable `math-nodes` with default features off and use `DtscFieldBuildNode`, `DtscFieldSampleNode` or `DtscReadoutNode` with host-provided DTSC inputs.

Wire the calibrated pool to port 1 of both `ObservedDtscReadoutNode` and `ThinkingNode`. Set `PoolCalibrateConfig::precision` to `Some(StrictLeftFold)` to share the exact query cosines between these readouts and River. DTSC computes its own strict query cosines when the shared pool uses fast SIMD.
