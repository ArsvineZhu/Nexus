# Frozen Nous benchmark

Export the frozen study with Python, NumPy and MessagePack installed. Pass the frozen component directory containing `assets.msgpack` and `math.msgpack`, the study directory containing `fixed-dense-groups.jsonl` and `query-vectors/`, and a local destination.

```sh
python scripts/export-nous-benchmark.py "$COMPONENT" "$STUDY" temp/frozen-study
NEXUS_FROZEN_FIXTURE=temp/frozen-study cargo bench -p nexus-graph --bench frozen --locked
```

```powershell
python scripts/export-nous-benchmark.py $component $study temp/frozen-study
$env:NEXUS_FROZEN_FIXTURE = (Resolve-Path temp/frozen-study).Path
cargo bench -p nexus-graph --bench frozen --locked
```

The exporter checks cached query digests and dimensions and preserves real candidate paths, tag vectors, edge weights and provenance. Keep the exported fixture local. The benchmark replays the result of exact, residual-dependent tag retrieval, computed before timing, so its latency covers Nexus operations and graph execution. It uses the complete candidate pool with dense host scores. The fan-out results use the subset of queries with a grounded Thinking route and print its size.

The benchmark runs sequentially by default. Set `NEXUS_BENCH_THREADS` to compare bounded Rayon pools. It prints query latency percentiles before running Criterion. Criterion writes its reports under `target/criterion/`.

For CPU profiling, build the benchmark with `CARGO_PROFILE_BENCH_DEBUG=line-tables-only` and `cargo bench -p nexus-graph --bench frozen --no-run --locked`. Profile the resulting executable with the same fixture and worker count. Pass `--bench --profile-time 30 shared_river_dtsc_thinking` for a steady window of fan-out execution. Compare CPU stacks during that window, then run the normal benchmark without the profiler to measure latency.
