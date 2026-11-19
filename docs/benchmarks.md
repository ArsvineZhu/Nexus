# Frozen Nous benchmark

Export a frozen study with Python, NumPy and MessagePack installed. Pass the component directory containing `assets.msgpack` and `math.msgpack`, the study directory containing `fixed-dense-groups.jsonl` and `query-vectors/`, and a local destination.

```sh
python scripts/export-nous-benchmark.py "$COMPONENT" "$STUDY" temp/frozen-study
NEXUS_FROZEN_FIXTURE=temp/frozen-study cargo bench -p nexus --bench frozen --locked
```

On PowerShell, set `$env:NEXUS_FROZEN_FIXTURE = (Resolve-Path temp/frozen-study).Path` before running the benchmark.

The benchmark borrows the frozen graph, candidates and provenance. Its host callback replays the Tag hits captured from residual queries before timing. The public River preset covers the entire query set. A custom plan shares the candidate batch and observation across River, DTSC and Thinking for queries with a grounded route. Each prints its query count and latency percentiles. Set `NEXUS_BENCH_THREADS` to supply a bounded Rayon pool.

For CPU sampling, set `CARGO_PROFILE_BENCH_DEBUG=2` and build with `cargo bench -p nexus --bench frozen --no-run --locked`. Run the resulting executable under the profiler with `--bench --profile-time 30 shared_river_dtsc_thinking` and the same fixture and worker count.
