# Nexus

Nexus is an in-memory Rust library for vector geometry, graph propagation and retrieval. Prepare assets once, then use your own recall provider or build the optional indexes. Pass business IDs and receive results with those IDs attached.

~~~rust
use nexus::{AssetConfig, CandidateInput, Dataset, ExecutionContext, Nexus, Query, RiverQueryConfig, TagRecall};

fn retrieve(
    data: Dataset,
    vector: &[f32],
    ids: &[u64],
    recall: &dyn TagRecall,
) -> nexus::core::VcpResult<nexus::RiverReadout> {
    let ctx = ExecutionContext::default();
    let nexus = Nexus::prepare(data, &AssetConfig::default(), &ctx)?;
    let query = Query::new(vector, "query text").with_tag_recall(recall);
    nexus.rank_river(&query, CandidateInput::Ids(ids), &RiverQueryConfig::default(), ctx)
}
~~~

For multiple readouts, build one candidate set and evaluate it against an observation. TagMemo, River, DTSC and Thinking share the query similarities. Use individual operators for custom numerical work, and use `Plan` to connect owned results when you need scheduling.

- [Host integration](docs/host-integration.md) covers asset import, borrowed candidates, recall and reuse.
- [Algorithm map](INDEX.md) links each operation to its implementation and numerical contract.
- [Live vector retrieval](docs/live-retrieval.md) runs on model-generated embeddings.
- [Numerical sources](docs/upstream.md) identifies the upstream algorithms and their licenses.

Run the self-contained host example with `cargo run -p nexus --example public_host --locked`. Run `just check` for the workspace checks.

`nexus` and `nexus-vcp` are CC BY-NC-SA 4.0. `nexus-math` and `nexus-core` are Apache 2.0. Depend on `nexus-math` directly for DTSC and RRF without the VCP dependency.
