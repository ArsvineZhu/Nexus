# Nexus

Nexus is an in-memory Rust library for vector geometry, graph propagation and retrieval readouts. Prepare data once, then run a preset or call the individual operators with the same assets.

```rust
use nexus::{AssetConfig, Candidates, Dataset, Nexus, Pipeline, Query, RiverPipelineConfig};

fn retrieve(data: Dataset, vector: &[f32]) -> nexus::core::VcpResult<()> {
    let nexus = Nexus::prepare(data, &AssetConfig::default())?;
    let result = nexus.run(
        &Query::new(vector, "query text"),
        &Candidates::recall(20),
        Pipeline::river(),
        &RiverPipelineConfig::default(),
    )?;
    for item in &result.scores.items {
        println!("{} {:.4}", result.candidate_ids[item.candidate.index()], item.quality_score);
    }
    Ok(())
}
```

Use `Pipeline::tag_memo()` for enhanced-query cosine ranking or `Pipeline::dtsc()` for curve-based scoring. `observe`, `fields`, `river`, `dtsc` and `thinking` expose the intermediate steps. `Plan` composes typed operator outputs within one request.

- [Algorithm map](INDEX.md) links mathematical definitions and public entry points.
- [Borrowed assets and custom composition](docs/nous-integration.md) covers asset preparation and host integration.
- [Live vector retrieval](docs/live-retrieval.md) runs all of the above against model-generated embeddings.
- [Numerical sources](docs/upstream.md) identifies upstream implementations and their licenses.

`nexus` and `nexus-vcp` are CC BY-NC-SA 4.0. `nexus-math` and `nexus-core` are Apache 2.0. Depend on `nexus-math` directly for DTSC and typed composition without the VCP dependency.

Run `just check` for the Rust workspace checks and `just math-only` for the standalone DTSC suite.
