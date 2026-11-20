# Nexus

Nexus is an in-memory Rust library for vector geometry, graph propagation and retrieval. Prepare a collection once, then use its built-in indexes or bring your own recall. Results carry the business IDs you supplied for candidates, documents, concepts and sources.

The workspace uses Rust 1.100.0.

~~~rust
use nexus::{Concept, Dataset, Document, ExecutionContext, Nexus, Query, RecallRequest};

fn main() -> nexus::core::VcpResult<()> {
    let ctx = ExecutionContext::default();
    let data = Dataset {
        concepts: vec![Concept::new(10, "battery", vec![1.0, 0.0])],
        documents: vec![Document::new(101, vec![1.0, 0.0])],
        ..Dataset::default()
    };
    let nexus = Nexus::prepare(data, &Default::default(), &ctx)?;
    let indexes = nexus.prepare_recall(&Default::default(), &ctx)?;

    let query = Query::new(&[1.0, 0.0], "battery");
    let observation = indexes.observe(&query, &Default::default(), &ctx)?;
    let candidates = observation.recall(&indexes, &RecallRequest::top_k(10), &ctx)?;
    let ranking = observation.evaluate(&candidates, &ctx)?.river(&Default::default(), &ctx)?;
    for item in ranking.items {
        println!("{} {}", item.id, item.scores.score);
    }
    Ok(())
}
~~~

Use one evaluation for TagMemo, River, DTSC and Thinking. Reuse River or DTSC evidence to vary scoring, or call the independent algorithms with their typed inputs.

- [Host integration](docs/host-integration.md) covers host-provided recall, multiple lanes, provenance and reusable evidence.
- [Algorithm map](INDEX.md) links each operation to its implementation and numerical contract.
- [Live vector retrieval](docs/live-retrieval.md) runs on model-generated embeddings.
- [Numerical sources](docs/upstream.md) identifies the upstream algorithms and their licenses.

Run the self-contained host example with `cargo run -p nexus --example public_host --locked`. Run `just check` for the workspace checks.

`nexus` and `nexus-vcp` are CC BY-NC-SA 4.0. `nexus-math` and `nexus-core` are Apache 2.0. Depend on `nexus-math` directly for DTSC and RRF without the VCP dependency.
