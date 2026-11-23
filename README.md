# Nexus

Nexus is an in-memory Rust library for vector geometry, graph propagation, and retrieval. Supply concept vectors, ordered source facts, a query, and candidate vectors. Nexus derives the graph, query directions, and ranking evidence. Results use your candidate, concept, and source IDs.

### Rank candidates you already have

~~~rust
use nexus::{Candidate, Concept, Model, Query, RelationData, Source};

fn main() -> nexus::Result<()> {
    let model = Model::build(
        RelationData {
            dimension: 2,
            concepts: vec![
                Concept::new(10, vec![1.0, 0.0]).with_label("cache"),
                Concept::new(20, vec![0.8, 0.6]).with_label("index"),
            ],
            sources: vec![Source::independent(100, vec![vec![10, 20]])],
            relations: vec![],
        },
        &Default::default(),
    )?;

    let candidates = model.candidates(vec![
        Candidate::new(101, vec![0.9, 0.6]).with_sources(vec![100]),
        Candidate::new(102, vec![0.0, 1.0]),
    ])?;

    let ranking = model.rank_river(
        Query::new(&[1.0, 0.2]).with_text("cache"),
        &candidates,
        &Default::default(),
    )?;
    for row in ranking.rows() {
        println!("{} {}", row.id, row.value);
    }
    Ok(())
}
~~~

For host recall, call **observe → search your index → select candidates → evaluate**. One observation exposes the original, enhanced and available field vectors. One evaluation supports cosine, TagMemo, River, DTSC and Thinking. Use the ordinary Rust functions in `nexus::math`, `nexus::operators` and `nexus::context` to compose individual computations.

The [host guide](docs/host-integration.md) explains input ownership, score semantics and how to reuse evidence. The [algorithm map](INDEX.md) and [numerical sources](docs/upstream.md) document the computation.

### Run the examples

~~~sh
cargo run --manifest-path examples/public-client/Cargo.toml --bin numeric --locked
cargo run --manifest-path examples/public-client/Cargo.toml --bin direct_candidates --locked
cargo run --manifest-path examples/public-client/Cargo.toml --bin host_recall --locked
cargo run --manifest-path examples/public-client/Cargo.toml --bin custom_readouts --locked
~~~

The examples depend only on `nexus`. The workspace uses Rust 1.100.0. Run `just check` for lint, tests, and formatting.

`nexus` distributes the VCP-derived implementation under [CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt) with file-level notices for code under other terms. Independent cosine, NNLS, DTSC, transport, deduplication and RRF are available under [Apache 2.0](LICENSES/Apache-2.0.txt) in `nexus-math`.
