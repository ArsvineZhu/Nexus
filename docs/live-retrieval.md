# Live vector retrieval

The Python host embeds the demo passages, concepts, and queries through an OpenAI-compatible gateway. The Rust example loads those vectors, builds the relation model, observes each query, and sends its available directions through independent cosine recall. It compares raw recall, expanded candidate pools, and several readouts.

Set the gateway's `/v1` endpoint and model, and supply its token in `NEXUS_EMBEDDING_KEY`.

~~~sh
python scripts/live-retrieval.py temp/live-vectors --endpoint "$EMBEDDING_ENDPOINT" --model "$EMBEDDING_MODEL"
cargo run -p nexus --example retrieval_study --release --locked -- temp/live-vectors
~~~

The exporter supports `--dimensions` for models that accept a requested dimension. Each run uses one embedding space for concepts, candidates, and queries. The exported directory can be replayed without another gateway call. For a larger labeled study, see [retrieval evaluation](benchmarks.md).
