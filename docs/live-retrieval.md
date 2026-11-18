# Live vector retrieval

The live verification embeds ten passages, their concept tags and seven queries through a host-supplied OpenAI-compatible endpoint. The Rust example constructs geometry, facts, transport and provenance from those vectors. Each query runs dense cosine retrieval, residual tag recall, observation and River scoring. It reports the dense winner, River winner, topology contribution and expected rank.

Provide an authorized model API key through `NEXUS_EMBEDDING_KEY`. Pass the gateway's `/v1` endpoint and embedding model name explicitly. The script writes the returned vectors into ignored local storage when the destination is under `temp/`.

```sh
python scripts/live-retrieval.py temp/live-retrieval/embeddings.json --endpoint "$EMBEDDING_ENDPOINT" --model "$EMBEDDING_MODEL"
cargo run -p nexus-graph --example live_retrieval --locked -- temp/live-retrieval/embeddings.json
```

The queries have fixed expected passages. This example tests that the model output can drive the full retrieval path and checks expected top results on the sample. Use the [frozen corpus benchmark](benchmarks.md) to study larger inputs.
