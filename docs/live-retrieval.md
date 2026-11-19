# Live vector retrieval

The example embeds ten passages, concept tags and seven queries through an OpenAI-compatible endpoint. Nexus prepares the geometry and graph, recalls through the warped Chunk and Tag fields, and ranks with RiverMemo. The example borrows the same assets for independent DTSC and Thinking calls and for one typed combination sharing the observation.

Supply the gateway's `/v1` endpoint and embedding model explicitly. Pass an authorized key in `NEXUS_EMBEDDING_KEY`. The generated vectors can be kept in the local `temp/` directory and replayed without calling the model again.

```sh
python scripts/live-retrieval.py temp/live-retrieval/embeddings.json --endpoint "$EMBEDDING_ENDPOINT" --model "$EMBEDDING_MODEL"
cargo run -p nexus --example live_retrieval --locked -- temp/live-retrieval/embeddings.json
```

The output shows the expected rank, score, topology contribution, observation size and number of grounded routes. Use the [frozen corpus benchmark](benchmarks.md) for larger host-style inputs.
