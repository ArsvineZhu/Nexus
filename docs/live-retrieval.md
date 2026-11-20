# Live vector retrieval

The example embeds ten passages, concept tags and seven queries through an OpenAI-compatible endpoint. Nexus prepares the math assets and optional warped Tag and Chunk indexes. Each query builds one observation, recalls an ordered candidate set, and evaluates that set once. River, DTSC and Thinking use the same evaluation.

Supply the gateway's `/v1` endpoint and embedding model explicitly. Pass an authorized key in `NEXUS_EMBEDDING_KEY`. The generated vectors can be kept in the local `temp/` directory and replayed without calling the model again.

```sh
python scripts/live-retrieval.py temp/live-retrieval/embeddings.json --endpoint "$EMBEDDING_ENDPOINT" --model "$EMBEDDING_MODEL"
cargo run -p nexus --example live_retrieval --locked -- temp/live-retrieval/embeddings.json
```

To compare embedding widths, run the exporter with `--dimensions` set to a width your model supports. Keep concepts, passages and queries in the same model and dimension for each run.

The output shows the expected rank, River score, DTSC result count and number of grounded routes. Use the [frozen corpus benchmark](benchmarks.md) for larger host-style inputs.
