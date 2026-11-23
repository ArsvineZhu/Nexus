# Retrieval evaluation

## Replaying a judged corpus

The scoring example builds Nexus assets from concept vectors, canonical sources, ordered paths, and positive relations. It runs host cosine recall over all candidate vectors, retains up to 64 candidates, then compares the available recall directions and readouts. The example uses the public API throughout.

For the frozen Nous study, use the component directory containing `assets.msgpack` and the study directory containing `fixed-dense-groups.jsonl`, `query-vectors/`, and its read-only snapshot database. The exporter needs Python, NumPy and MessagePack.

~~~sh
python scripts/export-nous-benchmark.py "$COMPONENT" "$STUDY" temp/computation-study
cargo run -p nexus --example retrieval_study --release --locked -- temp/computation-study
~~~

The replay format has a `fixture.json` containing a dimension, concept metadata, canonical sources, relations, candidate metadata and query judgments. The three vector matrices are row-major, little-endian f32 in `concepts.f32`, `candidates.f32` and `queries.f32`, in the corresponding metadata order. The exporter is also a working example of binding candidate references against declared source paths. [Exporter](../scripts/export-nous-benchmark.py) · [Rust replay](../crates/nexus/examples/retrieval_study.rs)

## Replayed model vectors

The comparison ran on 720 public-corpus queries and 446 frozen Nous queries. It reused cached model embeddings and relevance judgments. English used BGE small at 384 dimensions, Chinese T2Retrieval used BGE small at 512, and Nous used Doubao at 2,048. The public pools contain labeled passages and distractors from the earlier study. The Nous component supplies four ordered source paths and 14 positive relations. Eight of its 31 annotated candidate paths resolve to those declared source paths. The other candidates retain their vectors and declared sources. Source independence is unknown in the replay. [SciFact](https://huggingface.co/datasets/mteb/scifact) · [FiQA](https://huggingface.co/datasets/mteb/fiqa) · [T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)

### Ranking on the held-out queries

nDCG@8 uses document IDs after candidate deduplication by document. River, DTSC, TagMemo and RRF in the *original pool* rows work on the identical 64-candidate pool. DTSC applies the default full correction to the TagMemo baseline. RRF equally weights the original, River and DTSC rankings.

| Strategy | SciFact (200) | FiQA (160) | T2Retrieval (120) | Nous (137) |
| --- | ---: | ---: | ---: | ---: |
| Original cosine | 0.8010 | 0.6141 | 0.9683 | 0.9166 |
| Enhanced cosine recall | 0.7956 | 0.6120 | 0.9634 | 0.8846 |
| Original pool → TagMemo | 0.7956 | 0.6120 | 0.9634 | 0.8936 |
| Original pool → River | 0.7777 | 0.6169 | 0.9653 | 0.9166 |
| Original pool → DTSC on TagMemo | 0.6591 | 0.4767 | 0.9465 | 0.8931 |
| Original pool → RRF | 0.7843 | 0.5981 | 0.9636 | 0.9100 |
| Dense 48 + two fields, then River | 0.7777 | 0.6169 | 0.9653 | 0.9166 |

### Which candidates enter the pool

The field expansion keeps the first 48 original-cosine hits. It then interleaves local and transfer searches, using up to 32 hits from either direction, and takes distinct IDs up to 64. Unused slots are filled with original hits. Metrics count the relevant documents covered by those 64 passages.

| Strategy, Recall@64 | SciFact | FiQA | T2Retrieval | Nous |
| --- | ---: | ---: | ---: | ---: |
| Original cosine | 0.9750 | 0.9030 | 0.9967 | 0.9773 |
| Enhanced cosine | 0.9750 | 0.9039 | 0.9967 | 0.9700 |
| Dense 48 + two fields | 0.9750 | 0.8891 | 0.9967 | 0.9586 |

River improved FiQA ranking slightly and tied original cosine on Nous. Full-gain DTSC on the enhanced baseline lowered held-out nDCG in all four corpora. The tested field expansion preserved the protected top eight and reduced pool recall on FiQA and Nous. These results use the specified River budget `0.08` and DTSC gain `1.0`.

## Candidate computation reuse

Release build, Rust 1.100.0, one thread on an Intel Core i9-12900H. Each paired measurement runs River, DTSC and Thinking twelve times against one fixed observation and 64 candidates, alternating the run order. The shared run builds one `Evaluation`; the separate run builds three. The runner compares the complete results, including selected paths and Thinking routes.

| Corpus | Dimensions | Model concepts | Sample nodes / edges / retained transitions | Candidate paths | Shared / separate mean |
| --- | ---: | ---: | --- | ---: | ---: |
| Nous | 2,048 | 50 | 4 / 2 / 2 | 0 | 0.252 / 0.631 ms |
| SciFact | 384 | 700 | 10 / 1 / 1 | 64 | 1.689 / 1.932 ms |
| FiQA | 384 | 700 | 17 / 10 / 10 | 64 | 2.009 / 2.327 ms |
| T2Retrieval | 512 | 700 | 12 / 3 / 3 | 63 | 1.760 / 2.023 ms |

Mean observe time across each full corpus was 1.527 ms, 1.374 ms, 1.790 ms, and 1.205 ms respectively, in table order. The corresponding mean shared evaluation and three readouts took 0.363 ms, 2.095 ms, 2.303 ms, and 1.675 ms. These timings exclude embedding and host candidate recall.
