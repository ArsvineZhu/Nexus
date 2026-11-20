# Retrieval evaluation

## Frozen Nous throughput

Export a frozen study with Python, NumPy and MessagePack installed. Pass the component directory containing `assets.msgpack` and `math.msgpack`, the study directory containing `fixed-dense-groups.jsonl` and `query-vectors/`, and a local destination.

```sh
python scripts/export-nous-benchmark.py "$COMPONENT" "$STUDY" temp/frozen-study
NEXUS_FROZEN_FIXTURE=temp/frozen-study cargo bench -p nexus --bench frozen --locked
```

On PowerShell, set `$env:NEXUS_FROZEN_FIXTURE = (Resolve-Path temp/frozen-study).Path` before running the benchmark.

The benchmark imports the frozen graph and candidate source roots, prepares directions, and constructs the ordered candidate set once. The frozen snapshot supplies path positions and candidate-level roots. Its result paths are identified by candidate ID and path position. The host callback replays Tag hits captured from residual queries before timing. River runs across all queries. The shared run reads River, DTSC and Thinking from one evaluation for queries with a grounded route. Each prints query counts and latency percentiles. Set `NEXUS_BENCH_THREADS` to supply a bounded Rayon pool.

For CPU sampling, set `CARGO_PROFILE_BENCH_DEBUG=2` and build with `cargo bench -p nexus --bench frozen --no-run --locked`. Run the resulting executable under the profiler with `--bench --profile-time 30 shared_river_dtsc_thinking` and the same fixture and worker count.

A Windows CPU sample on the 1,961-candidate, 2,048-dimensional fixture attributed 33.7% of sampled time to the query similarities and 24.1% to the River field similarities. Both now pack two query directions once and accumulate their FP64 dot products together, in the original addition order for each direction. In a paired one-thread run, median River time went from 15.01 ms to 10.11 ms. Median time for the shared River, DTSC and Thinking workload went from 17.05 ms to 11.34 ms.

## Labeled quality

The public study used 300 SciFact claims, 240 FiQA questions and 180 Chinese T2Retrieval queries. Each pool contained the labeled relevant passages plus distractors. SciFact and FiQA included lexical hard candidates. Tag names and ordered paths came from the passages. A deterministic query split reserved two thirds of each corpus for evaluation. [SciFact](https://huggingface.co/datasets/mteb/scifact) · [FiQA](https://huggingface.co/datasets/mteb/fiqa) · [T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)

The table reports nDCG@8 on those held-out pools. River, DTSC and TagMemo used an exact-dense top-64 candidate pool. Warped recall used its own top 64. English vectors came from BGE small English at 384 dimensions, and Chinese vectors from BGE small Chinese at 512 dimensions.

| Held-out pool | Queries | Dense cosine | Warped recall | TagMemo | River before → current | DTSC before → current | Dense + River + DTSC RRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 200 | 0.8010 | 0.8034 | 0.7979 | 0.7776 → 0.7902 | 0.6723 → 0.7938 | 0.8010 |
| FiQA | 160 | 0.6141 | 0.6153 | 0.6048 | 0.6142 → 0.6091 | 0.4744 → 0.6097 | 0.6134 |
| T2Retrieval | 120 | 0.9683 | 0.9703 | 0.9627 | 0.9640 → 0.9656 | 0.9367 → 0.9684 | 0.9662 |

The measured defaults use a River topology budget of 0.02 and a DTSC geometry weight of 0.15. DTSC distinguishes inferred seeds from explicit lexical and core evidence. Its cosine fallback matched the dense baseline when Pyramid was disabled and the supplied catalog order carried no baseline ranking. In the frozen Nous study, both River budgets had identical ranking over 446 queries.

The frozen evaluation also checked known source families and, where present, exact document revisions. Across the 137 development queries, DTSC family nDCG@8 moved from 0.9114 to 0.9166 on dense top 64 and from 0.7588 to 0.9166 on catalog-ordered input. Its final ranking retrieved the exact revision in the top eight for 431 of 437 queries with exact labels across the whole study.

For a direct width comparison, the Docker gateway requested native 1,024- and 2,048-dimensional vectors from the same Doubao model for 32 SciFact queries, 65 passages and 48 concepts. Dense cosine, warped recall, River, DTSC and their tested RRF combination each reached nDCG@8 of 0.9644 and 0.9707, respectively. Both widths found a relevant first result for 30 of the 32 queries.

## Recall and reranking compositions

These comparisons reuse the same vectors, judged pools and held-out queries. The baseline runs exact cosine on original model vectors and keeps 64 passages. TagMemo, River and DTSC can reorder that pool. Separate Warped searches use the original, enhanced, local-field or transfer-field direction and can change which passages enter the pool.

| Pipeline, nDCG@8 | SciFact 200 | FiQA 160 | T2Retrieval 120 | Nous development 137 |
| --- | ---: | ---: | ---: | ---: |
| Exact cosine | 0.8010 | 0.6141 | 0.9683 | 0.9166 |
| Warped, original direction | 0.8034 | 0.6153 | 0.9703 | 0.9069 |
| Warped, enhanced direction | 0.8021 | 0.6039 | 0.9645 | 0.9090 |
| Warped, local field | 0.5285 | 0.1942 | 0.7332 | 0.4483 |
| Warped, transfer field | 0.4895 | 0.1166 | 0.6576 | 0.4349 |
| Dense pool → TagMemo | 0.7979 | 0.6047 | 0.9627 | 0.9086 |
| Dense pool → River | 0.7902 | 0.6091 | 0.9656 | 0.9168 |
| Dense pool → DTSC | 0.7938 | 0.6097 | 0.9684 | 0.9166 |
| Dense, River, DTSC RRF | 0.8010 | 0.6134 | 0.9662 | 0.9166 |
| Dense and both original/enhanced Warped results, RRF | 0.8033 | 0.6152 | 0.9661 | 0.9127 |
| Add both field searches to that RRF | 0.7138 | 0.4790 | 0.8803 | 0.8200 |
| Five-list RRF → River | 0.7902 | 0.6091 | 0.9656 | 0.9168 |
| Five-list RRF → DTSC | 0.7938 | 0.6097 | 0.9684 | 0.9166 |
| Five-list RRF combined with its River and DTSC results | 0.7446 | 0.5634 | 0.9283 | 0.8786 |
| Dense and BM25 RRF | 0.7748 | 0.5059 | 0.9320 | — |

The recall fusions used weights 1, 1, 0.5 for dense, original Warped and enhanced Warped, then 0.5 for each field where present. The result fusions used 1 for the base list and 0.5 each for River and DTSC. BM25 and dense had equal weights. RRF used its default rank constant of 60.

An expansion strategy kept the first 48 exact-cosine results in order, then filled up to 64 with unseen IDs from the five-list fusion. It preserved nDCG@8 and raised Recall@64 from 0.9030 to 0.9133 on FiQA and from 0.9773 to 0.9805 on Nous development. SciFact and T2Retrieval kept the same Recall@64. Both coverage comparisons count results from a 64-passage pool and deduplicate document IDs before measuring relevance. The two field directions contributed useful candidates in some pools and performed poorly as standalone rankings.
