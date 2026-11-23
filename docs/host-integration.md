# Host integration

## Build the model and give candidates their own identity

`Model::build` takes a `RelationData` with a positive dimension, concept vectors, source records and optional positive relations. Use one embedding space for concepts, queries and candidates. Nexus checks dimensions, finite values and nonzero vectors at the input boundary, then prepares the unit directions and graph.

Use stable `u64` IDs. A source ID identifies one canonical body of evidence. Copies and extra candidate passages can refer to the same source. Declare an origin with `Source::independent`, `Source::derived`, or `Source::unknown_origin`. Use a derived source for known ancestry. Unknown sources still supply ordinary paths and relations. Confirmed roots determine independent support and cross-source witnesses. Keep the original concept order when providing paths.

Use `Candidate::new` for a vector and ID, `with_document_id` for document grouping and `with_sources` for full source paths. Specify `with_paths` when a candidate covers a particular path or fragment:

~~~rust
use nexus::{Candidate, PathRef};

let fragment = Candidate::new(501, vec![0.9, 0.2, 0.1]).with_paths(vec![
    PathRef { source: 100, path: 0, positions: Some(vec![1, 2]) },
]);
~~~

Path numbers and positions are zero-based in the original source. A source path `[10, 20, 10, 40]` with positions `[1, 2]` yields `[20, 10]`; the selected path returns the same positions. `positions: None` selects the whole path. `with_paths(vec![])` keeps any declared source ownership and selects no paths.

`Model::candidates` binds owned candidate data once. Use `Model::candidate_view` with `CandidateView::new(math::UnitVectors::new(...)? )` when the host already owns a row-major matrix of unit vectors. Optional `CandidateMetadata` supplies sources and document ID for specific rows. `Candidates::select(ids)` keeps the requested ID order and shares the original vectors and resolved evidence. A cloned model can use the same bound candidates.

## Observe before host recall

The observation is ready to use across an asynchronous host call. It owns the query results. The library supplies `Original`, `Enhanced`, and whichever of `LocalField` and `TransferField` have a direction.

~~~rust
use nexus::{Candidates, Model, Query, Ranking, RecallKind};

fn rank_with_host(
    model: &Model,
    catalog: &Candidates<'_>,
    query: &[f32],
    recall: impl FnOnce(&[f32]) -> nexus::Result<Vec<u64>>,
) -> nexus::Result<Ranking> {
    let observation = model.observe(Query::new(query), &Default::default())?;
    let ids = recall(observation.vector(RecallKind::Enhanced).expect("enhanced query"))?;
    let candidates = catalog.select(ids)?;
    let evaluation = observation.evaluate(&candidates, &Default::default())?;
    Ok(evaluation.river(&Default::default())?.top(Some(10)))
}
~~~

To use several recall lanes, iterate `observation.recall_vectors()` and combine their ID lists with `math::merge_ranked_ids`. An independent `math::WarpedIndex` can search document directions supplied by an observation. Its hits carry distances ordered nearest first. `math::cosine_scores` provides exact cosine without building an index.

`Query::with_text` supplies the text for deterministic lexical matching against labels and aliases. Explicit core IDs participate even if concept search misses them. Vector anchors affect query fusion and need no graph identity. `with_edge_gates` takes concept pairs for this observation, with factors in `[0, 1]`. Duplicate factors combine by minimum. A positive edge gated to zero carries no flow; the other outgoing weights keep their mass.

Default Pyramid performs internal numeric concept search. `ObserveOptions::concept_search` allows a synchronous index to replace only that search. Its request contains the current residual, level, limit and excluded concept IDs. Return a unique set of permitted model IDs. Nexus fits those concepts using the original vectors. Use `ObserveConfig::pyramid = None` to disable the residual decomposition.

## Reuse one evaluation and choose a score

`Evaluation::cosine` and `tag_memo` return original and enhanced cosine. River reports a nonnegative score with separate baseline, path and graph contributions and the complete winning path. The topology budget defaults to `0.08`. Cross-source witnesses identify the positive wormhole, both sources, and the common third concept.

DTSC applies geometric correction to a baseline you choose. On a trusted field the result is `baseline + correction_gain * geo_bonus`, with gain `1` by default. A low-trust decision preserves your baseline values and ordering. Use `evaluation.baseline([(id, value), ...])` to import finite scores that cover every candidate exactly once. This is also how you use a score from another evaluation.

~~~rust
let river = evaluation.river(&Default::default())?;
let enhanced = evaluation.tag_memo();
let curves = evaluation.dtsc_evidence(&Default::default())?;
let on_river = curves.apply(&river, &Default::default())?;
let on_enhanced = curves.apply(&enhanced, &Default::default())?;
~~~

For River, `evaluation.river_evidence` computes every path once. `RiverEvidence::readout` can then apply several topology budgets or Ω exponents and select the winning path each time. `operators::bridge_witnesses` exposes qualified source pairs before candidate scoring.

`Scores` retains every candidate in the input order. `top(Some(k))` produces an owned ranking, `top(None)` takes all scores, and `top(Some(0))` returns an empty ranking. Equal scores follow the baseline's stable order. Use `math::reciprocal_rank_fusion` to combine rank positions from algorithms with different score scales.

`Evaluation::thinking` takes ordered `ThinkingStage` values with candidate IDs and per-stage limits. It returns selected paths, parents, and retained routes. Already selected documents are skipped throughout the sequence. Candidates without a document ID are identified by their own candidate IDs.

## Configure the computation that owns each parameter

- `BuildConfig` sets local and global geometry, fact weights, the transport graph and Warped concept-index geometry.
- `ObserveConfig` sets EPA, Pyramid, seed selection, Sense, fusion, and both fields. A finite beam or iteration count gives a usable approximation. Field solutions report the returned iterate's residual and L1 error bound.
- `RiverConfig`, DTSC field/evidence/readout configuration, and `ThinkingConfig` affect their corresponding readout. Change DTSC's geometry options to rebuild evidence. Change trust or gain to read the same evidence again.
- `RunOptions` controls per-call cancellation and parallelism. Sequential is the default. Use `Parallelism::Rayon` inside your own `rayon::ThreadPool::install` to bound worker count.

All configuration structs expose their numerical parameters and support serde. The optional `schemars` feature generates configuration schemas. Consult the [algorithm map](../INDEX.md) and the [runnable consumer](../examples/public-client/) for individual operators and complete usage.
