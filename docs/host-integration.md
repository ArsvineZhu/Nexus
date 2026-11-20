# Calling Nexus from a host

## Prepare assets once

Use `Nexus::prepare(data, &config, &ctx)` with raw concept and document vectors, ordered paths and source facts. Pass vectors from one embedding model and space, and keep the Nexus across queries. Raw vectors must be finite and nonzero. Preparation normalizes them. The host supplies embeddings, document text and any authorization or metadata filters.

Assign each candidate a unique `Document::id`. Fragments can share a `document_id` so Thinking groups them by document. List each source referenced by a path or relation in `Dataset::sources`. A ranked `PathRef` identifies the path by candidate ID and its zero-based position in that candidate's input. It includes the business source ID when attribution is available. Position ties among the steps of a path preserve input order.

Import immutable unit vectors and a checked base transport graph with `Nexus::from_prepared`. The borrowed `nexus.prepared()` view carries optional tag residuals for a concept index and optional per-path source roots. Keep the backing arrays alive for the lifetime of the borrowed Nexus. Stable business IDs can map directly to original files, passages or application records.

## Built-in recall and ranking

Build `nexus.prepare_recall(&RecallConfig::default(), &ctx)` once. The concept index supplies Pyramid recall and the document index selects candidates for ranking. Set the corresponding configuration field to `None` when only one index is needed.

~~~rust
let observation = indexes.observe(&Query::new(vector, text), &Default::default(), &ctx)?;
let request = RecallRequest {
    allowed: Some(permitted_candidate_ids),
    ..RecallRequest::top_k(100)
};
let candidates = observation.recall(&indexes, &request, &ctx)?;
let evaluation = observation.evaluate(&candidates, &ctx)?;
let river = evaluation.river(&Default::default(), &ctx)?;
let dtsc = evaluation.dtsc(&Default::default(), &ctx)?;
~~~

Document recall uses the original direction and returns up to the requested number of candidates. To run multiple recall lanes, call `indexes.search_documents(vector, &request, &ctx)` with each lane's vector. For example, use the raw embedding in one lane and `observation.values().fused.enhanced_query` in another. Results are ordered by increasing warped distance and carry candidate and document IDs. Combine the ordered IDs with `math::ranking::rrf` or your own selection policy, then prepare one `CandidateInput::Ids` from the combined order.

~~~rust
use nexus::math::{WeightedRanking, rrf};

let id_order = [
    indexes.search_documents(vector, &request, &ctx)?,
    indexes.search_documents(&observation.values().fused.enhanced_query, &request, &ctx)?,
].map(|hits| hits.into_iter().map(|hit| hit.id).collect::<Vec<_>>());
let lanes = id_order.each_ref().map(|ids| WeightedRanking { ids, weight: 1.0 });
let fused = rrf(&lanes, &Default::default(), &ctx)?;
let ids: Vec<_> = fused.items.iter().take(request.limit).map(|hit| hit.id).collect();
let candidates = nexus.candidates(CandidateInput::Ids(&ids))?;
~~~

To add field lanes, compute `observation.fields` before putting together the candidate set. Its basin output provides optional `local_direction` and `transfer_direction` vectors. Search with the directions that are present, then pass the same fields to `evaluation.river_evidence` after combining the lanes.

## Host recall and supplied candidates

A host provider fills each Pyramid layer with ordered concept IDs. The callback receives the current residual with its magnitude, the excluded IDs and the requested limit. Return distinct known IDs and skip excluded IDs. An empty output ends decomposition. The default observation enables Pyramid and uses either the prepared concept index or the host provider. When using core concepts alone, call `Query::with_core_concepts` and set `ObservationConfig::pyramid` to `None`.

~~~rust
let query = Query::new(vector, text).with_tag_recall(&recall);
let config = RiverQueryConfig::default();
let result = nexus.rank_river(&query, CandidateInput::Ids(ids), &config, &ctx)?;
for item in result.items {
    println!("{} {}", item.id, item.scores.score);
}
~~~

Use `rank_dtsc` or `rank_tag_memo` for the corresponding one-call readout. Use `CandidateInput::Ids` for catalog IDs, `Owned` for `Document` rows with raw vectors, or `Borrowed` for a row-major matrix of unit vectors. The supplied order breaks exact scoring ties. When the host provides candidates, pass only rows allowed in the current query. For imported assets using host concept recall, the tag residual column is optional.

Use `query.with_edge_gates(gates)` for request-specific attenuation. Gates follow the base CSR edge order and range from zero to one. The observation materializes the effective weights once and keeps the same edge slots and provenance. A static candidate set can be reused across requests with different gates.

## Share observations and score fixed evidence

Call `evaluation.tag_memo`, `evaluation.river`, `evaluation.dtsc` and `evaluation.thinking` on one evaluation to share candidate similarities. Thinking takes ordered named stages with candidate IDs from that evaluation. The staged form below keeps River and DTSC evidence for multiple scoring passes.

~~~rust
use nexus::{BorrowedCandidates, CandidateInput, RiverConfig};
use nexus::math::DtscConfig;
use nexus::vcp::{ObservationConfig, RiverScoringConfig, ThinkingConfig};

let rows = BorrowedCandidates::new(ids, unit_matrix).with_paths(path_rows);
let candidates = nexus.candidates(CandidateInput::Borrowed(rows))?;
let observation = nexus.observe(&query, &ObservationConfig::default(), &ctx)?;
let evaluation = observation.evaluate(&candidates, &ctx)?;

let config = RiverConfig::default();
let fields = observation.fields(&config.fields, &ctx)?;
let river = evaluation.river_evidence(&fields, &config.analysis, &ctx)?;
let standard = river.readout(&config.scoring, &ctx)?;
let retuned = river.readout(
    &RiverScoringConfig { unified_budget_ct: 0.04, ..config.scoring },
    &ctx,
)?;

let dtsc_config = DtscConfig::default();
let field = observation.dtsc_field(&dtsc_config.field, &ctx)?;
let evidence = evaluation.dtsc_evidence(&field, &dtsc_config.scoring.evidence, &ctx)?;
let dtsc = evidence.readout(&dtsc_config.scoring.readout, &ctx)?;
let thinking = evaluation.thinking(&stages, &ThinkingConfig::default(), &ctx)?;
~~~

An evaluation computes the original and enhanced candidate similarities once. River and DTSC evidence own their scores and business identity mapping. You can release the observation, fields and candidates, then read out the evidence with a fresh execution context.

| Change | Work to repeat |
| --- | --- |
| River budget or omega exponent | Read out the same evidence. Each readout picks the best complete path under its own settings |
| River evidence policy, path thresholds or cross-file configuration | Rebuild River evidence |
| River field configuration | Solve fields and rebuild River evidence |
| DTSC geometry weight, fallback order, pool trust or top-k | Read out the same evidence |
| DTSC path scoring or auxiliary policy | Rebuild DTSC evidence |
| Candidate set or order | Evaluate the new set. Reuse the observation and fields |
| Query, core concepts or edge gates | Observe and evaluate again |

River defaults to a topology budget of 0.02. Set `config.scoring.unified_budget_ct` to `nexus::vcp::VCP_TOPOLOGY_BUDGET` to use VCP's 0.08. The DTSC setting `dtsc_config.scoring.readout.geometry_weight` defaults to 0.15 of the prepared reward. When evidence is insufficient, candidates are ordered by original cosine before taking `top_k`. Set `dtsc_config.scoring.readout.fallback_order` to `nexus::math::DtscFallbackOrder::InputOrder` when candidate order already encodes your host's baseline ranking. `DtscDecision::Fallback` reports the reason and the order used.

Propagation completeness is reported by every ordinary readout. River also reports projection limits. Thinking uses complete propagation and a complete recorded trace. A finite field iteration budget can return `NotConverged` before projection.

Set `ExecutionContext::cancel` from the host's request token to cancel observations, search and readouts. Operations return `core::VcpError` for invalid input, cancellation and algorithm failures. Run synchronous queries on the host's blocking executor when integrating with an async service. Use `ParallelPolicy::CustomPool` when the service owns a bounded Rayon pool.

## Compose individual algorithms

Use each function's typed input and borrow the results you need. For example, pass the `effective_vector` from `vcp::context::aggregate` to `vcp::observation::analyze_epa`. For ordered lists of business IDs, `math::ranking::rrf` returns rank scores over the union. `vcp::river::wormhole_rrf` uses a River baseline and chosen paths to compute rank scores from the baseline and supported routes.

The [public host example](../crates/nexus/examples/public_host.rs) uses host recall, catalog and borrowed candidates, shared evidence, Thinking and independent fusion.
