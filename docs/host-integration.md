# Calling Nexus from a host

## Prepare assets once

Use `Nexus::prepare(data, &config, &ctx)` with raw concept and document vectors, ordered paths, and source facts. List each source referenced by a path or relation in `Dataset::sources`. Assign unique candidate IDs and set `document_id` on fragments from the same document so Thinking can deduplicate them. Nexus remembers the caller's concept, candidate and source IDs.

Raw vectors must be finite, nonzero and in the same embedding space. Preparation normalizes them. For existing immutable assets, use `Nexus::from_prepared` with unit vector matrices, explicit concept and source IDs, and a `TransportGraphView`. Construct the checked transport view once from CSR, and use `with_edge_gates` to bind effective request weights. Every effective row must have total mass at most one.

## Use host recall or build an index

Nexus attaches the recall provider to the query. At each residual layer the provider gets the current residual, a limit and excluded concept IDs. Fill the supplied buffer with business concept IDs in recall order. An empty buffer ends decomposition. The callback may use the execution context for cancellation and returns its errors directly.

~~~rust
let query = Query::new(vector, text).with_tag_recall(&recall);
let result = nexus.rank_river(
    &query, CandidateInput::Ids(ids), &RiverQueryConfig::default(), ctx,
)?;
for item in result.items {
    println!("{} {}", item.id, item.score);
}
~~~

With your own recall provider, no index preparation is necessary. For built-in recall, call `nexus.prepare_recall`, attach the returned object to the query, then use `observation.recall(&indexes, &RecallRequest::top_k(limit))` for documents. `RecallConfig` can enable concept and document indexes independently. `RecallRequest.allowed` restricts document recall to business IDs.

## Share a candidate set and observation

Use `CandidateInput::Ids` for catalog IDs, `Owned` for new rows with raw vectors and `Borrowed` for immutable unit vectors. `BorrowedCandidates::new` needs only IDs and a matrix. Add document IDs and paths when you need deduplication and structural evidence. Candidate row order breaks exact scoring ties; equal positions within a path retain their input order.

~~~rust
let rows = BorrowedCandidates::new(ids, unit_matrix).with_paths(path_rows);
let candidates = nexus.candidates(CandidateInput::Borrowed(rows))?;
let observation = nexus.observe(&query, &ObservationConfig::default(), ctx)?;
let evaluation = observation.evaluate(&candidates)?;
let tag_memo = evaluation.tag_memo()?;

let fields = observation.fields(&FieldConfig::default())?;
let river = evaluation.prepare_river(&fields)?;
let result = river.readout(&RiverConfig::default())?;

let config = DtscConfig::default();
let field = observation.dtsc_field(&config.field)?;
let evidence = evaluation.dtsc_evidence(&field, &config.scoring.evidence)?;
let dtsc = evidence.readout(&config.scoring.readout)?;
let thinking = evaluation.thinking(&stages, &ThinkingConfig::default())?;
~~~

All work after `observe` uses its execution context. Thinking stages name candidate IDs and use the observation's complete propagation trace. A disabled or truncated trace leaves Thinking unavailable. A dual field that exceeds its iteration budget returns `NotConverged` before basin projection.

| Change | Preparation to repeat |
| --- | --- |
| River scoring or evidence policy | Call `PreparedRiver::readout` again |
| River field settings | Build fields and prepare River again |
| DTSC trust, fallback policy or top-k | Call `DtscEvidence::readout` again |
| DTSC path or auxiliary settings | Build evidence again |
| Candidate order or candidate set | Evaluate the new set. Reuse observation fields |
| Query text, core concepts or observation config with the same normalized direction | Observe again and use `evaluate_reusing` to share original cosines |

The candidate set has no query scores. Evaluation binds it to an observation, and computes shared original and enhanced cosine columns. Prepared River adds two field cosine columns. The readouts derive their results from their own bound inputs and return business IDs, including local path indices and Thinking transitions.

## Compose independent operations

Call an `Operator` directly, then borrow its output in the next operation. For example, pass a context aggregate's `effective_vector` directly to EPA. A `Plan` uses typed handles and runs closures while their dependencies are borrowed. A closure can capture a prepared evaluation and return its owned Thinking result. To combine River with an external ordered list, pass the two lists of IDs and their weights to `math::RrfOperator`. RRF returns IDs from their union and leaves the River evidence available in the original result.

The [public host example](../crates/nexus/examples/public_host.rs) uses catalog and borrowed candidates, shared readouts and custom fusion.
