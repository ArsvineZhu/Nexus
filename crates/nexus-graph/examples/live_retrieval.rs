// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use nexus_core::kernel::cosine_f32;
use nexus_core::*;
use nexus_graph::nexus_vcp::*;
use serde::Deserialize;
use std::error::Error;
use std::marker::PhantomData;

#[derive(Deserialize)]
struct Tag {
    name: String,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct Document {
    title: String,
    tags: Vec<NodeId>,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct Query {
    text: String,
    expected_document: usize,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct Fixture {
    model: String,
    dimension: usize,
    tags: Vec<Tag>,
    documents: Vec<Document>,
    queries: Vec<Query>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("expected an embedding fixture path")?;
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    if fixture.documents.is_empty() || fixture.tags.is_empty() || fixture.queries.is_empty() {
        return Err("retrieval needs documents, tags and queries".into());
    }
    let ctx = ExecutionContext::default();
    let mut scratch = ScratchPad::new();
    let dimension = fixture.dimension;
    let tag_values: Vec<_> = fixture
        .tags
        .iter()
        .flat_map(|tag| tag.vector.iter().copied())
        .collect();
    let tags = VectorMatrixView::new(&tag_values, fixture.tags.len(), dimension)?;
    let names: Vec<_> = fixture.tags.iter().map(|tag| tag.name.as_str()).collect();
    let alias_offsets = vec![0; tags.rows + 1];
    let lexical = TagLexicalMetaView {
        names: &names,
        aliases: &[],
        alias_offsets: &alias_offsets,
        node_count: tags.rows,
    };
    let count = fixture.documents.len();
    let documents: Vec<_> = (0..count as u32).map(DocumentId).collect();
    let roots: Vec<_> = (0..count as u32).map(RootId).collect();
    let mut path_tags = Vec::new();
    let mut positions = Vec::new();
    let mut offsets = vec![0];
    for document in &fixture.documents {
        path_tags.extend_from_slice(&document.tags);
        positions.extend(1..=document.tags.len() as u32);
        offsets.push(path_tags.len() as u32);
    }
    let sequences = SequenceSliceView {
        sequence_count: count,
        seq_offsets: &offsets,
        seq_documents: &documents,
        seq_source_roots: &roots,
        tag_ids: &path_tags,
        positions: &positions,
    };
    let geometry = LocalConeAgsOperator.execute(
        &ctx,
        &LocalConeAgsConfig {
            min_neighbors: 2,
            ..LocalConeAgsConfig::default()
        },
        LocalConeAgsInput {
            tag_vectors: tags,
            sequences,
            explicit_edges: None,
        },
        &mut scratch,
    )?;
    let basis = GlobalEpaOperator.execute(
        &ctx,
        &GlobalEpaConfig::default(),
        GlobalEpaInput {
            tag_vectors: tags,
            geometry: &geometry,
        },
        &mut scratch,
    )?;
    let facts = FactMatrixOperator.execute(
        &ctx,
        &FactMatrixConfig::default(),
        FactMatrixInput {
            tag_vectors: tags,
            sequences,
            anchor_gains: &geometry.anchor_gains,
            positive_relations: None,
            negative_relations: None,
        },
        &mut scratch,
    )?;
    let transport = TransportCalibrateOperator.execute(
        &ctx,
        &TransportConfig::default(),
        TransportCalibrateInput {
            fact_matrix: &facts.fact_matrix,
            anchor_gains: &geometry.anchor_gains,
            edge_gates: None,
            _marker: PhantomData,
        },
        &mut scratch,
    )?;
    let graph = transport.as_csr_view();
    let provenance = ProvenanceBuildOperator.execute(
        &ctx,
        &ProvenanceConfig::default(),
        ProvenanceBuildInput {
            transport_row_offsets: graph.row_offsets,
            transport_targets: graph.targets,
            sequences,
            positive_relations: None,
            support_groups: None,
        },
        &mut scratch,
    )?;

    let values: Vec<_> = fixture
        .documents
        .iter()
        .flat_map(|doc| doc.vector.iter().copied())
        .collect();
    let candidate_vectors = VectorMatrixView::new(&values, count, dimension)?;
    let rows: Vec<_> = (0..=count as u32).collect();
    let zeros = vec![0.0; count];
    println!(
        "{} · {} dimensions · {count} documents · {} transport edges",
        fixture.model,
        dimension,
        graph.nnz()
    );
    let mut hits = 0;
    for query in &fixture.queries {
        let recall = |_, residual: &[f32], k, result: &mut Vec<_>| {
            result.extend(
                tags.data
                    .chunks_exact(dimension)
                    .enumerate()
                    .map(|(index, tag)| {
                        (
                            NodeId(index as u32),
                            cosine_f32(residual, tag, ctx.precision),
                        )
                    }),
            );
            result.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            result.truncate(k);
            Ok(())
        };
        let state = observe(
            &ctx,
            &ObservationConfig::default(),
            ObservationInput {
                query: &query.vector,
                query_text: &query.text,
                candidates: &recall,
                graph,
                tag_vectors: tags,
                lexical_meta: lexical,
                epa_basis: Some(basis.as_view()),
                core_tags: &[],
                ghost_vectors: None,
                ghost_is_core: &[],
            },
            &mut scratch,
        )?;
        let scores: Vec<_> = candidate_vectors
            .data
            .chunks_exact(dimension)
            .map(|row| cosine_f32(&query.vector, row, ctx.precision))
            .collect();
        let mut ordered: Vec<_> = (0..count).collect();
        ordered.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let mut ranks = vec![0; count];
        for (i, &document) in ordered.iter().enumerate() {
            ranks[document] = i as u32 + 1;
        }
        let candidates = CandidatePoolView {
            candidate_count: count,
            baseline_scores: &scores,
            union_scores: &scores,
            union_ranks: &ranks,
            bm25_scores: &zeros,
            bm25_reference: 0.0,
            anchor_scores: &zeros,
            curve_document_ids: &documents,
            candidate_vectors,
            candidate_root_offsets: &rows,
            candidate_roots: &roots,
            candidate_path_offsets: &rows,
            path_tag_offsets: &offsets,
            path_tags: &path_tags,
            path_positions: &positions,
        };
        let pool = PoolCalibrateOperator.execute(
            &ctx,
            &PoolCalibrateConfig::default(),
            PoolCalibrateInput {
                observation: &state,
                candidates,
            },
            &mut scratch,
        )?;
        let output = RiverReadoutOperator.execute(
            &ctx,
            &RiverReadoutConfig::default(),
            RiverReadoutInput {
                observation: &state,
                pool: &pool,
                candidates,
                graph,
                provenance: provenance.as_view(),
                support_groups: SupportGroupView::empty(),
                tag_vectors: tags,
            },
            &mut scratch,
        )?;
        let top = output
            .items
            .first()
            .ok_or("readout returned no candidates")?;
        let expected = query.expected_document;
        let rank = output
            .items
            .iter()
            .position(|item| item.candidate.index() == expected)
            .ok_or("expected document missing from readout")?
            + 1;
        if rank == 1 {
            hits += 1;
        }
        println!(
            "{} | dense: {} | River: {} ({:.4}, topology {:.4}) | expected rank: {} | observed nodes: {}",
            query.text,
            fixture.documents[ordered[0]].title,
            fixture.documents[top.candidate.index()].title,
            top.quality_score,
            top.omega_contribution,
            rank,
            state.sense.nodes.len()
        );
    }
    println!("Top-1: {hits}/{}", fixture.queries.len());
    if hits < fixture.queries.len() {
        return Err("at least one expected document did not rank first".into());
    }
    Ok(())
}
