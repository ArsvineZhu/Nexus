// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use nexus_core::kernel::{cosine_f32, normalize_direction};
use nexus_core::*;
use nexus_vcp::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/vcp_5adb12b.json")).unwrap()
}

fn rows<'a>(fixture: &'a Value, section: &str) -> &'a [Value] {
    fixture[section].as_array().unwrap()
}

fn decode<T: DeserializeOwned>(value: &Value) -> T {
    serde_json::from_value(value.clone()).unwrap()
}

fn close(case: &Value, what: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{} {what}: Nexus {actual:.16}, upstream {expected:.16}",
        case["case"]
    );
}

fn matching_values(case: &Value, what: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{}", case["case"]);
    for (&actual, &expected) in actual.iter().zip(expected) {
        close(case, what, actual, expected, tolerance);
    }
}

#[test]
fn upstream_nnls_projection_and_cone_geometry() {
    let source = fixture();
    assert_eq!(
        source["revision"],
        "5adb12bb55e616e22bd0e0dbfdc13032c069941a"
    );
    let ctx = ExecutionContext::default();

    for case in rows(&source, "nnls") {
        let query: Vec<f64> = decode(&case["query"]);
        let ids: Vec<i64> = decode(&case["ids"]);
        let vectors: Vec<Vec<f64>> = decode(&case["vectors"]);
        let columns: Vec<_> = ids
            .into_iter()
            .zip(vectors.iter().map(Vec::as_slice))
            .collect();
        let fit = nexus_core::nnls::solve_nnls_with_ids_f64(
            &query,
            &columns,
            columns.len(),
            &ctx.cancel,
            &mut NnlsWorkspace::default(),
        )
        .unwrap();
        matching_values(
            case,
            "projection",
            &fit.projection,
            &decode::<Vec<_>>(&case["projection"]),
            2e-7,
        );
        matching_values(
            case,
            "residual",
            &fit.residual,
            &decode::<Vec<_>>(&case["residual"]),
            2e-7,
        );
        matching_values(
            case,
            "unit coefficients",
            &fit.direction_weights,
            &decode::<Vec<_>>(&case["weights"]),
            1e-10,
        );
        close(
            case,
            "residual ratio",
            fit.residual_ratio,
            decode(&case["residual_ratio"]),
            1e-10,
        );
    }

    for case in rows(&source, "geometry") {
        let vectors: Vec<Vec<f32>> = decode(&case["vectors"]);
        let dim = vectors[0].len();
        let flat: Vec<_> = vectors.iter().flatten().copied().collect();
        let weights: Vec<f64> = decode(&case["weights"]);
        let edges: Vec<_> = weights
            .into_iter()
            .enumerate()
            .map(|(index, weight)| (NodeId(0), NodeId(index as u32 + 1), weight))
            .collect();
        let geometry = local_cone(
            LocalConeAgsInput {
                tag_vectors: VectorMatrixView::new(&flat, vectors.len(), dim).unwrap(),
                sequences: SequenceSliceView::empty(),
                explicit_edges: Some(&edges),
            },
            &LocalConeAgsConfig {
                min_neighbors: 1,
                semantic_gate_enabled: false,
                ..LocalConeAgsConfig::default()
            },
            &ctx,
        )
        .unwrap();
        let target = &geometry.intrinsics[0];
        assert_eq!(
            target.basis_used,
            decode::<usize>(&case["basis_used"]),
            "{}",
            case["case"]
        );
        close(
            case,
            "cone residual",
            target.residual_ratio,
            decode(&case["residual_ratio"]),
            1e-10,
        );
    }
}

#[test]
fn upstream_hypernym_and_dedup_contracts() {
    let source = fixture();
    let ctx = ExecutionContext::default();

    for case in rows(&source, "hypernym") {
        let raw_query: Vec<f32> = decode(&case["query"]);
        let query = normalize_direction(&raw_query, raw_query.len(), "test").unwrap();
        let names: Vec<String> = decode(&case["names"]);
        let names: Vec<_> = names.iter().map(String::as_str).collect();
        let vectors: Vec<Vec<f32>> = decode(&case["vectors"]);
        let flat: Vec<_> = vectors.iter().flatten().copied().collect();
        let dim = query.len();
        let level = PyramidLevel {
            level: 0,
            tags: vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| PyramidTagEntry {
                    node: NodeId(index as u32),
                    similarity: cosine_f32(&query, vector, PrecisionMode::FastSimd),
                    contribution: if index == 0 {
                        decode(&case["contribution"])
                    } else {
                        1.0
                    },
                    handshake_magnitude: 0.0,
                })
                .collect(),
            projection_magnitude: 0.0,
            residual_magnitude: 0.0,
            residual_energy_ratio: 0.0,
            energy_explained: 0.0,
            handshake: HandshakeFeatures::default(),
        };
        let pyramid = PyramidResidual {
            levels: vec![level],
            ..PyramidResidual::default()
        };
        let cores: &[NodeId] = if decode(&case["explicit"]) {
            &[NodeId(0)]
        } else {
            &[]
        };
        let config = HypernymConfig::default();
        let result = select_seeds(
            SeedSelectionInput {
                epa: &EpaAnalysis::default(),
                query_text: "",
                query: &query,
                pyramid: &pyramid,
                tag_vectors: VectorMatrixView::new(&flat, vectors.len(), dim).unwrap(),
                lexical_meta: TagLexicalMetaView {
                    node_count: vectors.len(),
                    names: TextRows::Borrowed(&names),
                    alias_offsets: &[0, 0, 0],
                    aliases: TextRows::Borrowed(&[]),
                },
                core_tags: cores,
            },
            &SeedSelectionConfig {
                hypernym: Some(config.clone()),
                ..Default::default()
            },
            &ctx,
        )
        .unwrap()
        .disambiguated;
        let parents: Vec<_> = decode::<Vec<u32>>(&case["parents"])
            .into_iter()
            .map(|id| NodeId(id - 1))
            .collect();
        assert_eq!(result.pruned_hypernyms, parents, "{}", case["case"]);
        let projected: Vec<f64> = result
            .differentia_q_perp
            .unwrap_or_default()
            .into_iter()
            .map(f64::from)
            .collect();
        matching_values(
            case,
            "differentia",
            &projected,
            &decode::<Vec<_>>(&case["projection"]),
            1e-7,
        );
        let mut confidence = decode(&case["confidence"]);
        if case["case"] == "weak_direction" {
            // Removing the first axis leaves the second component of the unit direction.
            let confidence_after_projection = |vector: &[f32]| {
                let [x, y] = [f64::from(vector[0]), f64::from(vector[1])];
                y.abs() / (x.hypot(y) * config.differentia_confidence_scale)
            };
            close(
                case,
                "raw differentia confidence",
                confidence_after_projection(&raw_query),
                confidence,
                1e-12,
            );
            confidence = confidence_after_projection(&query);
        }
        close(
            case,
            "differentia confidence",
            result.differentia_confidence,
            confidence,
            1e-12,
        );
    }

    for case in rows(&source, "dedup") {
        let vectors: Vec<Vec<f32>> = decode(&case["vectors"]);
        let flat: Vec<_> = vectors.iter().flatten().copied().collect();
        let ranks: Vec<_> = (0..vectors.len() as u32).map(CandidateId).collect();
        let result = cosine_dedup(
            CosineDedupInput {
                ranked: &ranks,
                candidate_vectors: VectorMatrixView::new(&flat, vectors.len(), vectors[0].len())
                    .unwrap(),
            },
            &CosineDedupConfig {
                threshold: decode(&case["threshold"]),
                ..CosineDedupConfig::default()
            },
            &ctx,
        )
        .unwrap();
        let upstream: Vec<_> = decode::<Vec<u32>>(&case["survivors"])
            .into_iter()
            .map(|id| CandidateId(id - 1))
            .collect();
        if case["case"] == "distinct_query_order" {
            assert_eq!(upstream, [CandidateId(1), CandidateId(2)]);
            assert_eq!(result.survivors, [CandidateId(0), CandidateId(2)]);
            assert_eq!(result.folded[0].representative, CandidateId(0));
        } else {
            assert_eq!(result.survivors, upstream, "{}", case["case"]);
        }
    }
}

fn sparse_field(value: &Value, count: usize) -> Vec<f64> {
    let mut result = vec![0.0; count];
    for (id, mass) in decode::<Vec<(usize, f64)>>(value) {
        result[id - 1] = mass;
    }
    result
}

fn observation(input: &Value, count: usize) -> (ObservationState, FieldValues) {
    let state = &input["queryState"];
    let calibration = &input["calibration"];
    let nodes = state["riverNodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| {
            let id = decode::<u32>(&node["id"]);
            ObservationNode {
                node: NodeId(id - 1),
                energy: decode(&node["energy"]),

                hop: decode(&node["hop"]),
                source_kind: match id {
                    1 => SeedSourceKind::Core,
                    2 => SeedSourceKind::LexicalSeed,
                    _ => SeedSourceKind::Emergent,
                },
                seed_node: Some(NodeId(0)),
            }
        })
        .collect();
    let edges = state["riverEdges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| ObservationEdge {
            source: NodeId(decode::<u32>(&edge["sourceId"]) - 1),
            target: NodeId(decode::<u32>(&edge["targetId"]) - 1),
            flow: decode(&edge["flow"]),
            normalized_flow: decode(&edge["normalizedFlow"]),
            max_flow: decode(&edge["flow"]),
            conductance: 0.8,
            min_hop: 1,
            wormhole: decode(&edge["wormhole"]),
            immediate_return: false,
        })
        .collect();
    let ids = |value: &Value| {
        decode::<Vec<u32>>(value)
            .into_iter()
            .map(|id| NodeId(id - 1))
            .collect()
    };
    (
        ObservationState {
            original_query: decode(&input["query"]["vector"]),
            epa: EpaAnalysis::default(),
            pyramid: PyramidResidual {
                coverage: decode(&calibration["explainedRatio"]),
                ..PyramidResidual::default()
            },
            disambiguated: DisambiguatedTags {
                differentia_confidence: decode(&calibration["differentiaConfidence"]),
                differentia_q_perp: Some(decode(&calibration["differentiaVector"])),
                pruned_hypernyms: ids(&calibration["supersededIds"]),
            },
            gated_seeds: GatedSeeds {
                exact_focus: Some(NodeId(decode::<u32>(&calibration["exactFocusId"]) - 1)),
                ..GatedSeeds::default()
            },
            fused: FusedObservation {
                enhanced_query: decode(&input["denoisedVector"]),
                ..FusedObservation::default()
            },
            sense: SenseOutput {
                nodes,
                edges,
                ..SenseOutput::default()
            },
        },
        FieldValues {
            dual_fields: DualFields {
                local_converged: true,
                transfer_converged: true,
                local_domain: ids(&state["localDomainIds"]),
                transfer_domain: ids(&state["transferDomainIds"]),
                local_field: sparse_field(&state["localField"], count),
                transfer_field: sparse_field(&state["transferField"], count),
                ..DualFields::default()
            },
            basins: BasinSupports {
                local_direction: Some(decode(&input["localVector"])),
                transfer_direction: Some(decode(&input["transferVector"])),
                ..BasinSupports::default()
            },
        },
    )
}

#[test]
fn upstream_river_and_cross_file_wormhole_scores() {
    let source = fixture();
    let ctx = ExecutionContext::default();

    for (case, wormhole) in rows(&source, "river")
        .iter()
        .map(|v| (v, false))
        .chain(rows(&source, "wormhole").iter().map(|v| (v, true)))
    {
        let (state, fields) = observation(&case["input"], 5);
        let tag_values: Vec<_> = decode::<Vec<Vec<f32>>>(&case["tags"])
            .into_iter()
            .flatten()
            .collect();
        let tag_vectors = VectorMatrixView::new(&tag_values, 5, 3).unwrap();
        let bodies_and_paths: Vec<(Vec<f32>, Vec<u32>)> = decode(&case["candidates"]);
        let body_values: Vec<_> = bodies_and_paths
            .iter()
            .flat_map(|(body, _)| body.iter().copied())
            .collect();
        let tags: Vec<_> = bodies_and_paths
            .iter()
            .flat_map(|(_, path)| path.iter().map(|id| NodeId(*id - 1)))
            .collect();
        let positions: Vec<_> = bodies_and_paths
            .iter()
            .flat_map(|(_, path)| 0..path.len() as u32)
            .collect();
        let path_offsets: Vec<u32> = std::iter::once(0)
            .chain(bodies_and_paths.iter().scan(0, |offset, (_, path)| {
                *offset += path.len() as u32;
                Some(*offset)
            }))
            .collect();
        let wormhole_flags = [wormhole];
        let graph = TransportGraphView::new(
            CsrGraphView::new(CsrParts {
                node_count: 5,
                row_offsets: &[0, 1, 1, 1, 1, 1],
                targets: &[NodeId(2)],
                weights: &[0.8],
                is_static_wormhole: &wormhole_flags,
                inbound: &[0.0, 0.0, 0.8, 0.0, 0.0],
                max_inbound: 0.8,
                anchor_gains: &[1.0; 5],
                target_sorted_indices: None,
                edge_gates: None,
            })
            .unwrap(),
        )
        .unwrap();
        let candidates = CandidateRowsView {
            candidate_count: 3,
            document_ids: &[0, 1, 2],
            candidate_vectors: VectorMatrixView::new(&body_values, 3, 3).unwrap(),
            candidate_root_offsets: &[0; 4],
            candidate_roots: &[],
            candidate_path_offsets: &[0, 1, 2, 3],
            path_tag_offsets: &path_offsets,
            path_tags: &tags,
            path_positions: &positions,
        };
        let query = QueryCosines::prepare(
            &ctx,
            candidates,
            &state.original_query,
            &state.fused.enhanced_query,
        )
        .unwrap();
        let has_provenance: bool = decode(&case["has_provenance"]);
        let provenance = if has_provenance {
            ProvenanceView {
                edge_count: 1,
                edge_prov_offsets: &[0, 1],
                prov_groups: &[GroupId(0)],
                prov_masses: &[1.0],
            }
        } else {
            ProvenanceView {
                edge_count: 1,
                edge_prov_offsets: &[0, 0],
                prov_groups: &[],
                prov_masses: &[],
            }
        };
        let roots: Vec<Vec<u32>> = if wormhole {
            decode(&case["witness_tags"])
        } else {
            Vec::new()
        };
        let root_tags: Vec<_> = roots.iter().flatten().map(|id| NodeId(*id - 1)).collect();
        let offsets = [
            0,
            roots.first().map_or(0, |tags| tags.len() as u32),
            root_tags.len() as u32,
        ];
        let root_facts = wormhole.then_some(RootFactView {
            root_count: 2,
            checksum_hashes: &[1, 2],
            root_tag_offsets: &offsets,
            root_tags: &root_tags,
            root_req_offsets: &[0, 1, 2],
            root_required_roots: &[RootId(0), RootId(1)],
        });
        let result = prepare_river_evidence(
            RiverAnalysisInput {
                observation: &state,
                query: &query,
                fields: &fields,
                graph,
                provenance,
                support_groups: SupportGroupView::empty(),
                tag_vectors,
                root_facts,
            },
            &RiverAnalysisConfig {
                evidence: EvidencePolicy::DiscountWhenUnattributed,
                ..Default::default()
            },
            &ctx,
        )
        .unwrap()
        .readout(
            &RiverScoringConfig {
                unified_budget_ct: VCP_TOPOLOGY_BUDGET,
                ..Default::default()
            },
            &ctx,
        )
        .unwrap();
        close(
            case,
            "omega",
            result.omega.omega,
            decode(&case["omega"]),
            1e-12,
        );
        let expected_order: Vec<_> = rows(case, "results")
            .iter()
            .map(|row| CandidateId(decode::<u32>(&row["id"]) - 1))
            .collect();
        assert_eq!(
            result
                .items
                .iter()
                .map(|item| item.candidate)
                .collect::<Vec<_>>(),
            expected_order,
            "{}",
            case["case"]
        );
        for (item, expected) in result.items.iter().zip(rows(case, "results")) {
            let name = |field| format!("candidate {} {field}", item.candidate.0);
            close(
                case,
                &name("Genus Gate"),
                item.scores.genus_gate,
                decode(&expected["genus_gate"]),
                2e-7,
            );
            close(
                case,
                &name("semantic base"),
                item.scores.base_score,
                decode(&expected["base_score"]),
                2e-7,
            );
            close(
                case,
                &name("path bonus"),
                item.scores.path_bonus,
                decode(&expected["path_bonus"]),
                2e-7,
            );
            if !has_provenance && item.candidate == CandidateId(1) {
                // The upstream assumes complete independence when attribution
                // is absent. The discount policy uses its 0.15 floor.
                assert!(item.scores.graph_bonus < decode::<f64>(&expected["graph_bonus"]));
                close(
                    case,
                    &name("quality sum"),
                    item.scores.score,
                    item.scores.base_score + item.scores.path_bonus + item.scores.graph_bonus,
                    1e-12,
                );
            } else {
                close(
                    case,
                    &name("graph and transport bonus"),
                    item.scores.graph_bonus,
                    decode(&expected["graph_bonus"]),
                    2e-7,
                );
                close(
                    case,
                    &name("quality"),
                    item.scores.score,
                    decode(&expected["score"]),
                    2e-7,
                );
            }
        }
    }
}
