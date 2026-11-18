// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::*;
use serde_json::json;

fn capture(
    case: &str,
    is_wormhole: bool,
    independent_bridge: bool,
    scale: f64,
    weak: bool,
) -> Value {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(
        "CREATE TABLE rivermemo_artifacts(
             artifact_sig TEXT PRIMARY KEY, schema_version TEXT, algorithm_version TEXT,
             payload_codec TEXT, payload_checksum TEXT, payload BLOB, status TEXT);
         CREATE TABLE kbm_artifact_dependencies(
             artifact_sig TEXT, dependency_key TEXT, dependency_version TEXT,
             PRIMARY KEY(artifact_sig,dependency_key));
         CREATE TABLE kbm_runtime_meta(key TEXT PRIMARY KEY,value TEXT);
         INSERT INTO rivermemo_artifacts VALUES(
             'fixture','rivermemo-persisted-artifact-v2','memo.native-artifact-v2-binary',
             'cached','checksum',X'00','ready');
         INSERT INTO kbm_artifact_dependencies VALUES
             ('fixture','memo_tag_facts','7'),('fixture','derived_geometry','11');
         INSERT INTO kbm_runtime_meta VALUES ('memo_tag_facts','7'),('derived_geometry','11');
         CREATE TABLE tags(id INTEGER PRIMARY KEY,name TEXT,vector BLOB);
         CREATE TABLE chunks(id INTEGER PRIMARY KEY,file_id INTEGER,vector BLOB);
         CREATE TABLE files(id INTEGER PRIMARY KEY,checksum TEXT);
         CREATE TABLE file_tags(file_id INTEGER,tag_id INTEGER,position INTEGER);
         INSERT INTO files VALUES (101,'root-a'), (102,'root-b'), (201,'candidate-a'),
                                  (202,'candidate-b'), (203,'candidate-c');",
    )
    .unwrap();
    let tags = [
        [1.0_f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.1],
        [0.0, 0.8, 0.6],
    ];
    let candidates = [
        ([0.1_f32, 0.0, 0.99], vec![3]),
        ([0.42, 0.2, 0.88], vec![1, 3]),
        ([0.0, 0.99, 0.1], vec![2, 5]),
    ];
    let bytes = |row: &[f32]| row.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    for (index, vector) in tags.iter().enumerate() {
        let id = index as i64 + 1;
        db.execute(
            "INSERT INTO tags VALUES (?1,?2,?3)",
            rusqlite::params![id, format!("tag-{id}"), bytes(vector)],
        )
        .unwrap();
    }
    for (index, (vector, path)) in candidates.iter().enumerate() {
        let id = index as i64 + 1;
        let file = 200 + id;
        db.execute(
            "INSERT INTO chunks VALUES (?1,?2,?3)",
            rusqlite::params![id, file, bytes(vector)],
        )
        .unwrap();
        for (position, tag) in path.iter().enumerate() {
            db.execute(
                "INSERT INTO file_tags VALUES (?1,?2,?3)",
                rusqlite::params![file, tag, position as i64],
            )
            .unwrap();
        }
    }
    let witness_rows = if independent_bridge {
        [vec![1, 2, 3, 4], vec![1, 3, 4]]
    } else {
        [vec![1, 2, 3], vec![1, 3]]
    };
    for (index, path) in witness_rows.iter().enumerate() {
        for (position, tag) in path.iter().enumerate() {
            db.execute(
                "INSERT INTO file_tags VALUES (?1,?2,?3)",
                rusqlite::params![101 + index as i64, tag, position as i64],
            )
            .unwrap();
        }
    }

    let runtime = MemoRuntime::new();
    let has_provenance = case != "unattributed_transport";
    let artifact = NativeArtifact {
        node_ids: (1..=5).collect(),
        node_index: (1..=5).enumerate().map(|(index, id)| (id, index)).collect(),
        row_offsets: vec![0, 1, 1, 1, 1, 1],
        targets: vec![2],
        weights: vec![0.8],
        inbound: HashMap::from([(3, 0.8)]),
        max_inbound: 0.8,
        provenance: if has_provenance {
            HashMap::from([((1, 3), vec![(101, 1.0)])])
        } else {
            HashMap::new()
        },
        ..NativeArtifact::default()
    }
    .prepared();
    runtime.publish("fixture", Arc::new(artifact)).unwrap();
    let flow = if weak { 0.01 } else { 0.8 };
    let input = json!({
        "dimension":3, "includeTrace":true, "query":{"vector":[0.8,0.6,0.0]},
        "localVector":[0.0,0.0,1.0], "transferVector":[0.0,0.0,1.0], "denoisedVector":[0.8,0.6,0.0],
        "candidates":[{"id":1,"score":0.91},{"id":2,"score":0.83},{"id":3,"score":0.67}],
        "calibration":{"exactFocusId":1, "differentiaVector":[0.0,0.6,0.8],
            "supersededIds":[5], "differentiaConfidence":0.8, "explainedRatio":0.9},
        "queryState":{
            "completeObservation":true,
            "sourceField":[[1,0.75],[2,0.25]],
            "localField":[[1,0.5*scale],[3,0.8*scale]],
            "transferField":[[1,0.35*scale],[3,0.65*scale]],
            "localDomainIds":[1,3], "transferDomainIds":[1,3],
            "riverNodes":[{"id":1,"hop":0,"energy":1.0,"normalizedEnergy":1.0},
                {"id":2,"hop":0,"energy":0.3,"normalizedEnergy":0.3},
                {"id":3,"hop":1,"energy":0.5,"normalizedEnergy":0.5}],
            "riverEdges":[{"sourceId":1,"targetId":3,"flow":flow,"normalizedFlow":flow,"wormhole":is_wormhole}],
            "fieldProvenance":[{"id":1,"hop":0,"sourceType":"core"},{"id":2,"hop":0,"sourceType":"lexical_seed"}]
        }
    });
    let output: Value = serde_json::from_str(
        &run_native_with_connection(&runtime, &db, "fixture", &input.to_string()).unwrap(),
    )
    .unwrap();
    let results: Vec<_> = output["results"].as_array().unwrap().iter().map(|row| {
        json!({"id":row["id"],"score":row["score"],"base_score":row["baseScore"],
            "path_bonus":row["pathBonus"],"graph_bonus":row["graphBonus"],"anchor_bonus":row["anchorBonus"],
            "genus_gate":row["topologyV3"]["genusGate"],
            "relative_topology":row["relativeTopology"], "geometry":row["geometry"],
            "wormhole_unified":row["topologyV3"]["wormholeUnified"]})
    }).collect();
    json!({"case":case, "input":input, "tags":tags, "candidates":candidates, "has_provenance":has_provenance,
        "witness_tags":witness_rows, "results":results, "omega":output["omega"]["omega"]})
}

#[test]
fn nexus_golden_capture() {
    let river = [
        capture("ordinary_focus", false, false, 1.0, false),
        capture("rescaled_fields", false, false, 0.001, false),
        capture("below_active_edge_floor", false, false, 1.0, true),
    ];
    let wormhole = [
        capture("independent_bridge", true, true, 1.0, false),
        capture("missing_bridge", true, false, 1.0, false),
        capture("unattributed_transport", true, false, 1.0, false),
    ];
    println!("NEXUS_GOLDEN:{}", json!(["river", river]));
    println!("NEXUS_GOLDEN:{}", json!(["wormhole", wormhole]));
}
