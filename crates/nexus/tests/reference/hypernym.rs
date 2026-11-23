// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::*;
use serde_json::json;

#[test]
fn nexus_golden_capture() {
    let cases = [
        (
            "redundant_parent",
            vec![0.8, 0.6],
            ["华为", "华为汽车"],
            [vec![1.0, 0.0], vec![0.8, 0.6]],
            0.0,
            false,
        ),
        (
            "explicit_parent",
            vec![0.8, 0.6],
            ["华为", "华为汽车"],
            [vec![1.0, 0.0], vec![0.8, 0.6]],
            0.0,
            true,
        ),
        (
            "independent_parent",
            vec![0.8, 0.6],
            ["car", "car assembly"],
            [vec![1.0, 0.0], vec![0.8, 0.6]],
            0.2,
            false,
        ),
        (
            "tight_alias",
            vec![1.0, 0.0],
            ["Lunarium", "Lunarium星之旅"],
            [vec![1.0, 0.0], vec![1.0, 0.0]],
            0.0,
            false,
        ),
        (
            "weak_direction",
            vec![1.0, 0.03],
            ["car", "car assembly"],
            [vec![1.0, 0.0], vec![0.8, 0.6]],
            0.0,
            false,
        ),
        (
            "low_parent_similarity",
            vec![0.0, 1.0],
            ["car", "car assembly"],
            [vec![1.0, 0.0], vec![0.8, 0.6]],
            0.0,
            false,
        ),
    ];
    let results: Vec<_> = cases.into_iter().map(|(case, query, names, vectors, contribution, explicit)| {
        let level: Vec<_> = names.iter().zip(&vectors).enumerate().map(|(index, (&name, vector))| {
            PyramidTagWork { id: index as i64 + 1, name: name.into(), vector: vector.clone().into(),
                similarity: cosine(&query, vector), contribution: if index == 0 { contribution } else { 1.0 },
                handshake_magnitude: 0.0 }
        }).collect();
        let names_to_keep = if explicit { HashSet::from([names[0].into()]) } else { HashSet::new() };
        let lexical_parent = superseded_hypernym(&level[0], &level, 0.1);
        let (projection, parents, confidence) = differentia_direction(
            &query, &[level], &PipelineConfig::default(), &names_to_keep);
        json!({"case":case, "query":query, "names":names, "vectors":vectors, "contribution":contribution,
            "explicit":explicit, "lexical_parent":lexical_parent, "projection":projection,
            "parents":parents, "confidence":confidence})
    }).collect();
    println!("NEXUS_GOLDEN:{}", json!(["hypernym", results]));
}
