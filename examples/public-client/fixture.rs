// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use nexus::{Candidate, Candidates, Concept, Model, PathRef, RelationData, Source};

pub const QUERY: [f32; 4] = [0.6, 0.6, 0.53, 0.0];

pub fn model() -> nexus::Result<Model> {
    let concepts = [
        (10, "缓存", [1.0, 0.0, 0.0, 0.0]),
        (20, "索引", [0.8, 0.6, 0.0, 0.0]),
        (30, "增量更新", QUERY),
        (40, "迁移", [0.2, 0.7, 0.5, 0.47]),
        (50, "内存占用", [0.8, 0.0, 0.6, 0.0]),
        (60, "并发", [0.0, 0.8, 0.2, 0.57]),
        (70, "无关主题", [0.0, 0.0, 0.0, 1.0]),
    ]
    .into_iter()
    .map(|(id, label, vector)| Concept::new(id, vector.to_vec()).with_label(label))
    .collect();
    Model::build(
        RelationData {
            dimension: 4,
            concepts,
            sources: vec![
                Source::independent(100, vec![vec![10, 20, 30], vec![30, 20, 10]]),
                Source::independent(200, vec![vec![20, 30, 40]]),
                Source::independent(300, vec![vec![10, 50, 20]]),
                Source::independent(400, vec![vec![60, 20, 40]]),
                Source::derived(500, vec![100, 200], vec![vec![10, 20, 40]]),
            ],
            relations: Vec::new(),
        },
        &Default::default(),
    )
}

pub const ROWS: [(u64, [f32; 4]); 5] = [
    (1001, [0.8, 0.62, 0.38, 0.0]),
    (1002, [0.57, 0.62, 0.53, 0.0]),
    (1003, [0.3, 0.8, 0.5, 0.4]),
    (1004, [0.0, 0.3, 0.0, 1.0]),
    (1005, [0.8, 0.0, 0.55, 0.0]),
];

pub fn candidates(model: &Model) -> nexus::Result<Candidates<'static>> {
    let candidate = |row: usize| Candidate::new(ROWS[row].0, ROWS[row].1.to_vec());
    model.candidates(vec![
        candidate(0).with_sources(vec![100]).with_document_id(501),
        candidate(1)
            .with_paths(vec![PathRef {
                source: 100,
                path: 0,
                positions: Some(vec![1, 2]),
            }])
            .with_document_id(501),
        candidate(2).with_sources(vec![200]),
        candidate(3),
        candidate(4).with_sources(vec![300]),
    ])
}
