// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::{ConceptId, Result};

#[derive(Clone, Copy, Debug)]
pub struct Query<'a> {
    pub vector: &'a [f32],
    pub text: Option<&'a str>,
    pub core_concepts: &'a [ConceptId],
    pub anchors: &'a [VectorAnchor<'a>],
    pub edge_gates: &'a [EdgeGate],
}

impl<'a> Query<'a> {
    pub fn new(vector: &'a [f32]) -> Self {
        Self {
            vector,
            text: None,
            core_concepts: &[],
            anchors: &[],
            edge_gates: &[],
        }
    }

    pub fn with_text(mut self, text: &'a str) -> Self {
        self.text = Some(text);
        self
    }

    pub fn with_core_concepts(mut self, concepts: &'a [ConceptId]) -> Self {
        self.core_concepts = concepts;
        self
    }

    pub fn with_anchors(mut self, anchors: &'a [VectorAnchor<'a>]) -> Self {
        self.anchors = anchors;
        self
    }

    pub fn with_edge_gates(mut self, gates: &'a [EdgeGate]) -> Self {
        self.edge_gates = gates;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VectorAnchor<'a> {
    pub vector: &'a [f32],
    pub role: AnchorRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorRole {
    Core,
    Related,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeGate {
    pub from: ConceptId,
    pub to: ConceptId,
    pub factor: f64,
}

pub struct ConceptSearchRequest<'a> {
    pub residual: &'a [f32],
    pub level: usize,
    pub limit: usize,
    pub excluded: &'a [ConceptId],
}

pub trait ConceptSearch: Sync {
    fn search(&self, request: ConceptSearchRequest<'_>) -> Result<Vec<ConceptId>>;
}
