// SPDX-License-Identifier: Apache-2.0

use nexus_core::{
    CandidatePoolView, CsrGraphView, EpaBasisView, NodeId, ProvenanceView, RootFactView,
    SupportGroupView, TagLexicalMetaView, ThinkingStageInputView, VectorMatrixView,
};

#[derive(Clone, Copy, Debug)]
pub struct AssetBundle<'asset> {
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub transport: Option<CsrGraphView<'asset>>,
    pub provenance: Option<ProvenanceView<'asset>>,
    pub support_groups: Option<SupportGroupView<'asset>>,
    pub root_facts: Option<RootFactView<'asset>>,
    pub epa_basis: Option<EpaBasisView<'asset>>,
    pub lexical_meta: Option<TagLexicalMetaView<'asset>>,
}

impl<'asset> AssetBundle<'asset> {
    pub fn new(tag_vectors: VectorMatrixView<'asset, f32>) -> Self {
        Self {
            tag_vectors,
            transport: None,
            provenance: None,
            support_groups: None,
            root_facts: None,
            epa_basis: None,
            lexical_meta: None,
        }
    }
}

/// Immutable host-owned data shared by every node of one graph invocation.
pub struct ExecutionEnv<'asset, 'query> {
    pub assets: AssetBundle<'asset>,
    pub query_vec: &'query [f32],
    pub query_text: &'query str,
    pub core_tags: &'query [NodeId],
    pub ghost_vectors: Option<VectorMatrixView<'query, f32>>,
    pub ghost_is_core: &'query [bool],
    pub candidates: Option<CandidatePoolView<'query>>,
    pub thinking_stages: &'query [ThinkingStageInputView<'query>],
    #[cfg(feature = "vcp-nodes")]
    pub pyramid_candidates: Option<&'query nexus_vcp::PyramidCandidates<'query>>,
    #[cfg(feature = "math-nodes")]
    pub dtsc_nodes: &'query [nexus_math::DtscNodeInput],
    #[cfg(feature = "math-nodes")]
    pub dtsc_query_ids: &'query [NodeId],
    #[cfg(feature = "math-nodes")]
    pub dtsc_query_vectors: Option<VectorMatrixView<'query, f32>>,
    #[cfg(feature = "math-nodes")]
    pub dtsc_enhanced_query: Option<&'query [f32]>,
    #[cfg(feature = "math-nodes")]
    pub dtsc_geometry: nexus_math::DtscGeometryState,
}

impl<'asset, 'query> ExecutionEnv<'asset, 'query> {
    pub fn new(assets: AssetBundle<'asset>, query_vec: &'query [f32]) -> Self {
        Self {
            assets,
            query_vec,
            query_text: "",
            core_tags: &[],
            ghost_vectors: None,
            ghost_is_core: &[],
            candidates: None,
            thinking_stages: &[],
            #[cfg(feature = "vcp-nodes")]
            pyramid_candidates: None,
            #[cfg(feature = "math-nodes")]
            dtsc_nodes: &[],
            #[cfg(feature = "math-nodes")]
            dtsc_query_ids: &[],
            #[cfg(feature = "math-nodes")]
            dtsc_query_vectors: None,
            #[cfg(feature = "math-nodes")]
            dtsc_enhanced_query: None,
            #[cfg(feature = "math-nodes")]
            dtsc_geometry: nexus_math::DtscGeometryState::default(),
        }
    }
}
