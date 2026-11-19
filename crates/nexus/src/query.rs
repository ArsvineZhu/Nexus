// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::dataset::{invalid, slot};
use nexus_core::{ExecutionContext, NodeId, VcpResult, VectorMatrixView};
use nexus_vcp::EpaAnalysis;
use std::cell::RefCell;
use std::collections::HashSet;

pub struct TagRecallRequest<'call> {
    pub layer: usize,
    pub residual: &'call [f32],
    pub limit: usize,
    pub epa: Option<&'call EpaAnalysis>,
    pub excluded_concepts: &'call [u64],
}

pub trait TagRecall {
    fn recall(
        &self,
        ctx: &ExecutionContext<'_>,
        request: TagRecallRequest<'_>,
        out: &mut Vec<u64>,
    ) -> VcpResult<()>;
}

impl<F> TagRecall for F
where
    F: Fn(&ExecutionContext<'_>, TagRecallRequest<'_>, &mut Vec<u64>) -> VcpResult<()>,
{
    fn recall(
        &self,
        ctx: &ExecutionContext<'_>,
        request: TagRecallRequest<'_>,
        out: &mut Vec<u64>,
    ) -> VcpResult<()> {
        self(ctx, request, out)
    }
}

#[derive(Clone, Copy)]
pub struct Query<'a> {
    pub vector: &'a [f32],
    pub text: &'a str,
    pub core_concepts: &'a [u64],
    pub ghost_vectors: Option<VectorMatrixView<'a>>,
    pub ghost_is_core: &'a [bool],
    pub tag_recall: Option<&'a dyn TagRecall>,
}

impl<'a> Query<'a> {
    pub fn new(vector: &'a [f32], text: &'a str) -> Self {
        Self {
            vector,
            text,
            core_concepts: &[],
            ghost_vectors: None,
            ghost_is_core: &[],
            tag_recall: None,
        }
    }

    pub fn with_tag_recall(mut self, recall: &'a dyn TagRecall) -> Self {
        self.tag_recall = Some(recall);
        self
    }
}

#[derive(Default)]
struct RecallBuffers {
    hits: Vec<u64>,
    excluded_ids: Vec<u64>,
    seen: HashSet<u64>,
}

pub(crate) struct PyramidRecall<'a, 'data> {
    nexus: &'a crate::Nexus<'data>,
    ctx: &'a ExecutionContext<'data>,
    recall: &'a dyn TagRecall,
    buffers: RefCell<RecallBuffers>,
}

impl<'a, 'data> PyramidRecall<'a, 'data> {
    pub(crate) fn new(
        nexus: &'a crate::Nexus<'data>,
        ctx: &'a ExecutionContext<'data>,
        recall: &'a dyn TagRecall,
    ) -> Self {
        Self {
            nexus,
            ctx,
            recall,
            buffers: RefCell::default(),
        }
    }

    pub(crate) fn recall(
        &self,
        layer: usize,
        residual: &[f32],
        limit: usize,
        out: &mut Vec<NodeId>,
        epa: Option<&EpaAnalysis>,
        excluded: &[NodeId],
    ) -> VcpResult<()> {
        const STAGE: &str = "query.tag_recall";
        self.ctx.check_cancelled(STAGE)?;
        let ids = self.nexus.prepared().concept_ids;
        let mut buffers = self.buffers.borrow_mut();
        let RecallBuffers {
            hits,
            excluded_ids,
            seen,
        } = &mut *buffers;
        hits.clear();
        excluded_ids.clear();
        excluded_ids.extend(excluded.iter().map(|id| ids[id.index()]));
        self.recall.recall(
            self.ctx,
            TagRecallRequest {
                layer,
                residual,
                limit,
                epa,
                excluded_concepts: excluded_ids,
            },
            hits,
        )?;
        self.ctx.check_cancelled(STAGE)?;
        if hits.len() > limit {
            return Err(invalid(
                STAGE,
                format!("host returned {} concepts for limit {limit}", hits.len()),
            ));
        }
        seen.clear();
        seen.extend(excluded_ids.iter().copied());
        for &id in hits.iter() {
            if !seen.insert(id) {
                return Err(invalid(
                    STAGE,
                    format!("concept {id} is repeated or excluded"),
                ));
            }
            let index = slot(&self.nexus.concept_slots, id)
                .map_err(|_| invalid(STAGE, format!("unknown concept {id}")))?;
            out.push(NodeId::from_index(index, STAGE)?);
        }
        Ok(())
    }
}
