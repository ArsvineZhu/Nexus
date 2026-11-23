// SPDX-License-Identifier: CC-BY-NC-SA-4.0
mod geometry;
mod input;
mod provenance;
mod transport;

use std::sync::Arc;

pub use geometry::{
    GlobalCandidateMode, GlobalPriorMode, GlobalSpan, GlobalSpanConfig, GlobalSpanOptions,
    LocalConeConfig, LocalConeOptions, LocalGeometry, LocalResidual, global_span, local_cone,
};
pub use input::{Concept, ConceptData, ConceptView, Relation, RelationData, Source, SourceOrigin};
pub(crate) use provenance::SourceBook;
pub use transport::{
    FactsConfig, GraphAssets, GraphConfig, GraphOptions, TransportConfig, compile_graph,
};

use crate::warped::{ConceptIndex, WarpedBuildOptions, WarpedGeometryConfig};
use crate::{Result, RunOptions};
use nexus_math::TransportGraph;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BuildConfig {
    pub geometry: GeometryConfig,
    pub graph: GraphConfig,
    pub warped_tag_geometry: WarpedGeometryConfig,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeometryConfig {
    pub local: LocalConeConfig,
    pub global: GlobalSpanConfig,
}

#[derive(Clone, Default)]
pub struct BuildOptions {
    pub config: BuildConfig,
    pub run: RunOptions,
}

struct Assets {
    concepts: ConceptData,
    graph: GraphAssets,
    span: GlobalSpan,
    concept_index: ConceptIndex,
    identity: Arc<()>,
}

#[derive(Clone)]
pub struct Model(Arc<Assets>);

impl Model {
    pub fn build(data: RelationData, options: &BuildOptions) -> Result<Self> {
        let run = &options.run;
        run.check_cancelled()?;
        let concepts = ConceptData::prepare(data.dimension, data.concepts, run)?;
        let view = concepts.view();
        let sources = SourceBook::new(view, &data.sources, &data.relations, run)?;
        let local = geometry::local_geometry(view, &sources, &options.config.geometry.local, run)?;
        let span = geometry::build_span(view, &local, &options.config.geometry.global, run)?;
        let graph = transport::build_graph(view, sources, &local, &options.config.graph, run)?;
        let concept_index = ConceptIndex::build(
            view,
            &local,
            graph.transport(),
            &WarpedBuildOptions {
                geometry: options.config.warped_tag_geometry,
                run: run.clone(),
            },
        )?;
        run.check_cancelled()?;
        Ok(Self(Arc::new(Assets {
            concepts,
            graph,
            span,
            concept_index,
            identity: Arc::new(()),
        })))
    }

    pub fn concepts(&self) -> ConceptView<'_> {
        self.0.concepts.view()
    }

    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.0.identity
    }
    pub(crate) fn transport(&self) -> &TransportGraph {
        self.0.graph.transport()
    }
    pub(crate) fn span(&self) -> &GlobalSpan {
        &self.0.span
    }
    pub(crate) fn concept_index(&self) -> &ConceptIndex {
        &self.0.concept_index
    }
    pub(crate) fn graph_assets(&self) -> &GraphAssets {
        &self.0.graph
    }
}
