// SPDX-License-Identifier: Apache-2.0

use super::{DtscFieldConfig, unit};
use nexus_core::kernel::{cosine_f32_with_norms, norm_sq_f32};
use nexus_core::{
    ExecutionContext, NodeId, Operator, PrecisionMode, ScratchPad, VcpError, VectorMatrixView,
};

#[derive(Clone, Debug)]
pub struct DtscNodeInput {
    pub node: NodeId,
    pub energy: f64,
    pub normalized_energy: f64,
    pub source_type: String,
}

pub struct DtscFieldBuildInput<'asset, 'query> {
    pub nodes: &'query [DtscNodeInput],
    pub tag_vectors: VectorMatrixView<'asset, f32>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscSample {
    pub node: NodeId,
    pub potential: f64,
    pub exact: bool,
    pub source_type: String,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscFieldNode {
    pub node: NodeId,
    pub potential: f64,
    pub vector: Vec<f32>,
    pub source_type: String,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscField {
    pub exact_field: Vec<DtscSample>,
    pub field_nodes: Vec<DtscFieldNode>,
    pub field_entropy: f64,
    pub field_trusted: bool,
    pub total_energy: f64,
    pub max_energy: f64,
}

pub struct DtscFieldSampleInput<'asset, 'query> {
    pub field: &'query DtscField,
    pub query_ids: &'query [NodeId],
    pub query_vectors: VectorMatrixView<'query, f32>,
    pub inbound: &'asset [f64],
}

/// Bit-exact cosine similarity under `PrecisionMode::StrictLeftFold` with `1e-12` squared-norm guard.
/// Does NOT clamp the ratio so `NaN` / `Inf` preserve reference semantics.
#[inline]
pub fn dtsc_cosine(a: &[f32], b: &[f32]) -> f64 {
    let aa = norm_sq_f32(a, PrecisionMode::StrictLeftFold);
    let bb = norm_sq_f32(b, PrecisionMode::StrictLeftFold);
    cosine_f32_with_norms(a, b, aa, bb, PrecisionMode::StrictLeftFold, 1e-12)
}

#[inline]
fn node_energy(n: &DtscNodeInput) -> f64 {
    if n.energy > 0.0 {
        n.energy
    } else {
        n.normalized_energy
    }
}

pub struct DtscFieldSampler<'a> {
    pub field: &'a DtscField,
    pub inbound: &'a [f64],
    pub max_inbound: f64,
    pub config: &'a DtscFieldConfig,
    pub node_specs: Vec<(f64, f64)>,
}

impl<'a> DtscFieldSampler<'a> {
    /// Builds a sampler over `field` using dense `inbound` lookup.
    /// Computes `max_inbound` via `f64::max` semantics (`fold(0.0, f64::max)`).
    pub fn new(field: &'a DtscField, inbound: &'a [f64], config: &'a DtscFieldConfig) -> Self {
        let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);
        let floor = config.public_hub_floor.clamp(0.05, 1.0);
        let node_specs = field
            .field_nodes
            .iter()
            .map(|n| {
                let specific = if max_inbound <= 0.0 {
                    1.0
                } else {
                    let w = inbound.get(n.node.index()).copied().unwrap_or(0.0);
                    floor.max(1.0 - unit(w / max_inbound).sqrt())
                };
                (
                    norm_sq_f32(&n.vector, PrecisionMode::StrictLeftFold),
                    specific,
                )
            })
            .collect();
        Self {
            field,
            inbound,
            max_inbound,
            config,
            node_specs,
        }
    }

    pub fn sample(&self, node: NodeId, vector: &[f32]) -> DtscSample {
        let mut neighbors = Vec::new();
        self.sample_with_buf(node, vector, &mut neighbors)
    }

    pub(crate) fn sample_with_buf<'s>(
        &'s self,
        node: NodeId,
        vector: &[f32],
        neighbors: &mut Vec<(f64, &'s DtscFieldNode)>,
    ) -> DtscSample {
        let exact = self.field.exact_field.iter().find(|s| s.node == node);
        let threshold = self.config.field_similarity_threshold.clamp(-1.0, 1.0);
        let vector_norm = norm_sq_f32(vector, PrecisionMode::StrictLeftFold);
        let exponent = self.config.field_kernel_exponent.max(0.25);

        neighbors.clear();
        for (n, &(node_norm, specific)) in self.field.field_nodes.iter().zip(&self.node_specs) {
            if n.node == node {
                continue;
            }
            let similarity = cosine_f32_with_norms(
                vector,
                &n.vector,
                vector_norm,
                node_norm,
                PrecisionMode::StrictLeftFold,
                1e-12,
            );
            if similarity < threshold {
                continue;
            }
            let local = unit((similarity - threshold) / (1.0 - threshold).max(1e-6));
            neighbors.push((n.potential * local.powf(exponent) * specific, n));
        }

        neighbors.sort_by(|a, b| b.0.total_cmp(&a.0));
        neighbors.truncate(self.config.max_field_neighbors.max(1));

        let interpolated = if neighbors.is_empty() {
            0.0
        } else {
            neighbors.iter().map(|(p, _)| *p).sum::<f64>() / (neighbors.len() as f64).sqrt()
        };

        DtscSample {
            node,
            potential: unit(exact.map_or(0.0, |s| s.potential).max(interpolated)),
            exact: exact.is_some(),
            source_type: exact
                .map(|s| s.source_type.clone())
                .or_else(|| neighbors.first().map(|(_, n)| n.source_type.clone()))
                .unwrap_or_else(|| "unknown".into()),
        }
    }
}

pub struct DtscFieldBuildOperator;

impl Operator for DtscFieldBuildOperator {
    type Config = DtscFieldConfig;
    type Input<'asset, 'query>
        = DtscFieldBuildInput<'asset, 'query>
    where
        Self: 'asset + 'query;
    type Output = DtscField;

    fn name(&self) -> &'static str {
        "dtsc_field_build"
    }

    fn execute<'asset, 'query>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'asset, 'query>,
        _scratch: &mut ScratchPad,
    ) -> Result<Self::Output, VcpError> {
        const STAGE: &str = "dtsc_field_build";
        ctx.check_cancelled(STAGE)?;
        input.tag_vectors.validate(STAGE)?;

        let minimum = config.min_field_tags.max(1);
        let input_order_nodes: Vec<&DtscNodeInput> = input
            .nodes
            .iter()
            .filter(|n| n.node.index() < input.tag_vectors.rows && node_energy(n) > 0.0)
            .collect();

        let mut sorted_nodes = input_order_nodes.clone();
        sorted_nodes.sort_by(|a, b| {
            node_energy(b)
                .total_cmp(&node_energy(a))
                .then_with(|| a.node.cmp(&b.node))
        });

        let total_energy: f64 = sorted_nodes.iter().map(|n| node_energy(n)).sum();
        let max_energy = sorted_nodes
            .iter()
            .map(|n| node_energy(n))
            .fold(0.0_f64, f64::max);

        let field_entropy = if sorted_nodes.len() > 1 && total_energy > 0.0 {
            sorted_nodes
                .iter()
                .map(|n| {
                    let p = node_energy(n) / total_energy;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum::<f64>()
                / (sorted_nodes.len() as f64).ln()
        } else {
            0.0
        };

        let distribution_trusted = sorted_nodes.len() >= minimum
            && total_energy > 0.0
            && (!config.fallback_to_knn_on_low_trust
                || field_entropy >= unit(config.min_field_entropy));

        let exact_field: Vec<DtscSample> = input_order_nodes
            .iter()
            .map(|n| DtscSample {
                node: n.node,
                potential: if max_energy > 0.0 {
                    unit(node_energy(n) / max_energy)
                } else {
                    0.0
                },
                exact: true,
                source_type: n.source_type.clone(),
            })
            .collect();

        let mut retained = 0;
        let mut mass = 0.0;
        let max_nodes = config.max_field_nodes.max(minimum);
        let mass_ratio = config.field_energy_mass_ratio.clamp(0.5, 1.0);
        for n in &sorted_nodes {
            if retained >= max_nodes
                || (retained >= minimum && mass / total_energy.max(1e-12) >= mass_ratio)
            {
                break;
            }
            retained += 1;
            mass += node_energy(n);
        }

        let field_nodes: Vec<DtscFieldNode> = if input.tag_vectors.dim > 0 {
            sorted_nodes
                .iter()
                .take(retained)
                .filter_map(|n| {
                    input
                        .tag_vectors
                        .row(n.node.index())
                        .map(|v| DtscFieldNode {
                            node: n.node,
                            potential: if max_energy > 0.0 {
                                unit(node_energy(n) / max_energy)
                            } else {
                                0.0
                            },
                            vector: v.to_vec(),
                            source_type: n.source_type.clone(),
                        })
                })
                .collect()
        } else {
            Vec::new()
        };

        let field_trusted = distribution_trusted && field_nodes.len() >= minimum;

        Ok(DtscField {
            exact_field,
            field_nodes,
            field_entropy,
            field_trusted,
            total_energy,
            max_energy,
        })
    }
}

pub struct DtscFieldSampleOperator;

impl Operator for DtscFieldSampleOperator {
    type Config = DtscFieldConfig;
    type Input<'asset, 'query>
        = DtscFieldSampleInput<'asset, 'query>
    where
        Self: 'asset + 'query;
    type Output = Vec<DtscSample>;

    fn name(&self) -> &'static str {
        "dtsc_field_sample"
    }

    fn execute<'asset, 'query>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'asset, 'query>,
        _scratch: &mut ScratchPad,
    ) -> Result<Self::Output, VcpError> {
        const STAGE: &str = "dtsc_field_sample";
        ctx.check_cancelled(STAGE)?;
        input.query_vectors.validate(STAGE)?;
        if input.query_ids.len() != input.query_vectors.rows {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: input.query_ids.len(),
                actual: input.query_vectors.rows,
            });
        }

        let sampler = DtscFieldSampler::new(input.field, input.inbound, config);
        let mut neighbors = Vec::new();
        let mut out = Vec::with_capacity(input.query_ids.len());
        for (idx, &node) in input.query_ids.iter().enumerate() {
            if idx % 64 == 0 {
                ctx.check_cancelled(STAGE)?;
            }
            let vec = input.query_vectors.row(idx).unwrap_or(&[]);
            out.push(sampler.sample_with_buf(node, vec, &mut neighbors));
        }
        Ok(out)
    }
}
