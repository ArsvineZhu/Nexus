// SPDX-License-Identifier: Apache-2.0

use super::{DtscFieldConfig, unit};
use nexus_core::kernel::{cosine_f32_with_norms, norm_sq_f32};
use nexus_core::{ExecutionContext, NodeId, PrecisionMode, VcpError, VectorMatrixView};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DtscSourceType {
    Core,
    /// A direct concept asserted by the caller or grounded in the query text.
    Seed,
    /// A zero-hop concept obtained from semantic inference.
    Inferred,
    /// A concept first reached by propagation.
    Emergent,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct DtscNodeInput {
    pub node: NodeId,
    pub mass: f64,
    pub source_type: DtscSourceType,
}

pub struct DtscFieldBuildInput<'call> {
    pub nodes: &'call [DtscNodeInput],
    pub tag_vectors: VectorMatrixView<'call, f32>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscSample {
    pub node: NodeId,
    pub potential: f64,
    pub exact: bool,
    pub source_type: DtscSourceType,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DtscFieldNode {
    pub node: NodeId,
    pub potential: f64,
    pub vector: Vec<f32>,
    pub source_type: DtscSourceType,
}

/// Candidate-independent energy and interpolation data with its sampling parameters.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize)]
pub struct DtscFieldValues {
    exact_field: Vec<DtscSample>,
    field_nodes: Vec<DtscFieldNode>,
    field_entropy: f64,
    total_energy: f64,
    max_energy: f64,
    config: DtscFieldConfig,
}

impl DtscFieldValues {
    pub fn exact_field(&self) -> &[DtscSample] {
        &self.exact_field
    }

    pub fn field_nodes(&self) -> &[DtscFieldNode] {
        &self.field_nodes
    }

    pub fn field_entropy(&self) -> f64 {
        self.field_entropy
    }

    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    pub fn max_energy(&self) -> f64 {
        self.max_energy
    }

    pub fn input_node_count(&self) -> usize {
        self.exact_field.len()
    }

    pub fn retained_node_count(&self) -> usize {
        self.field_nodes.len()
    }

    pub fn config(&self) -> &DtscFieldConfig {
        &self.config
    }
}

pub struct DtscFieldSampleInput<'call> {
    pub field: &'call DtscFieldValues,
    pub query_ids: &'call [NodeId],
    pub query_vectors: VectorMatrixView<'call, f32>,
    pub inbound: &'call [f64],
}

/// Bit-exact cosine similarity under `PrecisionMode::StrictLeftFold` with `1e-12` squared-norm guard.
/// Does NOT clamp the ratio so `NaN` / `Inf` preserve reference semantics.
#[inline]
pub fn dtsc_cosine(a: &[f32], b: &[f32]) -> f64 {
    let aa = norm_sq_f32(a, PrecisionMode::StrictLeftFold);
    let bb = norm_sq_f32(b, PrecisionMode::StrictLeftFold);
    cosine_f32_with_norms(a, b, aa, bb, PrecisionMode::StrictLeftFold, 1e-12)
}

pub struct DtscFieldSampler<'a> {
    field: &'a DtscFieldValues,
    pub(super) max_inbound: f64,
    node_specs: Vec<(f64, f64)>,
}

impl<'a> DtscFieldSampler<'a> {
    /// Builds a sampler using the field's configuration and dense inbound masses.
    pub fn new(field: &'a DtscFieldValues, inbound: &[f64]) -> Self {
        let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);
        let floor = field.config.public_hub_floor;
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
            max_inbound,
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
        let config = self.field.config();
        let threshold = config.field_similarity_threshold;
        let vector_norm = norm_sq_f32(vector, PrecisionMode::StrictLeftFold);
        let exponent = config.field_kernel_exponent.max(0.25);

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
        neighbors.truncate(config.max_field_neighbors);

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
                .map(|s| s.source_type)
                .or_else(|| neighbors.first().map(|(_, n)| n.source_type))
                .unwrap_or(DtscSourceType::Unknown),
        }
    }
}

pub fn build_dtsc_field<'call>(
    input: DtscFieldBuildInput<'call>,
    config: &DtscFieldConfig,
    ctx: &ExecutionContext<'_>,
) -> Result<DtscFieldValues, VcpError> {
    const STAGE: &str = "dtsc_field_build";
    ctx.check_cancelled(STAGE)?;
    config.validate()?;
    input.tag_vectors.validate(STAGE)?;

    let mut seen = std::collections::HashSet::with_capacity(input.nodes.len());
    for node in input.nodes {
        ctx.check_cancelled(STAGE)?;
        if node.node.index() >= input.tag_vectors.rows || !seen.insert(node.node) {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: format!(
                    "node {} is repeated or outside the tag matrix",
                    node.node.index()
                ),
            });
        }
        if !node.mass.is_finite() || node.mass < 0.0 {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: format!(
                    "node {} mass must be finite and nonnegative",
                    node.node.index()
                ),
            });
        }
    }
    let minimum = config.min_field_tags;
    let input_order_nodes: Vec<&DtscNodeInput> =
        input.nodes.iter().filter(|n| n.mass > 0.0).collect();

    let mut sorted_nodes = input_order_nodes.clone();
    sorted_nodes.sort_by(|a, b| b.mass.total_cmp(&a.mass).then_with(|| a.node.cmp(&b.node)));

    let total_energy: f64 = sorted_nodes.iter().map(|n| n.mass).sum();
    if !total_energy.is_finite() {
        return Err(VcpError::NumericDivergence {
            stage: STAGE,
            detail: "total field mass overflowed".into(),
        });
    }
    let max_energy = sorted_nodes.iter().map(|n| n.mass).fold(0.0_f64, f64::max);

    let field_entropy = if sorted_nodes.len() > 1 && total_energy > 0.0 {
        sorted_nodes
            .iter()
            .map(|n| {
                let p = n.mass / total_energy;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum::<f64>()
            / (sorted_nodes.len() as f64).ln()
    } else {
        0.0
    };

    let exact_field: Vec<DtscSample> = input_order_nodes
        .iter()
        .map(|n| DtscSample {
            node: n.node,
            potential: if max_energy > 0.0 {
                unit(n.mass / max_energy)
            } else {
                0.0
            },
            exact: true,
            source_type: n.source_type,
        })
        .collect();

    let mut retained = 0;
    let mut mass = 0.0;
    let max_nodes = config.max_field_nodes;
    let mass_ratio = config.field_energy_mass_ratio;
    for n in &sorted_nodes {
        if retained >= max_nodes || (retained >= minimum && mass / total_energy >= mass_ratio) {
            break;
        }
        retained += 1;
        mass += n.mass;
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
                            unit(n.mass / max_energy)
                        } else {
                            0.0
                        },
                        vector: v.to_vec(),
                        source_type: n.source_type,
                    })
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(DtscFieldValues {
        exact_field,
        field_nodes,
        field_entropy,
        total_energy,
        max_energy,
        config: config.clone(),
    })
}

pub fn sample_dtsc_field<'call>(
    input: DtscFieldSampleInput<'call>,
    ctx: &ExecutionContext<'_>,
) -> Result<Vec<DtscSample>, VcpError> {
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

    let sampler = DtscFieldSampler::new(input.field, input.inbound);
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
