// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::time::Instant;

use nexus_core::kernel::{dot_f64, normalize_f64};
use nexus_core::*;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FactMatrixConfig {
    pub forward_gain: f64,
    pub reverse_gain: f64,
    pub min_reverse_gain: f64,
    pub max_reverse_gain: f64,
    pub distance_decay: f64,
    pub reverse_inversion_guard: f64,
    pub reverse_anchor_boost: bool,
    pub reverse_anchor_max: f64,
    pub semantic_gain_enabled: bool,
    pub semantic_gain_peak: f64,
    pub semantic_gain_sigma: f64,
    pub semantic_gain_low_fallback: f64,
}

impl Default for FactMatrixConfig {
    fn default() -> Self {
        Self {
            forward_gain: 1.0,
            reverse_gain: 0.42,
            min_reverse_gain: 0.25,
            max_reverse_gain: 0.70,
            distance_decay: 0.0,
            reverse_inversion_guard: 0.95,
            reverse_anchor_boost: false,
            reverse_anchor_max: 1.5,
            semantic_gain_enabled: false,
            semantic_gain_peak: 0.65,
            semantic_gain_sigma: 0.25,
            semantic_gain_low_fallback: 0.1,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransportConfig {
    pub wormhole_evidence_threshold: f64,
    pub wormhole_gain: f64,
    pub hub_penalty_exponent: f64,
    pub hub_penalty_min: f64,
    pub hub_penalty_max: f64,
    pub outbound_mass: f64,
    pub association_reserve_mass: f64,
    pub evidence_compression: f64,
    pub hub_smoothing_ratio: f64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            wormhole_evidence_threshold: 1.0,
            wormhole_gain: 1.35,
            hub_penalty_exponent: -0.3,
            hub_penalty_min: 0.55,
            hub_penalty_max: 1.8,
            outbound_mass: 0.95,
            association_reserve_mass: 0.05,
            evidence_compression: 1.0,
            hub_smoothing_ratio: 0.1,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProvenanceConfig {
    pub forward_gain: f64,
    pub provenance_reverse_gain: f64,
    pub provenance_distance_decay: f64,
}

impl Default for ProvenanceConfig {
    fn default() -> Self {
        Self {
            forward_gain: 1.0,
            provenance_reverse_gain: 0.35,
            provenance_distance_decay: 0.08,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FactMatrixInput<'asset, 'query> {
    pub tag_vectors: VectorMatrixView<'asset, f32>,
    pub sequences: SequenceSliceView<'asset>,
    pub anchor_gains: &'query [f64],
    pub positive_relations: Option<PositiveRelationView<'asset>>,
    pub negative_relations: Option<NegativeRelationView<'asset>>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FactMatrix {
    pub node_count: usize,
    pub row_offsets: Vec<usize>,
    pub targets: Vec<NodeId>,
    pub raw_weights: Vec<f64>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CounterevidenceAdj {
    pub sources: Vec<NodeId>,
    pub targets: Vec<NodeId>,
    pub roots: Vec<RootId>,
    pub relation_kinds: Vec<u16>,
    pub strengths: Vec<f64>,
    pub confidences: Vec<f64>,
}

impl CounterevidenceAdj {
    pub fn as_view(&self) -> CounterevidenceAdjView<'_> {
        CounterevidenceAdjView {
            edge_count: self.sources.len(),
            sources: &self.sources,
            targets: &self.targets,
            roots: &self.roots,
            relation_kinds: &self.relation_kinds,
            strengths: &self.strengths,
            confidences: &self.confidences,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FactMatrixOutput {
    pub fact_matrix: FactMatrix,
    pub counterevidence: CounterevidenceAdj,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Copy, Debug)]
pub struct TransportCalibrateInput<'query> {
    pub fact_matrix: &'query FactMatrix,
    pub anchor_gains: &'query [f64],
    pub edge_gates: Option<&'query [f64]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportGraph {
    pub graph: nexus_core::CsrGraph,
    pub diagnostics: Diagnostics,
}

impl TransportGraph {
    pub fn as_csr_view(&self) -> CsrGraphView<'_> {
        self.graph.view()
    }
}

impl std::ops::Deref for TransportGraph {
    type Target = nexus_core::CsrGraph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProvenanceBuildInput<'asset, 'query> {
    pub transport_row_offsets: &'query [usize],
    pub transport_targets: &'query [NodeId],
    pub sequences: SequenceSliceView<'asset>,
    pub positive_relations: Option<PositiveRelationView<'asset>>,
    pub support_groups: Option<SupportGroupView<'asset>>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProvenanceData {
    pub edge_count: usize,
    pub edge_prov_offsets: Vec<u32>,
    pub prov_groups: Vec<GroupId>,
    pub prov_masses: Vec<f64>,
    pub diagnostics: Diagnostics,
}

impl ProvenanceData {
    pub fn as_view(&self) -> ProvenanceView<'_> {
        ProvenanceView {
            edge_count: self.edge_count,
            edge_prov_offsets: &self.edge_prov_offsets,
            prov_groups: &self.prov_groups,
            prov_masses: &self.prov_masses,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FactMatrixOperator;

#[derive(Clone, Copy, Debug, Default)]
pub struct TransportCalibrateOperator;

#[derive(Clone, Copy, Debug, Default)]
pub struct ProvenanceBuildOperator;

fn stable_sum_edges(mut entries: Vec<(NodeId, NodeId, f64)>) -> Vec<(NodeId, NodeId, f64)> {
    entries.sort_by_key(|&(s, t, _)| (s, t));
    let mut out = Vec::new();
    let mut cur: Option<(NodeId, NodeId, f64)> = None;
    for (s, t, w) in entries {
        match cur {
            Some((cs, ct, ref mut cw)) if cs == s && ct == t => {
                *cw += w;
            }
            Some((cs, ct, cw)) => {
                if cw > 0.0 && cw.is_finite() {
                    out.push((cs, ct, cw));
                }
                cur = Some((s, t, 0.0 + w));
            }
            None => {
                cur = Some((s, t, 0.0 + w));
            }
        }
    }
    if let Some((cs, ct, cw)) = cur
        && cw > 0.0
        && cw.is_finite()
    {
        out.push((cs, ct, cw));
    }
    out
}

impl Operator for FactMatrixOperator {
    type Config = FactMatrixConfig;
    type Input<'asset, 'query> = FactMatrixInput<'asset, 'query>;
    type Output = FactMatrixOutput;

    fn name(&self) -> &'static str {
        "fact_matrix"
    }

    fn execute<'asset, 'query>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'asset, 'query>,
    ) -> VcpResult<Self::Output> {
        const STAGE: &str = "FactMatrixOperator::execute";
        let start_time = Instant::now();
        ctx.check_cancelled(STAGE)?;
        for (name, value, min, max) in [
            ("forward_gain", config.forward_gain, 0.0, f64::INFINITY),
            (
                "min_reverse_gain",
                config.min_reverse_gain,
                0.0,
                f64::INFINITY,
            ),
            (
                "max_reverse_gain",
                config.max_reverse_gain,
                config.min_reverse_gain,
                f64::INFINITY,
            ),
            (
                "reverse_gain",
                config.reverse_gain,
                config.min_reverse_gain,
                config.max_reverse_gain,
            ),
            ("distance_decay", config.distance_decay, 0.0, f64::INFINITY),
            (
                "reverse_inversion_guard",
                config.reverse_inversion_guard,
                0.0,
                1.0,
            ),
            (
                "reverse_anchor_max",
                config.reverse_anchor_max,
                1.0,
                f64::INFINITY,
            ),
            (
                "semantic_gain_sigma",
                config.semantic_gain_sigma,
                0.001,
                f64::INFINITY,
            ),
            ("semantic_gain_peak", config.semantic_gain_peak, -1.0, 1.0),
            (
                "semantic_gain_low_fallback",
                config.semantic_gain_low_fallback,
                0.0,
                f64::INFINITY,
            ),
        ] {
            crate::parameter(STAGE, name, value, min, max)?;
        }

        input.tag_vectors.validate(STAGE)?;
        let node_count = input.tag_vectors.rows;
        if input.tag_vectors.data.iter().any(|v| !v.is_finite()) {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "tag_vectors contains non-finite values".into(),
            });
        }
        input.sequences.validate(node_count, STAGE)?;

        if input.anchor_gains.len() != node_count {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: node_count,
                actual: input.anchor_gains.len(),
            });
        }
        if input
            .anchor_gains
            .iter()
            .any(|g| !g.is_finite() || *g < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "anchor_gains must be finite and non-negative".into(),
            });
        }
        if let Some(pos) = input.positive_relations {
            pos.validate(node_count, STAGE)?;
        }
        if let Some(neg) = input.negative_relations {
            neg.validate(node_count, STAGE)?;
        }

        // Route NegativeRelationView exclusively into CounterevidenceAdj (0-bit isolation).
        let counterevidence = match input.negative_relations {
            Some(neg) => CounterevidenceAdj {
                sources: neg.sources.to_vec(),
                targets: neg.targets.to_vec(),
                roots: neg.roots.to_vec(),
                relation_kinds: neg.relation_kinds.to_vec(),
                strengths: neg.strengths.to_vec(),
                confidences: neg.confidences.to_vec(),
            },
            None => CounterevidenceAdj::default(),
        };

        let unit_vecs: Vec<Option<Vec<f64>>> = if config.semantic_gain_enabled {
            let dim = input.tag_vectors.dim;
            (0..node_count)
                .map(|r| {
                    if dim == 0 {
                        return None;
                    }
                    let slice = input.tag_vectors.row(r).unwrap_or(&[]);
                    let mut v: Vec<f64> = slice.iter().map(|&x| f64::from(x)).collect();
                    normalize_f64(&mut v, 1e-12, PrecisionMode::StrictLeftFold).then_some(v)
                })
                .collect()
        } else {
            Vec::new()
        };

        let semantic_sigma = config.semantic_gain_sigma;
        let semantic_peak = config.semantic_gain_peak;
        let low_fallback = config.semantic_gain_low_fallback;
        let compute_semantic_gain = |a: NodeId, b: NodeId| -> f64 {
            if !config.semantic_gain_enabled {
                return 1.0;
            }
            let sim = unit_vecs
                .get(a.index())
                .and_then(|o| o.as_deref())
                .zip(unit_vecs.get(b.index()).and_then(|o| o.as_deref()))
                .map(|(va, vb)| dot_f64(va, vb, ctx.precision).clamp(-1.0, 1.0))
                .unwrap_or(low_fallback);
            if sim < 0.15 {
                0.4 + sim
            } else {
                0.5 + 0.8
                    * (-((sim - semantic_peak).powi(2)) / (2.0 * semantic_sigma * semantic_sigma))
                        .exp()
            }
        };

        let forward_gain = config.forward_gain;
        let base_reverse_gain = config.reverse_gain;
        let distance_decay = config.distance_decay;
        let reverse_guard = config.reverse_inversion_guard;
        let reverse_anchor_max = config.reverse_anchor_max;

        let mut pair_gains: BTreeMap<(NodeId, NodeId), f64> = BTreeMap::new();
        let mut raw_entries: Vec<(NodeId, NodeId, f64)> = Vec::new();
        let mut seq_buf: Vec<(u32, NodeId)> = Vec::new();
        let pair_limit = ctx.limits.pair_work_limit;
        let mut pair_work = 0usize;

        for seq_idx in 0..input.sequences.sequence_count {
            ctx.check_cancelled(STAGE)?;
            let (tags, positions) = input.sequences.sequence(SequenceId(seq_idx as u32));
            if tags.len() < 2 {
                continue;
            }
            seq_buf.clear();
            seq_buf.extend(positions.iter().copied().zip(tags.iter().copied()));
            if !seq_buf.windows(2).all(|w| w[0] <= w[1]) {
                seq_buf.sort_unstable();
            }

            let count = seq_buf.len() as f64;
            for i in 0..seq_buf.len() {
                let (left_pos, left_tag) = seq_buf[i];
                for &(right_pos, right_tag) in &seq_buf[i + 1..] {
                    if pair_work.is_multiple_of(1024) {
                        ctx.check_cancelled(STAGE)?;
                    }
                    pair_work += 1;
                    if pair_limit > 0 && pair_work > pair_limit {
                        return Err(VcpError::ResourceLimitExceeded {
                            stage: STAGE,
                            limit_name: "pair_work_limit",
                            actual: pair_work,
                            limit: pair_limit,
                        });
                    }
                    if left_tag == right_tag {
                        continue;
                    }
                    let semantic = if config.semantic_gain_enabled {
                        let pair_key = (left_tag.min(right_tag), left_tag.max(right_tag));
                        *pair_gains
                            .entry(pair_key)
                            .or_insert_with(|| compute_semantic_gain(left_tag, right_tag))
                    } else {
                        1.0
                    };

                    if left_pos > 0 && right_pos > 0 {
                        let phi_left = 0.9 - 0.4 * ((left_pos - 1) as f64) / (count - 1.0).max(1.0);
                        let phi_right =
                            0.9 - 0.4 * ((right_pos - 1) as f64) / (count - 1.0).max(1.0);
                        let delta = right_pos.saturating_sub(left_pos).max(1) as f64;
                        let distance = if distance_decay > 0.0 {
                            (-distance_decay * (delta - 1.0)).exp()
                        } else {
                            1.0
                        };
                        let base = phi_left * phi_right * distance;
                        let forward = base * forward_gain * semantic;
                        let mut dynamic_reverse = base_reverse_gain;
                        if config.reverse_anchor_boost {
                            let boost =
                                input.anchor_gains[left_tag.index()].min(reverse_anchor_max);
                            dynamic_reverse *= boost;
                        }
                        dynamic_reverse =
                            dynamic_reverse.clamp(config.min_reverse_gain, config.max_reverse_gain);
                        let backward =
                            (base * dynamic_reverse * semantic).min(forward * reverse_guard);
                        if forward.is_finite() && forward > 0.0 {
                            raw_entries.push((left_tag, right_tag, forward));
                        }
                        if backward.is_finite() && backward > 0.0 {
                            raw_entries.push((right_tag, left_tag, backward));
                        }
                    } else {
                        let weight = 0.7 * 0.7 * semantic;
                        if weight.is_finite() && weight > 0.0 {
                            raw_entries.push((left_tag, right_tag, weight));
                            raw_entries.push((right_tag, left_tag, weight));
                        }
                    }
                }
            }
        }

        // Deduplicate repeated (source, target, group) positive relations via max, then merge.
        if let Some(pos) = input.positive_relations {
            let mut dedup_pos: BTreeMap<(NodeId, NodeId, GroupId), f64> = BTreeMap::new();
            for k in 0..pos.sources.len() {
                let src = pos.sources[k];
                let dst = pos.targets[k];
                let grp = pos.groups[k];
                let mass = pos.masses[k];
                if src == dst || !mass.is_finite() || mass <= 0.0 {
                    continue;
                }
                let entry = dedup_pos.entry((src, dst, grp)).or_insert(0.0);
                *entry = entry.max(mass);
            }
            for ((src, dst, _grp), mass) in dedup_pos {
                raw_entries.push((src, dst, mass));
            }
        }

        let summed = stable_sum_edges(raw_entries);
        let mut row_offsets = vec![0usize; node_count + 1];
        let mut targets = Vec::with_capacity(summed.len());
        let mut raw_weights = Vec::with_capacity(summed.len());

        for (src, dst, w) in summed {
            row_offsets[src.index() + 1] += 1;
            targets.push(dst);
            raw_weights.push(w);
        }
        for r in 0..node_count {
            row_offsets[r + 1] += row_offsets[r];
        }

        Ok(FactMatrixOutput {
            fact_matrix: FactMatrix {
                node_count,
                row_offsets,
                targets,
                raw_weights,
            },
            counterevidence,
            diagnostics: Diagnostics {
                truncated: false,
                truncation_reason: None,
                elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            },
        })
    }
}

impl Operator for TransportCalibrateOperator {
    type Config = TransportConfig;
    type Input<'asset, 'query> = TransportCalibrateInput<'query>;
    type Output = TransportGraph;

    fn name(&self) -> &'static str {
        "transport_calibrate"
    }

    fn execute<'asset, 'query>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'asset, 'query>,
    ) -> VcpResult<Self::Output> {
        const STAGE: &str = "TransportCalibrateOperator::execute";
        let start_time = Instant::now();
        ctx.check_cancelled(STAGE)?;
        for (name, value, min, max) in [
            ("outbound_mass", config.outbound_mass, 0.01, 0.999_999),
            (
                "association_reserve_mass",
                config.association_reserve_mass,
                0.0,
                config.outbound_mass,
            ),
            (
                "evidence_compression",
                config.evidence_compression,
                0.01,
                f64::INFINITY,
            ),
            ("wormhole_gain", config.wormhole_gain, 1.0, f64::INFINITY),
            (
                "wormhole_evidence_threshold",
                config.wormhole_evidence_threshold,
                0.0,
                f64::INFINITY,
            ),
            (
                "hub_penalty_min",
                config.hub_penalty_min,
                f64::MIN_POSITIVE,
                f64::INFINITY,
            ),
            (
                "hub_penalty_max",
                config.hub_penalty_max,
                config.hub_penalty_min,
                f64::INFINITY,
            ),
            (
                "hub_penalty_exponent",
                config.hub_penalty_exponent,
                f64::NEG_INFINITY,
                f64::INFINITY,
            ),
            ("hub_smoothing_ratio", config.hub_smoothing_ratio, 0.01, 2.0),
        ] {
            crate::parameter(STAGE, name, value, min, max)?;
        }

        let fm = input.fact_matrix;
        let node_count = fm.node_count;
        let nnz = fm.targets.len();

        if fm.row_offsets.len() != node_count + 1 {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: node_count + 1,
                actual: fm.row_offsets.len(),
            });
        }
        if fm.row_offsets.first().copied() != Some(0)
            || fm.row_offsets.last().copied() != Some(nnz)
            || fm.row_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail:
                    "fact_matrix.row_offsets must start at 0, be non-decreasing, and end at nnz"
                        .into(),
            });
        }
        if fm.raw_weights.len() != nnz {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: nnz,
                actual: fm.raw_weights.len(),
            });
        }
        if fm.targets.iter().any(|t| t.index() >= node_count) {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "fact_matrix.targets contains out-of-bounds NodeId".into(),
            });
        }
        if fm.raw_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "fact_matrix.raw_weights must be finite and non-negative".into(),
            });
        }
        if input.anchor_gains.len() != node_count {
            return Err(VcpError::DimensionMismatch {
                stage: STAGE,
                expected: node_count,
                actual: input.anchor_gains.len(),
            });
        }
        if input
            .anchor_gains
            .iter()
            .any(|g| !g.is_finite() || *g < 0.0)
        {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "anchor_gains must be finite and non-negative".into(),
            });
        }
        if let Some(gates) = input.edge_gates {
            if gates.len() != nnz {
                return Err(VcpError::DimensionMismatch {
                    stage: STAGE,
                    expected: nnz,
                    actual: gates.len(),
                });
            }
            if gates
                .iter()
                .any(|g| !g.is_finite() || !(0.0..=1.0).contains(g))
            {
                return Err(VcpError::InvalidInput {
                    stage: STAGE,
                    detail: "edge_gates must be finite and in [0.0, 1.0]".into(),
                });
            }
        }

        let outbound_mass = config.outbound_mass;
        let reserve_mass = config.association_reserve_mass;
        let evidence_compression = config.evidence_compression;
        let wormhole_gain = config.wormhole_gain;
        let tension_threshold = config.wormhole_evidence_threshold;
        let hub_exp = {
            let e = config.hub_penalty_exponent;
            let neg = if e > 0.0 { -e } else { e };
            neg.clamp(-1.0, 0.0)
        };
        let hub_floor = config
            .hub_penalty_min
            .min(config.hub_penalty_max)
            .clamp(0.05, 1.0);
        let hub_ceiling = config
            .hub_penalty_min
            .max(config.hub_penalty_max)
            .clamp(1.0, 4.0);
        let smoothing_ratio = config.hub_smoothing_ratio;

        let mut target_inflow = vec![0.0_f64; node_count];
        let mut raw_rows: Vec<Vec<(NodeId, f64, bool, f64)>> = Vec::with_capacity(node_count);

        for u in 0..node_count {
            ctx.check_cancelled(STAGE)?;
            let start = fm.row_offsets[u];
            let end = fm.row_offsets[u + 1];
            let mut edges = Vec::with_capacity(end - start);
            for e in start..end {
                let target = fm.targets[e];
                let compat = fm.raw_weights[e];
                let gate = input.edge_gates.map_or(1.0, |g| g[e]);
                edges.push((target, compat, gate));
            }
            edges.sort_unstable_by_key(|&(t, _, _)| t);

            let mut row_conductances = Vec::with_capacity(edges.len());
            for (target, compat, gate) in edges {
                let evidence = (1.0 + compat.max(0.0) * evidence_compression).ln();
                let anchor_gain = input.anchor_gains[target.index()];
                let wormhole = (evidence * anchor_gain) >= tension_threshold;
                let conductance = evidence * if wormhole { wormhole_gain } else { 1.0 };
                if conductance > 0.0 && conductance.is_finite() {
                    row_conductances.push((target, conductance, wormhole, gate));
                    target_inflow[target.index()] += conductance;
                }
            }
            raw_rows.push(row_conductances);
        }

        let mut positive: Vec<f64> = target_inflow
            .iter()
            .copied()
            .filter(|&v| v > 0.0 && v.is_finite())
            .collect();
        let median = if positive.is_empty() {
            1.0
        } else {
            let middle = positive.len() / 2;
            *positive.select_nth_unstable_by(middle, f64::total_cmp).1
        };
        let smoothing = (median * smoothing_ratio).max(1e-9);
        let penalties: Vec<f64> = target_inflow
            .iter()
            .map(|&inflow| {
                let relative = inflow / (median + smoothing);
                let raw_penalty = if hub_exp < 0.0 {
                    relative.max(1e-9).powf(hub_exp)
                } else {
                    1.0
                };
                raw_penalty.clamp(hub_floor, hub_ceiling)
            })
            .collect();

        let mut row_offsets = Vec::with_capacity(node_count + 1);
        let mut targets = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);
        let mut is_static_wormhole = Vec::with_capacity(nnz);
        let mut inbound = vec![0.0_f64; node_count];
        row_offsets.push(0);

        for edges in raw_rows {
            let mut adjusted = Vec::with_capacity(edges.len());
            let mut total = 0.0;
            let mut wormhole_total = 0.0;

            for (target, conductance, wormhole, gate) in edges {
                let value = conductance * penalties[target.index()];
                if value > 0.0 && value.is_finite() {
                    total += value;
                    if wormhole {
                        wormhole_total += value;
                    }
                    adjusted.push((target, value, wormhole, gate));
                }
            }

            if total > 0.0 {
                let reserved = if wormhole_total > 0.0 {
                    reserve_mass
                } else {
                    0.0
                };
                let main = outbound_mass - reserved;
                let mut calibrated: Vec<(NodeId, f64, bool)> = adjusted
                    .into_iter()
                    .map(|(target, value, wormhole, gate)| {
                        let base_weight = main * value / total
                            + if wormhole && wormhole_total > 0.0 {
                                reserved * value / wormhole_total
                            } else {
                                0.0
                            };
                        // Apply edge_gates post-normalization without redistributing attenuated mass.
                        let weight = base_weight * gate;
                        (target, weight, wormhole)
                    })
                    .collect();

                // Primary CSR rows sorted descending by weight, tie-broken ascending by target NodeId.
                calibrated.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

                for (target, weight, wormhole) in calibrated {
                    inbound[target.index()] += weight;
                    targets.push(target);
                    weights.push(weight);
                    is_static_wormhole.push(wormhole);
                }
            }
            row_offsets.push(targets.len());
        }

        let max_inbound = inbound.iter().copied().fold(0.0_f64, f64::max);

        let target_sorted_indices = {
            let out_nnz = targets.len();
            let mut sorted = Vec::with_capacity(out_nnz);
            let mut row_buf = Vec::new();
            for u in 0..node_count {
                let start = row_offsets[u];
                let end = row_offsets[u + 1];
                row_buf.clear();
                for e in start..end {
                    row_buf.push(e as u32);
                }
                row_buf.sort_unstable_by(|&e0, &e1| {
                    targets[e0 as usize]
                        .cmp(&targets[e1 as usize])
                        .then_with(|| e0.cmp(&e1))
                });
                sorted.extend_from_slice(&row_buf);
            }
            Some(sorted)
        };

        let out = TransportGraph {
            graph: nexus_core::CsrGraph::new(nexus_core::CsrStorage {
                node_count,
                row_offsets,
                targets,
                weights,
                is_static_wormhole,
                inbound,
                max_inbound,
                anchor_gains: input.anchor_gains.to_vec(),
                target_sorted_indices,
            })?,
            diagnostics: Diagnostics {
                truncated: false,
                truncation_reason: None,
                elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            },
        };
        Ok(out)
    }
}

impl Operator for ProvenanceBuildOperator {
    type Config = ProvenanceConfig;
    type Input<'asset, 'query> = ProvenanceBuildInput<'asset, 'query>;
    type Output = ProvenanceData;

    fn name(&self) -> &'static str {
        "provenance_build"
    }

    fn execute<'asset, 'query>(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'asset, 'query>,
    ) -> VcpResult<Self::Output> {
        const STAGE: &str = "ProvenanceBuildOperator::execute";
        let start_time = Instant::now();
        ctx.check_cancelled(STAGE)?;
        for (name, value, min, max) in [
            ("forward_gain", config.forward_gain, 0.0, f64::INFINITY),
            (
                "provenance_reverse_gain",
                config.provenance_reverse_gain,
                0.0,
                f64::INFINITY,
            ),
            (
                "provenance_distance_decay",
                config.provenance_distance_decay,
                0.0,
                f64::INFINITY,
            ),
        ] {
            crate::parameter(STAGE, name, value, min, max)?;
        }

        if input.transport_row_offsets.is_empty() {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "transport_row_offsets must have at least 1 element".into(),
            });
        }
        let node_count = input.transport_row_offsets.len() - 1;
        let nnz = input.transport_targets.len();
        if input.transport_row_offsets.first().copied() != Some(0)
            || input.transport_row_offsets.last().copied() != Some(nnz)
            || input.transport_row_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "transport_row_offsets must start at 0, be non-decreasing, and end at nnz"
                    .into(),
            });
        }
        if input
            .transport_targets
            .iter()
            .any(|t| t.index() >= node_count)
        {
            return Err(VcpError::InvalidInput {
                stage: STAGE,
                detail: "transport_targets contains out-of-bounds NodeId".into(),
            });
        }
        input.sequences.validate(node_count, STAGE)?;
        if let Some(pos) = input.positive_relations {
            pos.validate(node_count, STAGE)?;
        }
        if let Some(sg) = input.support_groups {
            sg.validate(STAGE)?;
        }

        // A source group's first root identifies the sequence source. Subsequent
        // roots are required evidence and participate in exclusion.
        let mut source_groups = BTreeMap::new();
        let sg_count = input.support_groups.map_or(0, |sg| sg.group_count);
        if let Some(sg) = input.support_groups {
            for g in 0..sg.group_count {
                let gid = GroupId(g as u32);
                if sg.kind(gid) == Some(SupportKind::Source)
                    && let Some(&source) = sg.required_roots(gid).first()
                {
                    source_groups.entry(source).or_insert(gid);
                }
            }
        }
        let resolve_root_group = |root: RootId| -> VcpResult<GroupId> {
            if let Some(&gid) = source_groups.get(&root) {
                return Ok(gid);
            }
            if root.index() < sg_count {
                return Err(VcpError::InvalidInput {
                    stage: STAGE,
                    detail: format!(
                        "sequence source_root {} collides with SupportGroupView indices 0..{} without a matching Source group",
                        root.0, sg_count
                    ),
                });
            }
            Ok(GroupId(root.0))
        };

        let forward_gain = config.forward_gain;
        let reverse_gain = config.provenance_reverse_gain;
        let distance_decay = config.provenance_distance_decay;

        let mut seq_entries: Vec<((NodeId, NodeId, GroupId), f64)> = Vec::new();
        let mut seq_buf: Vec<(u32, NodeId)> = Vec::new();
        let pair_limit = ctx.limits.pair_work_limit;
        let mut pair_work = 0usize;

        for seq_idx in 0..input.sequences.sequence_count {
            ctx.check_cancelled(STAGE)?;
            let source_root = input.sequences.seq_source_roots[seq_idx];
            let (tags, positions) = input.sequences.sequence(SequenceId(seq_idx as u32));
            if tags.len() < 2 {
                continue;
            }
            let group_id = resolve_root_group(source_root)?;

            seq_buf.clear();
            seq_buf.extend(positions.iter().copied().zip(tags.iter().copied()));
            if !seq_buf.windows(2).all(|w| w[0] <= w[1]) {
                seq_buf.sort_unstable();
            }

            for i in 0..seq_buf.len() {
                let (left_pos, left_tag) = seq_buf[i];
                for &(right_pos, right_tag) in &seq_buf[i + 1..] {
                    if pair_work.is_multiple_of(1024) {
                        ctx.check_cancelled(STAGE)?;
                    }
                    pair_work += 1;
                    if pair_limit > 0 && pair_work > pair_limit {
                        return Err(VcpError::ResourceLimitExceeded {
                            stage: STAGE,
                            limit_name: "pair_work_limit",
                            actual: pair_work,
                            limit: pair_limit,
                        });
                    }
                    if left_tag == right_tag {
                        continue;
                    }
                    let ordered = left_pos > 0 && right_pos > 0;
                    let delta = right_pos.saturating_sub(left_pos).max(1) as f64;
                    let distance = if ordered {
                        (-distance_decay * (delta - 1.0)).exp()
                    } else {
                        1.0
                    };
                    let fwd_mass = forward_gain * distance;
                    let bwd_mass = if ordered {
                        reverse_gain * distance
                    } else {
                        forward_gain
                    };
                    if fwd_mass.is_finite() && fwd_mass > 0.0 {
                        seq_entries.push(((left_tag, right_tag, group_id), fwd_mass));
                    }
                    if bwd_mass.is_finite() && bwd_mass > 0.0 {
                        seq_entries.push(((right_tag, left_tag, group_id), bwd_mass));
                    }
                }
            }
        }

        // Stable sum over (source, target, group_id) for sequence occurrences.
        seq_entries.sort_by_key(|&(k, _)| k);
        let mut edge_map: BTreeMap<(NodeId, NodeId), BTreeMap<GroupId, f64>> = BTreeMap::new();
        let mut cur_key: Option<(NodeId, NodeId, GroupId)> = None;
        let mut cur_sum = 0.0_f64;
        for (key, mass) in seq_entries {
            if cur_key == Some(key) {
                cur_sum += mass;
            } else {
                if let Some((s, t, g)) = cur_key
                    && cur_sum > 0.0
                    && cur_sum.is_finite()
                {
                    *edge_map.entry((s, t)).or_default().entry(g).or_insert(0.0) += cur_sum;
                }
                cur_key = Some(key);
                cur_sum = 0.0 + mass;
            }
        }
        if let Some((s, t, g)) = cur_key
            && cur_sum > 0.0
            && cur_sum.is_finite()
        {
            *edge_map.entry((s, t)).or_default().entry(g).or_insert(0.0) += cur_sum;
        }

        // Deduplicate repeated positive relations for the same (source, target, group) via max.
        if let Some(pos) = input.positive_relations {
            let mut dedup_pos: BTreeMap<(NodeId, NodeId, GroupId), f64> = BTreeMap::new();
            for k in 0..pos.sources.len() {
                let s = pos.sources[k];
                let t = pos.targets[k];
                let g = pos.groups[k];
                let m = pos.masses[k];
                if s == t || !m.is_finite() || m <= 0.0 {
                    continue;
                }
                let entry = dedup_pos.entry((s, t, g)).or_insert(0.0);
                *entry = entry.max(m);
            }
            for ((s, t, g), m) in dedup_pos {
                *edge_map.entry((s, t)).or_default().entry(g).or_insert(0.0) += m;
            }
        }

        let mut edge_prov_offsets = Vec::with_capacity(nnz + 1);
        let mut prov_groups = Vec::new();
        let mut prov_masses = Vec::new();
        edge_prov_offsets.push(0);

        for u in 0..node_count {
            let src = NodeId(u as u32);
            let start = input.transport_row_offsets[u];
            let end = input.transport_row_offsets[u + 1];
            for &dst in &input.transport_targets[start..end] {
                if let Some(groups) = edge_map.get(&(src, dst)) {
                    for (&gid, &mass) in groups {
                        if mass > 0.0 && mass.is_finite() {
                            prov_groups.push(gid);
                            prov_masses.push(mass);
                        }
                    }
                }
                let offset = u32::try_from(prov_groups.len()).map_err(|_| {
                    VcpError::ResourceLimitExceeded {
                        stage: STAGE,
                        limit_name: "provenance_u32_offsets",
                        actual: prov_groups.len(),
                        limit: u32::MAX as usize,
                    }
                })?;
                edge_prov_offsets.push(offset);
            }
        }

        let out = ProvenanceData {
            edge_count: nnz,
            edge_prov_offsets,
            prov_groups,
            prov_masses,
            diagnostics: Diagnostics {
                truncated: false,
                truncation_reason: None,
                elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            },
        };
        out.as_view().validate(nnz, STAGE)?;
        Ok(out)
    }
}
