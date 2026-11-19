// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::{FusedObservation, invalid, numeric_failure};
use nexus_core::kernel::{csr_spmv_dual, relax_field};
use nexus_core::{CsrGraphView, ExecutionContext, NodeId, Operator, VcpResult};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DualFieldConfig {
    pub local_alpha: f64,
    pub transfer_alpha: f64,
    pub max_iterations: usize,
    pub local_tolerance: f64,
    pub transfer_tolerance: f64,
    pub local_mass_ratio: f64,
    pub transfer_mass_ratio: f64,
}

impl Default for DualFieldConfig {
    fn default() -> Self {
        Self {
            local_alpha: 0.15,
            transfer_alpha: 0.55,
            max_iterations: 80,
            local_tolerance: 1e-9,
            transfer_tolerance: 1e-9,
            local_mass_ratio: 0.8,
            transfer_mass_ratio: 0.9,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DualFieldSolveInput<'asset, 'query> {
    pub graph: CsrGraphView<'asset>,
    pub fused: &'query FusedObservation,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DualFields {
    pub local_field: Vec<f64>,
    pub transfer_field: Vec<f64>,
    pub local_domain: Vec<NodeId>,
    pub transfer_domain: Vec<NodeId>,
    pub iterations: usize,
    pub local_converged: bool,
    pub transfer_converged: bool,
    pub local_residual: f64,
    pub transfer_residual: f64,
}

fn domain(field: &[f64], ratio: f64, stage: &'static str) -> VcpResult<Vec<NodeId>> {
    if ratio == 0.0 {
        return Ok(Vec::new());
    }
    let mut ranked = Vec::new();
    for (index, &mass) in field.iter().enumerate().filter(|&(_, mass)| *mass > 0.0) {
        ranked.push((NodeId::from_index(index, stage)?, mass));
    }
    let target = field.iter().sum::<f64>() * ratio;
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut mass = 0.0;
    let mut result = Vec::new();
    for (node, value) in ranked {
        result.push(node);
        mass += value;
        if mass >= target {
            break;
        }
    }
    Ok(result)
}

pub struct DualFieldSolveOperator;

impl Operator for DualFieldSolveOperator {
    type Config = DualFieldConfig;
    type Input<'asset, 'query> = DualFieldSolveInput<'asset, 'query>;
    type Output = DualFields;

    fn name(&self) -> &'static str {
        "field.dual_solve"
    }

    fn execute(
        &self,
        ctx: &ExecutionContext<'_>,
        config: &Self::Config,
        input: Self::Input<'_, '_>,
    ) -> VcpResult<Self::Output> {
        let stage = self.name();
        ctx.check_cancelled(stage)?;
        if ![config.local_alpha, config.transfer_alpha]
            .into_iter()
            .all(|v| (0.0..1.0).contains(&v))
            || ![config.local_mass_ratio, config.transfer_mass_ratio]
                .into_iter()
                .all(|v| (0.0..=1.0).contains(&v))
            || ![config.local_tolerance, config.transfer_tolerance]
                .into_iter()
                .all(|v| v.is_finite() && v > 0.0)
            || config.max_iterations == 0
        {
            return Err(invalid(stage, "invalid dual-field solve configuration"));
        }

        let n = input.graph.node_count;
        let mut source_buffer = vec![0.0; n];
        let source = source_buffer.as_mut_slice();
        for &(node, mass) in &input.fused.source_field {
            if node.index() >= n || !mass.is_finite() || mass < 0.0 {
                return Err(invalid(stage, "source field node or mass is invalid"));
            }
            source[node.index()] += mass;
        }
        let sum = source.iter().sum::<f64>();
        if !sum.is_finite() {
            return Err(numeric_failure(stage, "source mass overflowed"));
        }
        if sum == 0.0 {
            return Ok(DualFields {
                local_field: vec![0.0; n],
                transfer_field: vec![0.0; n],
                local_converged: true,
                transfer_converged: true,
                ..DualFields::default()
            });
        }

        for value in source.iter_mut() {
            *value /= sum;
        }
        let mut local = source.to_vec();
        let mut transfer = source.to_vec();
        let mut next_local_buffer = vec![0.0; n];
        let next_local = next_local_buffer.as_mut_slice();
        let mut next_transfer_buffer = vec![0.0; n];
        let next_transfer = next_transfer_buffer.as_mut_slice();
        let mut propagated_local_buffer = vec![0.0; n];
        let propagated_local = propagated_local_buffer.as_mut_slice();
        let mut propagated_transfer_buffer = vec![0.0; n];
        let propagated_transfer = propagated_transfer_buffer.as_mut_slice();
        let mut report = DualFields::default();
        // Also evaluate the residual of the final returned iterate when the work limit is hit.
        for iteration in 0..=config.max_iterations {
            ctx.check_cancelled(stage)?;
            csr_spmv_dual(
                &input.graph,
                &local,
                &transfer,
                propagated_local,
                propagated_transfer,
                ctx.parallel,
            )?;
            report.iterations = iteration;
            report.local_residual = relax_field(
                source,
                propagated_local,
                config.local_alpha,
                &local,
                next_local,
            );
            report.transfer_residual = relax_field(
                source,
                propagated_transfer,
                config.transfer_alpha,
                &transfer,
                next_transfer,
            );
            if !report.local_residual.is_finite() || !report.transfer_residual.is_finite() {
                return Err(numeric_failure(stage, "dual-field iteration diverged"));
            }
            report.local_converged = report.local_residual < config.local_tolerance;
            report.transfer_converged = report.transfer_residual < config.transfer_tolerance;
            if (report.local_converged && report.transfer_converged)
                || iteration == config.max_iterations
            {
                break;
            }
            if !report.local_converged {
                local.copy_from_slice(next_local);
            }
            if !report.transfer_converged {
                transfer.copy_from_slice(next_transfer);
            }
        }
        report.local_domain = domain(&local, config.local_mass_ratio, stage)?;
        report.transfer_domain = domain(&transfer, config.transfer_mass_ratio, stage)?;
        report.local_field = local;
        report.transfer_field = transfer;
        Ok(report)
    }
}
