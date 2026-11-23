// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use super::FieldSolveOptions;
use crate::{ConceptId, Error, Result, RunOptions, config, input};
use nexus_math::kernel::{csr_spmv_dual, relax_field};
use nexus_math::{TransportGraph, TransportGraphView};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldSolveConfig {
    pub local_alpha: f64,
    pub transfer_alpha: f64,
    pub max_iterations: usize,
    pub local_tolerance: f64,
    pub transfer_tolerance: f64,
    pub local_mass_ratio: f64,
    pub transfer_mass_ratio: f64,
}

impl Default for FieldSolveConfig {
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldSolution {
    pub values: Vec<(ConceptId, f64)>,
    pub domain: Vec<ConceptId>,
    pub iterations: usize,
    pub residual_l1: f64,
    pub error_bound_l1: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DualFields {
    pub local: FieldSolution,
    pub transfer: FieldSolution,
}

fn domain(values: &[(ConceptId, f64)], ratio: f64) -> Vec<ConceptId> {
    if ratio == 0.0 {
        return Vec::new();
    }
    let target = values.iter().map(|(_, mass)| mass).sum::<f64>() * ratio;
    let mut ranked: Vec<_> = values
        .iter()
        .copied()
        .filter(|&(_, mass)| mass > 0.0)
        .collect();
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
    result
}

pub fn solve_fields(
    graph: &TransportGraph,
    source: &[(ConceptId, f64)],
    options: &FieldSolveOptions,
) -> Result<DualFields> {
    solve(graph.view(), graph, source, &options.config, &options.run)
}

struct FieldIteration {
    alpha: f64,
    tolerance: f64,
    mass_ratio: f64,
    current: Vec<f64>,
    next: Vec<f64>,
    propagated: Vec<f64>,
    iterations: usize,
    residual: f64,
}

impl FieldIteration {
    fn new(source: &[f64], alpha: f64, tolerance: f64, mass_ratio: f64) -> Self {
        Self {
            alpha,
            tolerance,
            mass_ratio,
            current: source.to_vec(),
            next: vec![0.0; source.len()],
            propagated: vec![0.0; source.len()],
            iterations: 0,
            residual: 0.0,
        }
    }

    fn measure(&mut self, source: &[f64]) -> Result<()> {
        self.residual = relax_field(
            source,
            &self.propagated,
            self.alpha,
            &self.current,
            &mut self.next,
        );
        if !self.residual.is_finite() {
            return Err(Error::numeric(
                "solve_fields",
                "fixed-point residual overflowed",
            ));
        }
        Ok(())
    }

    fn done(&self) -> bool {
        self.residual <= self.tolerance
    }

    fn step(&mut self) {
        if !self.done() {
            self.iterations += 1;
            std::mem::swap(&mut self.current, &mut self.next);
        }
    }

    fn finish(self, ids: &[ConceptId], max_row_sum: f64) -> FieldSolution {
        let values: Vec<_> = ids.iter().copied().zip(self.current).collect();
        FieldSolution {
            domain: domain(&values, self.mass_ratio),
            values,
            iterations: self.iterations,
            residual_l1: self.residual,
            error_bound_l1: self.residual / (1.0 - self.alpha * max_row_sum),
        }
    }
}

pub(super) fn solve(
    graph: TransportGraphView<'_>,
    transport: &TransportGraph,
    source: &[(ConceptId, f64)],
    cfg: &FieldSolveConfig,
    run: &RunOptions,
) -> Result<DualFields> {
    run.check_cancelled()?;
    if ![cfg.local_alpha, cfg.transfer_alpha]
        .into_iter()
        .all(|v| (0.0..1.0).contains(&v))
        || ![cfg.local_mass_ratio, cfg.transfer_mass_ratio]
            .into_iter()
            .all(|v| (0.0..=1.0).contains(&v))
        || ![cfg.local_tolerance, cfg.transfer_tolerance]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
        || cfg.max_iterations == 0
    {
        return Err(config(
            "fields.solve",
            "invalid field damping, mass ratio, tolerance or iteration count",
        ));
    }
    let mut distribution = vec![0.0; graph.node_count];
    for &(concept, mass) in source {
        run.check_cancelled()?;
        let node = transport
            .slot(concept)
            .ok_or_else(|| input("fields.source", Some(concept), "unknown concept"))?;
        if !mass.is_finite() || mass < 0.0 {
            return Err(input(
                "fields.source",
                Some(concept),
                "mass must be finite and non-negative",
            ));
        }
        distribution[node.index()] += mass;
    }
    let total = distribution.iter().sum::<f64>();
    if !total.is_finite() {
        return Err(Error::numeric("solve_fields", "source mass overflowed"));
    }

    let ids = transport.nodes();
    if total == 0.0 {
        let zero = FieldSolution {
            values: ids.iter().map(|&concept| (concept, 0.0)).collect(),
            ..Default::default()
        };
        return Ok(DualFields {
            local: zero.clone(),
            transfer: zero,
        });
    }
    for mass in &mut distribution {
        *mass /= total;
    }
    let mut local = FieldIteration::new(
        &distribution,
        cfg.local_alpha,
        cfg.local_tolerance,
        cfg.local_mass_ratio,
    );
    let mut transfer = FieldIteration::new(
        &distribution,
        cfg.transfer_alpha,
        cfg.transfer_tolerance,
        cfg.transfer_mass_ratio,
    );

    // The final measurement belongs to the returned iterate, including when either budget expires.
    for iteration in 0..=cfg.max_iterations {
        run.check_cancelled()?;
        csr_spmv_dual(
            &graph,
            &local.current,
            &transfer.current,
            &mut local.propagated,
            &mut transfer.propagated,
            run,
        )?;
        local.measure(&distribution)?;
        transfer.measure(&distribution)?;
        if (local.done() && transfer.done()) || iteration == cfg.max_iterations {
            break;
        }
        local.step();
        transfer.step();
    }
    let mut max_row_sum = 0.0_f64;
    for &concept in ids {
        run.check_cancelled()?;
        let node = transport.slot(concept).expect("transport node slot exists");
        max_row_sum = max_row_sum.max(graph.row_statistics(node).0);
    }
    Ok(DualFields {
        local: local.finish(ids, max_row_sum),
        transfer: transfer.finish(ids, max_row_sum),
    })
}
