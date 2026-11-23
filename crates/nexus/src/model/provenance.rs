// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use nexus_math::NodeId;

use super::{ConceptView, Relation, Source, SourceOrigin};
use crate::{Result, RunOptions, SourceId, input, numeric};

pub(super) type Edge = (NodeId, NodeId);
pub(super) type SourceSlot = usize;

#[derive(Clone)]
pub(crate) struct SourcePath {
    pub(crate) original: Vec<NodeId>,
    pub(crate) unique: Vec<(usize, NodeId)>,
}

pub(crate) struct SourceRecord {
    pub(crate) id: SourceId,
    pub(crate) paths: Vec<SourcePath>,
    pub(crate) nodes: BTreeSet<NodeId>,
    pub(crate) roots: Option<BTreeSet<SourceId>>,
    pub(super) relations: BTreeMap<Edge, f64>,
}

pub(crate) struct SourceBook {
    pub(crate) records: Vec<SourceRecord>,
    slots: HashMap<SourceId, SourceSlot>,
    pub(crate) by_node: Vec<Vec<SourceSlot>>,
}

impl SourceBook {
    pub(super) fn new(
        concepts: ConceptView<'_>,
        sources: &[Source],
        relations: &[Relation],
        run: &RunOptions,
    ) -> Result<Self> {
        let mut slots = HashMap::with_capacity(sources.len());
        for (i, source) in sources.iter().enumerate() {
            if slots.insert(source.id, i).is_some() {
                return Err(input("source.id", Some(source.id), "duplicate source ID"));
            }
        }
        let mut parents = vec![Vec::new(); sources.len()];
        let mut children = vec![Vec::new(); sources.len()];
        for (i, source) in sources.iter().enumerate() {
            run.check_cancelled()?;
            if let SourceOrigin::DerivedFrom(ids) = &source.origin {
                if ids.is_empty() {
                    return Err(input(
                        "source.origin",
                        Some(source.id),
                        "derived source needs a parent",
                    ));
                }
                let mut seen = HashSet::new();
                for &id in ids {
                    let parent = *slots.get(&id).ok_or_else(|| {
                        input(
                            "source.origin",
                            Some(source.id),
                            format!("unknown parent {id}"),
                        )
                    })?;
                    if parent == i {
                        return Err(input(
                            "source.origin",
                            Some(source.id),
                            "source cannot be its own parent",
                        ));
                    }
                    if seen.insert(parent) {
                        parents[i].push(parent);
                        children[parent].push(i);
                    }
                }
            }
        }
        let mut pending: Vec<_> = parents.iter().map(Vec::len).collect();
        let mut ready: VecDeque<_> = (0..sources.len()).filter(|&i| pending[i] == 0).collect();
        let mut roots: Vec<Option<BTreeSet<SourceId>>> = vec![None; sources.len()];
        while let Some(i) = ready.pop_front() {
            run.check_cancelled()?;
            roots[i] = match sources[i].origin {
                SourceOrigin::Independent => Some(BTreeSet::from([sources[i].id])),
                SourceOrigin::Unknown => None,
                SourceOrigin::DerivedFrom(_) => {
                    parents[i]
                        .iter()
                        .try_fold(BTreeSet::new(), |mut out, &parent| {
                            out.extend(roots[parent].as_ref()?);
                            Some(out)
                        })
                }
            };
            for &child in &children[i] {
                pending[child] -= 1;
                if pending[child] == 0 {
                    ready.push_back(child);
                }
            }
        }
        if let Some(i) = pending.iter().position(|&count| count > 0) {
            return Err(input(
                "source.origin",
                Some(sources[i].id),
                "cycle in source derivation",
            ));
        }

        let mut records = Vec::with_capacity(sources.len());
        for (source, roots) in sources.iter().zip(roots) {
            run.check_cancelled()?;
            let mut paths = Vec::with_capacity(source.paths.len());
            let mut nodes = BTreeSet::new();
            for path in &source.paths {
                let original: Vec<_> = path
                    .iter()
                    .map(|&id| {
                        concepts.node(id).map_err(|_| {
                            input(
                                "source.paths",
                                Some(source.id),
                                format!("unknown concept {id}"),
                            )
                        })
                    })
                    .collect::<Result<_>>()?;
                nodes.extend(original.iter().copied());
                let mut seen = HashSet::new();
                let unique = original
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, node)| seen.insert(node))
                    .collect();
                paths.push(SourcePath { original, unique });
            }
            records.push(SourceRecord {
                id: source.id,
                paths,
                roots,
                nodes,
                relations: BTreeMap::new(),
            });
        }
        for relation in relations {
            run.check_cancelled()?;
            let source = *slots.get(&relation.source).ok_or_else(|| {
                input(
                    "relation.source",
                    Some(relation.source),
                    "unknown source ID",
                )
            })?;
            if !relation.support.is_finite() || relation.support < 0.0 {
                return Err(input(
                    "relation.support",
                    Some(relation.source),
                    "expected finite nonnegative support",
                ));
            }
            let from = concepts
                .node(relation.from)
                .map_err(|_| input("relation.from", Some(relation.from), "unknown concept ID"))?;
            let to = concepts
                .node(relation.to)
                .map_err(|_| input("relation.to", Some(relation.to), "unknown concept ID"))?;
            if relation.support > 0.0 {
                let record = &mut records[source];
                record.nodes.extend([from, to]);
                let mass = record.relations.entry((from, to)).or_default();
                *mass = mass.max(relation.support);
            }
        }
        let mut by_node = vec![Vec::new(); concepts.ids().len()];
        for (slot, record) in records.iter().enumerate() {
            for node in &record.nodes {
                by_node[node.index()].push(slot);
            }
        }
        for posting in &mut by_node {
            posting.sort_unstable_by_key(|&slot| records[slot].id);
        }
        Ok(Self {
            records,
            slots,
            by_node,
        })
    }

    pub(crate) fn get(&self, id: SourceId) -> Option<&SourceRecord> {
        self.slots.get(&id).map(|&slot| &self.records[slot])
    }

    pub(super) fn local_neighbors(
        &self,
        count: usize,
        decay: f64,
        run: &RunOptions,
    ) -> Result<Vec<BTreeMap<usize, f64>>> {
        let mut adjacency = vec![BTreeMap::new(); count];
        for source in &self.records {
            let mut mass = source.relations.clone();
            for path in &source.paths {
                for (i, &(_, a)) in path.unique.iter().enumerate() {
                    run.check_cancelled()?;
                    for (distance, &(_, b)) in path.unique[i + 1..].iter().enumerate() {
                        let weight = (-decay * distance as f64).exp();
                        for edge in [(a, b), (b, a)] {
                            let current = mass.entry(edge).or_default();
                            *current = current.max(weight);
                        }
                    }
                }
            }
            for ((a, b), weight) in mass {
                if a != b {
                    let value = adjacency[a.index()].entry(b.index()).or_insert(0.0);
                    *value += weight;
                    if !value.is_finite() {
                        return Err(numeric("local_cone", "neighbor support overflowed"));
                    }
                }
            }
        }
        Ok(adjacency)
    }
}

#[derive(Default)]
pub(crate) struct EdgeSupport {
    pub(crate) roots: BTreeMap<SourceId, f64>,
    pub(crate) known: f64,
    pub(crate) total: f64,
}

impl EdgeSupport {
    pub(crate) fn factor(&self, owner: Option<&BTreeSet<SourceId>>) -> f64 {
        let Some(owner) = owner.filter(|_| self.total > 0.0) else {
            return 0.15;
        };
        let own = owner
            .iter()
            .filter_map(|root| self.roots.get(root))
            .sum::<f64>();
        ((self.known - own) / self.total).max(0.15)
    }
}
