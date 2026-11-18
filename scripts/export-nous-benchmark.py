# SPDX-License-Identifier: Apache-2.0
"""Export the frozen Nous study as dense inputs for Nexus benchmarks."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import msgpack
import numpy as np


def unpack(path):
    return msgpack.unpackb(path.read_bytes(), strict_map_key=False, use_list=False)


def flatten(rows):
    values = []
    offsets = [0]
    for row in rows:
        values.extend(row)
        offsets.append(len(values))
    return values, offsets


def export(component, study, destination):
    assets = unpack(component / "assets.msgpack")
    math = unpack(component / "math.msgpack")
    candidates = assets["candidates"]
    concepts = assets["concepts"]
    matrix = candidates["vectors"]
    tag_matrix = concepts["vectors"]
    dim = matrix["cols"]
    if tag_matrix["cols"] != dim:
        raise ValueError("concept and candidate vector dimensions differ")
    graph = math["graph"]
    node_ids = graph["node_ids"]
    slots = {node: index for index, node in enumerate(node_ids)}
    if len(slots) != tag_matrix["rows"]:
        raise ValueError("frozen concept rows do not cover the graph")
    concept_meta = json.loads(concepts["metadata_json"])
    tag_rows = np.asarray(tag_matrix["values"], dtype="<f4").reshape(-1, dim)
    concept_rows = [node - 1 for node in node_ids]
    labels = [concept_meta["labels"][row] for row in concept_rows]
    aliases, alias_offsets = flatten(concept_meta["aliases"][row] for row in concept_rows)
    candidate_meta = json.loads(candidates["metadata_json"])
    docs = {key: index for index, key in enumerate(sorted(set(candidate_meta["documents"])))}
    root_count = len(json.loads(assets["facts"]["metadata_json"])["root_keys"])
    group_ids = list(range(1, root_count + 1)) + sorted(graph["support_groups"])
    group_slots = {key: index for index, key in enumerate(group_ids)}
    groups = [[root] for root in range(root_count)]
    kinds = ["Source"] * root_count
    for key in group_ids[root_count:]:
        group = graph["support_groups"][key]
        groups.append([root - 1 for root in group["required_roots"]])
        kinds.append(group["kind"].capitalize())
    group_roots, group_offsets = flatten(groups)
    provenance_groups = []
    provenance_masses = []
    provenance_offsets = [0]
    sorted_indices = []
    for row, source in enumerate(node_ids):
        start, end = graph["row_offsets"][row:row + 2]
        sorted_indices.extend(sorted(range(start, end), key=lambda edge: graph["targets"][edge]))
        for edge in range(start, end):
            target = node_ids[graph["targets"][edge]]
            for group, mass in sorted(graph["provenance"].get((source, target), ())):
                provenance_groups.append(group_slots[group])
                provenance_masses.append(mass)
            provenance_offsets.append(len(provenance_groups))
    diagnostics = {"truncated": False, "truncation_reason": None, "elapsed_ms": 0.0}
    basis = None
    epa = json.loads(math["epa_json"]) if math["epa_json"] else {}
    if epa.get("basis"):
        decode = lambda encoded: np.frombuffer(base64.b64decode(encoded), dtype="<f4").tolist()
        axes, _ = flatten(decode(axis) for axis in epa["basis"])
        basis = {
            "dim": dim, "k_sem": epa["kSem"], "basis_count": len(epa["basis"]),
            "mean": decode(epa["mean"]), "basis": axes, "coverage": epa["coverage"],
            "diagnostics": diagnostics,
        }
    rows = []
    query_texts = []
    space = json.loads((study / "query-vectors" / "Q0001.json").read_text(encoding="utf-8"))["space"]
    for line in (study / "fixed-dense-groups.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["split"] not in {"tune", "development"}:
            continue
        question = entry["question"]
        text, key = question["question"], question["id"]
        prefix = study / "query-vectors" / key
        meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        if meta["space"] != space or meta["input_sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError(f"cached query differs from study: {key}")
        vector = np.load(prefix.with_suffix(".npy"), allow_pickle=False)
        if vector.shape != (dim,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid query vector: {key}")
        rows.append(vector)
        query_texts.append(text)
    path_positions = candidates["path_positions"]
    offsets = candidates["path_step_indptr"]
    if any(not 0 <= position <= 2**32 - 1 for position in path_positions):
        raise ValueError("frozen path position does not fit a Nexus slot")
    fixture = {
        "dim": dim, "docs": [docs[key] for key in candidate_meta["documents"]],
        "candidate_root_offsets": candidates["root_indptr"], "candidate_roots": candidates["root_indices"],
        "candidate_path_offsets": candidates["path_indptr"], "path_tag_offsets": offsets,
        "path_tags": [slots[tag + 1] for tag in candidates["path_indices"]], "path_positions": path_positions,
        "labels": labels, "alias_offsets": alias_offsets, "aliases": aliases,
        "graph": {
            "node_count": len(node_ids), "row_offsets": graph["row_offsets"], "targets": graph["targets"],
            "weights": graph["weights"], "is_static_wormhole": graph["edge_is_wormhole"],
            "inbound": [graph["inbound"].get(node, 0.0) for node in node_ids], "max_inbound": graph["max_inbound"],
            "anchor_gains": graph["node_anchor_gain"], "target_sorted_indices": sorted_indices,
            "diagnostics": diagnostics,
        },
        "provenance_offsets": provenance_offsets, "provenance_groups": provenance_groups,
        "provenance_masses": provenance_masses, "group_offsets": group_offsets, "group_roots": group_roots,
        "group_kinds": kinds, "basis": basis, "query_texts": query_texts,
    }
    destination.mkdir(parents=True, exist_ok=True)
    np.asarray(matrix["values"], dtype="<f4").tofile(destination / "candidates.f32")
    tag_rows[concept_rows].tofile(destination / "tags.f32")
    np.asarray(rows, dtype="<f4").tofile(destination / "queries.f32")
    (destination / "fixture.json").write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} queries, {matrix['rows']} × {dim} candidate matrix, {len(node_ids)} graph nodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", type=Path, help="frozen component with assets.msgpack and math.msgpack")
    parser.add_argument("study", type=Path, help="directory containing the study and cached query embeddings")
    parser.add_argument("destination", type=Path, help="local output directory for the exported fixture")
    args = parser.parse_args()
    export(args.component, args.study, args.destination)
