# SPDX-License-Identifier: Apache-2.0
"""Export raw semantic inputs and judged queries from a frozen Nous study."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import msgpack
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def entries(offsets, values, row):
    start, end = offsets[row:row + 2]
    return values[start:end]


def source_material(assets):
    facts = assets["facts"]
    keys = json.loads(facts["metadata_json"])
    source_ids = []
    by_content = {}
    sources = {}
    for index, (key, checksum) in enumerate(zip(keys["root_keys"], keys["root_checksums"])):
        identity = ("checksum", checksum) if checksum else ("key", key)
        source = by_content.setdefault(identity, index)
        source_ids.append(source)
        sources.setdefault(source, {"id": source, "paths": []})

    for row, slot in enumerate(facts["sequence_roots"]):
        if not facts["ordered"][row]:
            continue
        nodes = entries(facts["indptr"], facts["indices"], row)
        positions = entries(facts["indptr"], facts["positions"], row)
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError(f"unordered positions in declared source sequence {row}")
        path = list(nodes)
        paths = sources[source_ids[slot]]["paths"]
        if path not in paths:
            paths.append(path)

    explicit = assets["associations"]
    relations = []
    if explicit:
        fields = ("sources", "targets", "strengths", "confidences", "polarities", "directed", "edge_groups")
        for a, b, strength, confidence, polarity, directed, group in zip(*(explicit[key] for key in fields)):
            if polarity <= 0:
                continue
            roots = sorted({source_ids[row] for row in entries(explicit["group_root_indptr"], explicit["group_root_indices"], group)})
            if not roots:
                raise ValueError(f"positive relation has no source in group {group}")
            if len(roots) == 1:
                source = roots[0]
            else:
                source = len(source_ids) + group
                sources.setdefault(source, {"id": source, "parents": roots, "paths": []})
            support = strength * confidence
            relations.append({"from": a, "to": b, "support": support, "source": source})
            if not directed and a != b:
                relations.append({"from": b, "to": a, "support": support, "source": source})
    return sources, source_ids, relations


def subpath(path, sequence):
    selected = []
    step = 0
    for i, node in enumerate(path):
        if step < len(sequence) and node == sequence[step]:
            selected.append(i)
            step += 1
    return selected if step == len(sequence) else None


def candidate_material(assets, sources, source_ids):
    rows = assets["candidates"]
    metadata = json.loads(rows["metadata_json"])
    documents = {key: i for i, key in enumerate(sorted(set(metadata["documents"])))}
    candidates = []
    matched = 0
    supplied = 0
    for i, key in enumerate(metadata["documents"]):
        roots = sorted({source_ids[row] for row in entries(rows["root_indptr"], rows["root_indices"], i)})
        references = []
        start, end = rows["path_indptr"][i:i + 2]
        for j in range(start, end):
            supplied += 1
            fragment = entries(rows["path_step_indptr"], rows["path_indices"], j)
            matches = [
                {"source": source, "path": ordinal, "positions": positions}
                for source in roots
                for ordinal, source_path in enumerate(sources[source]["paths"])
                if (positions := subpath(source_path, fragment)) is not None and fragment
            ]
            references.extend(matches)
            matched += bool(matches)
        candidates.append({"id": i, "document": documents[key], "sources": roots, "paths": references})
    print(f"Resolved {matched}/{supplied} candidate paths against actual source sequences")
    return candidates


def study_queries(study, candidates, dimension):
    snapshot = read_json(study / "fine-snapshot.json")["snapshot"]["id"]
    families = read_json(study / "recall-family-map.json")["families"]
    database = (study / "fixed-instance/data/nous.sqlite3").resolve().as_uri()
    with sqlite3.connect(database + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "select row_index, group_key from wave_snapshot_candidates where snapshot_id=? order by row_index",
            (snapshot,),
        ).fetchall()
        groups = dict(connection.execute(
            "select key, context from wave_snapshot_groups where snapshot_id=?", (snapshot,),
        ))
    if [row for row, _ in rows] != list(range(len(candidates))):
        raise ValueError("the component does not cover the study's candidate rows")
    group_documents = {(key, row["document"]) for (_, key), row in zip(rows, candidates)}
    if len(group_documents) != len(groups) or len({doc for _, doc in group_documents}) != len(groups):
        raise ValueError("candidate documents and judged snapshot groups differ")
    revisions = {key: (json.loads(context).get("revision_ref") or {}).get("revision_id") for key, context in groups.items()}
    values = []
    queries = []
    embedding_space = None
    producer = None
    for line in (study / "fixed-dense-groups.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["split"] not in {"tune", "development"}:
            continue
        question = entry["question"]
        text = question["question"]
        prefix = study / "query-vectors" / question["id"]
        meta = read_json(prefix.with_suffix(".json"))
        if embedding_space is None:
            embedding_space = meta["space"]
            producer = meta["producer"]["returned_model_name"]
        if meta["space"] != embedding_space or meta["input_sha256"] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError(f"cached vector does not match {question['id']}")
        vector = np.load(prefix.with_suffix(".npy"), allow_pickle=False)
        if vector.shape != (dimension,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError(f"invalid query vector {question['id']}")
        values.append(vector)
        family_set = set(entry["families"])
        relevant = sorted({doc for group, doc in group_documents if families.get(revisions[group], {}).get("family") in family_set})
        if not relevant:
            raise ValueError(f"no relevant document for {question['id']}")
        queries.append({"split": entry["split"], "text": text, "relevant_documents": relevant})
    return values, queries, producer


def export(component, study, destination):
    assets = msgpack.unpackb((component / "assets.msgpack").read_bytes(), strict_map_key=False, use_list=False)
    candidate_vectors = assets["candidates"]["vectors"]
    concept_vectors = assets["concepts"]["vectors"]
    dim = candidate_vectors["cols"]
    if concept_vectors["cols"] != dim:
        raise ValueError("concept and candidate vector dimensions differ")
    names = json.loads(assets["concepts"]["metadata_json"])
    concepts = [
        {"id": i, "label": label, "aliases": aliases}
        for i, (label, aliases) in enumerate(zip(names["labels"], names["aliases"]))
    ]
    sources, source_ids, relations = source_material(assets)
    candidates = candidate_material(assets, sources, source_ids)
    values, queries, producer = study_queries(study, candidates, dim)
    fixture = {
        "name": "Nous frozen study", "producer": producer, "dimension": dim,
        "concepts": concepts, "sources": list(sources.values()), "relations": relations,
        "candidates": candidates, "queries": queries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for file, rows in [("concepts", concept_vectors["values"]), ("candidates", candidate_vectors["values"]), ("queries", values)]:
        np.asarray(rows, dtype="<f4").tofile(destination / f"{file}.f32")
    (destination / "fixture.json").write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    print(f"{len(queries)} queries, {len(candidates)} candidates, {len(concepts)} concepts, {len(sources)} sources, {dim} dimensions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.component, args.study, args.destination)
