"""Capture conformance fixtures by running the pinned upstream tests."""

import argparse
import json
import os
from pathlib import Path
import subprocess

REVISION = "5adb12bb55e616e22bd0e0dbfdc13032c069941a"
MODULES = {
    "memo_nnls": "nnls",
    "memo_geometry": "geometry",
    "memo_pipeline": "hypernym",
    "rivermemo_topology_v3": "river",
    "result_deduplicator": "dedup",
}
SECTIONS = {"nnls", "geometry", "hypernym", "river", "wormhole", "dedup"}
ROOT = Path(__file__).resolve().parents[1]


def git_source(checkout, path):
    return subprocess.check_output(["git", "-C", str(checkout), "show", f"{REVISION}:{path}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="disposable VCPToolBox checkout at the reference revision")
    parser.add_argument("destination", type=Path, help="output fixture JSON")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise SystemExit(f"expected upstream {REVISION}, got {revision}")
    project = checkout / "rust-vexus-lite"
    sources = ROOT / "crates/nexus-vcp/tests/reference"
    changes = {
        project / "Cargo.toml": b"\n[workspace]\n",
        **{
            project / "src" / f"{module}.rs":
            f'\n#[cfg(test)]\n#[path = {json.dumps((sources / (name + ".rs")).as_posix())}]\n'
            .encode("utf-8") + b"mod nexus_reference;\n"
            for module, name in MODULES.items()
        },
    }
    originals = {}
    for path, addition in changes.items():
        relative = path.relative_to(checkout).as_posix()
        upstream = git_source(checkout, relative)
        current = path.read_bytes()
        if current.replace(b"\r\n", b"\n") not in (upstream, upstream + addition):
            raise SystemExit(f"preserve unexpected modifications in {relative}; use a clean temporary checkout")
        originals[path] = current
    try:
        for path, addition in changes.items():
            upstream = git_source(checkout, path.relative_to(checkout).as_posix())
            path.write_bytes(upstream + addition)
        env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "temp/upstream-target"))
        run = subprocess.run(
            ["cargo", "+1.97.1", "test", "--manifest-path", str(project / "Cargo.toml"),
             "--locked", "--lib", "nexus_golden_capture", "--", "--nocapture", "--test-threads=1"],
            env=env, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", check=False,
        )
        if run.returncode:
            print(run.stdout)
            raise SystemExit(run.returncode)
    finally:
        for path, current in originals.items():
            path.write_bytes(current)

    sections = {}
    marker = "NEXUS_GOLDEN:"
    for line in run.stdout.splitlines():
        if marker in line:
            key, value = json.loads(line.split(marker, 1)[1])
            if key in sections:
                raise SystemExit(f"duplicate upstream fixture section {key}")
            sections[key] = value
    if set(sections) != SECTIONS:
        raise SystemExit(f"expected {sorted(SECTIONS)}, got {sorted(sections)}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps({"revision": REVISION, **sections}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Captured " + ", ".join(f"{key} ({len(value)})" for key, value in sorted(sections.items())))


if __name__ == "__main__":
    main()
