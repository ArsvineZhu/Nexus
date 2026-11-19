set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

default:
    @just --list

fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

test *args:
    cargo test --workspace --all-features --locked {{args}}

math-only:
    cargo test -p nexus-math --all-features --locked

check: lint test math-only
    cargo fmt --all -- --check

index:
    codegraph init --yes
