# ADR 0001: Core Boundaries

## Decision

SIEL is split into small Rust crates:

- `siel-core`: domain types, errors, policy, calibrator.
- `siel-db`: SQLite schema, repository, per-item encryption.
- `siel-vector`: vector abstraction and backends.
- `siel-retrieval`: retrieval pipeline and deterministic embedder baseline.
- `siel-llm`: llama.cpp client, prompt builder, output validation.
- `siel-api`: local HTTP API.
- `siel-ffi`: minimal bridge surface for Flutter spike.

## Rationale

The UI and model runtime must not own the architecture. The engine must remain
usable through CLI/API tests even when Flutter, llama.cpp, or mobile tooling is
not installed.

