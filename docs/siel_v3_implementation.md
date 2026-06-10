# SIEL v3 Implementation Notes

This repository implements SIEL as a local-first knowledge engine where the
database is the source of truth and language models are bounded helpers.

## Current Milestone

- Rust workspace with separated crates.
- SQLite schema with encrypted payload table and audit log.
- Deterministic retrieval path: exact, FTS5, fuzzy, vector, RRF, policy.
- Vector abstraction with fake backend and SQLite baseline backend.
- LLM module with OpenAI-compatible llama.cpp client and strict JSON validation.
- Axum HTTP API exposing the v1 contract.
- Python lab skeleton for calibration and retrieval evaluation.
- Flutter shell that talks to the local HTTP API.

## Production Gates

Before enabling real user data:

1. Install Rust and run all cargo checks.
2. Replace the development master secret with OS keystore integration.
3. Validate SQLCipher if whole-database encryption is required.
4. Run the fts5-snowball spike on Windows, Android, and iOS.
5. Replace `HashEmbedder` with the selected ONNX/Qwen embedder.
6. Replace or extend `SqliteVecIndex` with the native sqlite-vec extension path.
7. Generate the full golden dataset and calibrate confidence profiles.

## Privacy Notes

`knowledge_payload` stores encrypted question and answer payloads. `knowledge_fts`
stores searchable plaintext for active items, so deletion must remove FTS rows and
operational procedures must cover WAL, backups, and snapshots. SQLCipher protects
the database at rest but does not replace per-item crypto-shredding.

