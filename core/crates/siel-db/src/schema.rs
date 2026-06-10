pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS source (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    uri             TEXT,
    author          TEXT,
    license         TEXT,
    acquired_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS confidence_profile (
    profile_id          TEXT PRIMARY KEY,
    domain              TEXT,
    threshold_high      REAL NOT NULL DEFAULT 0.82,
    threshold_low       REAL NOT NULL DEFAULT 0.62,
    calibration_method  TEXT NOT NULL DEFAULT 'fixed',
    calibration_json    TEXT,
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO confidence_profile(profile_id, calibration_method)
VALUES ('default', 'fixed');

CREATE TABLE IF NOT EXISTS encryption_key (
    id              TEXT PRIMARY KEY,
    wrapped_key     BLOB NOT NULL,
    destroyed_at    DATETIME,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_item (
    id                    TEXT PRIMARY KEY,
    lang                  TEXT NOT NULL DEFAULT 'it',
    domain                TEXT,
    status                TEXT NOT NULL DEFAULT 'approved',
    reliability           REAL NOT NULL DEFAULT 1.0,
    confidence_profile_id TEXT NOT NULL DEFAULT 'default'
        REFERENCES confidence_profile(profile_id),
    valid_until           DATETIME,
    freshness_policy      TEXT DEFAULT 'static',
    source_id             TEXT REFERENCES source(id),
    version               INTEGER NOT NULL DEFAULT 1,
    checksum_sha256       TEXT NOT NULL,
    created_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at            DATETIME,
    encryption_key_id     TEXT REFERENCES encryption_key(id)
);

CREATE TABLE IF NOT EXISTS knowledge_payload (
    item_id          TEXT PRIMARY KEY REFERENCES knowledge_item(id),
    question_cipher  BLOB NOT NULL,
    answer_cipher    BLOB NOT NULL,
    nonce            BLOB NOT NULL,
    aad              BLOB NOT NULL,
    payload_version  INTEGER NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    item_id UNINDEXED,
    question,
    answer,
    tokenize = 'unicode61 remove_diacritics 1'
);

CREATE TABLE IF NOT EXISTS question_variant (
    id              TEXT PRIMARY KEY,
    item_id         TEXT NOT NULL REFERENCES knowledge_item(id),
    variant_text    TEXT NOT NULL,
    origin          TEXT NOT NULL DEFAULT 'human',
    deleted_at      DATETIME,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS answer_claim (
    id              TEXT PRIMARY KEY,
    item_id         TEXT NOT NULL REFERENCES knowledge_item(id),
    claim_text      TEXT NOT NULL,
    deleted_at      DATETIME,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embedding_model_registry (
    model_id        TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    dimensions      INTEGER NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'available',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deprecated_at   DATETIME,
    PRIMARY KEY (model_id, model_version)
);

CREATE TABLE IF NOT EXISTS current_embedding_model (
    model_id        TEXT PRIMARY KEY,
    model_version   TEXT NOT NULL,
    FOREIGN KEY (model_id, model_version)
        REFERENCES embedding_model_registry(model_id, model_version)
);

CREATE TABLE IF NOT EXISTS embedding (
    id              TEXT PRIMARY KEY,
    item_id         TEXT NOT NULL REFERENCES knowledge_item(id),
    model_id        TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    dimensions      INTEGER NOT NULL,
    quantization    TEXT,
    vector          BLOB NOT NULL,
    indexed_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id, model_version)
        REFERENCES embedding_model_registry(model_id, model_version)
);

CREATE VIEW IF NOT EXISTS stale_embeddings AS
    SELECT e.item_id, e.model_id, e.model_version
    FROM embedding e
    JOIN current_embedding_model c USING (model_id)
    WHERE e.model_version != c.model_version;

CREATE TABLE IF NOT EXISTS memory_candidate (
    id                  TEXT PRIMARY KEY,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    lang                TEXT NOT NULL DEFAULT 'it',
    domain              TEXT,
    source_id           TEXT REFERENCES source(id),
    status              TEXT NOT NULL DEFAULT 'pending',
    promoted_item_id    TEXT REFERENCES knowledge_item(id),
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at          DATETIME
);

CREATE TABLE IF NOT EXISTS contradiction (
    id                  TEXT PRIMARY KEY,
    candidate_id        TEXT REFERENCES memory_candidate(id),
    item_id             TEXT REFERENCES knowledge_item(id),
    description         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'open',
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_event (
    id              TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    item_id         TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    snapshot_zstd   BLOB,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_item_status
    ON knowledge_item(status, deleted_at);
CREATE INDEX IF NOT EXISTS idx_embedding_model
    ON embedding(model_id, model_version);
CREATE INDEX IF NOT EXISTS idx_audit_item
    ON audit_event(item_id, created_at);
"#;

