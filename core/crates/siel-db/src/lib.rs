mod crypto;
mod schema;

use std::path::Path;

pub use crypto::CryptoBox;
use siel_core::{
    new_id, ApproveResponse, SielError, Evidence, KnowledgeItem, KnowledgePayload,
    KnowledgeStatus, Result, TeachProposalKind, TeachRequest, TeachResponse,
};
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

pub struct SielDb {
    path: String,
    crypto: CryptoBox,
}

impl SielDb {
    pub fn open(path: impl AsRef<Path>, master_secret: &[u8]) -> Result<Self> {
        let db = Self {
            path: path.as_ref().to_string_lossy().to_string(),
            crypto: CryptoBox::new(master_secret),
        };
        db.with_conn(|conn| {
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "foreign_keys", "ON")?;
            conn.execute_batch(schema::SCHEMA)
        })?;
        Ok(db)
    }

    pub fn memory(master_secret: &[u8]) -> Result<Self> {
        let path = std::env::temp_dir().join(format!("siel_test_{}.db", new_id("db")));
        let db = Self {
            path: path.to_string_lossy().to_string(),
            crypto: CryptoBox::new(master_secret),
        };
        db.with_conn(|conn| {
            conn.pragma_update(None, "foreign_keys", "ON")?;
            conn.execute_batch(schema::SCHEMA)
        })?;
        Ok(db)
    }

    pub fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Result<T> {
        let conn = Connection::open(&self.path).map_err(|err| SielError::Storage(err.to_string()))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|err| SielError::Storage(err.to_string()))?;
        f(&conn).map_err(|err| SielError::Storage(err.to_string()))
    }

    pub fn create_source(&self, title: &str, uri: Option<&str>) -> Result<String> {
        let id = new_id("src");
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO source(id, title, uri) VALUES (?1, ?2, ?3)",
                params![id, title, uri],
            )
        })?;
        Ok(id)
    }

    pub fn create_item(
        &self,
        question: &str,
        answer: &str,
        lang: &str,
        domain: Option<&str>,
        source_id: Option<&str>,
    ) -> Result<String> {
        let id = new_id("item");
        let key_id = new_id("key");
        let item_key = self.crypto.generate_item_key();
        let wrapped_key = self.crypto.wrap_key(&item_key);
        let encrypted = self.crypto.encrypt_payload(&item_key, question, answer, id.as_bytes())?;
        let checksum = checksum(question, answer);

        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "INSERT INTO encryption_key(id, wrapped_key, destroyed_at)
                 VALUES (?1, ?2, NULL)",
                params![key_id, wrapped_key],
            )?;
            tx.execute(
                "INSERT INTO knowledge_item(
                    id, lang, domain, status, reliability, confidence_profile_id, source_id,
                    version, checksum_sha256, encryption_key_id
                 ) VALUES (?1, ?2, ?3, 'approved', 1.0, 'default', ?4, 1, ?5, ?6)",
                params![id, lang, domain, source_id, checksum, key_id],
            )?;
            tx.execute(
                "INSERT INTO knowledge_payload(
                    item_id, question_cipher, answer_cipher, nonce, aad, payload_version
                 ) VALUES (?1, ?2, ?3, ?4, ?5, 1)",
                params![
                    id,
                    encrypted.question_cipher,
                    encrypted.answer_cipher,
                    encrypted.nonce,
                    encrypted.aad
                ],
            )?;
            tx.execute(
                "INSERT INTO knowledge_fts(rowid, item_id, question, answer)
                 VALUES ((SELECT rowid FROM knowledge_item WHERE id = ?1), ?1, ?2, ?3)",
                params![id, question, answer],
            )?;
            tx.execute(
                "INSERT INTO audit_event(id, event_type, item_id, metadata_json)
                 VALUES (?1, 'item_created', ?2, ?3)",
                params![new_id("audit"), id, "{}"],
            )?;
            tx.commit()
        })?;

        Ok(id)
    }

    pub fn get_payload(&self, item_id: &str) -> Result<KnowledgePayload> {
        self.with_conn(|conn| {
            let row = conn
                .query_row(
                    "SELECT k.wrapped_key, p.question_cipher, p.answer_cipher, p.nonce, p.aad
                     FROM knowledge_payload p
                     JOIN knowledge_item i ON i.id = p.item_id
                     JOIN encryption_key k ON k.id = i.encryption_key_id
                     WHERE p.item_id = ?1 AND i.deleted_at IS NULL AND k.destroyed_at IS NULL",
                    params![item_id],
                    |row| {
                        Ok((
                            row.get::<_, Vec<u8>>(0)?,
                            row.get::<_, Vec<u8>>(1)?,
                            row.get::<_, Vec<u8>>(2)?,
                            row.get::<_, Vec<u8>>(3)?,
                            row.get::<_, Vec<u8>>(4)?,
                        ))
                    },
                )
                .optional()?;
            Ok(row)
        })?
        .ok_or_else(|| SielError::NotFound(item_id.to_string()))
        .and_then(|(wrapped_key, question_cipher, answer_cipher, nonce, aad)| {
            let item_key = self.crypto.unwrap_key(&wrapped_key)?;
            let (question, answer) = self.crypto.decrypt_payload(
                &item_key,
                &question_cipher,
                &answer_cipher,
                &nonce,
                &aad,
            )?;
            Ok(KnowledgePayload {
                item_id: item_id.to_string(),
                question,
                answer,
            })
        })
    }

    pub fn get_item(&self, item_id: &str) -> Result<KnowledgeItem> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT id, lang, domain, status, reliability, confidence_profile_id,
                        source_id, version, checksum_sha256, deleted_at, encryption_key_id
                 FROM knowledge_item WHERE id = ?1",
                params![item_id],
                map_item,
            )
            .optional()
        })?
        .ok_or_else(|| SielError::NotFound(item_id.to_string()))
    }

    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<Evidence>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT f.item_id, f.question, f.answer, i.source_id, bm25(knowledge_fts) AS rank
                 FROM knowledge_fts f
                 JOIN knowledge_item i ON i.id = f.item_id
                 WHERE knowledge_fts MATCH ?1
                   AND i.deleted_at IS NULL
                   AND i.status = 'approved'
                 ORDER BY rank
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![query, limit as i64], |row| {
                let rank: f64 = row.get(4)?;
                Ok(Evidence {
                    item_id: row.get(0)?,
                    question: row.get(1)?,
                    answer: row.get(2)?,
                    source_id: row.get(3)?,
                    score: (1.0 / (1.0 + rank.abs() as f32)).clamp(0.0, 1.0),
                    stale_embedding: false,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
        })
    }

    pub fn exact_search(&self, query: &str) -> Result<Option<Evidence>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT f.item_id, f.question, f.answer, i.source_id
                 FROM knowledge_fts f
                 JOIN knowledge_item i ON i.id = f.item_id
                 WHERE lower(f.question) = lower(?1)
                   AND i.deleted_at IS NULL
                   AND i.status = 'approved'
                 LIMIT 1",
                params![query],
                |row| {
                    Ok(Evidence {
                        item_id: row.get(0)?,
                        question: row.get(1)?,
                        answer: row.get(2)?,
                        source_id: row.get(3)?,
                        score: 1.0,
                        stale_embedding: false,
                    })
                },
            )
            .optional()
        })
    }

    pub fn active_search_docs(&self) -> Result<Vec<Evidence>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT f.item_id, f.question, f.answer, i.source_id
                 FROM knowledge_fts f
                 JOIN knowledge_item i ON i.id = f.item_id
                 WHERE i.deleted_at IS NULL AND i.status = 'approved'",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(Evidence {
                    item_id: row.get(0)?,
                    question: row.get(1)?,
                    answer: row.get(2)?,
                    source_id: row.get(3)?,
                    score: 0.0,
                    stale_embedding: false,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
        })
    }

    pub fn teach(&self, request: TeachRequest) -> Result<TeachResponse> {
        let candidate_id = new_id("candidate");
        let audit_id = new_id("audit");
        let metadata = serde_json::json!({
            "candidate_id": candidate_id,
            "lang": request.lang.as_deref().unwrap_or("it"),
            "domain": request.domain,
            "source_id": request.source_id
        });
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "INSERT INTO memory_candidate(
                    id, question, answer, lang, domain, source_id, status
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending')",
                params![
                    candidate_id,
                    request.question,
                    request.answer,
                    request.lang.as_deref().unwrap_or("it"),
                    request.domain,
                    request.source_id
                ],
            )?;
            tx.execute(
                "INSERT INTO audit_event(id, event_type, item_id, metadata_json)
                 VALUES (?1, 'candidate_created', NULL, ?2)",
                params![audit_id, metadata.to_string()],
            )?;
            tx.commit()
        })?;

        Ok(TeachResponse {
            candidate_id,
            proposal: TeachProposalKind::NewItem,
            related_item_ids: Vec::new(),
            audit_id,
        })
    }

    pub fn approve(&self, candidate_id: &str) -> Result<ApproveResponse> {
        let candidate = self.with_conn(|conn| {
            conn.query_row(
                "SELECT question, answer, lang, domain, source_id
                 FROM memory_candidate WHERE id = ?1 AND status = 'pending'",
                params![candidate_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )
            .optional()
        })?;

        let (question, answer, lang, domain, source_id) =
            candidate.ok_or_else(|| SielError::NotFound(candidate_id.to_string()))?;
        let item_id = self.create_item(
            &question,
            &answer,
            &lang,
            domain.as_deref(),
            source_id.as_deref(),
        )?;
        let audit_id = new_id("audit");
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "UPDATE memory_candidate SET status = 'approved', promoted_item_id = ?1
                 WHERE id = ?2",
                params![item_id, candidate_id],
            )?;
            tx.execute(
                "INSERT INTO audit_event(id, event_type, item_id, metadata_json)
                 VALUES (?1, 'candidate_approved', ?2, ?3)",
                params![
                    audit_id,
                    item_id,
                    serde_json::json!({ "candidate_id": candidate_id }).to_string()
                ],
            )?;
            tx.commit()
        })?;
        Ok(ApproveResponse { item_id, audit_id })
    }

    pub fn delete_item(&self, item_id: &str) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "UPDATE encryption_key
                 SET wrapped_key = X'', destroyed_at = CURRENT_TIMESTAMP
                 WHERE id = (SELECT encryption_key_id FROM knowledge_item WHERE id = ?1)",
                params![item_id],
            )?;
            tx.execute(
                "UPDATE knowledge_item SET deleted_at = CURRENT_TIMESTAMP, status = 'deleted'
                 WHERE id = ?1",
                params![item_id],
            )?;
            tx.execute("DELETE FROM knowledge_fts WHERE item_id = ?1", params![item_id])?;
            tx.execute("DELETE FROM embedding WHERE item_id = ?1", params![item_id])?;
            tx.execute(
                "INSERT INTO audit_event(id, event_type, item_id, metadata_json)
                 VALUES (?1, 'item_erased', ?2, '{}')",
                params![new_id("audit"), item_id],
            )?;
            tx.commit()
        })?;
        Ok(())
    }
}

fn map_item(row: &rusqlite::Row<'_>) -> rusqlite::Result<KnowledgeItem> {
    let status: String = row.get(3)?;
    let status = match status.as_str() {
        "approved" => KnowledgeStatus::Approved,
        "candidate" => KnowledgeStatus::Candidate,
        "rejected" => KnowledgeStatus::Rejected,
        "deleted" => KnowledgeStatus::Deleted,
        _ => KnowledgeStatus::Candidate,
    };
    Ok(KnowledgeItem {
        id: row.get(0)?,
        lang: row.get(1)?,
        domain: row.get(2)?,
        status,
        reliability: row.get::<_, f32>(4)?,
        confidence_profile_id: row.get(5)?,
        source_id: row.get(6)?,
        version: row.get(7)?,
        checksum_sha256: row.get(8)?,
        deleted_at: row.get(9)?,
        encryption_key_id: row.get(10)?,
    })
}

fn checksum(question: &str, answer: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(question.as_bytes());
    hasher.update(b"\0");
    hasher.update(answer.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_read_delete_item() {
        let db = SielDb::memory(b"test master secret").unwrap();
        let id = db
            .create_item("Chi sei?", "Sono SIEL.", "it", None, None)
            .unwrap();
        let payload = db.get_payload(&id).unwrap();
        assert_eq!(payload.answer, "Sono SIEL.");

        db.delete_item(&id).unwrap();
        assert!(db.get_payload(&id).is_err());
        assert!(db.exact_search("Chi sei?").unwrap().is_none());
    }
}
