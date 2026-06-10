use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use siel_core::{Result, SielError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorScoreKind {
    CosineSimilarity,
    DotProduct,
    Distance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchFilter {
    pub model_id: String,
    pub model_version: String,
    pub lang: Option<String>,
    pub domain: Option<String>,
    pub include_statuses: Vec<String>,
    pub exclude_deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHit {
    pub id: String,
    pub score: f32,
    pub score_kind: VectorScoreKind,
    pub model_id: String,
    pub model_version: String,
}

pub trait VectorIndex: Send + Sync {
    fn upsert_batch(
        &self,
        model_id: &str,
        model_version: &str,
        dimensions: usize,
        vectors: &[(&str, &[f32])],
    ) -> Result<()>;

    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &VectorSearchFilter,
    ) -> Result<Vec<VectorHit>>;
    fn delete_id(&self, id: &str) -> Result<()>;
    fn delete_model_version(&self, model_id: &str, model_version: &str) -> Result<()>;
    fn list_stale(&self, model_id: &str, current_version: &str) -> Result<Vec<String>>;
}

#[derive(Debug, Clone)]
struct StoredVector {
    model_id: String,
    model_version: String,
    dimensions: usize,
    vector: Vec<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct FakeVectorIndex {
    vectors: Arc<RwLock<HashMap<String, StoredVector>>>,
}

impl VectorIndex for FakeVectorIndex {
    fn upsert_batch(
        &self,
        model_id: &str,
        model_version: &str,
        dimensions: usize,
        vectors: &[(&str, &[f32])],
    ) -> Result<()> {
        let mut guard = self
            .vectors
            .write()
            .map_err(|_| SielError::Storage("vector lock poisoned".to_string()))?;
        for (id, vector) in vectors {
            validate_dimensions(dimensions, vector)?;
            guard.insert(
                (*id).to_string(),
                StoredVector {
                    model_id: model_id.to_string(),
                    model_version: model_version.to_string(),
                    dimensions,
                    vector: vector.to_vec(),
                },
            );
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &VectorSearchFilter,
    ) -> Result<Vec<VectorHit>> {
        let guard = self
            .vectors
            .read()
            .map_err(|_| SielError::Storage("vector lock poisoned".to_string()))?;
        let mut hits = guard
            .iter()
            .filter(|(_, stored)| {
                stored.model_id == filter.model_id && stored.model_version == filter.model_version
            })
            .map(|(id, stored)| {
                validate_dimensions(stored.dimensions, query)?;
                Ok(VectorHit {
                    id: id.clone(),
                    score: cosine_similarity(query, &stored.vector),
                    score_kind: VectorScoreKind::CosineSimilarity,
                    model_id: stored.model_id.clone(),
                    model_version: stored.model_version.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(k);
        Ok(hits)
    }

    fn delete_id(&self, id: &str) -> Result<()> {
        self.vectors
            .write()
            .map_err(|_| SielError::Storage("vector lock poisoned".to_string()))?
            .remove(id);
        Ok(())
    }

    fn delete_model_version(&self, model_id: &str, model_version: &str) -> Result<()> {
        self.vectors
            .write()
            .map_err(|_| SielError::Storage("vector lock poisoned".to_string()))?
            .retain(|_, stored| {
                !(stored.model_id == model_id && stored.model_version == model_version)
            });
        Ok(())
    }

    fn list_stale(&self, model_id: &str, current_version: &str) -> Result<Vec<String>> {
        Ok(self
            .vectors
            .read()
            .map_err(|_| SielError::Storage("vector lock poisoned".to_string()))?
            .iter()
            .filter(|(_, stored)| {
                stored.model_id == model_id && stored.model_version != current_version
            })
            .map(|(id, _)| id.clone())
            .collect())
    }
}

pub struct SqliteVecIndex {
    path: String,
}

impl SqliteVecIndex {
    pub fn open(path: impl Into<String>) -> Result<Self> {
        let index = Self { path: path.into() };
        index.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS vector_index (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vector_index_model
                    ON vector_index(model_id, model_version);",
            )
        })?;
        Ok(index)
    }

    fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Result<T> {
        let conn =
            Connection::open(&self.path).map_err(|err| SielError::Storage(err.to_string()))?;
        f(&conn).map_err(|err| SielError::Storage(err.to_string()))
    }
}

impl VectorIndex for SqliteVecIndex {
    fn upsert_batch(
        &self,
        model_id: &str,
        model_version: &str,
        dimensions: usize,
        vectors: &[(&str, &[f32])],
    ) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO vector_index(id, model_id, model_version, dimensions, vector)
                     VALUES (?1, ?2, ?3, ?4, ?5)
                     ON CONFLICT(id) DO UPDATE SET
                        model_id=excluded.model_id,
                        model_version=excluded.model_version,
                        dimensions=excluded.dimensions,
                        vector=excluded.vector",
                )?;
                for (id, vector) in vectors {
                    validate_dimensions(dimensions, vector)
                        .map_err(|err| rusqlite::Error::ToSqlConversionFailure(Box::new(err)))?;
                    stmt.execute(params![
                        id,
                        model_id,
                        model_version,
                        dimensions as i64,
                        encode_vector(vector)
                    ])?;
                }
            }
            tx.commit()
        })
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &VectorSearchFilter,
    ) -> Result<Vec<VectorHit>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, dimensions, vector FROM vector_index
                 WHERE model_id = ?1 AND model_version = ?2",
            )?;
            let rows = stmt.query_map(params![filter.model_id, filter.model_version], |row| {
                let id: String = row.get(0)?;
                let dimensions: i64 = row.get(1)?;
                let vector_blob: Vec<u8> = row.get(2)?;
                Ok((id, dimensions as usize, vector_blob))
            })?;

            let mut hits = Vec::new();
            for row in rows {
                let (id, dimensions, blob) = row?;
                let vector = decode_vector(&blob).map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Blob,
                        Box::new(err),
                    )
                })?;
                validate_dimensions(dimensions, query).map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Integer,
                        Box::new(err),
                    )
                })?;
                hits.push(VectorHit {
                    id,
                    score: cosine_similarity(query, &vector),
                    score_kind: VectorScoreKind::CosineSimilarity,
                    model_id: filter.model_id.clone(),
                    model_version: filter.model_version.clone(),
                });
            }
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
            hits.truncate(k);
            Ok(hits)
        })
    }

    fn delete_id(&self, id: &str) -> Result<()> {
        self.with_conn(|conn| conn.execute("DELETE FROM vector_index WHERE id = ?1", params![id]))?;
        Ok(())
    }

    fn delete_model_version(&self, model_id: &str, model_version: &str) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "DELETE FROM vector_index WHERE model_id = ?1 AND model_version = ?2",
                params![model_id, model_version],
            )
        })?;
        Ok(())
    }

    fn list_stale(&self, model_id: &str, current_version: &str) -> Result<Vec<String>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id FROM vector_index WHERE model_id = ?1 AND model_version != ?2",
            )?;
            let rows = stmt.query_map(params![model_id, current_version], |row| row.get(0))?;
            rows.collect::<rusqlite::Result<Vec<String>>>()
        })
    }
}

fn validate_dimensions(dimensions: usize, vector: &[f32]) -> Result<()> {
    if vector.len() != dimensions {
        return Err(SielError::InvalidInput(format!(
            "vector dimension mismatch: expected {dimensions}, got {}",
            vector.len()
        )));
    }
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (left, right) in a.iter().zip(b) {
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(SielError::Storage(
            "vector blob length is not a multiple of 4".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_index_respects_model_version() {
        let index = FakeVectorIndex::default();
        index
            .upsert_batch("m", "1", 2, &[("a", &[1.0, 0.0]), ("b", &[0.0, 1.0])])
            .unwrap();
        index
            .upsert_batch("m", "2", 2, &[("c", &[1.0, 0.0])])
            .unwrap();

        let hits = index
            .search(
                &[1.0, 0.0],
                10,
                &VectorSearchFilter {
                    model_id: "m".to_string(),
                    model_version: "1".to_string(),
                    lang: None,
                    domain: None,
                    include_statuses: vec!["approved".to_string()],
                    exclude_deleted: true,
                },
            )
            .unwrap();

        assert_eq!(hits[0].id, "a");
        assert!(hits.iter().all(|hit| hit.model_version == "1"));
    }

    #[test]
    fn detects_stale_vectors() {
        let index = FakeVectorIndex::default();
        index
            .upsert_batch("m", "1", 2, &[("a", &[1.0, 0.0])])
            .unwrap();
        assert_eq!(index.list_stale("m", "2").unwrap(), vec!["a".to_string()]);
    }
}
