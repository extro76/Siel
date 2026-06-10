use std::{collections::HashMap, sync::Arc};

use siel_core::{
    decide, ConfidenceProfile, SielError, Evidence, ModelRef, QueryRequest, QueryResponse,
    QueryStatus, Result,
};
use siel_db::SielDb;
use siel_vector::{VectorIndex, VectorSearchFilter};
use sha2::{Digest, Sha256};

pub trait Embedder: Send + Sync {
    fn model_ref(&self) -> ModelRef;
    fn dimensions(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone)]
pub struct HashEmbedder {
    model_ref: ModelRef,
    dimensions: usize,
}

impl HashEmbedder {
    pub fn new(model_id: &str, model_version: &str, dimensions: usize) -> Self {
        Self {
            model_ref: ModelRef {
                model_id: model_id.to_string(),
                model_version: model_version.to_string(),
            },
            dimensions,
        }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new("hash-embedder", "0.1.0", 64)
    }
}

impl Embedder for HashEmbedder {
    fn model_ref(&self) -> ModelRef {
        self.model_ref.clone()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0_f32; self.dimensions];
        for token in normalize_tokens(text) {
            let digest = Sha256::digest(token.as_bytes());
            let idx = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize
                % self.dimensions;
            vector[idx] += 1.0;
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        Ok(vector)
    }
}

pub struct RetrievalService<V: VectorIndex, E: Embedder> {
    db: Arc<SielDb>,
    vector_index: Arc<V>,
    embedder: Arc<E>,
    profile: ConfidenceProfile,
}

impl<V: VectorIndex, E: Embedder> RetrievalService<V, E> {
    pub fn new(
        db: Arc<SielDb>,
        vector_index: Arc<V>,
        embedder: Arc<E>,
        profile: ConfidenceProfile,
    ) -> Self {
        Self {
            db,
            vector_index,
            embedder,
            profile,
        }
    }

    pub fn query_deterministic(&self, request: QueryRequest) -> Result<QueryResponse> {
        let query = request.query.trim();
        if query.is_empty() {
            return Err(SielError::InvalidInput("query cannot be empty".to_string()));
        }

        let mut evidences = Vec::new();
        if let Some(exact) = self.db.exact_search(query)? {
            evidences.push(exact);
        }

        evidences.extend(self.db.fts_search(query, request.max_hits.unwrap_or(8))?);
        evidences.extend(self.fuzzy_search(query, 6)?);
        evidences.extend(self.vector_search(query, request.embedding_model.as_ref(), 8)?);

        let ranked = reciprocal_rank_fusion(evidences, 60.0);
        let Some(best) = ranked.first() else {
            return Ok(QueryResponse::unknown("no supporting evidence found"));
        };

        let decision = decide(best.score, &self.profile)?;
        if decision.status == QueryStatus::Unknown {
            return Ok(QueryResponse {
                answer: "Non lo so con sufficiente certezza.".to_string(),
                status: QueryStatus::Unknown,
                confidence: decision.confidence,
                confidence_calibrated: decision.confidence_calibrated,
                confidence_profile_id: self.profile.profile_id.clone(),
                support_ids: Vec::new(),
                source_ids: Vec::new(),
                missing_information: vec![decision
                    .reason
                    .unwrap_or_else(|| "confidence below threshold".to_string())],
                learning_suggestion: Some(
                    "Puoi insegnarmi questa informazione con /teach.".to_string(),
                ),
                stale_embeddings_detected: ranked.iter().any(|e| e.stale_embedding),
            });
        }

        let support_ids = ranked
            .iter()
            .take(3)
            .map(|e| e.item_id.clone())
            .collect::<Vec<_>>();
        let source_ids = ranked
            .iter()
            .filter_map(|e| e.source_id.clone())
            .collect::<Vec<_>>();

        Ok(QueryResponse {
            answer: best.answer.clone(),
            status: decision.status,
            confidence: decision.confidence,
            confidence_calibrated: decision.confidence_calibrated,
            confidence_profile_id: self.profile.profile_id.clone(),
            support_ids,
            source_ids,
            missing_information: Vec::new(),
            learning_suggestion: None,
            stale_embeddings_detected: ranked.iter().any(|e| e.stale_embedding),
        })
    }

    pub fn index_item(&self, item_id: &str) -> Result<()> {
        let payload = self.db.get_payload(item_id)?;
        let model = self.embedder.model_ref();
        let vector = self
            .embedder
            .embed(&format!("{}\n{}", payload.question, payload.answer))?;
        self.vector_index.upsert_batch(
            &model.model_id,
            &model.model_version,
            self.embedder.dimensions(),
            &[(item_id, vector.as_slice())],
        )
    }

    fn vector_search(
        &self,
        query: &str,
        requested_model: Option<&ModelRef>,
        k: usize,
    ) -> Result<Vec<Evidence>> {
        let active_model = requested_model
            .cloned()
            .unwrap_or_else(|| self.embedder.model_ref());
        let query_vector = self.embedder.embed(query)?;
        let hits = self.vector_index.search(
            &query_vector,
            k,
            &VectorSearchFilter {
                model_id: active_model.model_id,
                model_version: active_model.model_version,
                lang: None,
                domain: None,
                include_statuses: vec!["approved".to_string()],
                exclude_deleted: true,
            },
        )?;

        let mut evidences = Vec::new();
        for hit in hits {
            if let Ok(payload) = self.db.get_payload(&hit.id) {
                let item = self.db.get_item(&hit.id)?;
                evidences.push(Evidence {
                    item_id: hit.id,
                    question: payload.question,
                    answer: payload.answer,
                    source_id: item.source_id,
                    score: hit.score.clamp(0.0, 1.0),
                    stale_embedding: false,
                });
            }
        }
        Ok(evidences)
    }

    fn fuzzy_search(&self, query: &str, limit: usize) -> Result<Vec<Evidence>> {
        let mut docs = self.db.active_search_docs()?;
        for doc in &mut docs {
            let q_score = similarity(query, &doc.question);
            let a_score = similarity(query, &doc.answer) * 0.75;
            doc.score = q_score.max(a_score);
        }
        docs.retain(|doc| doc.score >= 0.45);
        docs.sort_by(|a, b| b.score.total_cmp(&a.score));
        docs.truncate(limit);
        Ok(docs)
    }
}

pub fn reciprocal_rank_fusion(evidences: Vec<Evidence>, k: f32) -> Vec<Evidence> {
    let mut fused: HashMap<String, Evidence> = HashMap::new();
    for (rank, evidence) in evidences.into_iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32 + 1.0);
        fused
            .entry(evidence.item_id.clone())
            .and_modify(|existing| {
                existing.score = (existing.score + contribution).min(1.0);
                existing.stale_embedding |= evidence.stale_embedding;
            })
            .or_insert_with(|| Evidence {
                score: evidence.score.max(contribution),
                ..evidence
            });
    }

    let mut out = fused.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

pub fn normalize_tokens(input: &str) -> Vec<String> {
    input
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

fn similarity(a: &str, b: &str) -> f32 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    if a == b {
        return 1.0;
    }
    let distance = levenshtein(&a, &b) as f32;
    let max_len = a.chars().count().max(b.chars().count()) as f32;
    if max_len == 0.0 {
        0.0
    } else {
        (1.0 - distance / max_len).clamp(0.0, 1.0)
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b_chars = b.chars().collect::<Vec<_>>();
    let mut costs = (0..=b_chars.len()).collect::<Vec<_>>();
    for (i, ca) in a.chars().enumerate() {
        let mut last = i;
        costs[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let old = costs[j + 1];
            costs[j + 1] = if ca == *cb {
                last
            } else {
                1 + last.min(costs[j]).min(costs[j + 1])
            };
            last = old;
        }
    }
    costs[b_chars.len()]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use siel_core::ConfidenceProfile;
    use siel_db::SielDb;
    use siel_vector::FakeVectorIndex;

    use super::*;

    #[test]
    fn deterministic_query_returns_support() {
        let db = Arc::new(SielDb::memory(b"master").unwrap());
        let item = db
            .create_item("Che cos'è SIEL?", "SIEL è un motore di conoscenza.", "it", None, None)
            .unwrap();
        let service = RetrievalService::new(
            db,
            Arc::new(FakeVectorIndex::default()),
            Arc::new(HashEmbedder::default()),
            ConfidenceProfile {
                threshold_high: 0.01,
                threshold_low: 0.0,
                ..ConfidenceProfile::default()
            },
        );
        service.index_item(&item).unwrap();
        let response = service
            .query_deterministic(QueryRequest {
                query: "Che cos'è SIEL?".to_string(),
                lang: Some("it".to_string()),
                domain: None,
                embedding_model: None,
                max_hits: Some(5),
            })
            .unwrap();
        assert_eq!(response.status, QueryStatus::Answered);
        assert_eq!(response.support_ids[0], item);
    }
}

