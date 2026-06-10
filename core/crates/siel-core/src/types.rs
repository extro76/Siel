use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryStatus {
    Answered,
    NeedsClarification,
    Unknown,
    Conflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeStatus {
    Approved,
    Candidate,
    Rejected,
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub model_id: String,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub lang: Option<String>,
    pub domain: Option<String>,
    pub embedding_model: Option<ModelRef>,
    pub max_hits: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub answer: String,
    pub status: QueryStatus,
    pub confidence: f32,
    pub confidence_calibrated: bool,
    pub confidence_profile_id: String,
    pub support_ids: Vec<String>,
    pub source_ids: Vec<String>,
    pub missing_information: Vec<String>,
    pub learning_suggestion: Option<String>,
    pub stale_embeddings_detected: bool,
}

impl QueryResponse {
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            answer: "Non lo so con sufficiente certezza.".to_string(),
            status: QueryStatus::Unknown,
            confidence: 0.0,
            confidence_calibrated: false,
            confidence_profile_id: "default".to_string(),
            support_ids: Vec::new(),
            source_ids: Vec::new(),
            missing_information: vec![reason.into()],
            learning_suggestion: None,
            stale_embeddings_detected: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub item_id: String,
    pub question: String,
    pub answer: String,
    pub source_id: Option<String>,
    pub score: f32,
    pub stale_embedding: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeItem {
    pub id: String,
    pub lang: String,
    pub domain: Option<String>,
    pub status: KnowledgeStatus,
    pub reliability: f32,
    pub confidence_profile_id: String,
    pub source_id: Option<String>,
    pub version: i64,
    pub checksum_sha256: String,
    pub deleted_at: Option<String>,
    pub encryption_key_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePayload {
    pub item_id: String,
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeachRequest {
    pub question: String,
    pub answer: String,
    pub lang: Option<String>,
    pub domain: Option<String>,
    pub source_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeachProposalKind {
    NewItem,
    VariantAdded,
    DuplicateCandidate,
    Conflict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeachResponse {
    pub candidate_id: String,
    pub proposal: TeachProposalKind,
    pub related_item_ids: Vec<String>,
    pub audit_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproveRequest {
    pub candidate_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproveResponse {
    pub item_id: String,
    pub audit_id: String,
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4())
}

