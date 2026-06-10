use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use siel_core::{ApproveRequest, ConfidenceProfile, QueryRequest, SielError, TeachRequest};
use siel_db::SielDb;
use siel_retrieval::{HashEmbedder, RetrievalService};
use siel_vector::{FakeVectorIndex, VectorIndex};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

type AppRetrieval = RetrievalService<FakeVectorIndex, HashEmbedder>;

#[derive(Clone)]
struct AppState {
    db: Arc<SielDb>,
    vector: Arc<FakeVectorIndex>,
    retrieval: Arc<AppRetrieval>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "siel_api=info,tower_http=info".to_string()),
        )
        .init();

    let db_path = std::env::var("SIEL_DB").unwrap_or_else(|_| "siel.db".to_string());
    let secret = std::env::var("SIEL_MASTER_SECRET")
        .unwrap_or_else(|_| "development-secret-change-me".to_string());
    let db = Arc::new(SielDb::open(db_path, secret.as_bytes())?);
    let vector = Arc::new(FakeVectorIndex::default());
    let embedder = Arc::new(HashEmbedder::default());
    let retrieval = Arc::new(RetrievalService::new(
        Arc::clone(&db),
        Arc::clone(&vector),
        embedder,
        ConfidenceProfile::default(),
    ));

    let state = AppState {
        db,
        vector,
        retrieval,
    };

    let app = router(state);
    let addr = std::env::var("SIEL_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8787".to_string())
        .parse::<SocketAddr>()?;
    tracing::info!("listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/query", post(query))
        .route("/teach", post(teach))
        .route("/approve", post(approve))
        .route("/items/:id", get(get_item).delete(delete_item))
        .route("/items/:id/rollback", post(rollback))
        .route("/items/:id/stale", get(stale))
        .route("/reindex", post(reindex))
        .route("/reindex/dry-run", post(reindex_dry_run))
        .route("/eval/run", post(eval_run))
        .route("/eval/calibrate", post(eval_calibrate))
        .route("/confidence-profiles", get(confidence_profiles))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(Arc::new(state))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<impl Serialize>, ApiError> {
    Ok(Json(state.retrieval.query_deterministic(request)?))
}

async fn teach(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TeachRequest>,
) -> Result<Json<impl Serialize>, ApiError> {
    Ok(Json(state.db.teach(request)?))
}

async fn approve(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApproveRequest>,
) -> Result<Json<impl Serialize>, ApiError> {
    let response = state.db.approve(&request.candidate_id)?;
    state.retrieval.index_item(&response.item_id)?;
    Ok(Json(response))
}

async fn get_item(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<impl Serialize>, ApiError> {
    let item = state.db.get_item(&id)?;
    let payload = state.db.get_payload(&id)?;
    Ok(Json(serde_json::json!({
        "item": item,
        "payload": payload
    })))
}

async fn delete_item(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.db.delete_item(&id)?;
    state.vector.delete_id(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn rollback(Path(_id): Path<String>) -> Result<Json<serde_json::Value>, ApiError> {
    Err(ApiError(SielError::InvalidInput(
        "rollback endpoint is reserved; snapshot restore is not enabled until audit snapshots are populated".to_string(),
    )))
}

async fn stale(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "item_id": id,
        "stale": false,
        "reason": "stale detection is available through VectorIndex::list_stale during reindex"
    }))
}

async fn reindex() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "accepted",
        "note": "wire this endpoint to the selected production embedder before enabling model downloads"
    }))
}

async fn reindex_dry_run() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "estimated_items": 0,
        "estimated_seconds": 0,
        "requires_model": true
    }))
}

async fn eval_run() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "not_configured",
        "tool": "lab/run_retrieval_eval.py"
    }))
}

async fn eval_calibrate() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "not_configured",
        "tool": "lab/calibrate_thresholds.py"
    }))
}

async fn confidence_profiles() -> Json<serde_json::Value> {
    Json(serde_json::json!([
        {
            "profile_id": "default",
            "threshold_high": 0.82,
            "threshold_low": 0.62,
            "calibration_method": "fixed"
        }
    ]))
}

struct ApiError(SielError);

impl From<SielError> for ApiError {
    fn from(value: SielError) -> Self {
        Self(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            SielError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            SielError::NotFound(_) => StatusCode::NOT_FOUND,
            SielError::Conflict(_) | SielError::Policy(_) => StatusCode::CONFLICT,
            SielError::Crypto(_)
            | SielError::Storage(_)
            | SielError::Model(_)
            | SielError::Serialization(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(serde_json::json!({
            "error": self.0.to_string()
        }));
        (status, body).into_response()
    }
}
