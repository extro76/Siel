use std::collections::HashSet;

use crate::{ConfidenceProfile, Evidence, QueryResponse, QueryStatus, Result};

#[derive(Debug, Clone)]
pub struct PolicyDecision {
    pub status: QueryStatus,
    pub confidence: f32,
    pub confidence_calibrated: bool,
    pub reason: Option<String>,
}

pub fn decide(raw_score: f32, profile: &ConfidenceProfile) -> Result<PolicyDecision> {
    let (confidence, calibrated) = profile.calibrator.predict(raw_score)?;
    let status = if confidence >= profile.threshold_high {
        QueryStatus::Answered
    } else if confidence >= profile.threshold_low {
        QueryStatus::NeedsClarification
    } else {
        QueryStatus::Unknown
    };

    Ok(PolicyDecision {
        status,
        confidence,
        confidence_calibrated: calibrated,
        reason: (status == QueryStatus::Unknown)
            .then(|| "confidence below low threshold".to_string()),
    })
}

pub fn validate_supported_response(
    mut response: QueryResponse,
    retrieved: &[Evidence],
) -> Result<QueryResponse> {
    let allowed: HashSet<&str> = retrieved.iter().map(|e| e.item_id.as_str()).collect();
    let has_invalid_support = response
        .support_ids
        .iter()
        .any(|id| !allowed.contains(id.as_str()));

    if has_invalid_support {
        return Ok(QueryResponse::unknown(
            "model cited support_ids that were not retrieved",
        ));
    }

    if response.status == QueryStatus::Answered && response.support_ids.is_empty() {
        return Ok(QueryResponse::unknown(
            "answered response had no supporting evidence",
        ));
    }

    response.stale_embeddings_detected =
        response.stale_embeddings_detected || retrieved.iter().any(|e| e.stale_embedding);
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answered_without_support_becomes_unknown() {
        let response = QueryResponse {
            answer: "test".to_string(),
            status: QueryStatus::Answered,
            confidence: 0.9,
            confidence_calibrated: false,
            confidence_profile_id: "default".to_string(),
            support_ids: Vec::new(),
            source_ids: Vec::new(),
            missing_information: Vec::new(),
            learning_suggestion: None,
            stale_embeddings_detected: false,
        };

        let checked = validate_supported_response(response, &[]).unwrap();
        assert_eq!(checked.status, QueryStatus::Unknown);
    }
}

