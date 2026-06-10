use thiserror::Error;

pub type Result<T> = std::result::Result<T, SielError>;

#[derive(Debug, Error)]
pub enum SielError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("crypto error: {0}")]
    Crypto(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("policy rejected output: {0}")]
    Policy(String),
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<serde_json::Error> for SielError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}
