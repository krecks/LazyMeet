//! API-facing error type. Mirrors the FastAPI HTTPException status codes the
//! old backend returned so the frontend's `err.status` checks keep working.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ApiError {
    /// HTTP-style status code (404, 409, 422, 500) — the frontend switches on it.
    pub code: u16,
    pub kind: ErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    NotFound,
    Conflict,
    Invalid,
    Internal,
}

impl ApiError {
    pub fn not_found(message: impl Into<String>) -> Self {
        Self { code: 404, kind: ErrorKind::NotFound, message: message.into() }
    }
    pub fn conflict(message: impl Into<String>) -> Self {
        Self { code: 409, kind: ErrorKind::Conflict, message: message.into() }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self { code: 422, kind: ErrorKind::Invalid, message: message.into() }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self { code: 500, kind: ErrorKind::Internal, message: message.into() }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.code, self.message)
    }
}

impl std::error::Error for ApiError {}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        ApiError::internal(e.to_string())
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
