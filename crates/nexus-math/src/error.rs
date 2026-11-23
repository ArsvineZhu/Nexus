// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input at {field} ({id:?}): {message}")]
    Input {
        field: &'static str,
        id: Option<u64>,
        message: String,
    },
    #[error("invalid configuration at {field}: {message}")]
    Config {
        field: &'static str,
        message: String,
    },
    #[error("association error: {message}")]
    Association { message: String },
    #[error("numeric error in {operator}: {message}")]
    Numeric {
        operator: &'static str,
        message: String,
    },
    #[error("external operation {operation} failed: {message}")]
    External {
        operation: &'static str,
        message: String,
    },
    #[error("cancelled")]
    Cancelled,
}

impl Error {
    pub fn input(field: &'static str, message: impl Into<String>) -> Self {
        Self::Input {
            field,
            id: None,
            message: message.into(),
        }
    }

    pub fn dimension(field: &'static str, expected: usize, actual: usize) -> Self {
        Self::input(field, format!("expected {expected}, got {actual}"))
    }

    pub fn config(field: &'static str, message: impl Into<String>) -> Self {
        Self::Config {
            field,
            message: message.into(),
        }
    }

    pub fn numeric(operator: &'static str, message: impl Into<String>) -> Self {
        Self::Numeric {
            operator,
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
