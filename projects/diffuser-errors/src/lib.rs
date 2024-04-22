mod errors;
mod types;

pub use crate::errors::{DiffuserErrorKind, MissingPartError, MissingPartKind};
pub use crate::types::{versions::ModelVersion, ModelInfo};

type Result<T> = core::result::Result<T, DiffuserError>;

pub struct DiffuserError {
    kind: Box<DiffuserErrorKind>,
}
