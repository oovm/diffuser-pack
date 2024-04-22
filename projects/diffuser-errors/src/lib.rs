
mod errors;

pub use crate::errors::{DiffuserErrorKind, MissingPartError, MissingPartKind};

type Result<T> = core::result::Result<T, DiffuserError>;

pub struct DiffuserError {
    kind: Box<DiffuserErrorKind>,
}
