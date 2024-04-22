
mod errors;

pub use crate::errors::{DiffuserErrorKind, MissingPartError, MissingPartKind};

pub struct DiffuserError {
    kind: Box<DiffuserErrorKind>,
}
