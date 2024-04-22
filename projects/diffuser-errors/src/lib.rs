mod errors;
mod types;

pub use crate::{
    errors::{DiffuserErrorKind, MissingPartError, MissingPartKind},
    types::{versions::ModelVersion, weights::WeightInfo, ModelStorage},
};
type Result<T> = core::result::Result<T, DiffuserError>;

pub struct DiffuserError {
    kind: Box<DiffuserErrorKind>,
}
