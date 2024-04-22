use crate::{
    DiffuserError,
};
use std::path::PathBuf;
use url::Url;

mod missing_part;

pub enum DiffuserErrorKind {
    MissingPart(MissingPartError),
}

pub struct MissingPartError {
    part: MissingPartKind,
    local: Option<PathBuf>,
    remote: Option<Url>,
}

pub enum MissingPartKind {
    UNetWeight,
}

impl From<DiffuserErrorKind> for DiffuserError {
    fn from(error: DiffuserErrorKind) -> Self {
        Self { kind: Box::new(error) }
    }
}