use crate::{
    DiffuserError,
};
use std::path::PathBuf;
use candle_core::Error;
use url::Url;

mod missing_part;

pub enum DiffuserErrorKind {
    MissingPart(MissingPartError),
    CustomError(String),
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

impl From<Error> for DiffuserError {
    fn from(value: Error) -> Self {
        DiffuserErrorKind::CustomError(value.to_string()).into()
    }
}

impl DiffuserError {
    pub fn custom(message: impl Into<String>) -> Self {
        DiffuserErrorKind::CustomError(message.into()).into()
    }
}