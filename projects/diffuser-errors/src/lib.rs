use std::path::{Path, PathBuf};

pub struct DiffuserError {
    kind: Box<DiffuserErrorKind>,
}

pub enum DiffuserErrorKind {
    MissingPart(MissingPartError)
}

pub struct MissingPartError {
    part: MissingPartKind,
    local: PathBuf,
    remote: Option<Ur>
}

pub enum MissingPartKind {
    UNetWeight
}


impl DiffuserError {
    pub fn missing_part<S: Into<String>>(path: MissingPartKind, part: S) -> Self {
        DiffuserErrorKind::MissingPartError { part, path: path.into() }.into()
    }
}

impl From<DiffuserErrorKind> for DiffuserError {
    fn from(value: DiffuserErrorKind) -> Self {
        Self {
            kind: Box::new(value),
        }
    }
}

impl From<DiffuserErrorKind> for DiffuserError {
    fn from(value: DiffuserErrorKind) -> Self {
        Self {
            kind: Box::new(value),
        }
    }
}