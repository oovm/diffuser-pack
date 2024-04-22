use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::path::PathBuf;


use url::Url;

mod missing_part;

pub type Result<T> = core::result::Result<T, DiffuserError>;

pub struct DiffuserError {
    kind: Box<DiffuserErrorKind>,
}


pub enum DiffuserErrorKind {
    MissingPart(MissingPartError),
    CustomError(String),
}



#[derive(Debug)]
pub struct MissingPartError {
    part: MissingPartKind,
    local: Option<PathBuf>,
    remote: Option<Url>,
}
#[derive(Debug)]
pub enum MissingPartKind {
    UNetWeight,
}
impl Error for DiffuserError {

}

impl Debug for DiffuserError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.kind, f)
    }
}

impl Display for DiffuserError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.kind, f)
    }
}
impl Error for DiffuserErrorKind {

}
impl Debug for DiffuserErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingPart(e) => {Debug::fmt(e, f)}
            Self::CustomError(e) => {
                f.write_str(e)
            }
        }
    }
}

impl Display for DiffuserErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingPart(e) => {Debug::fmt(e, f)}
            Self::CustomError(e) => {
                f.write_str(e)
                
            }
        }
    }
}



impl From<DiffuserErrorKind> for DiffuserError {
    fn from(error: DiffuserErrorKind) -> Self {
        Self { kind: Box::new(error) }
    }
}

impl From<candle_core::Error> for DiffuserError {
    fn from(value: candle_core::Error) -> Self {
        DiffuserErrorKind::CustomError(value.to_string()).into()
    }
}

impl DiffuserError {
    pub fn custom(message: impl Display) -> Self {
        DiffuserErrorKind::CustomError(message.to_string()).into()
    }
}

impl From<serde::de::value::Error> for DiffuserError {
    fn from(error: serde::de::value::Error) -> Self {
        DiffuserErrorKind::CustomError(error.to_string()).into()
    }
}

impl From<std::io::Error> for DiffuserError {
    fn from(error: std::io::Error) -> Self {
        DiffuserErrorKind::CustomError(error.to_string()).into()
    }
}