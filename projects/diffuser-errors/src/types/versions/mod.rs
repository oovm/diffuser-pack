use std::fmt::{Display, Formatter};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Error;

mod ser;
mod der;


#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVersion {
    V1_5,
    V2_1,
    XL,
    XL_Turbo,
}

