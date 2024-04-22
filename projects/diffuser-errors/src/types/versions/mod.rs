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


impl ModelVersion {
    pub fn vae_scale(&self) -> f32 {
        match self {
            Self::V1_5 => 0.18215,
            Self::V2_1 => 0.18215,
            Self::XL => 0.18215,
            Self::XL_Turbo => 0.13025,
        }
    }
}
