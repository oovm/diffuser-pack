use std::fmt::{Display, Formatter};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Error;
use crate::ModelVersion;

pub mod versions;


pub struct ModelInfo {
    pub name: String,
    pub version: ModelVersion,
    pub tokenizer1: String,
    pub tokenizer2: String,
    pub clip1: String,
    pub clip2: String,
    pub unet: String,
    pub vae: String,
}