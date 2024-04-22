use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    path::{Path, PathBuf},
    str::FromStr,
};

use candle_core::DType;
use serde::{
    de::{Error, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use url::Url;

use crate::{DiffuserError, ModelVersion, WeightInfo};

pub mod versions;
pub mod weights;

#[derive(Debug, Default)]
pub struct ModelStorage {
    pub root: PathBuf,
    pub models: BTreeMap<String, ModelVersion>,
    pub weights: BTreeMap<String, WeightInfo>,
}

impl Serialize for ModelStorage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}

impl<'de> Deserialize<'de> for ModelStorage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}

pub struct ModelStorageVisitor {
    place: ModelStorage,
}

impl<'i, 'de> Visitor<'de> for &'i mut ModelStorageVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        todo!()
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.to_ascii_lowercase().as_str() {
                "models" => self.place.models = map.next_value()?,
                "weights" => self.place.weights = map.next_value()?,
                _ => {}
            }
        }
        Ok(())
    }
}

impl ModelStorage {
    pub fn load<'de, P, D>(root: P, format: D) -> Result<Self, DiffuserError>
    where
        P: AsRef<Path>,
        D: Deserializer<'de>,
    {
        let path = root.as_ref();
        if !path.exists() {
            return Err(DiffuserError::ModelNotFound(path.to_path_buf()));
        }
        Ok(Self::deserialize(format)?)
    }
    pub fn load_weight(&self, id: &str) -> Result<(PathBuf, DType), DiffuserError> {
        let (local, dtype) = match self.weights.get(id) {
            Some(s) => (self.root.join("models").join(&s.local), s.r#type),
            None => Err(DiffuserError::custom(format!("model `{}` not found", id)))?,
        };
        if local.exists() {
            Ok((local, dtype))
        }
        else {
            Err(DiffuserError::custom(format!("weight `{}` not found, down load first", local.display())))
        }
    }
}
