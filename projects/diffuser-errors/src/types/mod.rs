use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    fs::{create_dir, read_to_string, File},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
};

use candle_core::DType;
use serde::{
    de::{Error, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_yaml2::de::YamlDeserializer;
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
        let mut visitor = ModelStorageVisitor { place: ModelStorage::default() };
        deserializer.deserialize_map(&mut visitor)?;
        Ok(visitor.place)
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
    pub fn load<P>(root: P) -> Result<Self, DiffuserError>
    where
        P: AsRef<Path>,
    {
        let root = root.as_ref();
        let yaml = root.join("models").join("models.yaml");
        if !yaml.exists() {
            create_dir(root.join("images"))?;
            create_dir(root.join("models"))?;
            let mut new = File::create(&yaml)?;
            new.write_all(include_bytes!("weights/builtin.yaml"))?;
        }
        let storage = read_to_string(&yaml)?;
        let mut der = YamlDeserializer::from_str(&storage)?;
        Ok(ModelStorage::deserialize(&mut der)?)
    }
    pub fn load_weight(&self, weight: &str) -> Result<(PathBuf, DType), DiffuserError> {
        let (local, dt) = match self.weights.get(weight) {
            Some(s) => (self.root.join("models").join(&s.local), s.r#type),
            None => Err(DiffuserError::custom(format!("找不到权重 `{}`, 请确认拼写", weight)))?,
        };
        if local.exists() {
            Ok((local, dt))
        }
        else {
            Err(DiffuserError::custom(format!("权重文件 `{}` 不存在, 使用 `diffuser download {}` 下载", local.display(), weight)))
        }
    }
    pub fn load_version<'a>(&'a self, model: &str) -> Result<&'a ModelVersion, DiffuserError> {
        match self.models.get(model) {
            Some(s) => {Ok(s)}
            None => {
                Err(DiffuserError::custom(format!("找不到模型 `{}`", model)))?
            }
        }
    }
}
