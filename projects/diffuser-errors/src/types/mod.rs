use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Display, Formatter},
    fs::{create_dir, read_to_string, File},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
};

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device,
};
use candle_transformers::models::{
    stable_diffusion,
    stable_diffusion::{clip::ClipTextTransformer, unet_2d::UNet2DConditionModel, vae::AutoEncoderKL, StableDiffusionConfig},
};
use serde::{
    de::{Error, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_yaml2::de::YamlDeserializer;
use tokenizers::Tokenizer;
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

pub struct DiffuserTokenizer {
    pub data_type: DType,
    tokenizer: Tokenizer,
    pub clip: ClipTextTransformer,
}

impl DiffuserTokenizer {
    pub fn pad_token(&self, config: &StableDiffusionConfig) -> u32 {
        let vocab = self.tokenizer.get_vocab(true);
        let token = match &config.clip.pad_with {
            Some(padding) => padding.as_str(),
            None => "<|endoftext|>",
        };
        match vocab.get(token) {
            Some(s) => *s,
            None => panic!("找不到填充 token `{}`", token),
        }
    }
    pub fn encode_token(&self, prompt: &str) -> Result<Vec<u32>, DiffuserError> {
        match self.tokenizer.encode(prompt, true) {
            Ok(o) => Ok(o.get_ids().to_vec()),
            Err(e) => Err(DiffuserError::custom(e)),
        }
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
            Err(DiffuserError::custom(format!(
                "权重文件 `{}` 不存在, 尝试使用 `diffuser download {}` 下载",
                local.display(),
                weight
            )))
        }
    }
    pub fn load_version(&self, model: &str) -> Result<&ModelVersion, DiffuserError> {
        match self.models.get(model) {
            Some(s) => Ok(s),
            None => Err(DiffuserError::custom(format!("找不到模型 `{}`", model)))?,
        }
    }
    pub fn load_config(
        &self,
        version: &ModelVersion,
        width: usize,
        height: usize,
    ) -> StableDiffusionConfig {
        match version {
            ModelVersion::V1_5 { .. } => StableDiffusionConfig::v1_5(None, Some(height), Some(width)),
            ModelVersion::V2_1 { .. } => StableDiffusionConfig::v2_1(None, Some(height), Some(width)),
            ModelVersion::XL { .. } => StableDiffusionConfig::sdxl(None, Some(height), Some(width)),
            ModelVersion::XL_Turbo { .. } => StableDiffusionConfig::sdxl_turbo(None, Some(height), Some(width)),
        }
    }

    pub fn load_tokenizer1(
        &self,
        model: &ModelVersion,
        sd: &StableDiffusionConfig,
    ) -> Result<DiffuserTokenizer, DiffuserError> {
        tracing::info!("正在构建 Tokenizer");
        let (tk, dt) = match model {
            ModelVersion::V1_5 { token, .. } => self.load_weight(token)?,
            ModelVersion::V2_1 { token, .. } => self.load_weight(token)?,
            ModelVersion::XL { token1, .. } => self.load_weight(token1)?,
            ModelVersion::XL_Turbo { token1, .. } => self.load_weight(token1)?,
        };
        let token = match Tokenizer::from_file(tk) {
            Ok(o) => o,
            Err(e) => Err(DiffuserError::custom(e))?,
        };
        Ok(DiffuserTokenizer { data_type: dt, tokenizer: token, clip: self.load_clip1(model, sd)? })
    }
    fn load_clip1(&self, model: &ModelVersion, sd: &StableDiffusionConfig) -> Result<ClipTextTransformer, DiffuserError> {
        tracing::info!("正在构建 Clip");
        let device = detect_device()?;
        let (clip, dt) = match model {
            ModelVersion::V1_5 { clip, .. } => self.load_weight(clip)?,
            ModelVersion::V2_1 { clip, .. } => self.load_weight(clip)?,
            ModelVersion::XL { clip1, .. } => self.load_weight(clip1)?,
            ModelVersion::XL_Turbo { clip1, .. } => self.load_weight(clip1)?,
        };
        Ok(stable_diffusion::build_clip_transformer(&sd.clip, clip, &device, dt)?)
    }

    pub fn load_tokenizer2(
        &self,
        model: &ModelVersion,
        sd: &StableDiffusionConfig,
    ) -> Result<Option<DiffuserTokenizer>, DiffuserError> {
        tracing::info!("正在构建 Tokenizer 2");
        let (tk, dt) = match model {
            ModelVersion::V1_5 { .. } => return Ok(None),
            ModelVersion::V2_1 { .. } => return Ok(None),
            ModelVersion::XL { token2, .. } => self.load_weight(token2)?,
            ModelVersion::XL_Turbo { token2, .. } => self.load_weight(token2)?,
        };
        let token = match Tokenizer::from_file(tk) {
            Ok(o) => o,
            Err(e) => Err(DiffuserError::custom(e))?,
        };
        Ok(Some(DiffuserTokenizer { data_type: dt, tokenizer: token, clip: self.load_clip2(model, sd)? }))
    }
    fn load_clip2(&self, model: &ModelVersion, sd: &StableDiffusionConfig) -> Result<ClipTextTransformer, DiffuserError> {
        tracing::info!("正在构建 UNet");
        let device = detect_device()?;
        let (clip, dt) = match model {
            ModelVersion::V1_5 { .. } => unreachable!(),
            ModelVersion::V2_1 { .. } => unreachable!(),
            ModelVersion::XL { clip2, .. } => self.load_weight(clip2)?,
            ModelVersion::XL_Turbo { clip2, .. } => self.load_weight(clip2)?,
        };
        let config = unsafe { sd.clip2.as_ref().unwrap_unchecked() };
        Ok(stable_diffusion::build_clip_transformer(&config, clip, &device, dt)?)
    }

    pub fn load_vae(&self, model: &ModelVersion, sd: &StableDiffusionConfig) -> Result<AutoEncoderKL, DiffuserError> {
        tracing::info!("正在构建 auto encoder");
        let device = detect_device()?;
        let (vae, dt) = match model {
            ModelVersion::V1_5 { vae, .. } => self.load_weight(vae)?,
            ModelVersion::V2_1 { vae, .. } => self.load_weight(vae)?,
            ModelVersion::XL { vae, .. } => self.load_weight(vae)?,
            ModelVersion::XL_Turbo { vae, .. } => self.load_weight(vae)?,
        };
        Ok(sd.build_vae(vae, &device, dt)?)
    }
    pub fn load_unet(&self, model: &ModelVersion, sd: &StableDiffusionConfig) -> Result<UNet2DConditionModel, DiffuserError> {
        tracing::info!("正在构建 UNet");
        let device = detect_device()?;
        let (unet, dt) = match model {
            ModelVersion::V1_5 { unet, .. } => self.load_weight(unet)?,
            ModelVersion::V2_1 { unet, .. } => self.load_weight(unet)?,
            ModelVersion::XL { unet, .. } => self.load_weight(unet)?,
            ModelVersion::XL_Turbo { unet, .. } => self.load_weight(unet)?,
        };
        Ok(sd.build_unet(unet, &device, 4, cfg!(feature = "flash"), dt)?)
    }
}

pub fn detect_device() -> candle_core::Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    }
    else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    }
    else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build this example with `--features metal`");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
