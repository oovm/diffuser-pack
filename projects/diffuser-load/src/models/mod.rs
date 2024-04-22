use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use safetensors::Dtype;
use url::Url;
use crate::StableDiffusionVersion;

pub struct StableDiffusionXLF32 {
    name: String,
    clip1: WeightInfo,
    clip2: WeightInfo,
    vae: WeightInfo,
    unet: WeightInfo,
}

pub struct StableDiffusionV21F32 {
    name: String,
    clip1: WeightInfo,
    clip2: WeightInfo,
    vae: WeightInfo,
    unet: WeightInfo,
}

pub struct ModelStorage {
    pub weights: BTreeMap<WeightID, WeightInfo>,
}


pub struct ModelInfo {
    name: String,
    version: StableDiffusionVersion,
    clip1: WeightID,
    clip2: WeightID,
    vae: WeightID,
    unet: WeightID,
}


pub struct DiffuserTask {
    width: u32,
    height: u32,
    steps: u32,
}

impl ModelInfo {
    pub fn build(&self, task: DiffuserTask) -> candle_core::Result<StableDiffusionConfig> {
        let config = self.version.build(0, task.width as usize, task.height as usize);
        
        config.build_vae();
        config.build_unet();
        let scheduler = config.build_scheduler(task.steps as usize)?;
        
        
        
    }
}


pub struct WeightInfo {
    id: WeightID,
    data: Dtype,
    remote: Url,
    hash: Option<u64>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeightID {
    name: Arc<str>,
}


