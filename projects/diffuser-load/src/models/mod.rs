use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use candle_transformers::models::stable_diffusion::ddim::{DDIMScheduler, DDIMSchedulerConfig};
use candle_transformers::models::stable_diffusion::euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig;
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
    id: ModelID,
    version: StableDiffusionVersion,
    clip1: WeightID,
    clip2: WeightID,
    vae: WeightID,
    unet: WeightID,
}

pub struct ModelID {
    name: Arc<String>,
}

pub struct DiffuserTask {
    model: ModelID,
    width: u32,
    height: u32,
    scheduler: DiffuserScheduler,
    steps: u32,
    seed: u64,
}

pub enum DiffuserScheduler {
    DDIM(DDIMSchedulerConfig),
    EulerAncestralDiscrete(EulerAncestralDiscreteSchedulerConfig),
}

impl ModelInfo {
    pub fn build(&self, task: DiffuserTask) -> candle_core::Result<StableDiffusionConfig> {
        let config = self.version.build(0, task.width as usize, task.height as usize);

        config.build_vae();
        config.build_unet();


        let steps= self.version.adapt_steps(task.steps);
        let scheduler = config.build_scheduler(steps)?;
        
        
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


