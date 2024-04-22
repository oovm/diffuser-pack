use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use candle_transformers::models::stable_diffusion::ddim::{DDIMScheduler, DDIMSchedulerConfig};
use candle_transformers::models::stable_diffusion::euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use safetensors::Dtype;
use url::Url;
use crate::ModelVersion;
use candle_core::{Error, Result};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;


pub struct ModelInfo {
    id: ModelID,
    version: ModelVersion,

}

pub struct ModelID {
    name: Arc<String>,
}

pub struct DiffuserTask {
    pub model: ModelID,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: u64,
    pub prompt_positive: String,
    pub prompt_negative: String,
}

pub struct DiffuserTaskRunner {
    model: ModelInfo,
    config: DiffuserTask,
    vae: AutoEncoderKL,
}

impl DiffuserTask {
    pub fn build(&self, config: &ModelStorage) -> Result<DiffuserTaskRunner> {
        let model = ModelInfo { id: ModelID { name: Arc::new("".to_string()) }, version: () };
        let mut config = self.clone();

        model.version.adapt_vae();
        
        DiffuserTaskRunner {
            model: ModelInfo { id: ModelID { name: Arc::new("".to_string()) }, version: () },
            config: self,
            vae: (),
        }
    }
}

impl ModelInfo {
    pub fn build(&self, task: DiffuserTask) -> candle_core::Result<StableDiffusionConfig> {
        let config = self.version.adapt_config(0, task.width as usize, task.height as usize);


        let steps = self.version.adapt_steps(task.steps);
        let scheduler = config.build_scheduler(steps)?;
    }
}


pub struct ModelStorage {
    pub weights: BTreeMap<WeightID, WeightInfo>,
}

impl ModelStorage {
    pub fn get_local_weight(&self, id: WeightID) -> Result<PathBuf> {
        Err(Error::Msg("todo".to_string()))
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


