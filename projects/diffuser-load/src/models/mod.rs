use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use safetensors::Dtype;
use url::Url;

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
pub struct StableDiffusion {
    name: String,
    version: StableDiffusionVersion,
    clip1: WeightID,
    clip2: WeightID,
    vae: WeightID,
    unet: WeightID,
}

pub enum StableDiffusionVersion {}



pub enum WeightKind {
    Clip1,
    Clip2,
    VaeV15 {
        data: Dtype
    },
    VaeV21 {
        data: Dtype
    },
    VaeV21F32,
    UnetV15F32,
    UnetV21F32,
    UnetV21F16,
}

pub struct WeightInfo {
    id: WeightID,
    remote: Url,
    hash: Option<u64>,
}

pub struct WeightID {
    name: Arc<str>,
}


