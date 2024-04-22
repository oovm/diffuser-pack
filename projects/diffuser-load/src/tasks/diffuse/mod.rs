use std::fmt::Formatter;
use std::num::NonZeroUsize;
use candle_core::DType;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Visitor;
use serde::ser::SerializeStruct;
use crate::ModelVersion;

mod ser;
mod der;


#[derive(Debug)]
pub struct DiffuseTask {
    /// 图像的正向提示词, 强调要出现的特征
    pub positive_prompt: String,
    /// 图像的负向提示词, 不希望出现的特征
    pub negative_prompt: String,
    /// 图像的宽度, 需要是 32 的倍数
    pub width: usize,
    /// 图像的高度, 需要是 32 的倍数
    pub height: usize,
    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    pub sliced_attention_size: usize,
    /// The number of steps to run the diffusion for.
    pub n_steps: Option<usize>,
    /// The numbers of samples to generate simultaneously.
    pub batch_size: NonZeroUsize,
    pub sd_version: ModelVersion,
    /// Force to use f16
    pub use_f16: bool,
    /// Generate intermediary images at each step.
    pub intermediary_images: bool,
    pub guidance_scale: Option<f64>,
    pub img2img: Option<String>,
    /// The strength, indicates how much to transform the initial image. 
    /// The    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    pub img2img_strength: f64,
    /// The seed to use when generating random samples.
    pub seed: Option<u64>,
}

const NON_ZERO_ONE: NonZeroUsize = unsafe {
    NonZeroUsize::new_unchecked(1)
};

impl Default for DiffuseTask {
    fn default() -> Self {
        Self {
            positive_prompt: "".to_string(),
            negative_prompt: "".to_string(),
            height: 512,
            width: 512,
            sliced_attention_size: 0,
            n_steps: None,
            batch_size: NON_ZERO_ONE,
            sd_version: ModelVersion::V1_5 { vae: "standard-v1.5-clip.safetensors".to_string(), unet: "".to_string() },
            intermediary_images: false,
            use_f16: true,
            guidance_scale: None,
            img2img: None,
            img2img_strength: 0.0,
            seed: None,
        }
    }
}

