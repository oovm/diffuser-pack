use std::fmt::Formatter;
use std::num::NonZeroUsize;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Visitor;
use serde::ser::SerializeStruct;
use crate::ModelVersion;

mod ser;
mod der;


#[derive(Debug)]
pub struct DiffuseTask {
    /// The prompt to be used for image generation.
    pub prompt_positive: String,
    pub prompt_negative: String,
    /// The height in pixels of the generated image.
    pub height: usize,
    /// The width in pixels of the generated image.
    pub width: usize,
    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    pub sliced_attention_size: Option<usize>,
    /// The number of steps to run the diffusion for.
    pub n_steps: Option<usize>,
    /// The numbers of samples to generate simultaneously.
    pub batch_size: NonZeroUsize,
    /// The name of the final image to generate.
    pub final_image: String,
    pub sd_version: ModelVersion,
    /// Generate intermediary images at each step.
    pub intermediary_images: bool,
    pub use_flash_attn: bool,
    pub use_f16: bool,
    pub guidance_scale: Option<f64>,
    pub img2img: Option<String>,
    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
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
            prompt_positive: "".to_string(),
            prompt_negative: "".to_string(),
            height: 512,
            width: 512,
            sliced_attention_size: None,
            n_steps: None,
            batch_size: NON_ZERO_ONE,
            final_image: "test.png".to_string(),
            sd_version: ModelVersion::V1_5,
            intermediary_images: false,
            use_flash_attn: false,
            use_f16: true,
            guidance_scale: None,
            img2img: None,
            img2img_strength: 0.0,
            seed: None,
        }
    }
}

