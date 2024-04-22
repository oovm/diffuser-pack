use std::fmt::Formatter;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Visitor;
use serde::ser::SerializeStruct;
use crate::StableDiffusionVersion;

mod ser;
mod der;


#[derive(Debug)]
pub struct DiffuseTask {
    /// The prompt to be used for image generation.
    pub prompt_positive: String,
    pub prompt_negative: String,
    /// The height in pixels of the generated image.
    pub height: Option<usize>,
    /// The width in pixels of the generated image.
    pub width: Option<usize>,
    /// The UNet weight file, in `.safetensors` format.
    pub unet_weights: Option<String>,
    /// The CLIP weight file, in `.safetensors` format.
    pub clip_weights: Option<String>,
    /// The VAE weight file, in `.safetensors` format.
    pub vae_weights: Option<String>,
    /// The file specifying the tokenizer to used for tokenization.
    pub tokenizer: Option<String>,
    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    pub sliced_attention_size: Option<usize>,
    /// The number of steps to run the diffusion for.
    pub n_steps: Option<usize>,
    /// The number of samples to generate iteratively.
    pub num_samples: usize,
    /// The numbers of samples to generate simultaneously.
    pub batch_size: usize,
    /// The name of the final image to generate.
    pub final_image: String,
    pub sd_version: StableDiffusionVersion,
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

impl Default for DiffuseTask {
    fn default() -> Self {
        Self {
            prompt_positive: "".to_string(),
            prompt_negative: "".to_string(),
            height: None,
            width: None,
            unet_weights: None,
            clip_weights: None,
            vae_weights: None,
            tokenizer: None,
            sliced_attention_size: None,
            n_steps: None,
            num_samples: 0,
            batch_size: 0,
            final_image: "".to_string(),
            sd_version: StableDiffusionVersion::V1_5,
            intermediary_images: false,
            use_flash_attn: false,
            use_f16: false,
            guidance_scale: None,
            img2img: None,
            img2img_strength: 0.0,
            seed: None,
        }
    }
}
