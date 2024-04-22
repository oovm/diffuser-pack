use candle_transformers::models::stable_diffusion::StableDiffusionConfig;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    XL_Turbo,
    XL_SSD1B,
}

impl StableDiffusionVersion {
    pub fn build(&self, sliced_attention_size: usize, width: usize, height: usize) -> StableDiffusionConfig {
        match self {
            Self::V1_5 => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::V2_1 => {
                StableDiffusionConfig::v2_1(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::Xl => {
                StableDiffusionConfig::sdxl(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL_Turbo => {
                StableDiffusionConfig::sdxl_turbo(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL_SSD1B => {
                StableDiffusionConfig::ssd1b(Some(sliced_attention_size), Some(width), Some(height))
            }
        }
    }
}