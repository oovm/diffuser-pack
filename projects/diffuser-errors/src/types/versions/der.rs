use super::*;


impl<'de> Deserialize<'de> for ModelVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        let text = String::deserialize(deserializer)?.to_ascii_lowercase();
        match text.as_str() {
            "v1.5" | "v1_5" => {
                Ok(ModelVersion::V1_5)
            }
            "v2.5" | "v2_1" => {
                Ok(ModelVersion::V2_1)
            }
            "xl"  => {
                Ok(ModelVersion::XL)
            }
            "xl turbo" | "xl_turbo" => {
                Ok(ModelVersion::XL_Turbo)
            }
            _ => Err(Error::custom("expect one of ('v1.5', 'v2.1', 'XL', 'XL Turbo')"))
        }
    }
}

impl ModelVersion {
    pub fn repo(&self) -> &'static str {
        match self {
            Self::XL => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::XL_Turbo => "stabilityai/sdxl-turbo",
        }
    }

    pub fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::XL | Self::XL_Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    pub fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::XL | Self::XL_Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    pub fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::XL | Self::XL_Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    pub fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::XL | Self::XL_Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}