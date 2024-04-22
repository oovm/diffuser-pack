use super::*;
use serde::de::{MapAccess, Visitor};

impl<'de> Deserialize<'de> for ModelVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut visitor = DiffuserVisitor::default();
        deserializer.deserialize_map(&mut visitor)?;
        let version = visitor.version.to_ascii_lowercase();
        let model = match version.as_str() {
            "v1.5" | "v1_5" => {
                ModelVersion::V1_5 { vae: visitor.vae, unet: visitor.unet, clip: visitor.clip1, token: visitor.tokenizer1 }
            }
            "v2.1" | "v2_1" => {
                ModelVersion::V2_1 { vae: visitor.vae, unet: visitor.unet, clip: visitor.clip1, token: visitor.tokenizer1 }
            }
            "xl" => ModelVersion::XL {
                vae: visitor.vae,
                unet: visitor.unet,
                clip1: visitor.clip1,
                token1: visitor.tokenizer1,
                clip2: visitor.clip2,
                token2: visitor.tokenizer2,
            },
            "xl turbo" | "xl_turbo" => ModelVersion::XL_Turbo {
                vae: visitor.vae,
                unet: visitor.unet,
                clip1: visitor.clip1,
                token1: visitor.tokenizer1,
                clip2: visitor.clip2,
                token2: visitor.tokenizer2,
            },
            _ => Err(Error::custom(format!("不支持 {version} 版本, 必须是 ('v1.5', 'v2.1', 'XL', 'XL Turbo') 之一")))?,
        };
        Ok(model)
    }
}

#[derive(Default)]
struct DiffuserVisitor {
    version: String,
    tokenizer1: String,
    tokenizer2: String,
    clip1: String,
    clip2: String,
    vae: String,
    unet: String,
}

impl<'i, 'de> Visitor<'de> for &'i mut DiffuserVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        formatter.write_str("expect a object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.to_ascii_lowercase().as_str() {
                "version" => self.version = map.next_value()?,
                "tokenizer1" | "tokenizer" => self.tokenizer1 = map.next_value()?,
                "tokenizer2" => self.tokenizer2 = map.next_value()?,
                "clip1" | "clip" => self.clip1 = map.next_value()?,
                "clip2" => self.clip2 = map.next_value()?,
                "vae" => self.vae = map.next_value()?,
                "unet" => self.unet = map.next_value()?,
                _ => {}
            }
        }
        Ok(())
    }
}

impl ModelVersion {
    pub fn repo(&self) -> &'static str {
        match self {
            Self::XL { .. } => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 { .. } => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 { .. } => "runwayml/stable-diffusion-v1-5",
            Self::XL_Turbo { .. } => "stabilityai/sdxl-turbo",
        }
    }

    pub fn unet_file(&self, use_f16: bool) -> &'static str {
        if use_f16 { "unet/diffusion_pytorch_model.fp16.safetensors" } else { "unet/diffusion_pytorch_model.safetensors" }
    }

    pub fn vae_file(&self, use_f16: bool) -> &'static str {
        if use_f16 { "vae/diffusion_pytorch_model.fp16.safetensors" } else { "vae/diffusion_pytorch_model.safetensors" }
    }

    pub fn clip_file(&self, use_f16: bool) -> &'static str {
        if use_f16 { "text_encoder/model.fp16.safetensors" } else { "text_encoder/model.safetensors" }
    }

    pub fn clip2_file(&self, use_f16: bool) -> &'static str {
        if use_f16 { "text_encoder_2/model.fp16.safetensors" } else { "text_encoder_2/model.safetensors" }
    }
}
