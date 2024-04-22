use super::*;
use candle_core::DType;
mod der;
mod ser;

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelVersion {
    V1_5 { vae: String, unet: String },
    V2_1 { vae: String, unet: String },
    XL { vae: String, unet: String },
    XL_Turbo { vae: String, unet: String },
}

impl ModelVersion {
    pub fn adapt_steps(&self, steps: Option<usize>) -> usize {
        match steps {
            None | Some(0) => match self {
                ModelVersion::V1_5 { .. } => 30,
                ModelVersion::V2_1 { .. } => 30,
                ModelVersion::XL { .. } => 30,
                ModelVersion::XL_Turbo { .. } => 1,
            },
            Some(n_steps) => n_steps,
        }
    }
    pub fn adapt_vae_scale(&self, scale: Option<f64>) -> f64 {
        match scale {
            Some(vae) => vae,
            None => match self {
                Self::V1_5 { .. } => 0.18215,
                Self::V2_1 { .. } => 0.18215,
                Self::XL { .. } => 0.18215,
                Self::XL_Turbo { .. } => 0.13025,
            },
        }
    }
    pub fn adapt_guidance_scale(&self, scale: Option<f64>) -> f64 {
        match scale {
            Some(guidance) => guidance,
            None => match self {
                ModelVersion::V1_5 { .. } => 7.5,
                ModelVersion::V2_1 { .. } => 7.5,
                ModelVersion::XL { .. } => 7.5,
                ModelVersion::XL_Turbo { .. } => 0.,
            },
        }
    }
}
