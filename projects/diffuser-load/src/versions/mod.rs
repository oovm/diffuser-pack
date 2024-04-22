use std::path::{Path, PathBuf};
use candle_core::{Device, DType, Error, Module, Tensor};
use candle_transformers::models::stable_diffusion::{build_clip_transformer, StableDiffusionConfig};
use candle_transformers::models::stable_diffusion::clip::Config;
use tokenizers::Tokenizer;

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
    pub fn adapt_steps(&self, steps: u32) -> usize {
        match steps {
            0 => {
                match self {
                    StableDiffusionVersion::V1_5 => { 30 }
                    StableDiffusionVersion::V2_1 => { 30 }
                    StableDiffusionVersion::Xl => { 30 }
                    StableDiffusionVersion::XL_Turbo => { 1 }
                    StableDiffusionVersion::XL_SSD1B => { 30 }
                }
            }
            s => s as usize
        }
    }
    pub fn adapt_token_id(&self, clip: &Config, prompt: String, weights: &Path) -> candle_core::Result<Vec<u32>> {
        let tokenizer = match Tokenizer::from_file(weights) {
            Ok(o) => { o }
            Err(e) => {
                Err(Error::Msg(e.to_string()).with_path(weights))?
            }
        };
        let pad_id = match &clip.pad_with {
            Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
        };
        println!("Running with prompt \"{prompt}\".");
        let encoding = match tokenizer.encode(prompt, true) {
            Ok(o) => { o }
            Err(e) => {
                Err(Error::Msg(e.to_string()))?
            }
        };
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.len() > clip.max_position_embeddings {
            //     anyhow::bail!(
            //     "the prompt is too long, {} > max-tokens ({})",
            //     tokens.len(),
            //     sd_config.clip.max_position_embeddings
            // )
            panic!()
        }
        while tokens.len() < clip.max_position_embeddings {
            tokens.push(pad_id)
        }
        Ok(tokens)
    }
}

fn text_embeddings(prompt: &str, uncond_prompt: &str, tokenizer: &Path, clip_weights: &Path, sd_config: &StableDiffusionConfig) -> candle_core::Result<Tensor> {
    let which = match sd_version {
        StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
        _ => vec![true],
    };
    let text_embeddings = which
        .iter()
        .map(|first| {
            text_embeddings(
                &prompt,
                &uncond_prompt,
                tokenizer.clone(),
                clip_weights.clone(),
                sd_version,
                &sd_config,
                use_f16,
                &device,
                dtype,
                use_guide_scale,
                *first,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
    let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;
    println!("{text_embeddings:?}");
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings_once(prompt: &str, uncond_prompt: &str, tokenizer: &Path, clip_weights: &Path, sd_config: &StableDiffusionConfig, first: bool) -> candle_core::Result<Tensor> {
    let device = Device::cuda_if_available(0)?;
    let tokenizer = match Tokenizer::from_file(tokenizer) {
        Ok(o) => { o }
        Err(e) => { Err(Error::Msg(e.to_string()).with_path(tokenizer))? }
    };
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = match tokenizer
        .encode(prompt, true) {
        Ok(o) => { o }
        Err(e) => {Err(Error::Msg(e.to_string()))?}
    };
    let mut tokens = tokens.get_ids().to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        // anyhow::bail!(
        //     "the prompt is too long, {} > max-tokens ({})",
        //     tokens.len(),
        //     sd_config.clip.max_position_embeddings
        // )
        panic!()
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_config = if first {
        &sd_config.clip
    } else {
        unsafe {
            sd_config.clip2.as_ref().unwrap_unchecked()
        }
    };
    let text_model = build_clip_transformer(clip_config, clip_weights, &device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let mut uncond_tokens = match tokenizer
        .encode(uncond_prompt, true) {
        Ok(o) => {o}
        Err(e) => {Err(Error::Msg(e.to_string()))?}
    };
    let mut uncond_tokens = uncond_tokens
        .get_ids()
        .to_vec();
    if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
        // anyhow::bail!(
        //         "the negative prompt is too long, {} > max-tokens ({})",
        //         uncond_tokens.len(),
        //         sd_config.clip.max_position_embeddings
        //     )
        panic!()
    }
    while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
        uncond_tokens.push(pad_id)
    }

    let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let uncond_embeddings = text_model.forward(&uncond_tokens)?;

    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?;

    Ok(text_embeddings)
}