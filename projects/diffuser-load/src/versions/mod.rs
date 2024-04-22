use std::path::{Path, PathBuf};
use candle_core::{D, Device, DType, Error, Module, Tensor};
use candle_transformers::models::stable_diffusion::{build_clip_transformer, StableDiffusionConfig};
use candle_transformers::models::stable_diffusion::clip::Config;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use tokenizers::Tokenizer;
use crate::models::WeightID;

#[allow(non_camel_case_types)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ModelVersion {
    V1_5 {
        clip: WeightID,
        vae: WeightID,
        unet: WeightID,
    },
    V2_1 {
        clip: WeightID,
        vae: WeightID,
        unet: WeightID,
    },
    XL {
        clip1: WeightID,
        clip2: WeightID,
        vae: WeightID,
        unet: WeightID,
    },
    XL_Turbo {
        clip1: WeightID,
        clip2: WeightID,
        vae: WeightID,
        unet: WeightID,
    },
    XL_SSD1B {
        clip1: WeightID,
        clip2: WeightID,
        vae: WeightID,
        unet: WeightID,
    },
}

impl ModelVersion {
    pub fn adapt_config(&self, sliced_attention_size: usize, width: usize, height: usize) -> StableDiffusionConfig {
        match self {
            Self::V1_5 { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::V2_1 { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL_Turbo { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL_SSD1B { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
        }
    }
    pub fn adapt_steps(&self, steps: u32) -> usize {
        match steps {
            0 => {
                match self {
                    ModelVersion::V1_5 { .. } => { 30 }
                    ModelVersion::V2_1 { .. } => { 30 }
                    ModelVersion::XL { .. } => { 30 }
                    ModelVersion::XL_Turbo { .. } => { 1 }
                    ModelVersion::XL_SSD1B { .. } => { 30 }
                }
            }
            s => s as usize
        }
    }
    pub fn adapt_vae(&self, config: &mut StableDiffusionConfig) -> Result<AutoEncoderKL> {
        let device = select_device()?;
        let vae = match self {
            Self::V1_5 { clip, vae, unet } => {
                config.build_vae(vae)?
            }
            Self::V2_1 { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL_Turbo { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
            Self::XL_SSD1B { .. } => {
                StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(width), Some(height))
            }
        };
        Ok(vae)
    }
    
    pub fn adapt_token_id(&self, sd_config: &StableDiffusionConfig, prompt: String, weights: &Path) -> candle_core::Result<Vec<u32>> {
        println!("Building the autoencoder.");

        let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
        let init_latent_dist = match &img2img {
            None => None,
            Some(image) => {
                let image = image_preprocess(image)?.to_device(&device)?;
                Some(vae.encode(&image)?)
            }
        };
        println!("Building the unet.");
        let unet_weights = ModelFile::Unet.get(unet_weights, sd_version, use_f16)?;
        let unet = sd_config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;

        let t_start = if img2img.is_some() {
            n_steps - (n_steps as f64 * img2img_strength) as usize
        } else {
            0
        };

        let vae_scale = match sd_version {
            ModelVersion::V1_5
            | ModelVersion::V2_1
            | ModelVersion::Xl => 0.18215,
            ModelVersion::Turbo => 0.13025,
        };

        for idx in 0..num_samples {
            let timesteps = scheduler.timesteps();
            let latents = match &init_latent_dist {
                Some(init_latent_dist) => {
                    let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        scheduler.add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }
                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        (bsize, 4, sd_config.height / 8, sd_config.width / 8),
                        &device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(dtype)?;

            println!("starting sampling");
            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let start_time = std::time::Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
                let noise_pred =
                    unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                let dt = start_time.elapsed().as_secs_f32();
                println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

                if args.intermediary_images {
                    save_image(
                        &vae,
                        &latents,
                        vae_scale,
                        bsize,
                        idx,
                        &final_image,
                        num_samples,
                        Some(timestep_index + 1),
                    )?;
                }
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );
            save_image(
                &vae,
                &latents,
                vae_scale,
                bsize,
                idx,
                &final_image,
                num_samples,
                None,
            )?;
        }
    }
}

fn text_embeddings(positive: &str, negative: &str, tokenizer: &Path, clip_weights: &Path, sd_config: &StableDiffusionConfig, sd_version: ModelVersion, batch_size: usize) -> candle_core::Result<Tensor> {
    let which = match sd_version {
        ModelVersion::XL | ModelVersion::XL_Turbo | ModelVersion::XL_SSD1B => vec![
            text_embeddings_once(
                &positive,
                &negative,
                tokenizer.clone(),
                clip_weights.clone(),
                &sd_config,
                true,
            )?,
            text_embeddings_once(
                &positive,
                &negative,
                tokenizer.clone(),
                clip_weights.clone(),
                &sd_config,
                false,
            )?,
        ],
        _ => vec![
            text_embeddings_once(
                &positive,
                &negative,
                tokenizer.clone(),
                clip_weights.clone(),
                &sd_config,
                true,
            )?
        ],
    };
    let text_embeddings = Tensor::cat(&which, D::Minus1)?;
    text_embeddings.repeat((batch_size, 1, 1))
}


pub fn select_device() -> candle_core::Result<Device> {
    if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings_once(positive: &str, negative: &str, tokenizer: &Path, clip: &Path, sd: &StableDiffusionConfig, first: bool) -> candle_core::Result<Tensor> {
    let device = Device::cuda_if_available(0)?;
    let tokenizer = match Tokenizer::from_file(tokenizer) {
        Ok(o) => { o }
        Err(e) => { Err(Error::Msg(e.to_string()).with_path(tokenizer))? }
    };
    let pad_id = match &sd.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{positive}\".");
    let mut tokens = match tokenizer
        .encode(positive, true) {
        Ok(o) => { o }
        Err(e) => { Err(Error::Msg(e.to_string()))? }
    };
    let mut tokens = tokens.get_ids().to_vec();
    if tokens.len() > sd.clip.max_position_embeddings {
        // anyhow::bail!(
        //     "the prompt is too long, {} > max-tokens ({})",
        //     tokens.len(),
        //     sd_config.clip.max_position_embeddings
        // )
        panic!()
    }
    while tokens.len() < sd.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_config = if first {
        &sd.clip
    } else {
        unsafe {
            sd.clip2.as_ref().unwrap_unchecked()
        }
    };
    let text_model = build_clip_transformer(clip_config, clip, &device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let mut uncond_tokens = match tokenizer
        .encode(negative, true) {
        Ok(o) => { o }
        Err(e) => { Err(Error::Msg(e.to_string()))? }
    };
    let mut uncond_tokens = uncond_tokens
        .get_ids()
        .to_vec();
    if uncond_tokens.len() > sd.clip.max_position_embeddings {
        // anyhow::bail!(
        //         "the negative prompt is too long, {} > max-tokens ({})",
        //         uncond_tokens.len(),
        //         sd_config.clip.max_position_embeddings
        //     )
        panic!()
    }
    while uncond_tokens.len() < sd.clip.max_position_embeddings {
        uncond_tokens.push(pad_id)
    }

    let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let uncond_embeddings = text_model.forward(&uncond_tokens)?;

    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?;

    Ok(text_embeddings)
}