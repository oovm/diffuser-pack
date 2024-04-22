use anyhow::{Error as E, Result};
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, IndexOp, Module, Tensor, D,
};
use candle_transformers::models::{stable_diffusion, stable_diffusion::unet_2d::UNet2DConditionModel};
use serde::de::value::Error;

use stable_diffusion::StableDiffusionConfig;
use std::{
    env::current_dir,
    path::{Path, PathBuf},
};

use crate::helpers::detect_device;
use diffuser_error::{DiffuserError, ModelStorage, ModelVersion};
use stable_diffusion::vae::AutoEncoderKL;
use tokenizers::Tokenizer;

pub mod helpers;
mod tasks;

pub use crate::tasks::diffuse::DiffuseTask;

fn output_filename(basename: &str, sample_idx: usize, num_samples: usize, timestep_idx: Option<usize>) -> PathBuf {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    }
    else {
        basename.to_string()
    };
    let filepath = match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    };
    PathBuf::from(filepath)
}

#[allow(clippy::too_many_arguments)]
fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    batch_size: usize,
    final_image: &str,
    timestep_ids: Option<usize>,
) -> Result<(), DiffuserError> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..batch_size {
        let img = images.i(batch)?;
        let image_filename = output_filename(final_image, (batch_size * 0) + batch + 1, batch + 1, timestep_ids);
        save_image_raw(&img, &image_filename)?;
    }
    Ok(())
}

pub fn save_image_raw(img: &Tensor, path: &Path) -> Result<(), DiffuserError> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        Err(DiffuserError::custom("save_image expects an input of shape (3, height, width)"))?
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => Err(DiffuserError::custom(format!("error saving image {}", path.display())))?,
        };
    image.save(path).map_err(candle_core::Error::wrap)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings1(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: &Path,
    clip_weights: &Path,
    sd_config: &StableDiffusionConfig,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    let device = detect_device()?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer.encode(prompt, true).map_err(E::msg)?.get_ids().to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!("the prompt is too long, {} > max-tokens ({})", tokens.len(), sd_config.clip.max_position_embeddings)
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_config = &sd_config.clip;
    let text_model = stable_diffusion::build_clip_transformer(clip_config, clip_weights, &device, dtype)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer.encode(uncond_prompt, true).map_err(E::msg)?.get_ids().to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            anyhow::bail!(
                "the negative prompt is too long, {} > max-tokens ({})",
                uncond_tokens.len(),
                sd_config.clip.max_position_embeddings
            )
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    }
    else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings2(
    prompt: &str,
    uncond_prompt: &str,
    clip2: &Path,
    tokenizer2: &Path,
    sd_config: &StableDiffusionConfig,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    let device = detect_device()?;
    let tokenizer = Tokenizer::from_file(tokenizer2).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer.encode(prompt, true).map_err(E::msg)?.get_ids().to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!("the prompt is too long, {} > max-tokens ({})", tokens.len(), sd_config.clip.max_position_embeddings)
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_config = unsafe { sd_config.clip2.as_ref().unwrap_unchecked() };
    let text_model = stable_diffusion::build_clip_transformer(clip_config, clip2, &device, dtype)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer.encode(uncond_prompt, true).map_err(E::msg)?.get_ids().to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            anyhow::bail!(
                "the negative prompt is too long, {} > max-tokens ({})",
                uncond_tokens.len(),
                sd_config.clip.max_position_embeddings
            )
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    }
    else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn image_preprocess<T: AsRef<Path>>(path: T) -> Result<Tensor, DiffuserError> {
    let img = image::io::Reader::open(path)?.decode().map_err(DiffuserError::custom)?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(width as u32, height as u32, image::imageops::FilterType::CatmullRom);
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

pub struct DiffuseRunner {
    vae: AutoEncoderKL,
    unet: UNet2DConditionModel,
}

impl DiffuseRunner {
    pub fn load(model: &ModelVersion, sd: &StableDiffusionConfig, store: &ModelStorage) -> Result<Self, DiffuserError> {
        Ok(Self { vae: Self::load_vae(model, sd, store)?, unet: Self::load_unet(model, sd, store)? })
    }

    fn load_vae(
        model: &ModelVersion,
        sd: &StableDiffusionConfig,
        store: &ModelStorage,
    ) -> Result<AutoEncoderKL, DiffuserError> {
        tracing::info!("正在构建 auto encoder");
        let device = detect_device()?;
        let (vae, dt) = match model {
            ModelVersion::V1_5 { vae, .. } => store.load_weight(vae)?,
            ModelVersion::V2_1 { vae, .. } => store.load_weight(vae)?,
            ModelVersion::XL { vae, .. } => store.load_weight(vae)?,
            ModelVersion::XL_Turbo { vae, .. } => store.load_weight(vae)?,
        };
        Ok(sd.build_vae(vae, &device, dt)?)
    }
    fn load_unet(
        model: &ModelVersion,
        sd: &StableDiffusionConfig,
        store: &ModelStorage,
    ) -> Result<UNet2DConditionModel, DiffuserError> {
        tracing::info!("正在构建 UNet");
        let device = detect_device()?;
        let (unet, dt) = match model {
            ModelVersion::V1_5 { unet, .. } => store.load_weight(unet)?,
            ModelVersion::V2_1 { unet, .. } => store.load_weight(unet)?,
            ModelVersion::XL { unet, .. } => store.load_weight(unet)?,
            ModelVersion::XL_Turbo { unet, .. } => store.load_weight(unet)?,
        };
        Ok(sd.build_unet(unet, &device, 4, cfg!(feature = "flash"), dt)?)
    }
}

pub fn run(args: DiffuseTask, out: &Path) -> Result<(), DiffuserError> {
    let here = current_dir()?;
    println!("Running on {}", here.display());
    let DiffuseTask {
        positive_prompt: prompt,
        negative_prompt: uncond_prompt,
        height,
        width,
        n_steps,
        sliced_attention_size,
        // num_samples,
        batch_size,
        model,
        use_f16,
        guidance_scale,
        img2img,
        img2img_strength,
        seed,
        ..
    } = args;

    let store = ModelStorage::load(&here)?;

    let bsize = batch_size.get();
    let dtype = if use_f16 { DType::F16 } else { DType::F32 };

    let img2img_strength = img2img_strength.max(0.0).min(1.0);
    let sd_version = store.load_version(&model)?;
    let sd_config = match sd_version {
        ModelVersion::V1_5 { .. } => StableDiffusionConfig::v1_5(Some(sliced_attention_size), Some(height), Some(width)),
        ModelVersion::V2_1 { .. } => StableDiffusionConfig::v2_1(Some(sliced_attention_size), Some(height), Some(width)),
        ModelVersion::XL { .. } => StableDiffusionConfig::sdxl(Some(sliced_attention_size), Some(height), Some(width)),
        ModelVersion::XL_Turbo { .. } => {
            StableDiffusionConfig::sdxl_turbo(Some(sliced_attention_size), Some(height), Some(width))
        }
    };
    let n_steps = sd_version.adapt_steps(n_steps);
    let guidance_scale = sd_version.adapt_guidance_scale(guidance_scale);

    let scheduler = sd_config.build_scheduler(n_steps)?;
    let device = detect_device()?;
    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }
    let use_guide_scale = guidance_scale > 1.0;

    let mut embeddings = vec![
        text_embeddings1(
            &prompt,
            &uncond_prompt,
            &here.join("models").join("standard-v1.5-tokenizer.json"),
            &here.join("models").join("standard-v1.5-clip.safetensors"),
            &sd_config,
            dtype,
            use_guide_scale,
        )
        .map_err(|e| DiffuserError::custom(e.to_string()))?,
    ];
    match sd_version {
        ModelVersion::XL { .. } | ModelVersion::XL_Turbo { .. } => embeddings.push(
            text_embeddings2(
                &prompt,
                &uncond_prompt,
                &here.join("models").join("standard-v1.5-tokenizer.json"),
                &here.join("models").join("standard-v1.5-tokenizer.json"),
                &sd_config,
                dtype,
                use_guide_scale,
            )
            .map_err(|e| DiffuserError::custom(e.to_string()))?,
        ),
        _ => {}
    };
    let text_embeddings = Tensor::cat(&embeddings, D::Minus1)?;
    let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;
    println!("{text_embeddings:?}");

    let runner = DiffuseRunner::load(&sd_version, &sd_config, &store)?;

    tracing::info!("正在构建 auto encoder");
    let vae_scale = sd_version.adapt_vae_scale(None);
    let init_latent_dist = match &img2img {
        None => None,
        Some(image) => {
            let image = image_preprocess(image)?.to_device(&device)?;
            Some(runner.vae.encode(&image)?)
        }
    };
    tracing::info!("正在构建 unet");
    let t_start = if img2img.is_some() { n_steps - (n_steps as f64 * img2img_strength) as usize } else { 0 };

    let timesteps = scheduler.timesteps();
    let latents = match &init_latent_dist {
        Some(init_latent_dist) => {
            let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&device)?;
            if t_start < timesteps.len() {
                let noise = latents.randn_like(0f64, 1f64)?;
                scheduler.add_noise(&latents, noise, timesteps[t_start])?
            }
            else {
                latents
            }
        }
        None => {
            let latents = Tensor::randn(0f32, 1f32, (bsize, 4, sd_config.height / 8, sd_config.width / 8), &device)?;
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
        let latent_model_input = if use_guide_scale { Tensor::cat(&[&latents, &latents], 0)? } else { latents.clone() };

        let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
        let noise_pred = runner.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
        }
        else {
            noise_pred
        };

        latents = scheduler.step(&noise_pred, timestep, &latents)?;
        let dt = start_time.elapsed().as_secs_f32();
        println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

        if args.intermediary_images {
            save_image(&runner.vae, &latents, vae_scale, bsize, "test.png", Some(timestep_index + 1))?;
        }
    }

    println!("Generating the final image for sample",);
    save_image(&runner.vae, &latents, vae_scale, bsize, "test.png", None)?;
    Ok(())
}
