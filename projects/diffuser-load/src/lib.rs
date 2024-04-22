use anyhow::{Error as E, Result};
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, IndexOp, Module, Tensor, D,
};
use candle_transformers::models::{stable_diffusion, stable_diffusion::unet_2d::UNet2DConditionModel};
use serde::de::value::Error;

use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use stable_diffusion::StableDiffusionConfig;
use std::{
    env::current_dir,
    path::{Path, PathBuf},
};

use crate::helpers::detect_device;
use diffuser_error::{DiffuserError, DiffuserTokenizer, ModelStorage, ModelVersion};
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
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    runner: &DiffuserTokenizer,
    sd_config: &StableDiffusionConfig,
    use_guide_scale: bool,
) -> Result<Tensor, DiffuserError> {
    let device = detect_device()?;
    let pad_id = runner.pad_token(sd_config);
    let mut positive_tokens = runner.encode_token(prompt)?;

    println!("Running with prompt \"{prompt}\".");
    if positive_tokens.len() > sd_config.clip.max_position_embeddings {
        tracing::warn!("提示词过多, 最多 {} 个, 当前 {} 个", sd_config.clip.max_position_embeddings, positive_tokens.len());
        positive_tokens.truncate(sd_config.clip.max_position_embeddings);

    }
    while positive_tokens.len() < sd_config.clip.max_position_embeddings {
        positive_tokens.push(pad_id)
    }
    let tokens = Tensor::new(positive_tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let text_embeddings = runner.clip.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut negative_token = runner.encode_token(uncond_prompt)?;
        if negative_token.len() > sd_config.clip.max_position_embeddings {
            tracing::warn!("提示词过多, 最多 {} 个, 当前 {} 个", sd_config.clip.max_position_embeddings, negative_token.len());
            positive_tokens.truncate(sd_config.clip.max_position_embeddings);
        }
        while negative_token.len() < sd_config.clip.max_position_embeddings {
            negative_token.push(pad_id)
        }

        let uncond_tokens = Tensor::new(negative_token.as_slice(), &device)?.unsqueeze(0)?;
        let uncond_embeddings = runner.clip.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(runner.data_type)?
    }
    else {
        text_embeddings.to_dtype(runner.data_type)?
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
    tokenizer1: DiffuserTokenizer,
    tokenizer2: Option<DiffuserTokenizer>,
}

impl DiffuseRunner {
    pub fn load(model: &ModelVersion, sd: &StableDiffusionConfig, store: &ModelStorage) -> Result<Self, DiffuserError> {
        Ok(Self {
            vae: store.load_vae(model, sd)?,
            unet: store.load_unet(model, sd)?,
            tokenizer1: store.load_tokenizer1(&model, sd)?,
            tokenizer2: store.load_tokenizer2(&model, &sd)?,
        })
    }
    pub fn text_embeddings(
        &self,
        positive: &str,
        negative: &str,
        sd: &StableDiffusionConfig,
        batch_size: usize,
        use_guide_scale: bool,
    ) -> Result<Tensor, DiffuserError> {
        let mut embedding1 = vec![text_embeddings(&positive, &negative, &self.tokenizer1, &sd, use_guide_scale)?];
        match self.tokenizer2.as_ref() {
            Some(s) => embedding1.push(text_embeddings(&positive, &negative, s, &sd, use_guide_scale)?),
            None => {}
        }
        let text_embeddings = Tensor::cat(&embedding1, D::Minus1)?;
        Ok(text_embeddings.repeat((batch_size, 1, 1))?)
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
        data_type,
        guidance_scale,
        img2img,
        img2img_strength,
        seed,
        ..
    } = args;

    let store = ModelStorage::load(&here)?;
    let sd_version = store.load_version(&model)?;
    let sd_config = store.load_config(&sd_version, sliced_attention_size, width, height);
    let runner = DiffuseRunner::load(&sd_version, &sd_config, &store)?;

    let batch_size = batch_size.get();
    let img2img_strength = img2img_strength.max(0.0).min(1.0);

    let n_steps = sd_version.adapt_steps(n_steps);
    let guidance_scale = sd_version.adapt_guidance_scale(guidance_scale);

    let scheduler = sd_config.build_scheduler(n_steps)?;
    let device = detect_device()?;
    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }
    let use_guide_scale = guidance_scale > 1.0;
    let text_embeddings = runner.text_embeddings(&prompt, &uncond_prompt, &sd_config, batch_size, use_guide_scale)?;
    println!("{text_embeddings:?}");

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
            let latents = Tensor::randn(0f32, 1f32, (batch_size, 4, sd_config.height / 8, sd_config.width / 8), &device)?;
            (latents * scheduler.init_noise_sigma())?
        }
    };
    let mut latents = latents.to_dtype(data_type)?;

    println!("开始采样");
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
            save_image(&runner.vae, &latents, vae_scale, batch_size, "test.png", Some(timestep_index + 1))?;
        }
    }

    println!("Generating the final image for sample",);
    save_image(&runner.vae, &latents, vae_scale, batch_size, "test.png", None)?;
    Ok(())
}
