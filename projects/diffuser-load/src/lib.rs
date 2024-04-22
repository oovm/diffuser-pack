use std::fmt::Formatter;
use candle_transformers::models::stable_diffusion;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_core::utils::{cuda_is_available, metal_is_available};

use stable_diffusion::vae::AutoEncoderKL;
use tokenizers::Tokenizer;

mod tasks;

pub use crate::tasks::diffuse::DiffuseTask;


pub struct DiffuserTaskRunner {
    /// The prompt to be used for image generation.
    pub prompt_positive: String,
    pub prompt_negative: String,
    /// The height in pixels of the generated image.
    pub height: usize,
    /// The width in pixels of the generated image.
    pub width: usize,
    /// The UNet weight file, in .safetensors format.
    pub unet_weights: Option<String>,
    /// The CLIP weight file, in .safetensors format.
    pub clip_weights: Option<String>,
    /// The VAE weight file, in .safetensors format.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    XL_Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::XL_Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::XL_Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::XL_Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::XL_Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::XL_Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl => {
                                "openai/clip-vit-large-patch14"
                            }
                            StableDiffusionVersion::XL_Turbo => {
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::XL_Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: usize,
    num_samples: usize,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> Result<()> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..bsize {
        let image = images.i(batch)?;
        let image_filename = output_filename(
            final_image,
            (bsize * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        save_image2(&image, image_filename)?;
    }
    Ok(())
}

pub fn save_image2<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> anyhow::Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        anyhow::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => anyhow::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle_core::Error::wrap)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        )
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
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

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

pub fn select_device() -> Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
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

pub fn run(args: DiffuseTask) -> Result<()> {
    let here = std::env::current_dir()?;
    println!("Running on {}", here.display());
    let DiffuseTask {
        prompt_positive: prompt,
        prompt_negative: uncond_prompt,
        height,
        width,
        n_steps,
        tokenizer,
        final_image,
        sliced_attention_size,
        num_samples,
        batch_size: bsize,
        sd_version,
        clip_weights,
        vae_weights,
        unet_weights,
        use_f16,
        guidance_scale,
        use_flash_attn,
        img2img,
        img2img_strength,
        seed,
        ..
    } = args;

    if !(0.0..=1.0).contains(&img2img_strength) {
        anyhow::bail!("img2img-strength should be between 0 and 1, got {img2img_strength}")
    }

    let guidance_scale = match guidance_scale {
        Some(guidance_scale) => guidance_scale,
        None => match sd_version {
            StableDiffusionVersion::V1_5 => 7.5,
            StableDiffusionVersion::V2_1 => 7.5,
            StableDiffusionVersion::Xl => 7.5,
            StableDiffusionVersion::XL_Turbo => 0.,
        },
    };
    let n_steps = match n_steps {
        Some(n_steps) => n_steps,
        None => match sd_version {
            StableDiffusionVersion::V1_5 => 30,
            StableDiffusionVersion::V2_1 => 30,
            StableDiffusionVersion::Xl => 30,
            StableDiffusionVersion::XL_Turbo => 1,
        },
    };
    let dtype = if use_f16 { DType::F16 } else { DType::F32 };
    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::Xl => {
            stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::XL_Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
            sliced_attention_size,
            height,
            width,
        ),
    };

    let scheduler = sd_config.build_scheduler(n_steps)?;
    let device = select_device()?;
    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }
    let use_guide_scale = guidance_scale > 1.0;

    let which = match sd_version {
        StableDiffusionVersion::Xl | StableDiffusionVersion::XL_Turbo => vec![true, false],
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

    println!("Building the autoencoder.");
    let vae_weights = ModelFile::Vae.get(vae_weights, sd_version, use_f16)?;
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
        StableDiffusionVersion::V1_5
        | StableDiffusionVersion::V2_1
        | StableDiffusionVersion::Xl => 0.18215,
        StableDiffusionVersion::XL_Turbo => 0.13025,
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
    Ok(())
}

