use std::collections::BTreeMap;

/// https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py
#[derive(Debug, Default)]
pub struct WebUILayerMapping {
    unet_conversion_map: BTreeMap<String, String>,
    unet_conversion_map_resnet: BTreeMap<String, String>,
    unet_conversion_map_layer: BTreeMap<String, String>,
    vae_conversion_map: BTreeMap<String, String>,
    vae_conversion_map_attn: BTreeMap<String, String>,
}

impl WebUILayerMapping {
    pub fn initialize_unet(&mut self) {
        self.unet_conversion_map.insert("time_embed.0.weight".to_string(), "time_embedding.linear_1.weight".to_string());
        self.unet_conversion_map.insert("time_embed.0.bias".to_string(), "time_embedding.linear_1.bias".to_string());
        self.unet_conversion_map.insert("time_embed.2.weight".to_string(), "time_embedding.linear_2.weight".to_string());
        self.unet_conversion_map.insert("time_embed.2.bias".to_string(), "time_embedding.linear_2.bias".to_string());
        self.unet_conversion_map.insert("input_blocks.0.0.weight".to_string(), "conv_in.weight".to_string());
        self.unet_conversion_map.insert("input_blocks.0.0.bias".to_string(), "conv_in.bias".to_string());
        self.unet_conversion_map.insert("out.0.weight".to_string(), "conv_norm_out.weight".to_string());
        self.unet_conversion_map.insert("out.0.bias".to_string(), "conv_norm_out.bias".to_string());
        self.unet_conversion_map.insert("out.2.weight".to_string(), "conv_out.weight".to_string());
        self.unet_conversion_map.insert("out.2.bias".to_string(), "conv_out.bias".to_string());
        self.unet_conversion_map_resnet.insert("in_layers.0".to_string(), "norm1".to_string());
        self.unet_conversion_map_resnet.insert("in_layers.2".to_string(), "conv1".to_string());
        self.unet_conversion_map_resnet.insert("out_layers.0".to_string(), "norm2".to_string());
        self.unet_conversion_map_resnet.insert("out_layers.3".to_string(), "conv2".to_string());
        self.unet_conversion_map_resnet.insert("emb_layers.1".to_string(), "time_emb_proj".to_string());
        self.unet_conversion_map_resnet.insert("skip_connection".to_string(), "conv_shortcut".to_string());
        for i in 0..4 {
            for j in 0..2 {
                let hf_down_res_prefix = format!("down_blocks.{}.resnets.{}", i, j);
                let sd_down_res_prefix = format!("input_blocks.{}.0.", 3 * i + j + 1);
                self.unet_conversion_map_layer.insert(sd_down_res_prefix, hf_down_res_prefix);
                if i < 3 {
                    let hf_down_atn_prefix = format!("down_blocks.{}.attentions.{}", i, j);
                    let sd_down_atn_prefix = format!("input_blocks.{}.1.", 3 * i + j + 1);
                    self.unet_conversion_map_layer.insert(sd_down_atn_prefix, hf_down_atn_prefix);
                }
            }
            for j in 0..3 {
                let hf_up_res_prefix = format!("up_blocks.{}.resnets.{}", i, j);
                let sd_up_res_prefix = format!("output_blocks.{}.0.", 3 * i + j);
                self.unet_conversion_map_layer.insert(sd_up_res_prefix, hf_up_res_prefix);
                if i > 0 {
                    let hf_up_atn_prefix = format!("up_blocks.{}.attentions.{}", i, j);
                    let sd_up_atn_prefix = format!("output_blocks.{}.1.", 3 * i + j);
                    self.unet_conversion_map_layer.insert(sd_up_atn_prefix, hf_up_atn_prefix);
                }
            }
            if i < 3 {
                let hf_downsample_prefix = format!("down_blocks.{}.downsamplers.0.conv.", i);
                let sd_downsample_prefix = format!("input_blocks.{}.0.op.", 3 * (i + 1));
                self.unet_conversion_map_layer.insert(sd_downsample_prefix, hf_downsample_prefix);
                let hf_upsample_prefix = format!("up_blocks.{}.upsamplers.0.", i);
                let sd_upsample_prefix = format!("output_blocks.{}.{}", 3 * i + 2, if i == 0 { 1 } else { 2 });
                self.unet_conversion_map_layer.insert(sd_upsample_prefix, hf_upsample_prefix);
            }
        }
        let hf_mid_atn_prefix = format!("mid_block.attentions.0.");
        let sd_mid_atn_prefix = format!("middle_block.1.");
        self.unet_conversion_map_layer.insert(sd_mid_atn_prefix, hf_mid_atn_prefix);
        for j in 0..2 {
            let hf_mid_res_prefix = format!("mid_block.resnets.{}", j);
            let sd_mid_res_prefix = format!("middle_block.{}", 2 * j);
            self.unet_conversion_map_layer.insert(sd_mid_res_prefix, hf_mid_res_prefix);
        }
    }
    pub fn initialize_vae(&mut self) {
        self.vae_conversion_map.insert("nin_shortcut.weight".to_string(), "conv_shortcut.weight".to_string());
        self.vae_conversion_map.insert("nin_shortcut.bias".to_string(), "conv_shortcut.bias".to_string());
        self.vae_conversion_map.insert("norm_out.weight".to_string(), "conv_norm_out.weight".to_string());
        self.vae_conversion_map.insert("norm_out.bias".to_string(), "conv_norm_out.bias".to_string());
        self.vae_conversion_map_attn.insert("norm.".to_string(), "group_norm.".to_string());
        self.vae_conversion_map_attn.insert("q.".to_string(), "query.".to_string());
        self.vae_conversion_map_attn.insert("k.".to_string(), "key.".to_string());
        self.vae_conversion_map_attn.insert("v.".to_string(), "value.".to_string());
        self.vae_conversion_map_attn.insert("proj_out.".to_string(), "proj_attn.".to_string());
        for i in 0..4 {
            for j in 0..2 {
                let hf_down_prefix = format!("encoder.down_blocks.{}.resnets.{}", i, j);
                let sd_down_prefix = format!("encoder.down.{}.block.{}", i, j);
                self.vae_conversion_map.insert(sd_down_prefix, hf_down_prefix);
            }
            if i < 3 {
                let hf_downsample_prefix = format!("down_blocks.{}.downsamplers.0.", i);
                let sd_downsample_prefix = format!("down.{}.downsample.", i);
                self.vae_conversion_map.insert(sd_downsample_prefix, hf_downsample_prefix);
                let hf_upsample_prefix = format!("up_blocks.{}.upsamplers.0.", i);
                let sd_upsample_prefix = format!("up.{}.upsample.", 3 - i);
                self.vae_conversion_map.insert(sd_upsample_prefix, hf_upsample_prefix);
            }
            for j in 0..3 {
                let hf_up_prefix = format!("decoder.up_blocks.{}.resnets.{}", i, j);
                let sd_up_prefix = format!("decoder.up.{}.block.{}", 3 - i, j);
                self.vae_conversion_map.insert(sd_up_prefix, hf_up_prefix);
            }
        }
        for i in 0..2 {
            let hf_mid_res_prefix = format!("mid_block.resnets.{}", i);
            let sd_mid_res_prefix = format!("mid.block_{}", i + 1);
            self.vae_conversion_map.insert(sd_mid_res_prefix, hf_mid_res_prefix);
        }
    }

}

#[test]
fn test() {
    let mut mapping = WebUILayerMapping::default();
    mapping.initialize_unet();
    println!("{:#?}", mapping);
}