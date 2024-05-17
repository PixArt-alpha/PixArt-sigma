
from safetensors import safe_open
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
import pathlib
import argparse
import gc
import torch
import sys

from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from scripts.diffusers_patches import pixart_sigma_init_patched_inputs

interpolation_scale_sigma = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
ckpt_id = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    safetensors_file = pathlib.Path(args.safetensors_path)
    image_size = args.image_size

    setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    transformer = Transformer2DModel(
        sample_size=image_size // 8,
        num_layers=28,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        attention_bias=True,
        num_attention_heads=16,
        cross_attention_dim=1152,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
        interpolation_scale=interpolation_scale_sigma[image_size],
    ).to('cuda')

    state_dict = {}
    with safe_open(safetensors_file, framework='pt') as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    transformer.load_state_dict(state_dict, strict=True)

    transformer.save_pretrained(pathlib.Path.joinpath(pathlib.Path(args.output_folder), 'transformer'))
    scheduler = DPMSolverMultistepScheduler()
    vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae")
    tokenizer = T5Tokenizer.from_pretrained(ckpt_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder")

    pipe = PixArtAlphaPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder)
    pipe.save_config(pathlib.Path(args.output_folder))
    del pipe
    del transformer
    del scheduler
    del vae
    del tokenizer
    del text_encoder
    flush_memory()

    scheduler = DPMSolverMultistepScheduler()
    scheduler.save_pretrained(pathlib.Path.joinpath(pathlib.Path(args.output_folder), 'scheduler'))
    del scheduler
    flush_memory()

    vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae").to('cuda')
    vae.save_pretrained(pathlib.Path.joinpath(pathlib.Path(args.output_folder), 'vae'))
    del vae
    flush_memory()

    tokenizer = T5Tokenizer.from_pretrained(ckpt_id, subfolder="tokenizer")
    tokenizer.save_pretrained(pathlib.Path.joinpath(pathlib.Path(args.output_folder), 'tokenizer'))
    del tokenizer
    flush_memory()

    text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder")
    text_encoder.save_pretrained(pathlib.Path.joinpath(pathlib.Path(args.output_folder), 'text_encoder'))
    del text_encoder
    flush_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--safetensors_path', required=True, type=str, help='Path to the .safetensors file to convert to diffusers folder structure')
    parser.add_argument('--image_size', required=False, default=512, type=int, choices=[256, 512, 1024, 2048], help='Image size of pretrained model')
    parser.add_argument('--output_folder', required=True, type=str, help='Path to the output folder')
    parser.add_argument('--multistep', required=False, type=bool, default=True, help='Multistep option')
    args = parser.parse_args()
    main(args)