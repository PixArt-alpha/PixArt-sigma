#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys
import json
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import random
import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from tqdm import tqdm

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@torch.no_grad()
@torch.inference_mode()
def generate(items):

    seed = int(randomize_seed_fn(0, randomize_seed=False))
    generator = torch.Generator().manual_seed(seed)

    for item in tqdm(items, "Generating: "):
        prompt = item['prompt']
        save_name = item['path'].split('.')[0]

        # noise
        latent_size = pipe.transformer.config.sample_size
        noise = torch.randn(
            (1, 4, latent_size, latent_size), generator=generator, dtype=torch.float32
        ).to(weight_dtype).to(device)

        # image
        img_latent = pipe(
            prompt=prompt,
            latents=noise.to(weight_dtype),
            generator=generator,
            output_type="latent",
            max_sequence_length=T5_token_max_length,
        ).images[0]

        # save noise-denoised latent features
        noise_save_path = os.path.join(noise_save_dir, f"{save_name}.npy")
        np.save(noise_save_path, noise[0].cpu().numpy())
        img_latent_save_path = os.path.join(img_latent_save_dir, f"{save_name}.npy")
        np.save(img_latent_save_path, img_latent.cpu().numpy())

        if args.save_img:
            image = pipe.vae.decode(img_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil")
            img_save_path = os.path.join(img_save_dir, f"{save_name}.png")
            image.save(img_save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--save_img', action='store_true', help='if save latents and images at the same time')
    parser.add_argument('--sample_nums', default=640_000, type=int, help='sample numbers')
    parser.add_argument('--T5_token_max_length', default=120, type=int, choices=[120, 300], help='T5 token length')
    parser.add_argument(
        '--model_path', default="PixArt-alpha/PixArt-XL-2-512x512", help='the dir to load a ckpt for teacher model')
    parser.add_argument(
        '--pipeline_load_from', default="PixArt-alpha/PixArt-XL-2-1024-MS", type=str,
        help="Download for loading text_encoder, "
             "tokenizer and vae from https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS")
    args = parser.parse_args()
    return args


# Use PixArt-Alpha to generate PixArt-Alpha-DMD training data (noise-image pairs).
if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metadata_json_list = ["pixart-sigma-toy-dataset/InternData/data_info.json",]
    # dataset
    meta_data_clean = []
    for json_file in metadata_json_list:
        meta_data = json.load(open(json_file, 'r'))
        meta_data_clean.extend([item for item in meta_data if item['ratio'] <= 4.5])

    weight_dtype = torch.float16
    T5_token_max_length = args.T5_token_max_length
    if torch.cuda.is_available():

        # Teacher Model
        pipe = PixArtAlphaPipeline.from_pretrained(
            args.pipeline_load_from,
            transformer=None,
            torch_dtype=weight_dtype,
        )
        pipe.transformer = Transformer2DModel.from_pretrained(
            args.model_path, subfolder="transformer", torch_dtype=weight_dtype
        )
        pipe.to(device)

    print(f"INFO: Select only first {args.sample_nums} samples")
    meta_data_clean = meta_data_clean[:args.sample_nums]

    # save path
    if args.save_img:
        img_save_dir = os.path.join(f'pixart-sigma-toy-dataset/InternData/InternImgs_DMD_images')
        os.makedirs(img_save_dir, exist_ok=True)
    img_latent_save_dir = os.path.join(f'pixart-sigma-toy-dataset/InternData/InternImgs_DMD_latents')
    noise_save_dir = os.path.join(f'pixart-sigma-toy-dataset/InternData/InternImgs_DMD_noises')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_latent_save_dir, exist_ok=True)
    os.makedirs(noise_save_dir, exist_ok=True)

    # generate
    generate(meta_data_clean)


