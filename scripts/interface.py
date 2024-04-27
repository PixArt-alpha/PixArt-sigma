import argparse
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import os
import random
import torch
from torchvision.utils import save_image
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusers.models import AutoencoderKL
from tools.download import find_model
from datetime import datetime
from typing import List, Union
import gradio as gr
import numpy as np
from gradio.components import Textbox, Image
from transformers import T5EncoderModel, T5Tokenizer
import gc

from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from torchvision.utils import _log_api_usage_once, make_grid
from diffusion.data.datasets.utils import *
from asset.examples import examples
from diffusion.utils.dist_utils import flush


MAX_SEED = np.iinfo(np.int32).max


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024-MS.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--port', default=7788, type=int)

    return parser.parse_args()


@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs,) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.inference_mode()
def generate_img(prompt, sampler, sample_steps, scale, seed=0, randomize_seed=False):
    flush()
    gc.collect()
    torch.cuda.empty_cache()

    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)

    os.makedirs(f'output/demo/online_demo_prompts/', exist_ok=True)
    save_promt_path = f'output/demo/online_demo_prompts/tested_prompts{datetime.now().date()}.txt'
    with open(save_promt_path, 'a') as f:
        f.write(prompt + '\n')
    print(prompt)
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)      # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
    emb_masks = caption_token.attention_mask

    caption_embs = caption_embs[:, None]
    null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)
    # Sample images:
    if sampler == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                            cfg_scale=scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        diffusion = IDDPM(str(sample_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    elif sampler == 'dpm-solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    elif sampler == 'sa-solver':
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sa_solver.sample(
            S=sample_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=scale,
            model_kwargs=model_kwargs,
        )[0]

    samples = samples.to(weight_dtype)
    samples = vae.decode(samples / vae.config.scaling_factor).sample
    samples = resize_and_crop_tensor(samples, custom_hw[0,1], custom_hw[0,0])
    display_model_info = f'Model path: {args.model_path},\nBase image size: {args.image_size}, \nSampling Algo: {sampler}'
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), prompt_show, display_model_info, seed


if __name__ == '__main__':
    from diffusion.utils.logger import get_root_logger
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    assert args.image_size in [256, 512, 1024, 2048], \
        "We only provide pre-trained models for 256x256, 512x512, 1024x1024 and 2048x2048 resolutions."
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    weight_dtype = torch.float16
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    if args.image_size in [512, 1024, 2048]:
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)
    else:
        model = PixArt_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            model_max_length=max_sequence_length,
        ).to(device)
    state_dict = find_model(args.model_path)
    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')
    model.to(weight_dtype)
    model.eval()
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    if args.sdvae:
        # pixart-alpha vae link: https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
    else:
        # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
        vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder").to(device)

    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    title = f"""
        '' Unleashing your Creativity \n ''
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src='https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/logo-sigma.png' style='width: 400px; height: auto; margin-right: 10px;' />
            {args.image_size}px
        </div>
    """
    DESCRIPTION = f"""# PixArt-Sigma {args.image_size}px
            ## If PixArt-Sigma is helpful, please help to ⭐ the [Github Repo](https://github.com/PixArt-alpha/PixArt-sigma) and recommend it to your friends ��'
            #### [PixArt-Sigma {args.image_size}px](https://github.com/PixArt-alpha/PixArt-sigma) is a transformer-based text-to-image diffusion system trained on text embeddings from T5. This demo uses the [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma) checkpoint.
            #### English prompts ONLY; 提示词仅限英文
            """
    if not torch.cuda.is_available():
        DESCRIPTION += "\n<p>Running on CPU �� This demo does not work on CPU.</p>"

    demo = gr.Interface(
        fn=generate_img,
        inputs=[Textbox(label="Note: If you want to specify a aspect ratio or determine a customized height and width, "
                              "use --ar h:w (or --aspect_ratio h:w) or --hw h:w. If no aspect ratio or hw is given, all setting will be default.",
                        placeholder="Please enter your prompt. \n"),
                gr.Radio(
                    choices=["iddpm", "dpm-solver", "sa-solver"],
                    label=f"Sampler",
                    interactive=True,
                    value='dpm-solver',
                ),
                gr.Slider(
                    label='Sample Steps',
                    minimum=1,
                    maximum=100,
                    value=14,
                    step=1
                ),
                gr.Slider(
                    label='Guidance Scale',
                    minimum=0.1,
                    maximum=30.0,
                    value=4.5,
                    step=0.1
                ),
                gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                ),
                gr.Checkbox(label="Randomize seed", value=True),
                ],
        outputs=[Image(type="numpy", label="Img"),
                 Textbox(label="clean prompt"),
                 Textbox(label="model info"),
                 gr.Slider(label='seed')],
        title=title,
        description=DESCRIPTION,
        examples=examples
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, debug=True)