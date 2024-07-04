# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import time
import subprocess
import os
from cog import BasePredictor, Input, Path
import torch
from diffusers import Transformer2DModel
from scripts.diffusers_patches import (
    pixart_sigma_init_patched_inputs,
    PixArtSigmaPipeline,
)


"""
# load pipeline with the following then upload to replicate.delivery for fasting loading on Replicate        
pixart_sigma_transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    subfolder='transformer', 
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=pixart_sigma_transformer,
    use_safetensors=True,
)
pipe.save_pretrained("model-cache")
"""


PIPELINE_URL = "https://weights.replicate.delivery/default/pixart_sigma.tar"
PIPELINE_CACHE = "model-cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        setattr(
            Transformer2DModel, "_init_patched_inputs", pixart_sigma_init_patched_inputs
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(PIPELINE_CACHE):
            download_weights(PIPELINE_URL, PIPELINE_CACHE)

        self.pipe = PixArtSigmaPipeline.from_pretrained(PIPELINE_CACHE).to(device)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A small cactus with a happy face in the Sahara desert.",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            use_resolution_binning=True,
            output_type="pil",
        ).images[0]
        out_path = "/tmp/out.png"
        image.save(out_path)
        return Path(out_path)
