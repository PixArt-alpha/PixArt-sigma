from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline, Transformer2DModel,DEISMultistepScheduler, DPMSolverMultistepScheduler
import torch
import gc
import argparse
import pathlib
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from scripts.diffusers_patches import pixart_sigma_init_patched_inputs

def main(args):
    setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)

    gc.collect()
    torch.cuda.empty_cache()

    repo_path = args.repo_path
    output_image = pathlib.Path(args.output)
    positive_prompt = args.positive_prompt
    negative_prompt = args.negative_prompt
    image_width = args.width
    image_height = args.height
    num_steps = args.num_steps
    guidance_scale = args.guidance_scale
    seed = args.seed
    low_vram = args.low_vram
    num_images = args.num_images
    scheduler_type = args.scheduler
    karras = args.karras
    algorithm_type = args.algorithm
    beta_schedule = args.beta_schedule
    use_lu_lambdas = args.use_lu_lambdas

    pipe = None
    if low_vram:
        print('low_vram')
        text_encoder = T5EncoderModel.from_pretrained(
            repo_path,
            subfolder="text_encoder",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        pipe = PixArtAlphaPipeline.from_pretrained(
            repo_path,
            text_encoder=text_encoder,
            transformer=None,
            torch_dtype=torch.float16,
        )

        with torch.no_grad():
            prompt = positive_prompt
            negative = negative_prompt
            prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt, negative_prompt=negative)

        def flush():
            gc.collect()
            torch.cuda.empty_cache()

        pipe.text_encoder = None
        del text_encoder
        flush()

        pipe.transformer = Transformer2DModel.from_pretrained(repo_path, subfolder='transformer',
                                                              load_in_8bit=True,
                                                              torch_dtype=torch.float16)
        pipe.to('cuda')
    else:
        print('low_vram=False')
        pipe = PixArtAlphaPipeline.from_pretrained(
            repo_path,
        ).to('cuda')

        with torch.no_grad():
            prompt = positive_prompt
            negative = negative_prompt
            prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt, negative_prompt=negative)

    generator = torch.Generator()
    
    if seed != -1:
        generator = generator.manual_seed(seed)
    else:
        generator = None

    prompt_embeds = prompt_embeds.to('cuda')
    negative_embeds = negative_embeds.to('cuda')
    prompt_attention_mask = prompt_attention_mask.to('cuda')
    negative_prompt_attention_mask = negative_prompt_attention_mask.to('cuda')
    
    if scheduler_type == 'deis':
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.scheduler.beta_schedule  = beta_schedule
    pipe.scheduler.algorithm_type = algorithm_type
    pipe.scheduler.use_karras_sigmas = karras
    pipe.scheduler.use_lu_lambdas = use_lu_lambdas
    latents = pipe(
        negative_prompt=None, 
        num_inference_steps=num_steps,
        height=image_height,
        width=image_width,
        prompt_embeds=prompt_embeds,
        guidance_scale=guidance_scale,
        negative_prompt_embeds=negative_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        num_images_per_prompt=num_images,
        output_type="latent",
        generator=generator,
    ).images

    words = str(output_image).split('.')
    filename = words[0]
    extension = words[1]

    with torch.no_grad():
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type="pil")

    i = 0
    for image in images:
        image.save(filename + str(i) + '.' + extension)
        i = i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', required=True, type=str, help='Local path or remote path to the pipeline folder')
    parser.add_argument('--output', required=False, type=str, default='out.png', help='Path to the generated output image. Supports most image formats i.e. .png, .jpg, .jpeg, .webp')
    parser.add_argument('--positive_prompt', required=True, type=str, help='Positive prompt to generate')
    parser.add_argument('--negative_prompt', required=False, type=str, default='', help='Negative prompt to generate')
    parser.add_argument('--seed', required=False, default=-1, type=int, help='Seed for the random generator')
    parser.add_argument('--width', required=False, default=512, type=int, help='Image width to generate')
    parser.add_argument('--height', required=False, default=512, type=int, help='Image height to generate')
    parser.add_argument('--num_steps', required=False, default=20, type=int, help='Number of inference steps')
    parser.add_argument('--guidance_scale', required=False, default=7.0, type=float, help='Guidance scale')
    parser.add_argument('--low_vram', required=False, action='store_true')
    parser.add_argument('--num_images', required=False, default=1, type=int, help='Number of images per prompt')
    parser.add_argument('--scheduler', required=False, default='dpm', type=str, choices=['dpm', 'deis'])
    parser.add_argument('--karras', required=False, action='store_true')
    parser.add_argument('--algorithm', required=False, default='sde-dpmsolver++', type=str, choices=['dpmsolver', 'dpmsolver++', 'sde-dpmsolver', 'sde-dpmsolver++'])
    parser.add_argument('--beta_schedule', required=False, default='linear', type=str, choices=['linear', 'scaled_linear', 'squaredcos_cap_v2'])
    parser.add_argument('--use_lu_lambdas', required=False, action='store_true')

    args = parser.parse_args()
    main(args)