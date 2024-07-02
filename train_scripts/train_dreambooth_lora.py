import sys
from pathlib import Path
import gc
from transformers import T5EncoderModel

from diffusers import PixArtAlphaPipeline, PixArtSigmaPipeline, PixArtTransformer2DModel
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, DDPMScheduler
from diffusers.image_processor import PixArtImageProcessor
from diffusers import StableDiffusionPipeline
import torch
from os import walk
import tqdm
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from accelerate import init_empty_weights
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import gc
from copy import deepcopy
from torch import nn
from torch import optim
import torch.distributed as dist
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel
from lycoris import create_lycoris, LycorisNetwork, create_lycoris_from_weights
from lycoris.utils import merge
from torchvision.transforms import Resize
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_256_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
from datetime import datetime

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from scripts.diffusers_patches import pixart_sigma_init_patched_inputs
from diffusers import Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor
setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)

def find_closest_resolution(ratio, resolution):
    if resolution == 512:
        ratio = min(ASPECT_RATIO_512_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_512_BIN[ratio]
    elif resolution == 256:
        ratio = min(ASPECT_RATIO_256_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_256_BIN[ratio]
    elif resolution == 1024:
        ratio = min(ASPECT_RATIO_1024_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_1024_BIN[ratio]
    else:
        ratio = min(ASPECT_RATIO_2048_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_2048_BIN[ratio]

def extract_embeddings(embeddings : torch.Tensor, device='cuda', dtype = torch.float32):
    return embeddings[0].to(dtype=dtype, device=device), embeddings[1].to(device=device),\
            embeddings[2].to(dtype=dtype, device=device), embeddings[3].to(device=device)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def compute_t5_validation_embeds(pretrained_model_path : str, validation_prompt : str):
    encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder='text_encoder', device_map='auto', load_in_8bit=True)
    encoder.requires_grad_(False)
    pipe = PixArtSigmaPipeline.from_pretrained(pretrained_model_path, text_encoder=encoder, transformer=None)

    embeds = pipe.encode_prompt(prompt=validation_prompt, device=pipe.device)

    # flush everything
    del encoder
    del pipe
    flush()

    return embeds



def compute_t5_features(pretrained_model_path : str, captions_folder : str, class_id : str, unique_id : str, output_path : str, read_from_files=True):
    # load the T5 encoder only
    encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder='text_encoder', device_map='auto', load_in_8bit=True)
    encoder.requires_grad_(False)
    pipe = PixArtSigmaPipeline.from_pretrained(pretrained_model_path, text_encoder=encoder, transformer=None)

    # navigate through the folder
    captions_folder = Path(captions_folder)

    # create an instance and class folder
    class_folder = Path(output_path).joinpath(class_id).joinpath('embeddings')
    class_folder.mkdir(parents=True, exist_ok=True)
    instance_folder = Path(output_path).joinpath(unique_id).joinpath('embeddings')
    instance_folder.mkdir(parents=True, exist_ok=True)

    for (dirpath, dirnames, filenames) in walk(captions_folder):
        for filename in tqdm.tqdm(filenames, desc='Extracting T5 Features'):
            suffix = Path(filename).suffix

            if suffix != '.txt':
                continue

            filepath = Path(dirpath).joinpath(filename)

            if read_from_files:
                with open(filepath) as f:
                    class_caption = f.read()
            else:
                class_caption = class_id
            
            # encode the original prompt
            class_embeds = pipe.encode_prompt(prompt=class_caption, device=pipe.device)

            # encode the prompt with the unique id inserted before the class_id
            instance_caption = class_caption.replace(class_id, f'{unique_id} {class_id}')
            instance_embeds = pipe.encode_prompt(instance_caption, pipe.device)

            # save those in the output directory
            filename = Path(filename).stem
            class_embeds_path = class_folder.joinpath(f'{filename}.npy')
            torch.save(class_embeds, class_embeds_path.absolute())

            instance_embeds_path = instance_folder.joinpath(f'{filename}.npy')
            torch.save(instance_embeds, instance_embeds_path.absolute())

    # flush everything
    del encoder
    del pipe
    flush()

def compute_vae_features(pretrained_model_path : str, images_path : str, unique_id : str, class_id : str,
                         output_path : str, batch_size : int, num_repeats : int):
    # load the pipe but without the encoder
    pipe = PixArtSigmaPipeline.from_pretrained(pretrained_model_path, text_encoder=None, torch_dtype=torch.float16)
    vae = pipe.vae
    interpolation_scale = pipe.transformer.config.interpolation_scale
    check_point_resolution = 512 * interpolation_scale

    pipe = pipe.to('cuda')
    image_processor = pipe.image_processor

    images_path = Path(images_path)
    unique_path = Path(output_path).joinpath(unique_id).joinpath('latents')
    unique_path.mkdir(parents=True, exist_ok=True)

    class_latent_path = Path(output_path).joinpath(class_id).joinpath('latents')
    class_latent_path.mkdir(parents=True, exist_ok=True)
    class_embed_path = Path(output_path).joinpath(class_id).joinpath('embeddings')
    latent_num_channels = pipe.transformer.config.in_channels
    
    # unique image vae features
    for (dirpath, dirnames, filenames) in walk(images_path):
        for filename in tqdm.tqdm(filenames, desc='Extracting Unique VAE Features'):
            try:
                # unique image vae
                filepath = Path(dirpath).joinpath(filename)
                pil_image = Image.open(filepath)
                image = image_processor.pil_to_numpy(pil_image)
                pil_image.close()
                image = torch.tensor(image, device='cuda', dtype=torch.float16)
                image = torch.moveaxis(image, -1, 1)

                # find the closest ratio
                ratio = image.shape[2] / image.shape[3]
                resolution = find_closest_resolution(ratio, check_point_resolution)

                image_width = int(resolution[1])
                image_height = int(resolution[0])

                resize_transform = Resize((image_height, image_width))
                image = resize_transform(image)

                with torch.no_grad():
                    image = image_processor.preprocess(image)
                    latent = vae.encode(image).latent_dist.sample()
                    latent = latent * vae.config.scaling_factor
                    del image
                    flush()

                    # test if this is good
                    #latent = latent / pipe.vae.config.scaling_factor
                    #image = pipe.vae.decode(latent).sample
                    #image = image_processor.postprocess(image.cpu().detach())
                    #image[0].show()

                # save the unique latent in the same directory as with the unique_id embedding
                filename = filepath.stem
                filepath = Path(unique_path).joinpath(filename + '.lt')

                latent_width = latent.shape[-1] - latent.shape[-1] % 2
                latent_height = latent.shape[-2] - latent.shape[-2] % 2

                torch.save(latent[:, :, :latent_height, :latent_width], filepath)

            except Exception as e:
                print(e)

    flush()
    for (dirpath, dirnames, filenames) in walk(images_path):
        for filename in tqdm.tqdm(filenames, desc='Generating Class VAE Features'):
            try:
                filename = Path(filename).stem
                filepath = Path(unique_path).joinpath(filename + '.lt')
                latent = torch.load(filepath)
                latents = randn_tensor(latent.shape, device=latent.device, dtype=torch.float16)

                filename = Path(filename).stem
                embeds_path = class_embed_path.joinpath(filename + '.npy')
                embeds = torch.load(embeds_path)
                prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
                        extract_embeddings(embeds, dtype=torch.float16, device=latents.device)
                
                output_path = class_latent_path.joinpath(filename)
                output_path.mkdir(parents=True, exist_ok=True)
                for i in range(batch_size * num_repeats):
                    latent = pipe(negative_prompt=None,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        negative_prompt_attention_mask=negative_prompt_attention_mask,
                        num_images_per_prompt=1,
                        output_type='latent',
                        latents=latents)[0]
                    output_file_path = output_path.joinpath(f'{i}.lt')

                    # test if this is good
                    #latent = latent / pipe.vae.config.scaling_factor
                    #image = pipe.vae.decode(latent).sample
                    #image = image_processor.postprocess(image.cpu().detach())
                    #image[0].show()

                    torch.save(latent, output_file_path)
                    del latent
                    flush()
                del prompt_embeds
                del prompt_attention_mask
                del negative_prompt_embeds
                del negative_prompt_attention_mask
                del embeds
                del latents
                flush()

            except Exception as e:
                print(e)
        del pipe
        flush()
            
class DreamboothDataset(Dataset):
    def __init__(self, output_folder : str, class_id : str, unique_id : str, batch_size : int, num_repeats):
        super().__init__()
        self.class_folder = Path(output_folder).joinpath(class_id)
        self.id_folder = Path(output_folder).joinpath(unique_id)
        self.filenames = []
        self.batch_size = batch_size
        self.num_repeats = num_repeats

        id_latents_folder = self.id_folder.joinpath('latents')

        for (dirpath, dirnames, filenames) in walk(id_latents_folder):
            for filename in filenames:
                for i in range(batch_size * self.num_repeats):
                    self.filenames.append(Path(filename).stem)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # load the unique latent
        unique_latent_path = self.id_folder.joinpath('latents').joinpath(filename + '.lt')
        unique_latent = torch.load(unique_latent_path)[0]

        # load the unique embedding
        unique_embedding_path = self.id_folder.joinpath('embeddings').joinpath(filename + '.npy')
        unique_embeddings = list(torch.load(unique_embedding_path))
        for i in range(len(unique_embeddings)):
            unique_embeddings[i] = unique_embeddings[i][0]

        # load the class embeddings
        class_embedding_path = self.class_folder.joinpath('embeddings').joinpath(filename + '.npy')
        class_embeddings = list(torch.load(class_embedding_path))
        for i in range(len(unique_embeddings)):
            class_embeddings[i] = class_embeddings[i][0]

        # load the class latents
        class_latent_folder = self.class_folder.joinpath('latents').joinpath(filename)
        class_latents = torch.zeros((self.batch_size, unique_latent.shape[-3], unique_latent.shape[-2], unique_latent.shape[-1]))

        index = idx % (self.batch_size * self.num_repeats)
        class_latent_file = class_latent_folder.joinpath(f'{index}.lt')
        class_latents = torch.load(class_latent_file)[0]
        
        return unique_latent, unique_embeddings, class_embeddings, class_latents

#https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py
def generate_noise_preds(transformer : PixArtTransformer2DModel, latent_model_input, embeds, 
                     embeds_attn, timesteps, added_cond_kwargs, guidance_scale = 4.5):
    noise_pred = transformer(latent_model_input,
                                        encoder_hidden_states=embeds,
                                        encoder_attention_mask=embeds_attn,
                                        timestep=timesteps,
                                        added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]
    return noise_pred

def show_latent(latent, pretrained_model_path):
    # test if this is good
    pipe = PixArtSigmaPipeline.from_pretrained(pretrained_model_path, transformer=None, text_encoder=None).\
        to(device=latent.device, dtype=latent.dtype)
    vae = pipe.vae
    latent = latent / vae.config.scaling_factor
    image = vae.decode(latent).sample
    
    image_processor = pipe.image_processor
    image = image_processor.postprocess(image.cpu().detach())
    image[0].show()

def validation_and_save(pretrained_model_path : str, transformer, validation_prompt : str, val_embeds, generator, output_folder, epoch, logger, global_step):
    with torch.no_grad():
        if validation_prompt is not None:
            transformer_copy = deepcopy(transformer)
            transformer_copy.merge_and_unload()
            transformer_copy = transformer_copy.get_base_model()
            transformer_copy = transformer_copy.to(dtype=torch.float16)

            pipe = PixArtSigmaPipeline.from_pretrained(pretrained_model_path, transformer=transformer_copy, text_encoder=None, torch_dtype=torch.float16)
            pipe = pipe.to(transformer_copy.device)

            generator.manual_seed(0)
            val_prompt_embeds, val_prompt_attention_mask, val_negative_embeds, val_negative_prompt_attention_mask = \
                extract_embeddings(val_embeds, dtype=torch.float16, device=transformer_copy.device)
            latents = pipe(
                negative_prompt=None,
                prompt_embeds=val_prompt_embeds,
                negative_prompt_embeds=val_negative_embeds,
                prompt_attention_mask=val_prompt_attention_mask,
                negative_prompt_attention_mask=val_negative_prompt_attention_mask,
                num_images_per_prompt=1,
                output_type="latent",
                generator=generator
            ).images

            del pipe.transformer
            del transformer_copy
            flush()

            image = pipe.vae.decode(latents.to(dtype=torch.float16) / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pt")[0]
            logger.add_image(f'epoch{epoch}', image, global_step)

        filename = Path(output_folder).stem
        transformer.save_pretrained(f'{filename}{epoch}')

        del pipe
        flush()
def train_dreambooth(pretrained_model_path,
                     images_folder,
                     captions_folder,
                     class_id, 
                     unique_id, 
                     output_folder, 
                     validation_prompt, 
                     num_epochs : int, 
                     num_regularization_passes, 
                     learning_rate,
                     num_regulatization_images,
                     class_specific_ratio,
                     extract_t5_features : bool = True,
                     extract_vae_features : bool = True,
                     lora_rank : int = 16,
                     epochs_per_save : int = 1):

    generator = torch.Generator()

    date = datetime.now().timestamp()
    logs_dir = Path(output_folder).joinpath(f'logs{date}')
    logger = SummaryWriter(logs_dir)

    # class-specific prior factor
    class_specific_ratio = 1.0

    # load the T5 encoder and tokenizer for:
    #   - the original captions i.e. the one we will use for the regularisation images
    #   - the original captions with the unique label placed in front of the class word i.e. C0rn dog
    if extract_t5_features:
        compute_t5_features(pretrained_model_path, captions_folder, class_id, unique_id, output_folder, True)

    # extract the vae features of the original images
    if extract_vae_features:
        compute_vae_features(pretrained_model_path, images_folder, unique_id, class_id, output_folder, num_regulatization_images, num_regularization_passes)

    if validation_prompt is not None:
        val_embeds = compute_t5_validation_embeds(pretrained_model_path, validation_prompt)

    # now that we have the t5 features, we can train
    # we have one frozen transformer2d and one transform2d unfrozen
    transformer = Transformer2DModel.from_pretrained(pretrained_model_path, subfolder='transformer')
    resolution = transformer.config.interpolation_scale * 512
    transformer.training = True
    transformer.gradient_checkpointing = True
    
    lora_config = LoraConfig(
        r=lora_rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ],
        use_dora=False,
        use_rslora=False
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')

    dataset = DreamboothDataset(output_folder, class_id, unique_id, num_regulatization_images, num_regularization_passes)
    dataloader = DataLoader(dataset, batch_size=num_regulatization_images)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(transformer.parameters(), lr=learning_rate)

    dtype = torch.bfloat16

    accelerator = Accelerator(mixed_precision='bf16')
    
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    generator = torch.Generator()

    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": None}

    global_step = 0
    epoch_bar = tqdm.tqdm(range(num_epochs), desc='Epochs', total=num_epochs)
    for epoch in epoch_bar:
        if epoch % epochs_per_save == 0:
            validation_and_save(pretrained_model_path, transformer, validation_prompt, val_embeds, generator,
                                output_folder, epoch, logger, global_step)

        transformer.train()
        epoch_bar.set_description('Epochs', refresh=True)
        # iterate through the unique latents
        batch_bar = tqdm.tqdm(dataloader, total=len(dataloader))
        for batch in dataloader:
            device = transformer.base_model.model.device

            # load the unique latent
            unique_latent, unique_embeddings, class_embeddings, original_class_latents = batch
            unique_latent = unique_latent.to(dtype=dtype, device=device)
            original_class_latents = original_class_latents.to(dtype=dtype, device=device)

            # load the unique embedding
            unique_prompt_embeds, unique_prompt_attention_mask, unique_negative_prompt_embeds, unique_negative_prompt_attention_mask = \
                extract_embeddings(unique_embeddings, dtype=dtype, device=device)
            unique_embeds = unique_prompt_embeds
            unique_embeds_attn = unique_prompt_attention_mask

            # load the class embeddings
            class_prompt_embeds, class_prompt_attention_mask, class_negative_prompt_embeds, class_negative_prompt_attention_mask = \
                extract_embeddings(class_embeddings, dtype=dtype, device=device)
            class_embeds = class_prompt_embeds
            class_embeds_attn = class_prompt_attention_mask

            with torch.no_grad():
                timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
                timesteps = timestep.expand(num_regulatization_images)

                # new class-specifc latents
                noise_class = randn_tensor(original_class_latents.shape, device=device, dtype=dtype)
                noisy_class_latents = scheduler.add_noise(original_class_latents, noise_class, timesteps)
                latent_model_input = noisy_class_latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)
                class_model_input = latent_model_input.to(dtype=class_embeds.dtype, device=device)

                # new unique id-specific latents
                noise_id = randn_tensor(unique_latent.shape, device=original_class_latents.device, dtype=dtype)
                noisy_id_latents = scheduler.add_noise(unique_latent, noise_id, timesteps)
                latent_model_input = noisy_id_latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)
                unique_model_input = latent_model_input.to(dtype=unique_embeds.dtype, device=device)

                timesteps = timestep.expand(latent_model_input.shape[0])
                timesteps = timesteps.to(device=device)
            
            optimizer.zero_grad()
            transformer.zero_grad()
            
            #with accelerator.accumulate(lycoris_net):
            
            class_noise_pred = generate_noise_preds(transformer, class_model_input, 
                                                    class_embeds,
                                                    class_embeds_attn,
                                                    timesteps, added_cond_kwargs)
            id_noise_pred = generate_noise_preds(transformer, 
                                                 unique_model_input,
                                                 unique_embeds,
                                                 unique_embeds_attn,
                                                 timesteps, added_cond_kwargs)

            # the class-specifc prior preservation loss is MSE between new_class_latents and original_class_latents
            class_loss = loss_fn(class_noise_pred.to(dtype=noise_class.dtype), noise_class)

            # the reconstruction loss is the error between the generated class latents and the one from the original dataset
            reconstruct_loss = loss_fn(id_noise_pred.to(dtype=noise_id.dtype), noise_id)

            # total loss is a weighted sum of those two losses
            total_loss = reconstruct_loss + class_specific_ratio * class_loss
            accelerator.backward(total_loss)
            optimizer.step()

            batch_bar.set_description(f'loss={total_loss}, repeat', refresh=True)
            batch_bar.update(1)

            logger.add_scalar("loss", total_loss.detach().item(), global_step)
            global_step = global_step + 1
        epoch_bar.update(1)
        


    transformer.save_pretrained(Path(output_folder))
    logger.close()

if __name__ == '__main__':
    # load the transformer2d
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', required=True, type=str)
    parser.add_argument('--images_folder', required=True, type=str)
    parser.add_argument('--captions_folder', required=True, type=str)
    parser.add_argument('--class_id', required=True, type=str)
    parser.add_argument('--unique_id', required=True, type=str)
    parser.add_argument('--output_folder', required=True, type=str)
    parser.add_argument('--validation_prompt', required=False, type=str, default=None)
    parser.add_argument('--num_epochs', required=False, type=int, default=10)
    parser.add_argument('--num_regularization_passes', required=False, type=int, default=10)
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-3)
    parser.add_argument('--num_regularization_images', required=False, type=int, default=4)
    parser.add_argument('--class_specific_ratio', required=False, type=float, default=1.0)
    parser.add_argument('--extract_t5_features', action='store_true')
    parser.add_argument('--extract_vae_features', action='store_true')
    parser.add_argument('--lora_rank', required=False, type=int, default=16)
    parser.add_argument('--epochs_per_save', required=False, type=int, default=2)
    parser.add_argument('--transformer_output_path', required=False, type=str, default=None)
    parser.add_argument('--bypass_training', action='store_true')
    parser.add_argument('--merge_path', required=False, type=str, default=None)
    
    args = parser.parse_args()

    if args.bypass_training == False:
        train_dreambooth(args.pretrained_model_path,
                        args.images_folder,
                        args.captions_folder,
                        args.class_id,
                        args.unique_id,
                        args.output_folder,
                        args.validation_prompt,
                        args.num_epochs,
                        args.num_regularization_passes,
                        args.learning_rate,
                        args.num_regularization_images,
                        args.class_specific_ratio,
                        args.extract_t5_features,
                        args.extract_vae_features,
                        args.lora_rank,
                        args.epochs_per_save)

    # merge the lora to the transformer
    # as of writing this, this is the only way to load the lora model correctly!
    if args.transformer_output_path is not None:
        transformer = PixArtTransformer2DModel.from_pretrained(args.pretrained_model_path, subfolder='transformer')

        if args.merge_path == None:
            model = PeftModel.from_pretrained(transformer, Path(args.output_folder))
        else:
            model = PeftModel.from_pretrained(transformer, Path(args.merge_path))
        model.merge_and_unload()
        transformer = model.get_base_model()
        transformer.save_pretrained(args.transformer_output_path)
