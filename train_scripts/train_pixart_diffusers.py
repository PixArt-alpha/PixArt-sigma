from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel, DDPMScheduler
from transformers import T5EncoderModel
from pathlib import Path
import tqdm
from os import walk
import os
from copy import copy, deepcopy
import torch
from torch import nn
import torch.distributed as dist
import torch.distributed
from torch import optim
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_256_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from datasets import load_dataset
from io import BytesIO
from tqdm.contrib.concurrent import process_map
from torchvision.transforms import Resize
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import datasets
from accelerate import Accelerator
import random
import gc
from datetime import datetime
import argparse
import requests
from datetime import timedelta
from torch.multiprocessing import set_start_method

import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path))

#from train_pixart_diffusers_map import add_embeddings_columns

# vae 16 channels test
# pipe = PixArtSigmaPipeline.from_pretrained('frutiemax/TwistedReality-pixart-512ms', transformer=None, text_encoder=None, torch_dtype=torch.float32)
# #vae = AutoencoderKL.from_pretrained('ostris/vae-kl-f8-d16').to(device='cuda', dtype=torch.float32)
# vae = pipe.vae.to('cuda')
# image_processor = pipe.image_processor

# bw_image = Image.open('6757458677_1d1feb56c9_b.jpg')
# image = Image.new('RGB', bw_image.size)
# image.paste(bw_image)

# image = image_processor.pil_to_numpy(image)
# image = torch.tensor(image, device='cuda', dtype=torch.float32)
# image = torch.moveaxis(image, -1, 1)

# with torch.no_grad():
#     image = image_processor.preprocess(image)
#     latent = vae.encode(image).latent_dist.sample()
#     #latent = latent * vae.config.scaling_factor

#     #latent = latent / vae.config.scaling_factor
#     image = vae.decode(latent).sample
#     image = image_processor.postprocess(image.cpu().detach())
#     image[0].show()

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
    
class EmbeddingsFeaturesDataset(Dataset):
    def __init__(self, folder, hf_dataset : datasets.Dataset, embeddings_column : str, vae_features_column : str):
        self.embeddings_column = embeddings_column
        self.vae_features_column = vae_features_column
        self.files = []
        self.hf_dataset = None
        if hf_dataset == None:
            self.folder = Path(folder)
            self.embeddings_folder = self.folder.joinpath('embeddings')
            self.features_folder = self.folder.joinpath('features')

            for (dirpath, dirnames, filenames) in walk(self.embeddings_folder):
                for filename in filenames:
                    embedding_filepath = self.embeddings_folder.joinpath(filename)
                    features_filepath = self.features_folder.joinpath(Path(filename).stem + '.lat')

                    if os.path.exists(features_filepath):
                        self.files.append((embedding_filepath, features_filepath))
        else:
            self.hf_dataset = hf_dataset
            self.hf_dataset = self.hf_dataset.with_format('torch')
    
    def __len__(self):
        if self.hf_dataset == None:
            return len(self.files)
        else:
            return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        if self.hf_dataset == None:
            embeddings, features = self.files[idx]
            embeddings = torch.load(embeddings)
            features = torch.load(features)
            return embeddings, features
        else:
            item = self.hf_dataset[idx]

            # we need to expand the embeddings back to 300 tokens
            # and create a mask to keep the valid tokens
            #  [1, 19, 4096]           [1, 300, 4096]
            embeddings = item['t5_prompt_embeds']

            # create the mask
            mask_length = embeddings.shape[1]
            embeddings_mask = torch.zeros((1, 300))
            embeddings_mask[:, :mask_length] = 1

            # right pad the embeddings
            embeddings_target = torch.zeros((1, 300, 4096))
            embeddings_target[:, :mask_length, :] = embeddings

            features = item[self.vae_features_column]
            return embeddings_target, embeddings_mask, features

class BucketSampler(BatchSampler):
    def __init__(self, files, max_batch_size, hf_dataset : datasets.Dataset, embeddings_column : str, vae_features_column : str):
        # initialize the same seed for every process
        random.seed(0)

        self.hf_dataset = hf_dataset
        self.embeddings_column = embeddings_column
        self.vae_features_column = vae_features_column
        self.max_batch_size = max_batch_size
        self.batch_size = max_batch_size
        self.buckets = {}
        self.num_images = 0

        if self.hf_dataset == None:
            self.files = files

            # we need to sort the files so they belong to the same features size
            for i in tqdm.tqdm(range(len(files)), desc='Calculating buckets'):
                file = files[i]
                embeddings, features = file
                latent = torch.load(Path(features))

                width = latent.shape[-1]
                height = latent.shape[-2]

                if not (height, width) in self.buckets.keys():
                    self.buckets[(height, width)] = []
                self.buckets[(height, width)].append(i)
                self.num_images = self.num_images + 1
        
        else:
            self.hf_dataset = self.hf_dataset.with_format('torch')
            for i in tqdm.tqdm(range(len(self.hf_dataset)), desc='Calculating buckets'):
                item = self.hf_dataset[i]
                ratio = float(item['ratio'])

                if not ratio in self.buckets.keys():
                    self.buckets[ratio] = []
                self.buckets[ratio].append(i)
                self.num_images = self.num_images + 1

            print(f'There are {len(self.buckets.keys())} buckets')

    def __iter__(self):
        batch = []
        for key, value in self.buckets.items():
            # randomize the value order
            value_copy = copy(value)
            random.shuffle(value_copy)

            for idx in value_copy:
                batch.append(idx)

                if len(batch) == self.max_batch_size:
                    yield batch
                    batch = []
            if batch != []:
                # create a batch with randomly sampled indices
                while len(batch) != self.max_batch_size:
                    idx = random.choice(value_copy)
                    batch.append(idx)
                yield batch
                batch = []
    def __len__(self):
        return int(self.num_images / self.max_batch_size)
        
def flush():
    gc.collect()
    torch.cuda.empty_cache()

class FilesDataset(Dataset):
    def __init__(self, folder, extensions):
        self.folder = Path(folder)
        self.files = []

        for (dirpath, dirnames, filenames) in walk(self.folder):
            for filename in filenames:
                if not Path(filename).suffix in extensions:
                    continue
                self.files.append(Path(dirpath).joinpath(filename))
    
    def __len__(self):
        test = len(self.files)
        return test
    
    def __getitem__(self, idx):
        test = str(self.files[idx])
        return test
    
def validation_and_save(repository_path : str, transformer : PixArtTransformer2DModel, output_folder : str, logger : SummaryWriter,
                        global_step : int, logging_path, push_to_hub = False):
    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float16,
                                                transformer=transformer).to(device=transformer.device)
    dataset = FilesDataset(Path(output_folder).joinpath('validation_embeddings'), extensions=['.emb'])
    dataloader = DataLoader(dataset)

    index = 0
    for batch, embeddings in tqdm.tqdm(enumerate(dataloader)):
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = torch.load(embeddings[0])
        prompt_embeds = prompt_embeds.to(device=transformer.device)
        prompt_attention_mask = prompt_attention_mask.to(device=transformer.device)
        negative_embeds = negative_embeds.to(device=transformer.device)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=transformer.device)
        
        with torch.no_grad():
            latents = pipe(negative_prompt=None, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                           prompt_attention_mask=prompt_attention_mask, negative_prompt_attention_mask=negative_prompt_attention_mask, 
                           output_type='latent').images
            image = pipe.vae.decode(latents.to(torch.float16) / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type='pt')[0]
            logger.add_image(batch, image, global_step)
        del image
        del latents
        flush()
        index = index + 1

    del pipe

    # optionnaly push the pipeline to the hub
    if push_to_hub == True:
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=transformer, torch_dtype=torch.float16)
        pipe.push_to_hub(repository_path)
    
    flush()
    
    # save the transformer
    transformer.save_pretrained(Path(logging_path).joinpath(f'{global_step}'))
    
def train(output_folder : str, num_epochs : int, batch_size : int, repository_path : str, learning_rate : float, steps_per_validation : int, epochs_per_validation : int,
          blank_transformer : bool, hf_dataset : datasets.Dataset, embeddings_column : str, vae_features_column : str, push_transformer_to_hub : bool):
    dataset = EmbeddingsFeaturesDataset(output_folder, hf_dataset, embeddings_column, vae_features_column)
    sampler = BucketSampler(dataset.files, batch_size, hf_dataset, embeddings_column, vae_features_column)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # load the transformer
    transformer = PixArtTransformer2DModel.from_pretrained(repository_path, subfolder='transformer').to(device='cuda')
    if blank_transformer == True:
        transformer = PixArtTransformer2DModel.from_config(transformer.config)
    transformer.gradient_checkpointing = True
    transformer.training = True
    scheduler = DDPMScheduler.from_pretrained(repository_path, subfolder='scheduler')

    # load the arguments optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(transformer.parameters(), lr=learning_rate)

    # prepare the logging dir
    date = datetime.now().timestamp()
    logs_dir = Path(output_folder).joinpath(f'logs{date}')
    logger = SummaryWriter(logs_dir)

    resolution = transformer.config.interpolation_scale * 512
    # prepare multi-gpu training
    accelerator = Accelerator()
    transformer = transformer.to(accelerator.device)
    dataloader, transformer, optimizer = accelerator.prepare(dataloader, transformer, optimizer)
    #dataloader = accelerator.prepare(dataloader)
    #transformer = accelerator.prepare(transformer)
    #optimizer = accelerator.prepare(optimizer)
    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": None}

    global_step = 0
    dtype = accelerator.unwrap_model(transformer).dtype
    device = accelerator.unwrap_model(transformer).device
    for epoch in tqdm.tqdm(range(num_epochs)):
        if epoch % epochs_per_validation == 0 and accelerator.is_main_process:
            validation_and_save(repository_path, accelerator.unwrap_model(transformer), output_folder, logger, global_step, logs_dir)
        
        for embeddings, embeddings_mask, features in tqdm.tqdm(dataloader, desc='Batch:'):
            if global_step % steps_per_validation == 0 and accelerator.is_main_process:
                validation_and_save(repository_path, accelerator.unwrap_model(transformer), output_folder, logger, global_step, logs_dir)

            with torch.no_grad():
                # strip the second dimension
                embeddings = torch.squeeze(embeddings, dim=1).to(dtype=dtype)
                embeddings_mask = torch.squeeze(embeddings_mask, dim=1).to(dtype=dtype)
                batch_features = torch.squeeze(features, dim=1).to(dtype=dtype)
                batch_size = batch_features.shape[0]
    
                timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
                timesteps = timestep.expand(batch_size)
    
                noise = randn_tensor(batch_features.shape, device=device, dtype=dtype)
                noisy_latents = scheduler.add_noise(batch_features, noise, timesteps)
                latent_model_input = noisy_latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)

            with accelerator.accumulate(transformer):
                noise_pred = accelerator.unwrap_model(transformer)(latent_model_input,
                                        encoder_hidden_states=embeddings,
                                        encoder_attention_mask=embeddings_mask,
                                        timestep=timesteps,
                                        added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]
                loss = loss_fn(noise_pred, noise).to(dtype=dtype)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                transformer.zero_grad()

            # some logging statistics
            logger.add_scalar('loss', loss.detach().item(), global_step)
            global_step = global_step + 1
    
    # final save
    if accelerator.is_main_process:
        validation_and_save(repository_path, accelerator.unwrap_model(transformer), output_folder, logger, global_step, logs_dir, push_transformer_to_hub)
            

if __name__ == '__main__':
    # if on windows, try to set to gloo for distributed training
    try:
        if dist.is_available() and not dist.is_initialized():
            if os.name == 'nt':
                torch.distributed.init_process_group(backend='gloo')
            else:
                print('torch.distributed.init_process_group(backend=''nccl'')')
                # set to an extremely large timeout so it's not possible to get past the barrier
                torch.distributed.init_process_group(backend='nccl', timeout=timedelta(days=30))
    except:
        pass
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--repository_path', required=True, type=str)
    parser.add_argument('--output_folder', required=True, type=str)
    parser.add_argument('--batch_size', required=False, type=int, default=8)
    parser.add_argument('--num_epochs', required=False, type=int, default=5)
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-5)
    parser.add_argument('--steps_per_validation', required=False, type=int, default=500)
    parser.add_argument('--epochs_per_validation', required=False, type=int, default=3)
    parser.add_argument('--blank_transformer', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--push_transformer_to_hub', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dataset_path', required=False, type=str, default=None)
    parser.add_argument('--dataset_split', required=False, type=str, default='train')

    args = parser.parse_args()

    repository_path = args.repository_path
    dataset_path = args.dataset_path
    output_folder = args.output_folder
    push_transformer_to_hub = args.push_transformer_to_hub
    dataset_split = args.dataset_split

    dataset = load_dataset(dataset_path, split=dataset_split)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    steps_per_validation = args.steps_per_validation
    epochs_per_validation = args.epochs_per_validation
    blank_transformer = args.blank_transformer

    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
    interpolation_scale = pipe.transformer.config.interpolation_scale
    checkpoint_resolution = 512 * interpolation_scale
    del pipe
    flush()

    train(output_folder, num_epochs, batch_size, repository_path, learning_rate, steps_per_validation, epochs_per_validation, blank_transformer, dataset,
          ['t5_prompt_embeds', 't5_prompt_attention_mask', 't5_negative_embeds', 't5_negative_prompt_attention_mask'],
          f'vae_{checkpoint_resolution}px', push_transformer_to_hub)